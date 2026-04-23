from __future__ import annotations

import argparse
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .metrics import MetricsTracker
from .ml_logic import ResolverEngine
from .schema import FeedbackRequest, PredictRequest, PredictResponse, SchemaError
import logging
#from aiohttp import web
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

STATUS_TEXT = {
    200: "OK",
    400: "Bad Request",
    404: "Not Found",
    405: "Method Not Allowed",
    408: "Request Timeout",
    413: "Payload Too Large",
    500: "Internal Server Error",
}


@dataclass(slots=True)
class QueuedPrediction:
    request: PredictRequest
    enqueued_at: float
    future: asyncio.Future[PredictResponse]
    expired: bool = False


class AsyncResolverServer:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        base_dir: str | Path | None = None,
        worker_count: int = 4,
        queue_size: int = 512,
        request_timeout: float = 0.10,
        max_request_age_seconds: float = 1.0,
        autosave_interval: float = 1.5,
        retrain_interval: float = 4.0,
    ) -> None:
        self.host = host
        self.port = port
        self.base_dir = Path(base_dir or Path.cwd())
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = MetricsTracker(
            self.base_dir / "resolver_metrics.json",
            self.base_dir / "resolver_metrics.jsonl",
        )
        self.engine = ResolverEngine(self.base_dir, self.metrics)
        self.worker_count = worker_count
        self.queue_size = queue_size
        self.request_timeout = request_timeout
        self.max_request_age_seconds = max_request_age_seconds
        self.autosave_interval = autosave_interval
        self.retrain_interval = retrain_interval
        self.queue: asyncio.PriorityQueue[tuple[Any, int, QueuedPrediction]] = asyncio.PriorityQueue(maxsize=queue_size)
        self.server: asyncio.AbstractServer | None = None
        self._ml_pool = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="resolver-ml")
        self._retrain_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="resolver-retrain")
        self._background_tasks: list[asyncio.Task[Any]] = []
        self._queue_counter = 0
        self._closed = False

    async def start(self) -> "AsyncResolverServer":
        self.engine.load_state()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._retrain_pool, self.engine.maybe_startup_warmup)
        self.server = await asyncio.start_server(self._handle_client, self.host, self.port)
        self._background_tasks.extend(asyncio.create_task(self._queue_worker(index)) for index in range(self.worker_count))
        self._background_tasks.append(asyncio.create_task(self._autosave_worker()))
        self._background_tasks.append(asyncio.create_task(self._autoretrain_worker()))
        return self

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

        await asyncio.to_thread(self.metrics.flush)
        await asyncio.to_thread(self.engine.close)
        self._ml_pool.shutdown(wait=True)
        self._retrain_pool.shutdown(wait=True)

    async def run_forever(self) -> None:
        await self.start()
        assert self.server is not None
        try:
            await self.server.serve_forever()
        finally:
            await self.close()

    def listening_port(self) -> int:
        if not self.server or not self.server.sockets:
            return self.port
        return int(self.server.sockets[0].getsockname()[1])

    def _next_queue_counter(self) -> int:
        self._queue_counter += 1
        return self._queue_counter

    def _queue_priority(self, request: PredictRequest) -> tuple[int, float, float]:
        state_rank = {"AIR": 0, "MOVING": 1, "STANDING": 2}.get(request.player_state, 3)
        distance = abs(request.derived_features[0] if request.derived_features else request.state_vector[17])
        return (state_rank, distance, request.timestamp)

    async def _queue_worker(self, worker_index: int) -> None:
        while True:
            _, _, queued = await self.queue.get()
            try:
                if queued.future.cancelled() or queued.expired:
                    continue

                age = time.time() - queued.request.timestamp
                if age > self.max_request_age_seconds:
                    print(
                        f"[Worker {worker_index}] Stale request: "
                        f"age={age:.3f}s, limit={self.max_request_age_seconds:.3f}s"
                    )
                    self.metrics.record_timeout("stale")
                    queue_latency_ms = (time.perf_counter() - queued.enqueued_at) * 1000.0
                    response = await asyncio.to_thread(
                        self.engine.register_fallback_prediction,
                        queued.request,
                        "stale",
                        queue_latency_ms=queue_latency_ms,
                    )
                else:
                    queue_latency_ms = (time.perf_counter() - queued.enqueued_at) * 1000.0
                    loop = asyncio.get_running_loop()
                    response = await loop.run_in_executor(self._ml_pool, self.engine.predict, queued.request, queue_latency_ms)

                if not queued.future.done():
                    queued.future.set_result(response)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not queued.future.done():
                    queued.future.set_exception(exc)
            finally:
                self.queue.task_done()

    async def _autosave_worker(self) -> None:
        while True:
            await asyncio.sleep(self.autosave_interval)
            await asyncio.to_thread(self.engine.persist_state)
            await asyncio.to_thread(self.metrics.flush)

    async def _autoretrain_worker(self) -> None:
        while True:
            await asyncio.sleep(self.retrain_interval)
            if self.engine.should_retrain():
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._retrain_pool, self.engine.retrain_from_storage)
    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            method, target, headers, body = await self._read_request(reader)
            path, query = self._split_target(target)
            # Лог входящего запроса в консоль
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {method} {path} from {writer.get_extra_info('peername')}")
            status, payload = await self._dispatch(method, path, query, headers, body)
        except SchemaError as exc:
            status, payload = 400, {"error": str(exc)}
        except asyncio.IncompleteReadError:
            status, payload = 400, {"error": "incomplete request"}
        except ValueError as exc:
            status, payload = 400, {"error": str(exc)}
        except Exception as exc:
            status, payload = 500, {"error": f"internal error: {exc}"}

        await self._send_json(writer, status, payload)
    async def _read_request(
        self,
        reader: asyncio.StreamReader,
    ) -> tuple[str, str, dict[str, str], bytes]:
        header_blob = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=1.0)
        if len(header_blob) > 65_536:
            raise ValueError("header too large")

        header_text = header_blob.decode("iso-8859-1")
        lines = header_text.split("\r\n")
        request_line = lines[0]
        method, target, _http_version = request_line.split(" ", 2)
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if not line:
                continue
            name, _, value = line.partition(":")
            headers[name.strip().lower()] = value.strip()

        content_length = int(headers.get("content-length", "0") or 0)
        if content_length > 1_000_000:
            raise ValueError("payload too large")
        body = await reader.readexactly(content_length) if content_length else b""
        return method.upper(), target, headers, body

    def _split_target(self, target: str) -> tuple[str, dict[str, list[str]]]:
        parsed = urlsplit(target)
        return parsed.path or "/", parse_qs(parsed.query, keep_blank_values=True)

    async def _dispatch(
        self,
        method: str,
        path: str,
        query: dict[str, list[str]],
        headers: dict[str, str],
        body: bytes,
    ) -> tuple[int, dict[str, Any]]:
        if path in {"/", "/predict"}:
            if method != "POST":
                return 405, {"error": "predict requires POST"}
            return await self._handle_predict(body)

        if path == "/feedback":
            if method != "POST":
                return 405, {"error": "feedback requires POST"}
            return await self._handle_feedback(body)

        if path == "/health":
            if method != "GET":
                return 405, {"error": "health requires GET"}
            return 200, self._health_payload()

        if path == "/metrics":
            if method != "GET":
                return 405, {"error": "metrics requires GET"}
            return 200, self._metrics_payload()

        if path == "/export":
            if method != "GET":
                return 405, {"error": "export requires GET"}
            limit = int((query.get("limit") or ["500"])[0] or 500)
            payload = await asyncio.to_thread(self.engine.export_training_data, limit)
            return 200, payload

        if path == "/retrain":
            if method != "POST":
                return 405, {"error": "retrain requires POST"}
            raw = self._decode_json(body) if body else {}
            limit = int(raw.get("limit", 256))
            loop = asyncio.get_running_loop()
            payload = await loop.run_in_executor(self._retrain_pool, self.engine.retrain_from_storage, limit)
            return 200, payload

        return 404, {"error": f"unknown path {path}"}

    def _decode_json(self, body: bytes) -> dict[str, Any]:
        if not body:
            return {}
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid json: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError("json body must be an object")
        return payload

    async def _handle_predict(self, body: bytes) -> tuple[int, dict[str, Any]]:
        request = PredictRequest.from_payload(self._decode_json(body))
        queued = QueuedPrediction(
            request=request,
            enqueued_at=time.perf_counter(),
            future=asyncio.get_running_loop().create_future(),
        )

        try:
            self.queue.put_nowait((self._queue_priority(request), self._next_queue_counter(), queued))
        except asyncio.QueueFull:
            self.metrics.record_timeout("queue_full")
            response = await asyncio.to_thread(
                self.engine.register_fallback_prediction,
                request,
                "queue_full",
                queue_latency_ms=0.0,
            )
            return 200, response.to_payload()

        try:
            response = await asyncio.wait_for(queued.future, timeout=self.request_timeout)
            return 200, response.to_payload()
        except asyncio.TimeoutError:
            queued.expired = True
            self.metrics.record_timeout("queue_timeout")
            response = await asyncio.to_thread(
                self.engine.register_fallback_prediction,
                request,
                "queue_timeout",
                queue_latency_ms=self.request_timeout * 1000.0,
            )
            return 200, response.to_payload()

    async def _handle_feedback(self, body: bytes) -> tuple[int, dict[str, Any]]:
        feedback = FeedbackRequest.from_payload(self._decode_json(body))
        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(self._ml_pool, self.engine.handle_feedback, feedback)
        return 200, payload

    def _health_payload(self) -> dict[str, Any]:
        payload = self.engine.health_snapshot()
        payload["queue_size"] = self.queue.qsize()
        payload["metrics"] = self.metrics.snapshot()
        return payload

    def _metrics_payload(self) -> dict[str, Any]:
        payload = self.metrics.snapshot()
        payload["queue_size"] = self.queue.qsize()
        payload["port"] = self.listening_port()
        return payload

    async def _send_json(self, writer: asyncio.StreamWriter, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        status_text = STATUS_TEXT.get(status, "OK")
        headers = [
            f"HTTP/1.1 {status} {status_text}",
            "Content-Type: application/json",
            f"Content-Length: {len(body)}",
            "Connection: close",
            "",
            "",
        ]
        writer.write("\r\n".join(headers).encode("ascii") + body)
        await writer.drain()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Async resolver server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    server = AsyncResolverServer(
        host=args.host,
        port=args.port,
        base_dir=args.base_dir,
        worker_count=args.workers,
    )
    asyncio.run(server.run_forever())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
