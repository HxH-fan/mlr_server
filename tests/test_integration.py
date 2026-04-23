from __future__ import annotations

import asyncio
import json
import shutil
import time
import unittest
import urllib.request
import uuid
from pathlib import Path

from server.async_server import AsyncResolverServer


def integration_payload(shot_id: int = 1, timestamp: float | None = None) -> dict[str, object]:
    return {
        "player_id": 101,
        "shot_id": shot_id,
        "timestamp": timestamp if timestamp is not None else time.time(),
        "state_vector": [
            0.15,
            -0.05,
            0.03,
            0.20,
            0.10,
            0.00,
            0.12,
            0.02,
            0.01,
            0.25,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.20,
            0.10,
            0.30,
        ],
        "history": [[0.0] * 18 for _ in range(8)],
        "player_state": "MOVING",
    }


class AsyncServerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        scratch_root = Path(__file__).resolve().parent.parent / ".tmp-tests"
        scratch_root.mkdir(parents=True, exist_ok=True)
        self.base = scratch_root / f"case-{uuid.uuid4().hex}"
        self.base.mkdir(parents=True, exist_ok=True)
        self.server = AsyncResolverServer(
            host="127.0.0.1",
            port=0,
            base_dir=self.base,
            request_timeout=0.30,
            autosave_interval=0.20,
            retrain_interval=0.20,
        )
        await self.server.start()
        self.base_url = f"http://127.0.0.1:{self.server.listening_port()}"

    async def asyncTearDown(self) -> None:
        await self.server.close()
        shutil.rmtree(self.base, ignore_errors=True)

    def _sync_json_request(self, method: str, path: str, payload: dict[str, object] | None = None) -> tuple[int, dict[str, object]]:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=3) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body)

    async def json_request(self, method: str, path: str, payload: dict[str, object] | None = None) -> tuple[int, dict[str, object]]:
        return await asyncio.to_thread(self._sync_json_request, method, path, payload)

    async def test_predict_feedback_export_and_retrain(self) -> None:
        status, prediction = await self.json_request("POST", "/predict", integration_payload())
        self.assertEqual(status, 200)
        self.assertIn("predicted_action", prediction)
        self.assertIn("confidence", prediction)
        self.assertIn("strategy_used", prediction)

        feedback_payload = {
            "player_id": 101,
            "shot_id": 1,
            "hit": True,
            "latency_ms": 11.5,
            "strategy_used": prediction["strategy_used"],
        }
        status, feedback = await self.json_request("POST", "/feedback", feedback_payload)
        self.assertEqual(status, 200)
        self.assertEqual(feedback["status"], "ok")

        status, health = await self.json_request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["metrics"]["success_rate"], 1.0)

        status, exported = await self.json_request("GET", "/export?limit=10")
        self.assertEqual(status, 200)
        self.assertEqual(exported["count"], 1)

        status, retrain = await self.json_request("POST", "/retrain", {"limit": 10})
        self.assertEqual(status, 200)
        self.assertGreaterEqual(retrain["samples"], 1)

    async def test_stale_request_returns_non_ml_fallback(self) -> None:
        status, prediction = await self.json_request(
            "POST",
            "/predict",
            integration_payload(shot_id=2, timestamp=time.time() - 5.0),
        )
        self.assertEqual(status, 200)
        self.assertIn(prediction["strategy_used"], {"BRUTE", "LAST", "FS"})


if __name__ == "__main__":
    unittest.main()
