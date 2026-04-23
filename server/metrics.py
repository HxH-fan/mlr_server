from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from .schema import STRATEGIES, clamp, confidence_bucket


class MetricsTracker:
    def __init__(self, snapshot_path: Path, log_path: Path, *, window: int = 128) -> None:
        self.snapshot_path = Path(snapshot_path)
        self.log_path = Path(log_path)
        self.window = window
        self.lock = threading.RLock()
        self.started_at = time.time()
        self.predict_requests = 0
        self.feedback_events = 0
        self.queue_timeouts = 0
        self.stale_discards = 0
        self.retrain_runs = 0
        self.latencies_ms: deque[float] = deque(maxlen=window)
        self.confidence_histogram = {confidence_bucket(index / 5): 0 for index in range(5)}
        self.strategy_success = {
            strategy: {
                "shots": 0,
                "hits": 0,
                "latencies_ms": deque(maxlen=window),
            }
            for strategy in STRATEGIES
        }
        self._pending_log_entries: list[dict[str, Any]] = []

    def _append_log(self, kind: str, payload: dict[str, Any]) -> None:
        entry = {"ts": round(time.time(), 6), "kind": kind}
        entry.update(payload)
        self._pending_log_entries.append(entry)

    def record_prediction(self, strategy: str, confidence: float, latency_ms: float) -> None:
        with self.lock:
            self.predict_requests += 1
            self.latencies_ms.append(float(latency_ms))
            bucket = confidence_bucket(confidence)
            self.confidence_histogram[bucket] = self.confidence_histogram.get(bucket, 0) + 1
            stats = self.strategy_success.setdefault(
                strategy,
                {"shots": 0, "hits": 0, "latencies_ms": deque(maxlen=self.window)},
            )
            stats["latencies_ms"].append(float(latency_ms))
            self._append_log(
                "prediction",
                {
                    "strategy_used": strategy,
                    "confidence": round(clamp(confidence, 0.0, 1.0), 4),
                    "latency_ms": round(float(latency_ms), 3),
                },
            )

    def record_feedback(self, strategy: str, hit: bool) -> None:
        with self.lock:
            self.feedback_events += 1
            stats = self.strategy_success.setdefault(
                strategy,
                {"shots": 0, "hits": 0, "latencies_ms": deque(maxlen=self.window)},
            )
            stats["shots"] += 1
            if hit:
                stats["hits"] += 1
            self._append_log("feedback", {"strategy_used": strategy, "hit": bool(hit)})

    def record_timeout(self, reason: str) -> None:
        with self.lock:
            if reason == "stale":
                self.stale_discards += 1
            else:
                self.queue_timeouts += 1
            self._append_log("timeout", {"reason": reason})

    def record_retrain(self, samples: int) -> None:
        with self.lock:
            self.retrain_runs += 1
            self._append_log("retrain", {"samples": int(samples)})

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            overall_shots = sum(item["shots"] for item in self.strategy_success.values())
            overall_hits = sum(item["hits"] for item in self.strategy_success.values())
            overall_success_rate = (overall_hits / overall_shots) if overall_shots else 0.0

            return {
                "started_at": round(self.started_at, 3),
                "uptime_seconds": round(time.time() - self.started_at, 3),
                "predict_requests": self.predict_requests,
                "feedback_events": self.feedback_events,
                "queue_timeouts": self.queue_timeouts,
                "stale_discards": self.stale_discards,
                "retrain_runs": self.retrain_runs,
                "latency": {
                    "last_ms": round(self.latencies_ms[-1], 3) if self.latencies_ms else 0.0,
                    "avg_ms": round(sum(self.latencies_ms) / len(self.latencies_ms), 3) if self.latencies_ms else 0.0,
                    "samples": len(self.latencies_ms),
                },
                "success_rate": round(overall_success_rate, 4),
                "confidence_histogram": dict(self.confidence_histogram),
                "strategies": {
                    strategy: {
                        "shots": stats["shots"],
                        "hits": stats["hits"],
                        "success_rate": round((stats["hits"] / stats["shots"]) if stats["shots"] else 0.0, 4),
                        "avg_latency_ms": round(sum(stats["latencies_ms"]) / len(stats["latencies_ms"]), 3)
                        if stats["latencies_ms"]
                        else 0.0,
                    }
                    for strategy, stats in self.strategy_success.items()
                },
            }

    def flush(self) -> None:
        snapshot = self.snapshot()
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")

        with self.lock:
            entries = list(self._pending_log_entries)
            self._pending_log_entries.clear()

        if not entries:
            return

        with self.log_path.open("a", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=True, separators=(",", ":")))
                handle.write("\n")
