from __future__ import annotations

import shutil
import time
import unittest
import uuid
from pathlib import Path

from server.metrics import MetricsTracker
from server.ml_logic import ResolverEngine
from server.schema import FeedbackRequest, PredictRequest


def sample_request(player_id: int = 7, shot_id: int = 1, timestamp: float | None = None) -> PredictRequest:
    return PredictRequest.from_payload(
        {
            "player_id": player_id,
            "shot_id": shot_id,
            "timestamp": timestamp if timestamp is not None else time.time(),
            "state_vector": [
                0.1,
                -0.1,
                0.05,
                0.2,
                0.0,
                0.1,
                0.15,
                0.05,
                0.02,
                0.35,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.25,
                0.1,
                0.35,
            ],
            "history": [
                [0.0] * 18,
                [0.0] * 18,
                [0.0] * 18,
                [0.0] * 18,
                [0.05] * 18,
                [0.10] * 18,
                [0.12] * 18,
                [0.14] * 18,
            ],
            "player_state": "MOVING",
        }
    )


class ResolverEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch_root = Path(__file__).resolve().parent.parent / ".tmp-tests"
        scratch_root.mkdir(parents=True, exist_ok=True)
        self.base = scratch_root / f"case-{uuid.uuid4().hex}"
        self.base.mkdir(parents=True, exist_ok=True)
        metrics = MetricsTracker(self.base / "resolver_metrics.json", self.base / "resolver_metrics.jsonl")
        self.engine = ResolverEngine(self.base, metrics)

    def tearDown(self) -> None:
        self.engine.close()
        shutil.rmtree(self.base, ignore_errors=True)

    def test_request_derives_missing_features(self) -> None:
        request = sample_request()
        self.assertEqual(len(request.derived_features), 12)
        self.assertGreater(request.derived_features[0], 0.0)
        self.assertEqual(request.player_state, "MOVING")

    def test_strategy_prefers_brute_when_ml_metrics_are_bad(self) -> None:
        request = sample_request()
        player = self.engine.get_player(request.player_id)
        player.model_recent.extend([0.0] * 12)
        brute_stats = player.strategy_stats["BRUTE"]
        brute_stats.shots = 10
        brute_stats.hits = 8
        brute_stats.recent.extend([1.0] * 8)

        strategy, scores = self.engine.choose_strategy(
            player,
            {
                "ML": {"action": 0.0, "confidence": 0.32},
                "BRUTE": {"action": 30.0, "confidence": 0.61},
                "LAST": {"action": 0.0, "confidence": 0.25, "available": False},
                "FS": {"action": -30.0, "confidence": 0.28},
            },
        )

        self.assertEqual(strategy, "BRUTE")
        self.assertGreater(scores["BRUTE"], scores["ML"])

    def test_feedback_is_saved_and_retrain_reads_sqlite(self) -> None:
        request = sample_request()
        response = self.engine.predict(request, 3.5)
        feedback = FeedbackRequest.from_payload(
            {
                "player_id": request.player_id,
                "shot_id": request.shot_id,
                "hit": True,
                "latency_ms": 12.0,
                "strategy_used": response.strategy_used,
            }
        )
        result = self.engine.handle_feedback(feedback)
        self.assertEqual(result["status"], "ok")

        exported = self.engine.export_training_data(limit=10)
        self.assertEqual(exported["count"], 1)
        retrain = self.engine.retrain_from_storage(limit=10)
        self.assertEqual(retrain["status"], "ok")
        self.assertGreaterEqual(retrain["samples"], 1)


if __name__ == "__main__":
    unittest.main()
