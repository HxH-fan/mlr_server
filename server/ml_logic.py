from __future__ import annotations

import json
import math
import pickle
import random
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .metrics import MetricsTracker
from .schema import (
    ACTIONS,
    BRUTE_ACTIONS,
    DERIVED_FEATURE_SIZE,
    HISTORY_LENGTH,
    PLAYER_STATES,
    STATE_INDEX_AIM_PITCH,
    STATE_INDEX_AIR,
    STATE_INDEX_AVG_DELTA,
    STATE_INDEX_DISTANCE,
    STATE_INDEX_LAST_DELTA,
    STATE_INDEX_RELATIVE_YAW,
    STATE_INDEX_SPEED,
    STATE_INDEX_VEL_Z,
    STATE_VECTOR_SIZE,
    STRATEGIES,
    FeedbackRequest,
    PredictRequest,
    PredictResponse,
    clamp,
    nearest_action,
)

RECENT_WINDOW = 16
REPLAY_CAPACITY = 4_096
MODEL_INPUT_SIZE = STATE_VECTOR_SIZE + DERIVED_FEATURE_SIZE
MODEL_HIDDEN_SIZE = 16
MIN_MODEL_ACCURACY = 0.38
PENDING_SHOT_TTL_SECONDS = 6.0
SUMMARY_VERSION = 4
RETRAIN_TRIGGER = 20
MIN_DYNAMIC_MODEL_ACCURACY = 0.22
ML_COLD_START_DECISIONS = 50
ML_COLD_START_MIN_CONFIDENCE = 0.18
ML_COLD_START_SCORE_MARGIN = 0.18
ML_UNCERTAINTY_BONUS_MAX = 0.12
ML_EXPLORATION_INTERVAL = 8
ML_EXPLORATION_MIN_CONFIDENCE = 0.25
ML_EXPLORATION_SCORE_MARGIN = 0.12
ML_EXPLORATION_MAX_SHOTS = 96
STARTUP_WARM_RETRAIN_LIMIT = 500
STARTUP_WARM_RETRAIN_MIN_ROWS = 50
STARTUP_WARM_RETRAIN_GROWTH = 50


def sigmoid(value: float) -> float:
    value = clamp(value, -40.0, 40.0)
    return 1.0 / (1.0 + math.exp(-value))


def softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    maximum = max(logits)
    exps = [math.exp(clamp(logit - maximum, -40.0, 40.0)) for logit in logits]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def recent_average(values: deque[float], default: float = 0.5) -> float:
    return (sum(values) / len(values)) if values else default


@dataclass(slots=True)
class ActionStats:
    shots: int = 0
    hits: int = 0
    recent: deque[float] = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))

    def record(self, hit: bool) -> None:
        self.shots += 1
        if hit:
            self.hits += 1
        self.recent.append(1.0 if hit else 0.0)

    @property
    def recent_success(self) -> float:
        return recent_average(self.recent, 0.5)

    @property
    def long_term_success(self) -> float:
        return (self.hits / self.shots) if self.shots else 0.5

    def to_snapshot(self) -> dict[str, Any]:
        return {"shots": self.shots, "hits": self.hits, "recent": list(self.recent)}

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> "ActionStats":
        item = cls()
        item.shots = int(payload.get("shots", 0))
        item.hits = int(payload.get("hits", 0))
        item.recent.extend(float(value) for value in list(payload.get("recent", []))[-RECENT_WINDOW:])
        return item


@dataclass(slots=True)
class StrategyStats:
    shots: int = 0
    hits: int = 0
    recent: deque[float] = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))

    def record(self, hit: bool, latency_ms: float) -> None:
        self.shots += 1
        if hit:
            self.hits += 1
        self.recent.append(1.0 if hit else 0.0)
        if latency_ms > 0:
            self.latencies_ms.append(float(latency_ms))

    @property
    def recent_success(self) -> float:
        return recent_average(self.recent, 0.5)

    @property
    def long_term_success(self) -> float:
        return (self.hits / self.shots) if self.shots else 0.5

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "shots": self.shots,
            "hits": self.hits,
            "recent": list(self.recent),
            "latencies_ms": list(self.latencies_ms),
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> "StrategyStats":
        item = cls()
        item.shots = int(payload.get("shots", 0))
        item.hits = int(payload.get("hits", 0))
        item.recent.extend(float(value) for value in list(payload.get("recent", []))[-RECENT_WINDOW:])
        item.latencies_ms.extend(float(value) for value in list(payload.get("latencies_ms", []))[-RECENT_WINDOW:])
        return item


@dataclass(slots=True)
class PendingShot:
    created_at_wall: float
    selected_action: float
    selected_confidence: float
    strategy_used: str
    hidden: list[float]
    request_payload: dict[str, Any]
    queue_latency_ms: float


@dataclass(slots=True)
class PlayerRecord:
    pending_shots: dict[int, PendingShot] = field(default_factory=dict)
    action_bias: dict[float, float] = field(default_factory=lambda: {action: 0.0 for action in ACTIONS})
    action_stats: dict[float, ActionStats] = field(default_factory=lambda: {action: ActionStats() for action in ACTIONS})
    strategy_stats: dict[str, StrategyStats] = field(default_factory=lambda: {strategy: StrategyStats() for strategy in STRATEGIES})
    last_action: float = 0.0
    last_success_action: float | None = None
    last_strategy: str = "ML"
    brute_cursor: int = 0
    recent_results: deque[float] = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))
    model_recent: deque[float] = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))
    decision_count: int = 0
    telemetry: dict[str, int] = field(
        default_factory=lambda: {
            "cache_hits": 0,
            "cache_misses": 0,
            "fallback_hits": 0,
            "fallback_misses": 0,
        }
    )

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "action_bias": {str(int(action)): value for action, value in self.action_bias.items()},
            "action_stats": {str(int(action)): stats.to_snapshot() for action, stats in self.action_stats.items()},
            "strategy_stats": {strategy: stats.to_snapshot() for strategy, stats in self.strategy_stats.items()},
            "last_action": self.last_action,
            "last_success_action": self.last_success_action,
            "last_strategy": self.last_strategy,
            "brute_cursor": self.brute_cursor,
            "recent_results": list(self.recent_results),
            "model_recent": list(self.model_recent),
            "decision_count": self.decision_count,
            "telemetry": dict(self.telemetry),
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> "PlayerRecord":
        item = cls()
        for action in ACTIONS:
            item.action_bias[action] = float((payload.get("action_bias") or {}).get(str(int(action)), 0.0))

        for action in ACTIONS:
            raw = (payload.get("action_stats") or {}).get(str(int(action)), {})
            item.action_stats[action] = ActionStats.from_snapshot(raw)

        for strategy in STRATEGIES:
            raw = (payload.get("strategy_stats") or {}).get(strategy, {})
            item.strategy_stats[strategy] = StrategyStats.from_snapshot(raw)

        item.last_action = float(payload.get("last_action", 0.0))
        last_success = payload.get("last_success_action")
        item.last_success_action = None if last_success is None else float(last_success)
        item.last_strategy = str(payload.get("last_strategy", "ML"))
        item.brute_cursor = int(payload.get("brute_cursor", 0))
        item.recent_results.extend(float(value) for value in list(payload.get("recent_results", []))[-RECENT_WINDOW:])
        item.model_recent.extend(float(value) for value in list(payload.get("model_recent", []))[-RECENT_WINDOW:])
        item.decision_count = int(payload.get("decision_count", 0))
        item.telemetry.update(payload.get("telemetry", {}))
        return item


class OnlineGRUModel:
    def __init__(self, input_size: int, hidden_size: int, actions: list[float]) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.actions = list(actions)
        rng = random.Random(1337)
        self.learning_rate = 0.035
        self.Wz = self._matrix(hidden_size, input_size, rng, 0.22)
        self.Uz = self._matrix(hidden_size, hidden_size, rng, 0.22)
        self.bz = [0.0] * hidden_size
        self.Wr = self._matrix(hidden_size, input_size, rng, 0.22)
        self.Ur = self._matrix(hidden_size, hidden_size, rng, 0.22)
        self.br = [0.0] * hidden_size
        self.Wh = self._matrix(hidden_size, input_size, rng, 0.22)
        self.Uh = self._matrix(hidden_size, hidden_size, rng, 0.22)
        self.bh = [0.0] * hidden_size
        self.Wo = self._matrix(len(actions), hidden_size, rng, 0.28)
        self.bo = [0.0] * len(actions)

    @staticmethod
    def _matrix(rows: int, cols: int, rng: random.Random, scale: float) -> list[list[float]]:
        return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

    def encode(self, sequence: list[list[float]]) -> list[float]:
        hidden = [0.0] * self.hidden_size
        for row in sequence:
            x = list(row[: self.input_size])
            if len(x) < self.input_size:
                x.extend([0.0] * (self.input_size - len(x)))
            z = [sigmoid(dot(self.Wz[index], x) + dot(self.Uz[index], hidden) + self.bz[index]) for index in range(self.hidden_size)]
            r = [sigmoid(dot(self.Wr[index], x) + dot(self.Ur[index], hidden) + self.br[index]) for index in range(self.hidden_size)]
            gated_hidden = [r[index] * hidden[index] for index in range(self.hidden_size)]
            proposal = [
                math.tanh(dot(self.Wh[index], x) + dot(self.Uh[index], gated_hidden) + self.bh[index])
                for index in range(self.hidden_size)
            ]
            hidden = [(1.0 - z[index]) * hidden[index] + z[index] * proposal[index] for index in range(self.hidden_size)]
        return hidden

    def logits(self, hidden: list[float]) -> list[float]:
        return [dot(weights, hidden) + bias for weights, bias in zip(self.Wo, self.bo)]

    def predict(
        self,
        sequence: list[list[float]],
        *,
        action_bias: dict[float, float] | None = None,
        priors: list[float] | None = None,
    ) -> tuple[list[float], list[float], list[float]]:
        hidden = self.encode(sequence)
        logits = self.logits(hidden)
        if action_bias:
            for index, action in enumerate(self.actions):
                logits[index] += action_bias.get(action, 0.0)
        if priors:
            for index, value in enumerate(priors[: len(logits)]):
                logits[index] += value
        probabilities = softmax(logits)
        return hidden, logits, probabilities

    def train_head(self, hidden: list[float], action: float, reward: float, alt_action: float | None = None) -> None:
        if not hidden:
            return

        try:
            chosen_index = self.actions.index(float(action))
        except ValueError:
            return

        probabilities = softmax(self.logits(hidden))
        target = list(probabilities)
        if reward >= 0:
            confidence = clamp(0.60 + reward * 0.18, 0.60, 0.92)
            remainder = (1.0 - confidence) / max(len(target) - 1, 1)
            target = [remainder] * len(target)
            target[chosen_index] = confidence
        else:
            penalty = clamp(abs(reward) * 0.25, 0.15, 0.40)
            spread = (1.0 - penalty) / max(len(target) - 1, 1)
            target = [spread] * len(target)
            target[chosen_index] = penalty * 0.10
            if alt_action in self.actions:
                alt_index = self.actions.index(float(alt_action))
                target[alt_index] += penalty * 0.55
                total = sum(target) or 1.0
                target = [value / total for value in target]

        gradients = [target[index] - probabilities[index] for index in range(len(target))]
        for action_index, gradient in enumerate(gradients):
            for hidden_index, hidden_value in enumerate(hidden):
                self.Wo[action_index][hidden_index] += self.learning_rate * gradient * hidden_value
            self.bo[action_index] += self.learning_rate * gradient

    def snapshot(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "actions": self.actions,
            "learning_rate": self.learning_rate,
            "Wz": self.Wz,
            "Uz": self.Uz,
            "bz": self.bz,
            "Wr": self.Wr,
            "Ur": self.Ur,
            "br": self.br,
            "Wh": self.Wh,
            "Uh": self.Uh,
            "bh": self.bh,
            "Wo": self.Wo,
            "bo": self.bo,
        }

    def restore(self, payload: dict[str, Any] | None) -> None:
        if not isinstance(payload, dict):
            return
        self.learning_rate = float(payload.get("learning_rate", self.learning_rate))
        for name in ("Wz", "Uz", "bz", "Wr", "Ur", "br", "Wh", "Uh", "bh", "Wo", "bo"):
            if name in payload:
                setattr(self, name, payload[name])


class ResolverEngine:
    def __init__(self, base_dir: str | Path, metrics: MetricsTracker) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = metrics
        self.lock = threading.RLock()
        self.model = OnlineGRUModel(MODEL_INPUT_SIZE, MODEL_HIDDEN_SIZE, ACTIONS)
        self.players: dict[str, PlayerRecord] = {}
        self.global_model_recent: deque[float] = deque(maxlen=64)
        self.replay_buffer: deque[dict[str, Any]] = deque(maxlen=REPLAY_CAPACITY)
        self.feedback_since_retrain = 0
        self.last_startup_warmup_rows = 0
        self.state_path = self.base_dir / "resolver_async_state.pkl"
        self.summary_path = self.base_dir / "resolver_data.json"
        self.events_path = self.base_dir / "resolver_events.jsonl"
        self.sqlite_path = self.base_dir / "resolver_learning.db"
        self.connection = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._init_db()

    def close(self) -> None:
        with self.lock:
            self.persist_state()
            self.connection.commit()
            self.connection.close()

    def _init_db(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS training_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                player_id INTEGER NOT NULL,
                shot_id INTEGER NOT NULL,
                strategy_used TEXT NOT NULL,
                predicted_action REAL NOT NULL,
                confidence REAL NOT NULL,
                hit INTEGER NOT NULL,
                reward REAL NOT NULL,
                latency_ms REAL NOT NULL,
                queue_latency_ms REAL NOT NULL,
                player_state TEXT NOT NULL,
                state_vector_json TEXT NOT NULL,
                derived_features_json TEXT NOT NULL,
                history_json TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def _append_event(self, kind: str, payload: dict[str, Any]) -> None:
        entry = {"ts": round(time.time(), 6), "kind": kind}
        entry.update(payload)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True, separators=(",", ":")))
            handle.write("\n")

    def _player_key(self, player_id: int) -> str:
        return str(player_id)

    def get_player(self, player_id: int) -> PlayerRecord:
        key = self._player_key(player_id)
        if key not in self.players:
            self.players[key] = PlayerRecord()
        return self.players[key]

    def prune_pending_shots(self, player: PlayerRecord) -> None:
        deadline = time.time() - PENDING_SHOT_TTL_SECONDS
        expired = [shot_id for shot_id, record in player.pending_shots.items() if record.created_at_wall < deadline]
        for shot_id in expired:
            player.pending_shots.pop(shot_id, None)

    def build_model_sequence(self, request: PredictRequest) -> list[list[float]]:
        sequence: list[list[float]] = []
        for row in request.history[-HISTORY_LENGTH:]:
            sequence.append(list(row) + [0.0] * DERIVED_FEATURE_SIZE)
        sequence.append(list(request.state_vector) + list(request.derived_features))
        return sequence[-(HISTORY_LENGTH + 1) :]

    def current_model_accuracy(self, player: PlayerRecord | None = None) -> float:
        if player is not None and player.model_recent:
            return recent_average(player.model_recent, recent_average(self.global_model_recent, 0.5))
        return recent_average(self.global_model_recent, 0.5)

    def action_bias_vector(self, player: PlayerRecord) -> dict[float, float]:
        priors: dict[float, float] = {}
        for action in ACTIONS:
            stats = player.action_stats[action]
            priors[action] = player.action_bias.get(action, 0.0) + (stats.recent_success - 0.5) * 0.35 + (stats.long_term_success - 0.5) * 0.20
        return priors

    def ml_sample_bonus(self, player: PlayerRecord) -> float:
        ml_shots = player.strategy_stats["ML"].shots
        if ml_shots >= ML_COLD_START_DECISIONS:
            return 0.0
        remaining = ML_COLD_START_DECISIONS - ml_shots
        return clamp((remaining / ML_COLD_START_DECISIONS) * ML_UNCERTAINTY_BONUS_MAX, 0.0, ML_UNCERTAINTY_BONUS_MAX)

    def state_priors(self, request: PredictRequest) -> list[float]:
        relative_yaw = request.derived_features[1]
        jitter = clamp(request.derived_features[9] / 180.0, 0.0, 1.0)
        air_bonus = 1.0 if request.player_state == "AIR" else 0.0
        priors: list[float] = []
        for action in ACTIONS:
            score = 0.0
            score -= abs((action / 120.0) - clamp(relative_yaw / 180.0, -1.0, 1.0)) * 0.06
            score += jitter * (abs(action) / 120.0) * 0.08
            if air_bonus > 0.5:
                score += abs(action) / 120.0 * 0.04
            priors.append(score)
        return priors

    def build_ml_candidate(self, player: PlayerRecord, request: PredictRequest, probabilities: list[float]) -> dict[str, Any]:
        best_index = max(range(len(probabilities)), key=lambda index: probabilities[index])
        confidence = probabilities[best_index]
        model_accuracy = self.current_model_accuracy(player)
        return {
            "action": ACTIONS[best_index],
            "confidence": clamp(confidence * 0.72 + model_accuracy * 0.28, 0.05, 0.99),
            "raw_confidence": confidence,
        }

    def build_brute_candidate(self, player: PlayerRecord, request: PredictRequest) -> dict[str, Any]:
        action = BRUTE_ACTIONS[player.brute_cursor % len(BRUTE_ACTIONS)]
        distance_norm = clamp(request.derived_features[0], 0.0, 1.0)
        stats = player.strategy_stats["BRUTE"]
        confidence = clamp(0.26 + (1.0 - distance_norm) * 0.18 + stats.recent_success * 0.22 + stats.long_term_success * 0.12, 0.16, 0.72)
        return {"action": action, "confidence": confidence}

    def build_last_candidate(self, player: PlayerRecord, request: PredictRequest) -> dict[str, Any]:
        action = player.last_success_action
        if action is None:
            action = player.last_action
        available = action is not None
        stats = player.strategy_stats["LAST"]
        confidence = clamp(0.18 + stats.recent_success * 0.30 + stats.long_term_success * 0.18, 0.10, 0.76)
        return {"action": action or 0.0, "confidence": confidence, "available": available}

    def build_fs_candidate(self, player: PlayerRecord, request: PredictRequest) -> dict[str, Any]:
        relative_yaw = request.derived_features[1]
        jitter = clamp(request.derived_features[9] / 180.0, 0.0, 1.0)
        angular_velocity = clamp(abs(request.derived_features[2]) / 180.0, 0.0, 1.0)
        raw_action = clamp(-relative_yaw * 0.65, -120.0, 120.0)
        action = nearest_action(raw_action)
        stats = player.strategy_stats["FS"]
        confidence = clamp(0.22 + (1.0 - jitter) * 0.20 + (1.0 - angular_velocity) * 0.14 + stats.recent_success * 0.16 + stats.long_term_success * 0.12, 0.14, 0.78)
        return {"action": action, "confidence": confidence}

    def effective_min_model_accuracy(self, player: PlayerRecord) -> float:
        total_feedback = sum(stats.shots for stats in player.strategy_stats.values())
        warmup = clamp(total_feedback / 80.0, 0.0, 1.0)
        lowered_threshold = MIN_MODEL_ACCURACY - 0.16 * warmup
        return clamp(lowered_threshold, MIN_DYNAMIC_MODEL_ACCURACY, MIN_MODEL_ACCURACY)

    def should_force_ml_exploration(
        self,
        player: PlayerRecord,
        ml_candidate: dict[str, Any] | None,
        strategy_scores: dict[str, float],
    ) -> bool:
        if ml_candidate is None:
            return False
        ml_shots = player.strategy_stats["ML"].shots
        confidence = float(ml_candidate.get("confidence", 0.0))
        best_non_ml_score = max(
            (score for strategy, score in strategy_scores.items() if strategy != "ML" and score > -900),
            default=-999.0,
        )
        ml_score = strategy_scores.get("ML", -999.0)
        if ml_shots < ML_COLD_START_DECISIONS:
            local_interval = 2 if ml_shots < 12 else 3 if ml_shots < 28 else 4
            if confidence < ML_COLD_START_MIN_CONFIDENCE:
                return False
            if best_non_ml_score > ml_score + ML_COLD_START_SCORE_MARGIN:
                return False
            return player.decision_count > 0 and player.decision_count % local_interval == 0

        if ml_shots >= ML_EXPLORATION_MAX_SHOTS:
            return False
        if confidence < ML_EXPLORATION_MIN_CONFIDENCE:
            return False
        if best_non_ml_score > ml_score + ML_EXPLORATION_SCORE_MARGIN:
            return False

        return player.decision_count > 0 and player.decision_count % ML_EXPLORATION_INTERVAL == 0

    def choose_strategy(self, player: PlayerRecord, candidates: dict[str, dict[str, Any]]) -> tuple[str, dict[str, float]]:
        best_strategy = "FS"
        best_score = -999.0
        strategy_scores: dict[str, float] = {}
        global_accuracy = self.current_model_accuracy()
        order_bias = {"BRUTE": 0.03, "LAST": 0.02, "FS": 0.01}
        min_model_accuracy = self.effective_min_model_accuracy(player)

        for strategy in ("ML", "BRUTE", "LAST", "FS"):
            candidate = candidates.get(strategy)
            if not candidate or not candidate.get("available", True):
                strategy_scores[strategy] = -999.0
                continue

            stats = player.strategy_stats[strategy]
            exploration = 0.14 / math.sqrt(stats.shots + 1)
            score = candidate["confidence"] * 0.48 + stats.recent_success * 0.28 + stats.long_term_success * 0.16 + exploration
            score += order_bias.get(strategy, 0.0)

            if strategy == "ML":
                player_accuracy = self.current_model_accuracy(player)
                ml_shots = stats.shots
                cold_start_bias = 0.15 if ml_shots < ML_COLD_START_DECISIONS else 0.04
                score += cold_start_bias
                score += self.ml_sample_bonus(player)
                score += (player_accuracy - 0.5) * 0.35
                score += (global_accuracy - min_model_accuracy) * 0.25
                confidence_floor = 0.45 if ml_shots < ML_COLD_START_DECISIONS else 0.55
                if player_accuracy < min_model_accuracy and candidate["confidence"] < confidence_floor:
                    score -= 0.12 if ml_shots < ML_COLD_START_DECISIONS else 0.22

            if strategy == "BRUTE" and recent_average(player.recent_results, 0.5) > 0.65:
                score -= 0.05

            if strategy == "LAST" and player.last_success_action is None:
                score -= 0.15

            strategy_scores[strategy] = score
            if score > best_score:
                best_score = score
                best_strategy = strategy

        if best_strategy != "ML" and self.should_force_ml_exploration(player, candidates.get("ML"), strategy_scores):
            strategy_scores["ML"] = max(strategy_scores.get("ML", -999.0), best_score + 0.001)
            best_strategy = "ML"

        return best_strategy, strategy_scores

    def register_prediction(
        self,
        request: PredictRequest,
        response: PredictResponse,
        *,
        queue_latency_ms: float,
        hidden: list[float] | None,
        strategy_scores: dict[str, float] | None = None,
        event_kind: str = "predict",
    ) -> PredictResponse:
        player = self.get_player(request.player_id)
        self.prune_pending_shots(player)
        player.pending_shots[request.shot_id] = PendingShot(
            created_at_wall=time.time(),
            selected_action=float(response.predicted_action),
            selected_confidence=float(response.confidence),
            strategy_used=response.strategy_used,
            hidden=list(hidden or []),
            request_payload=request.to_payload(),
            queue_latency_ms=float(queue_latency_ms),
        )
        player.last_action = float(response.predicted_action)
        player.last_strategy = response.strategy_used
        player.decision_count += 1

        self.metrics.record_prediction(response.strategy_used, response.confidence, queue_latency_ms)
        event_payload = {
            "player_id": request.player_id,
            "shot_id": request.shot_id,
            "strategy_used": response.strategy_used,
            "predicted_action": response.predicted_action,
            "confidence": response.confidence,
            "queue_latency_ms": round(queue_latency_ms, 3),
        }
        if strategy_scores is not None:
            event_payload["strategy_scores"] = strategy_scores
        self._append_event(event_kind, event_payload)
        return response

    def register_fallback_prediction(
        self,
        request: PredictRequest,
        reason: str,
        *,
        queue_latency_ms: float = 0.0,
    ) -> PredictResponse:
        player = self.get_player(request.player_id)
        self.prune_pending_shots(player)
        candidates = {
            "BRUTE": self.build_brute_candidate(player, request),
            "LAST": self.build_last_candidate(player, request),
            "FS": self.build_fs_candidate(player, request),
        }
        strategy_used, strategy_scores = self.choose_strategy(player, candidates)
        if strategy_used not in candidates:
            strategy_used = "BRUTE"
        candidate = candidates[strategy_used]
        response = PredictResponse(
            shot_id=request.shot_id,
            predicted_action=float(candidate["action"]),
            confidence=float(candidate["confidence"]),
            strategy_used=strategy_used,
        )
        return self.register_prediction(
            request,
            response,
            queue_latency_ms=queue_latency_ms,
            hidden=[],
            strategy_scores={**strategy_scores, "fallback_reason": reason},
            event_kind="fallback_predict",
        )

    def fallback_response(self, request: PredictRequest) -> PredictResponse:
        with self.lock:
            return self.register_fallback_prediction(request, "direct_fallback", queue_latency_ms=0.0)

    def predict(self, request: PredictRequest, queue_latency_ms: float = 0.0) -> PredictResponse:
        with self.lock:
            player = self.get_player(request.player_id)
            self.prune_pending_shots(player)
            sequence = self.build_model_sequence(request)
            hidden, _, probabilities = self.model.predict(
                sequence,
                action_bias=self.action_bias_vector(player),
                priors=self.state_priors(request),
            )
            candidates = {
                "ML": self.build_ml_candidate(player, request, probabilities),
                "BRUTE": self.build_brute_candidate(player, request),
                "LAST": self.build_last_candidate(player, request),
                "FS": self.build_fs_candidate(player, request),
            }
            strategy_used, strategy_scores = self.choose_strategy(player, candidates)
            selected = candidates[strategy_used]
            response = PredictResponse(
                shot_id=request.shot_id,
                predicted_action=float(selected["action"]),
                confidence=float(selected["confidence"]),
                strategy_used=strategy_used,
            )
            return self.register_prediction(
                request,
                response,
                queue_latency_ms=queue_latency_ms,
                hidden=hidden,
                strategy_scores=strategy_scores,
                event_kind="predict",
            )

    def apply_online_bias(self, player: PlayerRecord, action: float, reward: float) -> None:
        for candidate in ACTIONS:
            player.action_bias[candidate] *= 0.995
        player.action_bias[action] = player.action_bias.get(action, 0.0) + clamp(reward * 0.08, -0.20, 0.20)

    def handle_feedback(self, feedback: FeedbackRequest) -> dict[str, Any]:
        with self.lock:
            player = self.get_player(feedback.player_id)
            self.prune_pending_shots(player)
            pending = player.pending_shots.pop(feedback.shot_id, None)
            if pending is None:
                return {"status": "no_action", "shot_id": feedback.shot_id}

            reward = 1.0 if feedback.hit else -0.75
            strategy_used = feedback.strategy_used or pending.strategy_used
            if strategy_used not in STRATEGIES:
                strategy_used = pending.strategy_used

            action = nearest_action(pending.selected_action)
            alt_action = player.last_success_action
            if feedback.hit:
                player.last_success_action = action
                player.brute_cursor = 0
            elif strategy_used == "BRUTE":
                player.brute_cursor = (player.brute_cursor + 1) % len(BRUTE_ACTIONS)

            player.strategy_stats[strategy_used].record(feedback.hit, feedback.latency_ms)
            player.action_stats[action].record(feedback.hit)
            player.recent_results.append(1.0 if feedback.hit else 0.0)
            if strategy_used == "ML":
                player.model_recent.append(1.0 if feedback.hit else 0.0)
                self.global_model_recent.append(1.0 if feedback.hit else 0.0)

            if feedback.cache_used:
                key = "cache_hits" if feedback.hit else "cache_misses"
                player.telemetry[key] += 1
            if feedback.fallback_used:
                key = "fallback_hits" if feedback.hit else "fallback_misses"
                player.telemetry[key] += 1

            self.apply_online_bias(player, action, reward)
            self.model.train_head(pending.hidden, action, reward, alt_action=alt_action)
            self.replay_buffer.append(
                {
                    "player_id": feedback.player_id,
                    "shot_id": feedback.shot_id,
                    "action": action,
                    "reward": reward,
                    "hidden": list(pending.hidden),
                    "alt_action": alt_action,
                }
            )
            self.feedback_since_retrain += 1
            self.metrics.record_feedback(strategy_used, feedback.hit)

            request_payload = pending.request_payload
            self.connection.execute(
                """
                INSERT INTO training_samples (
                    created_at, player_id, shot_id, strategy_used, predicted_action, confidence,
                    hit, reward, latency_ms, queue_latency_ms, player_state,
                    state_vector_json, derived_features_json, history_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    round(time.time(), 6),
                    feedback.player_id,
                    feedback.shot_id,
                    strategy_used,
                    float(action),
                    float(pending.selected_confidence),
                    1 if feedback.hit else 0,
                    float(reward),
                    float(feedback.latency_ms),
                    float(pending.queue_latency_ms),
                    request_payload["player_state"],
                    json.dumps(request_payload["state_vector"], ensure_ascii=True, separators=(",", ":")),
                    json.dumps(request_payload["derived_features"], ensure_ascii=True, separators=(",", ":")),
                    json.dumps(request_payload["history"], ensure_ascii=True, separators=(",", ":")),
                ),
            )
            self.connection.commit()
            self._append_event(
                "feedback",
                {
                    "player_id": feedback.player_id,
                    "shot_id": feedback.shot_id,
                    "strategy_used": strategy_used,
                    "hit": feedback.hit,
                    "reward": reward,
                    "latency_ms": round(feedback.latency_ms, 3),
                },
            )
            return {
                "status": "ok",
                "shot_id": feedback.shot_id,
                "strategy_used": strategy_used,
                "reward": reward,
                "model_accuracy": round(self.current_model_accuracy(), 4),
            }

    def export_training_data(self, limit: int = 500) -> dict[str, Any]:
        with self.lock:
            rows = self.connection.execute(
                """
                SELECT created_at, player_id, shot_id, strategy_used, predicted_action, confidence,
                       hit, reward, latency_ms, queue_latency_ms, player_state,
                       state_vector_json, derived_features_json, history_json
                FROM training_samples
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()

        items = []
        for row in reversed(rows):
            items.append(
                {
                    "created_at": row["created_at"],
                    "player_id": row["player_id"],
                    "shot_id": row["shot_id"],
                    "strategy_used": row["strategy_used"],
                    "predicted_action": row["predicted_action"],
                    "confidence": row["confidence"],
                    "hit": bool(row["hit"]),
                    "reward": row["reward"],
                    "latency_ms": row["latency_ms"],
                    "queue_latency_ms": row["queue_latency_ms"],
                    "player_state": row["player_state"],
                    "state_vector": json.loads(row["state_vector_json"]),
                    "derived_features": json.loads(row["derived_features_json"]),
                    "history": json.loads(row["history_json"]),
                }
            )
        return {"count": len(items), "items": items}

    def retrain_from_storage(self, limit: int = 256) -> dict[str, Any]:
        exported = self.export_training_data(limit=limit)
        applied = 0
        with self.lock:
            for sample in exported["items"]:
                request = PredictRequest(
                    player_id=int(sample["player_id"]),
                    shot_id=int(sample["shot_id"]),
                    timestamp=time.time(),
                    state_vector=list(sample["state_vector"]),
                    derived_features=list(sample["derived_features"]),
                    history=[list(row) for row in sample["history"]],
                    player_state=str(sample["player_state"]) if sample["player_state"] in PLAYER_STATES else "STANDING",
                )
                hidden, _, _ = self.model.predict(self.build_model_sequence(request))
                self.model.train_head(hidden, float(sample["predicted_action"]), float(sample["reward"]), alt_action=None)
                applied += 1

            if applied:
                self.feedback_since_retrain = 0
                self.metrics.record_retrain(applied)
                self._append_event("retrain", {"samples": applied})

        return {"status": "ok", "samples": applied}

    def training_row_count(self) -> int:
        with self.lock:
            return int(self.connection.execute("SELECT COUNT(*) FROM training_samples").fetchone()[0])

    def maybe_startup_warmup(self, limit: int = STARTUP_WARM_RETRAIN_LIMIT) -> dict[str, Any]:
        total_rows = self.training_row_count()
        if total_rows < STARTUP_WARM_RETRAIN_MIN_ROWS:
            return {"status": "skipped", "reason": "insufficient_rows", "rows": total_rows}
        if self.last_startup_warmup_rows > 0 and total_rows < (self.last_startup_warmup_rows + STARTUP_WARM_RETRAIN_GROWTH):
            return {"status": "skipped", "reason": "no_significant_growth", "rows": total_rows}

        result = self.retrain_from_storage(limit=min(limit, total_rows))
        if result.get("status") == "ok" and int(result.get("samples", 0)) > 0:
            with self.lock:
                self.last_startup_warmup_rows = total_rows
        return result

    def should_retrain(self) -> bool:
        with self.lock:
            return self.feedback_since_retrain >= RETRAIN_TRIGGER

    def health_snapshot(self) -> dict[str, Any]:
        with self.lock:
            pending_shots = sum(len(player.pending_shots) for player in self.players.values())
            training_rows = self.connection.execute("SELECT COUNT(*) FROM training_samples").fetchone()[0]
            return {
                "status": "ok",
                "players": len(self.players),
                "pending_shots": pending_shots,
                "training_rows": int(training_rows),
                "replay_size": len(self.replay_buffer),
                "model_accuracy": round(self.current_model_accuracy(), 4),
            }

    def persist_state(self) -> None:
        payload = {
            "summary_version": SUMMARY_VERSION,
            "model": self.model.snapshot(),
            "players": {player_id: player.to_snapshot() for player_id, player in self.players.items()},
            "replay_buffer": list(self.replay_buffer)[-512:],
            "global_model_recent": list(self.global_model_recent),
            "feedback_since_retrain": self.feedback_since_retrain,
            "last_startup_warmup_rows": self.last_startup_warmup_rows,
        }
        self.state_path.write_bytes(pickle.dumps(payload))
        summary = {
            "_meta": {
                "summary_version": SUMMARY_VERSION,
                "saved_at": round(time.time(), 4),
                "model_accuracy": round(self.current_model_accuracy(), 4),
            },
            "players": {player_id: player.to_snapshot() for player_id, player in self.players.items()},
            "metrics": self.metrics.snapshot(),
        }
        self.summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")

    def load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = pickle.loads(self.state_path.read_bytes())
        except Exception:
            return

        with self.lock:
            self.model.restore(payload.get("model"))
            self.players = {
                str(player_id): PlayerRecord.from_snapshot(raw_player)
                for player_id, raw_player in (payload.get("players") or {}).items()
            }
            self.replay_buffer.clear()
            for item in list(payload.get("replay_buffer", []))[-REPLAY_CAPACITY:]:
                if isinstance(item, dict):
                    self.replay_buffer.append(item)
            self.global_model_recent.clear()
            self.global_model_recent.extend(float(value) for value in list(payload.get("global_model_recent", []))[-64:])
            self.feedback_since_retrain = int(payload.get("feedback_since_retrain", 0))
            self.last_startup_warmup_rows = int(payload.get("last_startup_warmup_rows", 0))
