from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

ACTIONS = [-120.0, -90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 120.0]
BRUTE_ACTIONS = [0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 120.0, -120.0]
PLAYER_STATES = ("STANDING", "MOVING", "AIR")
STRATEGIES = ("ML", "BRUTE", "LAST", "FS")

STATE_VECTOR_SIZE = 18
DERIVED_FEATURE_SIZE = 12
HISTORY_LENGTH = 8

STATE_INDEX_TARGET_X = 0
STATE_INDEX_TARGET_Y = 1
STATE_INDEX_TARGET_Z = 2
STATE_INDEX_VEL_X = 3
STATE_INDEX_VEL_Y = 4
STATE_INDEX_VEL_Z = 5
STATE_INDEX_YAW = 6
STATE_INDEX_LAST_DELTA = 7
STATE_INDEX_AVG_DELTA = 8
STATE_INDEX_SPEED = 9
STATE_INDEX_DUCK = 10
STATE_INDEX_AIR_TIME = 11
STATE_INDEX_STANDING = 12
STATE_INDEX_MOVING = 13
STATE_INDEX_AIR = 14
STATE_INDEX_RELATIVE_YAW = 15
STATE_INDEX_AIM_PITCH = 16
STATE_INDEX_DISTANCE = 17


class SchemaError(ValueError):
    pass


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def norm(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def angle_delta(current: float, previous: float) -> float:
    return norm(float(current) - float(previous))


def parse_float(value: Any, *, default: float | None = None, field_name: str = "value") -> float:
    if value in (None, ""):
        if default is None:
            raise SchemaError(f"missing {field_name}")
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{field_name} must be a float") from exc


def parse_int(value: Any, *, default: int | None = None, field_name: str = "value") -> int:
    if value in (None, ""):
        if default is None:
            raise SchemaError(f"missing {field_name}")
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{field_name} must be an int") from exc


def vector_pad(values: Any, size: int, *, field_name: str) -> list[float]:
    if values in (None, ""):
        return [0.0] * size
    if not isinstance(values, (list, tuple)):
        raise SchemaError(f"{field_name} must be an array")

    normalized = [parse_float(value, default=0.0, field_name=field_name) for value in list(values)[:size]]
    if len(normalized) < size:
        normalized.extend([0.0] * (size - len(normalized)))
    return normalized


def history_rows(rows: Any) -> list[list[float]]:
    if rows in (None, ""):
        rows = []
    if not isinstance(rows, (list, tuple)):
        raise SchemaError("history must be an array of arrays")

    normalized = [vector_pad(row, STATE_VECTOR_SIZE, field_name="history[]") for row in list(rows)[-HISTORY_LENGTH:]]
    while len(normalized) < HISTORY_LENGTH:
        normalized.insert(0, [0.0] * STATE_VECTOR_SIZE)
    return normalized


def _recent_deltas(history: list[list[float]], state_vector: list[float]) -> list[float]:
    yaw_series = [row[STATE_INDEX_YAW] for row in history]
    yaw_series.append(state_vector[STATE_INDEX_YAW])
    deltas: list[float] = []
    for index in range(1, len(yaw_series)):
        deltas.append(angle_delta(yaw_series[index], yaw_series[index - 1]))
    return deltas


def derive_features(
    state_vector: list[float],
    history: list[list[float]],
    player_state: str,
) -> list[float]:
    deltas = _recent_deltas(history, state_vector)
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    variance = 0.0
    if deltas:
        variance = sum((value - mean_delta) ** 2 for value in deltas) / len(deltas)

    state_standing = 1.0 if player_state == "STANDING" else 0.0
    state_moving = 1.0 if player_state == "MOVING" else 0.0
    state_air = 1.0 if player_state == "AIR" else 0.0

    distance = abs(state_vector[STATE_INDEX_DISTANCE])
    relative_angle = state_vector[STATE_INDEX_RELATIVE_YAW]
    angle_rate = state_vector[STATE_INDEX_LAST_DELTA] or mean_delta
    jitter = max((abs(value) for value in deltas), default=0.0)
    trend = state_vector[STATE_INDEX_AVG_DELTA] or mean_delta

    return [
        float(distance),
        float(relative_angle),
        float(angle_rate),
        state_standing,
        state_moving,
        state_air,
        float(abs(state_vector[STATE_INDEX_SPEED])),
        float(abs(state_vector[STATE_INDEX_VEL_Z])),
        float(trend),
        float(jitter),
        float(min(variance, 360.0)),
        float(abs(state_vector[STATE_INDEX_AIM_PITCH])),
    ]


def merged_derived_features(
    payload_features: Any,
    state_vector: list[float],
    history: list[list[float]],
    player_state: str,
) -> list[float]:
    auto = derive_features(state_vector, history, player_state)
    if payload_features in (None, ""):
        return vector_pad(auto, DERIVED_FEATURE_SIZE, field_name="derived_features")
    return vector_pad(payload_features, DERIVED_FEATURE_SIZE, field_name="derived_features")


def nearest_action(action: float) -> float:
    value = float(action)
    return min(ACTIONS, key=lambda candidate: abs(candidate - value))


@dataclass(slots=True)
class PredictRequest:
    player_id: int
    shot_id: int
    timestamp: float
    state_vector: list[float]
    derived_features: list[float]
    history: list[list[float]]
    player_state: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PredictRequest":
        if not isinstance(payload, Mapping):
            raise SchemaError("predict payload must be an object")

        player_id = parse_int(payload.get("player_id"), field_name="player_id")
        shot_id = parse_int(payload.get("shot_id"), field_name="shot_id")
        timestamp = parse_float(payload.get("timestamp"), default=time.time(), field_name="timestamp")
        now = time.time()
        if timestamp < now - 5.0 or timestamp > now + 1.0:
            timestamp = now
        player_state = str(payload.get("player_state", "STANDING")).upper()
        if player_state not in PLAYER_STATES:
            raise SchemaError("player_state must be STANDING, MOVING or AIR")

        state_vector = vector_pad(payload.get("state_vector"), STATE_VECTOR_SIZE, field_name="state_vector")
        history = history_rows(payload.get("history"))
        derived_features = merged_derived_features(payload.get("derived_features"), state_vector, history, player_state)
        return cls(
            player_id=player_id,
            shot_id=shot_id,
            timestamp=timestamp,
            state_vector=state_vector,
            derived_features=derived_features,
            history=history,
            player_state=player_state,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "shot_id": self.shot_id,
            "timestamp": self.timestamp,
            "state_vector": list(self.state_vector),
            "derived_features": list(self.derived_features),
            "history": [list(row) for row in self.history],
            "player_state": self.player_state,
        }


@dataclass(slots=True)
class PredictResponse:
    shot_id: int
    predicted_action: float
    confidence: float
    strategy_used: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "shot_id": int(self.shot_id),
            "predicted_action": float(nearest_action(self.predicted_action)),
            "confidence": round(clamp(self.confidence, 0.0, 1.0), 4),
            "strategy_used": self.strategy_used if self.strategy_used in STRATEGIES else "FS",
        }


@dataclass(slots=True)
class FeedbackRequest:
    player_id: int
    shot_id: int
    hit: bool
    latency_ms: float = 0.0
    strategy_used: str | None = None
    cache_used: bool = False
    fallback_used: bool = False

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FeedbackRequest":
        if not isinstance(payload, Mapping):
            raise SchemaError("feedback payload must be an object")

        strategy_used = payload.get("strategy_used")
        if strategy_used is not None:
            strategy_used = str(strategy_used).upper()
            if strategy_used not in STRATEGIES:
                strategy_used = None

        hit_value = payload.get("hit", False)
        if isinstance(hit_value, bool):
            hit = hit_value
        elif isinstance(hit_value, (int, float)):
            hit = bool(hit_value)
        else:
            hit = str(hit_value).strip().lower() in {"1", "true", "yes", "y", "on"}

        return cls(
            player_id=parse_int(payload.get("player_id"), field_name="player_id"),
            shot_id=parse_int(payload.get("shot_id"), field_name="shot_id"),
            hit=hit,
            latency_ms=parse_float(payload.get("latency_ms"), default=0.0, field_name="latency_ms"),
            strategy_used=strategy_used,
            cache_used=bool(payload.get("cache_used", False)),
            fallback_used=bool(payload.get("fallback_used", False)),
        )


def confidence_bucket(confidence: float) -> str:
    confidence = clamp(confidence, 0.0, 1.0)
    start = math.floor(confidence * 5) / 5
    end = min(start + 0.2, 1.0)
    return f"{start:.1f}-{end:.1f}"
