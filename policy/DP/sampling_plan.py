from __future__ import annotations

import dataclasses
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_PLAN_META_FILENAME = "plan_meta.json"
DEFAULT_PLAN_ARRAYS_FILENAME = "plan_arrays.npz"


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _resolve_plan_paths(path: str | Path) -> tuple[Path, Path]:
    path = Path(path)
    if path.is_dir():
        return path / DEFAULT_PLAN_META_FILENAME, path / DEFAULT_PLAN_ARRAYS_FILENAME
    if path.suffix == ".json":
        return path, path.with_name(DEFAULT_PLAN_ARRAYS_FILENAME)
    if path.suffix == ".npz":
        return path.with_name(DEFAULT_PLAN_META_FILENAME), path
    raise ValueError(
        f"Unsupported plan path: {path}. Provide a directory, {DEFAULT_PLAN_META_FILENAME}, or {DEFAULT_PLAN_ARRAYS_FILENAME}."
    )


def compute_plan_sha256(*, meta: dict[str, Any], arrays_bytes: bytes) -> str:
    meta_for_hash = dict(meta)
    meta_for_hash.pop("plan_sha256", None)
    h = hashlib.sha256()
    h.update(_canonical_json_bytes(meta_for_hash))
    h.update(arrays_bytes)
    return h.hexdigest()


@dataclasses.dataclass(frozen=True)
class SamplingPlan:
    meta: dict[str, Any]
    episode_ids: np.ndarray
    decision_ts: np.ndarray
    buffer_start_idx: np.ndarray
    buffer_end_idx: np.ndarray
    sample_start_idx: np.ndarray
    sample_end_idx: np.ndarray
    stage_ids: np.ndarray
    weights: np.ndarray

    @property
    def plan_sha256(self) -> str | None:
        value = self.meta.get("plan_sha256")
        return str(value) if value is not None else None

    def validate(self) -> None:
        arrays = {
            "episode_ids": self.episode_ids,
            "decision_ts": self.decision_ts,
            "buffer_start_idx": self.buffer_start_idx,
            "buffer_end_idx": self.buffer_end_idx,
            "sample_start_idx": self.sample_start_idx,
            "sample_end_idx": self.sample_end_idx,
            "stage_ids": self.stage_ids,
            "weights": self.weights,
        }
        lengths = {k: int(len(v)) for k, v in arrays.items()}
        n_values = set(lengths.values())
        if len(n_values) != 1:
            raise ValueError(f"Length mismatch among arrays: {lengths}")
        n = int(next(iter(n_values)))
        if n <= 0:
            raise ValueError("Plan has zero anchors.")

        int_arrays = [
            ("episode_ids", self.episode_ids),
            ("decision_ts", self.decision_ts),
            ("buffer_start_idx", self.buffer_start_idx),
            ("buffer_end_idx", self.buffer_end_idx),
            ("sample_start_idx", self.sample_start_idx),
            ("sample_end_idx", self.sample_end_idx),
            ("stage_ids", self.stage_ids),
        ]
        for name, arr in int_arrays:
            if arr.ndim != 1:
                raise ValueError(f"{name} must be 1D.")
            if not np.issubdtype(arr.dtype, np.integer):
                raise ValueError(f"{name} must be integer dtype.")

        if self.weights.ndim != 1:
            raise ValueError("weights must be 1D.")
        if not np.issubdtype(self.weights.dtype, np.floating):
            raise ValueError("weights must be floating dtype.")
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("weights contain non-finite values.")
        if np.any(self.weights < 0):
            raise ValueError("weights must be non-negative.")
        if float(self.weights.sum()) <= 0:
            raise ValueError("Sum of weights must be > 0.")

    def _arrays_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            episode_ids=self.episode_ids.astype(np.int64, copy=False),
            decision_ts=self.decision_ts.astype(np.int64, copy=False),
            buffer_start_idx=self.buffer_start_idx.astype(np.int64, copy=False),
            buffer_end_idx=self.buffer_end_idx.astype(np.int64, copy=False),
            sample_start_idx=self.sample_start_idx.astype(np.int64, copy=False),
            sample_end_idx=self.sample_end_idx.astype(np.int64, copy=False),
            stage_ids=self.stage_ids.astype(np.int32, copy=False),
            weights=self.weights.astype(np.float32, copy=False),
        )
        return buf.getvalue()

    def recompute_sha256(self) -> str:
        return compute_plan_sha256(meta=self.meta, arrays_bytes=self._arrays_bytes())

    def save(self, out_dir: str | Path) -> tuple[Path, Path]:
        self.validate()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        meta_path = out_dir / DEFAULT_PLAN_META_FILENAME
        arrays_path = out_dir / DEFAULT_PLAN_ARRAYS_FILENAME

        arrays_bytes = self._arrays_bytes()
        meta = dict(self.meta)
        meta["plan_sha256"] = compute_plan_sha256(meta=meta, arrays_bytes=arrays_bytes)

        arrays_path.write_bytes(arrays_bytes)
        meta_path.write_text(_canonical_json_bytes(meta).decode("utf-8"), encoding="utf-8")
        return meta_path, arrays_path

    @classmethod
    def load(cls, path: str | Path, *, verify_sha256: bool = True) -> "SamplingPlan":
        meta_path, arrays_path = _resolve_plan_paths(path)
        if not meta_path.exists():
            raise FileNotFoundError(f"Plan meta not found: {meta_path}")
        if not arrays_path.exists():
            raise FileNotFoundError(f"Plan arrays not found: {arrays_path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        arrays_bytes = arrays_path.read_bytes()
        npz = np.load(io.BytesIO(arrays_bytes))

        required = [
            "episode_ids",
            "decision_ts",
            "buffer_start_idx",
            "buffer_end_idx",
            "sample_start_idx",
            "sample_end_idx",
            "stage_ids",
            "weights",
        ]
        missing = [k for k in required if k not in npz.files]
        if missing:
            raise ValueError(f"Plan arrays missing required fields: {missing}")

        plan = cls(
            meta=meta,
            episode_ids=np.asarray(npz["episode_ids"], dtype=np.int64),
            decision_ts=np.asarray(npz["decision_ts"], dtype=np.int64),
            buffer_start_idx=np.asarray(npz["buffer_start_idx"], dtype=np.int64),
            buffer_end_idx=np.asarray(npz["buffer_end_idx"], dtype=np.int64),
            sample_start_idx=np.asarray(npz["sample_start_idx"], dtype=np.int64),
            sample_end_idx=np.asarray(npz["sample_end_idx"], dtype=np.int64),
            stage_ids=np.asarray(npz["stage_ids"], dtype=np.int32),
            weights=np.asarray(npz["weights"], dtype=np.float32),
        )
        plan.validate()

        if verify_sha256:
            expected = plan.plan_sha256
            computed = compute_plan_sha256(meta=meta, arrays_bytes=arrays_bytes)
            if expected is None:
                raise ValueError(f"Plan meta missing plan_sha256: {meta_path}")
            if computed != expected:
                raise ValueError(f"Plan sha256 mismatch: expected={expected}, computed={computed}")

        return plan
