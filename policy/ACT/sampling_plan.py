from __future__ import annotations

"""
ACT 的 Sampling Plan（采样计划）与 Anchor（锚点）基础协议。

动机（为什么要有这个文件）：
- 现在 ACT 的数据采样是“按 episode 抽样 + 在 __getitem__ 里随机 start_ts（起点帧）”，
  这会导致：你无法对“某个具体片段（segment）”做稳定的加权/去冗余/优先采样，也很难复现与审计。
- Sampling Plan 把“训练时究竟看哪些片段、每个片段权重是多少、随机种子是什么”等信息固化成静态文件，
  训练阶段只读这个 plan 来构造 sampler，从而让后续的课程学习/去冗余/优先采样都变成“换 plan”。

文件格式（尽量向 pi0 的 plan 形式对齐，但更简单）：
- plan_meta.json：人类可读的元信息（dataset_dir、stride、超参、sha256 等）
- plan_arrays.npz：数组（episode_ids / start_ts / stage_ids / weights）

说明：
- episode_ids + start_ts 唯一确定一个 anchor（锚点：从某条 demo 的某个时刻开始的训练片段）。
- stage_ids 当前仅作为“预留字段”（用于阶段均衡/课程学习/统计），基础地基里先不强依赖它。
"""

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
    """生成可复现的 JSON bytes（键排序 + 紧凑分隔符）。"""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _resolve_plan_paths(path: str | Path) -> tuple[Path, Path]:
    """支持三种输入：目录 / plan_meta.json / plan_arrays.npz。"""
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
    """计算 plan 的内容哈希；meta 内的 plan_sha256 字段不参与哈希。"""
    meta_for_hash = dict(meta)
    meta_for_hash.pop("plan_sha256", None)
    h = hashlib.sha256()
    h.update(_canonical_json_bytes(meta_for_hash))
    h.update(arrays_bytes)
    return h.hexdigest()


@dataclasses.dataclass(frozen=True)
class SamplingPlan:
    """采样计划：anchors（episode_ids/start_ts）+ stage_ids + weights + meta。"""

    meta: dict[str, Any]
    episode_ids: np.ndarray
    start_ts: np.ndarray
    stage_ids: np.ndarray
    weights: np.ndarray

    @property
    def plan_sha256(self) -> str | None:
        value = self.meta.get("plan_sha256")
        return str(value) if value is not None else None

    def validate(self) -> None:
        """基础一致性检查：形状、dtype、非负、非 NaN 等。"""
        if self.episode_ids.ndim != 1 or self.start_ts.ndim != 1:
            raise ValueError("episode_ids and start_ts must be 1D arrays.")
        n = int(len(self.episode_ids))
        if len(self.start_ts) != n:
            raise ValueError(f"Length mismatch: episode_ids={n}, start_ts={len(self.start_ts)}")
        if n == 0:
            raise ValueError("Plan has zero anchors.")

        if len(self.stage_ids) != n:
            raise ValueError(f"Length mismatch: stage_ids={len(self.stage_ids)} vs n={n}")
        if len(self.weights) != n:
            raise ValueError(f"Length mismatch: weights={len(self.weights)} vs n={n}")

        if not np.issubdtype(self.episode_ids.dtype, np.integer):
            raise ValueError("episode_ids must be integer dtype.")
        if not np.issubdtype(self.start_ts.dtype, np.integer):
            raise ValueError("start_ts must be integer dtype.")
        if not np.issubdtype(self.stage_ids.dtype, np.integer):
            raise ValueError("stage_ids must be integer dtype.")
        if not np.issubdtype(self.weights.dtype, np.floating):
            raise ValueError("weights must be floating dtype.")
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("weights contain non-finite values.")
        if np.any(self.weights < 0):
            raise ValueError("weights must be non-negative.")
        if float(self.weights.sum()) <= 0:
            raise ValueError("Sum of weights must be > 0.")

    def _arrays_bytes(self) -> bytes:
        """序列化 arrays 为 npz bytes（用于哈希与 save）。"""
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            episode_ids=self.episode_ids.astype(np.int64, copy=False),
            start_ts=self.start_ts.astype(np.int64, copy=False),
            stage_ids=self.stage_ids.astype(np.int32, copy=False),
            weights=self.weights.astype(np.float32, copy=False),
        )
        return buf.getvalue()

    def recompute_sha256(self) -> str:
        return compute_plan_sha256(meta=self.meta, arrays_bytes=self._arrays_bytes())

    def save(self, out_dir: str | Path) -> tuple[Path, Path]:
        """写出 plan_meta.json 与 plan_arrays.npz，并自动写入 plan_sha256。"""
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
        """读取 plan，并可选校验 sha256（默认开启，建议保持开启）。"""
        meta_path, arrays_path = _resolve_plan_paths(path)
        if not meta_path.exists():
            raise FileNotFoundError(f"Plan meta not found: {meta_path}")
        if not arrays_path.exists():
            raise FileNotFoundError(f"Plan arrays not found: {arrays_path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        arrays_bytes = arrays_path.read_bytes()
        npz = np.load(io.BytesIO(arrays_bytes))

        # 兼容：如果某些字段缺失，给出默认值（尽量不要让地基因为格式小差异而挂掉）
        episode_ids = np.asarray(npz["episode_ids"], dtype=np.int64)
        start_ts = np.asarray(npz["start_ts"], dtype=np.int64)
        stage_ids = np.asarray(npz["stage_ids"], dtype=np.int32) if "stage_ids" in npz.files else np.zeros_like(episode_ids, dtype=np.int32)
        weights = np.asarray(npz["weights"], dtype=np.float32) if "weights" in npz.files else np.ones_like(episode_ids, dtype=np.float32)

        plan = cls(meta=meta, episode_ids=episode_ids, start_ts=start_ts, stage_ids=stage_ids, weights=weights)
        plan.validate()

        if verify_sha256:
            expected = plan.plan_sha256
            computed = compute_plan_sha256(meta=meta, arrays_bytes=arrays_bytes)
            if expected is None:
                raise ValueError(f"Plan meta missing plan_sha256: {meta_path}")
            if computed != expected:
                raise ValueError(f"Plan sha256 mismatch: expected={expected}, computed={computed}")

        return plan
