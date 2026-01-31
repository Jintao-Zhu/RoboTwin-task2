#!/usr/bin/env python3
from __future__ import annotations

"""
为 ACT 构建 Sampling Plan（采样计划）：
- 输入：一个 processed_data 目录（里面有 episode_*.hdf5）
- 输出：一个 plan 目录（plan_meta.json + plan_arrays.npz）

这个脚本只做“离线生成 plan”，不修改训练逻辑：
训练时把 `--sampling_plan_path <plan_dir>` 传给 imitate_episodes.py 即可启用。

设计目标（地基阶段的取舍）：
- 代码尽量短、逻辑尽量直观（便于 review）。
- 默认行为尽量接近 baseline：不做尾部过滤，不做激进偏置；只把“随机 start_ts”变成“可枚举 anchor”。
- 预留 stage_ids 与 stage_balance 字段，为后续课程学习/阶段均衡/去冗余提供统一入口。
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

# 直接复用 policy/ACT 下的 plan 协议实现。
# 注意：这个脚本可能在 policy/ACT 目录下被执行（此时 python 的 sys.path 不包含 repo root），
# 所以这里显式把 repo root 加到 sys.path，保证 `import policy.*` 稳定可用。
import sys

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from policy.ACT.sampling_plan import SamplingPlan  # noqa: E402


def _stable_hash_u64(text: str) -> int:
    """稳定哈希：跨进程/跨平台一致，用于 stride 对齐 offset。"""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _parse_episode_id(path: Path) -> int:
    # episode_123.hdf5 -> 123
    stem = path.stem  # episode_123
    if not stem.startswith("episode_"):
        raise ValueError(f"Unexpected episode filename: {path.name}")
    return int(stem.split("_", 1)[1])


def _time_quantile_stage(*, start_ts: np.ndarray, episode_len: int, num_stages: int) -> np.ndarray:
    """按时间分位打阶段标签：stage = floor(K * t / (L-1))。"""
    if num_stages <= 1:
        return np.zeros_like(start_ts, dtype=np.int32)
    denom = max(1, int(episode_len) - 1)
    u = start_ts.astype(np.float32) / float(denom)
    stage = np.floor(float(num_stages) * u).astype(np.int64)
    stage = np.clip(stage, 0, num_stages - 1)
    return stage.astype(np.int32)


def _apply_stage_balance(
    *,
    stage_ids: np.ndarray,
    mode: str,
    inv_freq_power: float,
) -> np.ndarray:
    """根据 stage 频次生成基础权重（越稀有的 stage 权重越大）。"""
    stage_ids = stage_ids.astype(np.int64, copy=False)
    uniq, cnt = np.unique(stage_ids, return_counts=True)
    count_map = {int(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}
    counts = np.array([count_map[int(s)] for s in stage_ids.tolist()], dtype=np.float64)

    if mode == "uniform_over_stage":
        # 让每个 stage 的总概率质量尽量相等：w_i ∝ 1 / count(stage_i)
        w = 1.0 / np.maximum(1.0, counts)
    elif mode == "inv_freq":
        # 更一般的频次逆权：w_i ∝ count(stage_i)^(-power)
        w = np.power(np.maximum(1.0, counts), -float(inv_freq_power))
    else:
        raise ValueError(f"Unknown stage_balance mode: {mode}")
    return w.astype(np.float64)


def _clip_and_mix_weights(*, w: np.ndarray, clip_ratio: float, epsilon_mix: float) -> np.ndarray:
    """权重安全护栏：裁剪 + ε 混合。

通俗解释：
- clip（裁剪）：防止出现“极少数样本权重特别大”，导致训练被少数片段绑架。
- ε-mix（保底均匀混合）：防止某些样本权重太小而几乎永远看不到，保证覆盖。
"""
    w = w.astype(np.float64, copy=False)
    if float(w.sum()) <= 0:
        raise ValueError("Invalid weights: sum <= 0.")

    clip_ratio = float(clip_ratio)
    if clip_ratio > 1.0:
        med = float(np.median(w))
        lo = med / clip_ratio
        hi = med * clip_ratio
        w = np.clip(w, lo, hi)

    eps = float(epsilon_mix)
    if eps > 0:
        mean_w = float(w.mean())
        w = (1.0 - eps) * w + eps * mean_w

    if float(w.sum()) <= 0 or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("Invalid weights after clip/mix.")
    return w.astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="例如 policy/ACT/processed_data/sim-open_laptop/demo_clean-200")
    parser.add_argument("--out_dir", type=str, required=True, help="输出 plan 目录，例如 policy/ACT/sampling_plans/open_laptop_demo200_uniform")

    # === anchor 构建参数 ===
    parser.add_argument("--stride", type=int, default=1, help="每条 episode 内隔 stride 帧取一个 anchor（1=全量）。")
    parser.add_argument("--plan_seed", type=int, default=0, help="影响每条 episode 的 stride 对齐 offset（保证可复现且去相关）。")

    # === stage（阶段）与权重 ===
    parser.add_argument("--num_stages", type=int, default=4, help="阶段数 K（用于阶段均衡/课程学习的基础字段）。")
    parser.add_argument(
        "--stage_balance",
        type=str,
        default="none",
        choices=["none", "uniform_over_stage", "inv_freq"],
        help="是否按阶段均衡：none=不均衡；uniform_over_stage=每个阶段总权重相等；inv_freq=按频次逆权。",
    )
    parser.add_argument("--inv_freq_power", type=float, default=1.0, help="inv_freq 的指数 power。")
    parser.add_argument("--clip_ratio", type=float, default=5.0, help="权重裁剪比（>1 才生效）。")
    parser.add_argument("--epsilon_mix", type=float, default=0.01, help="ε 混合比例（0=关闭）。")

    # === 训练采样相关元信息（训练阶段可覆盖，但写进 meta 便于复现）===
    # 默认采用“有放回采样”（replacement=True），因为这更接近当前 ACT baseline 的随机片段抽样方式。
    parser.add_argument("--no_replacement", action="store_true", help="关闭有放回采样（默认：有放回采样=True）。")
    parser.add_argument("--sampler_seed", type=int, default=0, help="训练时 sampler 的随机种子（写入 meta 便于复现）。")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    # 读取所有 episode 文件
    episode_files = sorted(dataset_dir.glob("episode_*.hdf5"))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.hdf5 found in: {dataset_dir}")

    # 这里不强依赖 episode_id 连续性：只要文件名里能解析出 id 即可
    episode_ids_all = np.array([_parse_episode_id(p) for p in episode_files], dtype=np.int64)

    stride = int(args.stride)
    if stride <= 0:
        raise ValueError("--stride must be positive.")

    all_episode_ids: list[int] = []
    all_start_ts: list[int] = []
    all_stage_ids: list[int] = []

    # === 构建 anchors ===
    for ep_id, ep_path in zip(episode_ids_all.tolist(), episode_files):
        with h5py.File(ep_path, "r") as root:
            episode_len = int(root["/action"].shape[0])
        if episode_len <= 0:
            continue

        # stride 对齐 offset：避免所有 demo 都在同一相位取样（更像 pi0 的做法）
        offset = int(_stable_hash_u64(f"{int(args.plan_seed)}:{int(ep_id)}") % stride) if stride > 1 else 0
        start_ts = np.arange(episode_len, dtype=np.int64)
        if stride > 1:
            start_ts = start_ts[((start_ts - offset) % stride) == 0]

        stage_ids = _time_quantile_stage(start_ts=start_ts, episode_len=episode_len, num_stages=int(args.num_stages))

        all_episode_ids.extend([int(ep_id)] * int(len(start_ts)))
        all_start_ts.extend(start_ts.astype(np.int64).tolist())
        all_stage_ids.extend(stage_ids.astype(np.int32).tolist())

    episode_ids = np.asarray(all_episode_ids, dtype=np.int64)
    start_ts = np.asarray(all_start_ts, dtype=np.int64)
    stage_ids = np.asarray(all_stage_ids, dtype=np.int32)
    if len(episode_ids) == 0:
        raise ValueError("Generated zero anchors. Check dataset files and parameters.")

    # === 生成 weights（默认全 1；可选按 stage 均衡）===
    if args.stage_balance == "none":
        w = np.ones((len(episode_ids),), dtype=np.float64)
    else:
        w = _apply_stage_balance(stage_ids=stage_ids, mode=str(args.stage_balance), inv_freq_power=float(args.inv_freq_power))
        w = _clip_and_mix_weights(w=w, clip_ratio=float(args.clip_ratio), epsilon_mix=float(args.epsilon_mix))

    weights = w.astype(np.float32)

    meta = {
        "plan_version": "act_anchor_v1",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_dir": str(dataset_dir),
        "num_episodes": int(len(np.unique(episode_ids))),
        "num_anchors": int(len(episode_ids)),
        "stride": int(stride),
        "plan_seed": int(args.plan_seed),
        "num_stages": int(args.num_stages),
        "stage_balance": str(args.stage_balance),
        "inv_freq_power": float(args.inv_freq_power),
        "clip_ratio": float(args.clip_ratio),
        "epsilon_mix": float(args.epsilon_mix),
        "replacement": (not bool(args.no_replacement)),
        "sampler_seed": int(args.sampler_seed),
    }

    plan = SamplingPlan(meta=meta, episode_ids=episode_ids, start_ts=start_ts, stage_ids=stage_ids, weights=weights)
    out_dir = Path(args.out_dir)
    meta_path, arrays_path = plan.save(out_dir)

    # 额外写一份简短统计（方便你快速 sanity check）
    stats = {
        "num_episodes": int(meta["num_episodes"]),
        "num_anchors": int(meta["num_anchors"]),
        "stage_hist": {str(k): int(v) for k, v in zip(*np.unique(stage_ids, return_counts=True))},
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
    }
    (out_dir / "plan_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[OK] Wrote plan:\n- {meta_path}\n- {arrays_path}\n- {out_dir/'plan_stats.json'}")
    print(f"[OK] plan_sha256={plan.recompute_sha256()}")


if __name__ == "__main__":
    main()
