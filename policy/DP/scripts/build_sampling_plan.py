#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime
import sys
from pathlib import Path

import numpy as np

dp_root = Path(__file__).resolve().parents[1]
if str(dp_root) not in sys.path:
    sys.path.insert(0, str(dp_root))

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import create_indices, downsample_mask, get_val_mask
from sampling_plan import SamplingPlan


def stable_phase_offset(zarr_path: str, episode_id: int) -> int:
    raw = f"{zarr_path}|{episode_id}|static_stride2".encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DP static window-level sampling plan.")
    parser.add_argument("--zarr_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--pad_before", type=int, required=True)
    parser.add_argument("--pad_after", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--val_ratio", type=float, required=True)
    parser.add_argument("--max_train_episodes", type=int, default=None)
    args = parser.parse_args()

    replay_buffer = ReplayBuffer.copy_from_path(
        args.zarr_path,
        keys=["head_camera", "state", "action"],
    )

    n_episodes = int(replay_buffer.n_episodes)
    val_mask = get_val_mask(n_episodes=n_episodes, val_ratio=args.val_ratio, seed=args.seed)
    train_mask = ~val_mask
    train_mask = downsample_mask(mask=train_mask, max_n=args.max_train_episodes, seed=args.seed)

    episode_ends = replay_buffer.episode_ends[:].astype(np.int64)
    indices = create_indices(
        episode_ends,
        sequence_length=int(args.horizon),
        episode_mask=train_mask,
        pad_before=int(args.pad_before),
        pad_after=int(args.pad_after),
    ).astype(np.int64)

    num_before = int(indices.shape[0])
    if num_before <= 0:
        raise ValueError("No train windows found before stride filtering.")

    episode_ids = np.searchsorted(episode_ends, indices[:, 0], side="right").astype(np.int64)
    prev_ends = np.zeros_like(episode_ends)
    prev_ends[1:] = episode_ends[:-1]
    episode_start_idx = prev_ends[episode_ids]
    episode_length = (episode_ends[episode_ids] - episode_start_idx).astype(np.int64)

    logical_start = (indices[:, 0] - episode_start_idx) - indices[:, 2]
    decision_ts = (logical_start + int(args.pad_before)).astype(np.int64)

    phase_offsets = np.array(
        [stable_phase_offset(args.zarr_path, int(ep_id)) for ep_id in episode_ids.tolist()],
        dtype=np.int64,
    )
    keep_mask = ((decision_ts - phase_offsets) % 2) == 0

    kept_indices = indices[keep_mask]
    kept_episode_ids = episode_ids[keep_mask]
    kept_decision_ts = decision_ts[keep_mask]
    kept_episode_length = episode_length[keep_mask]

    num_after = int(kept_indices.shape[0])
    if num_after <= 0:
        raise ValueError("No windows left after stride=2 filtering.")

    denom = np.maximum(1, kept_episode_length - 1)
    stage_ids = np.floor(4.0 * kept_decision_ts.astype(np.float64) / denom.astype(np.float64)).astype(np.int64)
    stage_ids = np.clip(stage_ids, 0, 3).astype(np.int32)

    stage_counts = np.bincount(stage_ids, minlength=4).astype(np.int64)
    inv_counts = 1.0 / stage_counts[stage_ids]
    weights = inv_counts.astype(np.float64)
    weights = weights / float(weights.mean())
    weights = weights.astype(np.float32)

    per_stage_weight = np.zeros(4, dtype=np.float64)
    for s in range(4):
        per_stage_weight[s] = float(weights[stage_ids == s].sum())

    meta = {
        "plan_type": "dp_window_static_v1",
        "zarr_path": str(args.zarr_path),
        "horizon": int(args.horizon),
        "pad_before": int(args.pad_before),
        "pad_after": int(args.pad_after),
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "max_train_episodes": None if args.max_train_episodes is None else int(args.max_train_episodes),
        "stride": 2,
        "num_stages": 4,
        "weighting": "uniform_over_stage_after_stride",
        "num_anchors_before_stride": num_before,
        "num_anchors_after_stride": num_after,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    plan = SamplingPlan(
        meta=meta,
        episode_ids=kept_episode_ids.astype(np.int64),
        decision_ts=kept_decision_ts.astype(np.int64),
        buffer_start_idx=kept_indices[:, 0].astype(np.int64),
        buffer_end_idx=kept_indices[:, 1].astype(np.int64),
        sample_start_idx=kept_indices[:, 2].astype(np.int64),
        sample_end_idx=kept_indices[:, 3].astype(np.int64),
        stage_ids=stage_ids.astype(np.int32),
        weights=weights.astype(np.float32),
    )
    out_dir = Path(args.out_dir)
    plan.save(out_dir)

    print(f"total windows before stride: {num_before}")
    print(f"total windows after stride: {num_after}")
    print("per-stage counts:")
    for s in range(4):
        print(f"  stage {s}: {int(stage_counts[s])}")
    print("per-stage total weight:")
    for s in range(4):
        print(f"  stage {s}: {per_stage_weight[s]:.6f}")


if __name__ == "__main__":
    main()
