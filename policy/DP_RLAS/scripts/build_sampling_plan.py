#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path

import numpy as np

dp_root = Path(__file__).resolve().parents[1]
if str(dp_root) not in sys.path:
    sys.path.insert(0, str(dp_root))

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler
from diffusion_policy.common.sampler import create_indices, downsample_mask, get_val_mask
from sampling_plan import SamplingPlan


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DP static full-window action-motion sampling plan.")
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
        raise ValueError("No train windows found.")

    # Keep a sampler aligned with dataset semantics so sampled action chunks match training windows exactly.
    sampler = SequenceSampler(
        replay_buffer=replay_buffer,
        sequence_length=int(args.horizon),
        pad_before=int(args.pad_before),
        pad_after=int(args.pad_after),
        episode_mask=train_mask,
    )
    sampler_indices = sampler.indices.astype(np.int64, copy=False)
    if sampler_indices.shape != indices.shape or not np.array_equal(sampler_indices, indices):
        raise ValueError("create_indices and SequenceSampler indices mismatch; cannot build consistent sampling plan.")

    episode_ids = np.searchsorted(episode_ends, indices[:, 0], side="right").astype(np.int64)
    prev_ends = np.zeros_like(episode_ends)
    prev_ends[1:] = episode_ends[:-1]
    episode_start_idx = prev_ends[episode_ids]

    logical_start = (indices[:, 0] - episode_start_idx) - indices[:, 2]
    decision_ts = (logical_start + int(args.pad_before)).astype(np.int64)

    num_after = int(indices.shape[0])

    actions_list = []
    for i in range(num_after):
        sampled = sampler.sample_sequence(i)
        action = np.asarray(sampled["action"], dtype=np.float64)
        actions_list.append(action)

    all_actions = np.concatenate(actions_list, axis=0)
    action_mean = all_actions.mean(axis=0)
    action_std = all_actions.std(axis=0)
    action_std = np.maximum(action_std, 1e-6)

    raw_scores = np.empty((num_after,), dtype=np.float64)
    for i, action in enumerate(actions_list):
        action_norm = (action - action_mean) / action_std
        t_steps = int(action_norm.shape[0]) if action_norm.ndim >= 1 else 0

        diff_score = float(np.mean(np.abs(action_norm[1:] - action_norm[:-1]))) if t_steps >= 2 else 0.0
        var_score = float(np.mean(np.var(action_norm, axis=0))) if t_steps >= 1 else 0.0
        raw_scores[i] = diff_score + 0.3 * var_score

    mean_raw = float(raw_scores.mean())
    std_raw = float(raw_scores.std())
    scores = (raw_scores - mean_raw) / max(std_raw, 1e-8)

    temperature = 1.5
    uniform_mix = 0.5
    max_weight = 5.0

    logits = scores / temperature
    logits = logits - float(np.max(logits))
    biased_prob = np.exp(logits)
    biased_prob = biased_prob / float(np.sum(biased_prob))

    uniform_prob = np.ones_like(biased_prob) / float(len(biased_prob))
    final_prob = (1.0 - uniform_mix) * biased_prob + uniform_mix * uniform_prob

    weights = final_prob / float(np.mean(final_prob))
    weights = np.clip(weights, 1e-8, max_weight)
    weights = weights / float(np.mean(weights))
    weights = weights.astype(np.float32)

    stage_ids = np.zeros((num_after,), dtype=np.int32)

    meta = {
        "plan_type": "dp_window_action_static_v1",
        "zarr_path": str(args.zarr_path),
        "horizon": int(args.horizon),
        "pad_before": int(args.pad_before),
        "pad_after": int(args.pad_after),
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "max_train_episodes": None if args.max_train_episodes is None else int(args.max_train_episodes),
        "score_type": "normalized_action_motion_only",
        "score_formula": "mean_abs_diff(normalized_action) + 0.3*mean_var(normalized_action)",
        "temperature": temperature,
        "uniform_mix": uniform_mix,
        "max_weight": max_weight,
        "sampler_mix": "uniform_0.8_priority_0.2",
        "num_anchors_before_filter": num_before,
        "num_anchors_after_filter": num_after,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    plan = SamplingPlan(
        meta=meta,
        episode_ids=episode_ids.astype(np.int64),
        decision_ts=decision_ts.astype(np.int64),
        buffer_start_idx=indices[:, 0].astype(np.int64),
        buffer_end_idx=indices[:, 1].astype(np.int64),
        sample_start_idx=indices[:, 2].astype(np.int64),
        sample_end_idx=indices[:, 3].astype(np.int64),
        stage_ids=stage_ids.astype(np.int32),
        weights=weights.astype(np.float32),
    )
    out_dir = Path(args.out_dir)
    plan.save(out_dir)

    print(f"total windows before filter: {num_before}")
    print(f"total windows after filter: {num_after}")
    print(
        "raw_score stats: "
        f"mean={float(raw_scores.mean()):.6f}, "
        f"std={float(raw_scores.std()):.6f}, "
        f"min={float(raw_scores.min()):.6f}, "
        f"max={float(raw_scores.max()):.6f}"
    )
    print("score_type: normalized_action_motion_only")
    print(
        "action_mean range: "
        f"min={float(action_mean.min()):.6f}, "
        f"max={float(action_mean.max()):.6f}"
    )
    print(
        "action_std range: "
        f"min={float(action_std.min()):.6f}, "
        f"max={float(action_std.max()):.6f}"
    )
    print(f"temperature: {temperature}")
    print(f"uniform_mix: {uniform_mix}")
    print(f"max_weight: {max_weight}")
    print(
        "weights stats: "
        f"mean={float(weights.mean()):.6f}, "
        f"std={float(weights.std()):.6f}, "
        f"min={float(weights.min()):.6f}, "
        f"max={float(weights.max()):.6f}"
    )


if __name__ == "__main__":
    main()
