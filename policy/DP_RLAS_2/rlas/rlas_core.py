import json
from pathlib import Path

import numpy as np


class RLASState:
    def __init__(
        self,
        num_anchors: int,
        window_to_group: np.ndarray,
        group_to_indices: list,
        temperature: float = 0.05,
        epsilon_mix: float = 0.2,
        ema_beta: float = 0.5,
        top_frac: float = 0.3,
    ):
        self.num_anchors = int(num_anchors)

        window_to_group = np.asarray(window_to_group, dtype=np.int64)
        if window_to_group.shape != (self.num_anchors,):
            raise ValueError(
                f"window_to_group shape mismatch: expected {(self.num_anchors,)}, got {window_to_group.shape}"
            )
        if len(group_to_indices) == 0:
            raise ValueError("group_to_indices must not be empty")

        self.window_to_group = window_to_group
        self.group_to_indices = [np.asarray(v, dtype=np.int64) for v in group_to_indices]
        self.num_groups = int(len(self.group_to_indices))

        for gid, idxs in enumerate(self.group_to_indices):
            if idxs.ndim != 1 or idxs.size == 0:
                raise ValueError(f"group {gid} must be a non-empty 1D index array")
            if np.any(idxs < 0) or np.any(idxs >= self.num_anchors):
                raise ValueError(f"group {gid} contains out-of-range window index")

        if np.any(self.window_to_group < 0) or np.any(self.window_to_group >= self.num_groups):
            raise ValueError("window_to_group contains invalid group id")

        self.temperature = float(temperature)
        self.epsilon_mix = float(epsilon_mix)
        self.ema_beta = float(ema_beta)
        self.top_frac = float(top_frac)

        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not (0.0 <= self.epsilon_mix <= 1.0):
            raise ValueError("epsilon_mix must be in [0, 1]")
        if not (0.0 <= self.ema_beta <= 1.0):
            raise ValueError("ema_beta must be in [0, 1]")
        if not (0.0 < self.top_frac <= 1.0):
            raise ValueError("top_frac must be in (0, 1]")

        self.baseline_losses = None
        self.window_weights = np.ones((self.num_anchors,), dtype=np.float32)
        self.group_weights = np.ones((self.num_groups,), dtype=np.float32)
        self.update_count = 0
        self.last_update_step = None
        self.last_reducible = None
        self.last_positive = None
        self.last_group_scores = np.zeros((self.num_groups,), dtype=np.float32)

    @property
    def initialized(self):
        return self.baseline_losses is not None

    def initialize(self, losses: np.ndarray, step: int):
        losses = np.asarray(losses, dtype=np.float32)
        self._check_losses(losses)
        self.baseline_losses = losses.copy()
        self.window_weights = np.ones((self.num_anchors,), dtype=np.float32)
        self.group_weights = np.ones((self.num_groups,), dtype=np.float32)
        self.update_count = 0
        self.last_update_step = int(step)
        self.last_reducible = np.zeros_like(losses, dtype=np.float32)
        self.last_positive = np.zeros_like(losses, dtype=np.float32)
        self.last_group_scores = np.zeros((self.num_groups,), dtype=np.float32)
        return self.group_weights.copy()

    def update(self, current_losses: np.ndarray, step: int):
        if self.baseline_losses is None:
            return self.initialize(current_losses, step)

        current_losses = np.asarray(current_losses, dtype=np.float32)
        self._check_losses(current_losses)

        reducible = current_losses - self.baseline_losses
        positive = np.maximum(reducible, 0.0)

        group_scores = np.zeros((self.num_groups,), dtype=np.float32)
        for gid, idxs in enumerate(self.group_to_indices):
            vals = positive[idxs]
            if vals.size == 0:
                group_scores[gid] = 0.0
                continue

            k = max(1, int(np.ceil(vals.size * self.top_frac)))
            top_vals = np.partition(vals, -k)[-k:]
            group_scores[gid] = float(top_vals.mean())

        logits = group_scores / self.temperature
        logits = logits - float(np.max(logits))
        prob = np.exp(logits)
        prob = prob / float(prob.sum())

        uniform = np.ones_like(prob) / len(prob)
        mixed = (1.0 - self.epsilon_mix) * prob + self.epsilon_mix * uniform

        group_weights = mixed / float(mixed.mean())
        group_weights = group_weights.astype(np.float32)
        window_weights = group_weights[self.window_to_group]

        self.baseline_losses = (
            self.ema_beta * self.baseline_losses + (1.0 - self.ema_beta) * current_losses
        ).astype(np.float32)

        self.group_weights = group_weights
        self.window_weights = window_weights.astype(np.float32)
        self.last_reducible = reducible.astype(np.float32)
        self.last_positive = positive.astype(np.float32)
        self.last_group_scores = group_scores.astype(np.float32)
        self.update_count += 1
        self.last_update_step = int(step)
        return self.group_weights.copy()

    def _check_losses(self, losses):
        if losses.shape != (self.num_anchors,):
            raise ValueError(f"loss shape mismatch: expected {(self.num_anchors,)}, got {losses.shape}")
        if not np.all(np.isfinite(losses)):
            raise ValueError("losses contain non-finite values")

    def stats(self):
        gw = self.group_weights
        ww = self.window_weights
        reducible = self.last_reducible
        positive = self.last_positive
        group_scores = self.last_group_scores
        return {
            "rlas_update_count": int(self.update_count),
            "rlas_last_update_step": None if self.last_update_step is None else int(self.last_update_step),
            "group_weight_mean": float(gw.mean()),
            "group_weight_std": float(gw.std()),
            "group_weight_min": float(gw.min()),
            "group_weight_max": float(gw.max()),
            "window_weight_mean": float(ww.mean()),
            "window_weight_std": float(ww.std()),
            "window_weight_min": float(ww.min()),
            "window_weight_max": float(ww.max()),
            "reducible_mean": 0.0 if reducible is None else float(reducible.mean()),
            "reducible_std": 0.0 if reducible is None else float(reducible.std()),
            "positive_ratio": 0.0 if positive is None else float((positive > 0).mean()),
            "positive_max": 0.0 if positive is None else float(positive.max()),
            "group_score_mean": 0.0 if group_scores is None else float(group_scores.mean()),
            "group_score_std": 0.0 if group_scores is None else float(group_scores.std()),
            "group_score_max": 0.0 if group_scores is None else float(group_scores.max()),
        }

    def save_snapshot(self, out_dir, step: int, current_losses=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / f"weights_update_{step}.npy", self.window_weights)
        np.save(out_dir / f"group_weights_update_{step}.npy", self.group_weights)
        if self.baseline_losses is not None:
            np.save(out_dir / f"baseline_update_{step}.npy", self.baseline_losses)
        if current_losses is not None:
            np.save(out_dir / f"current_losses_update_{step}.npy", current_losses)
        np.save(out_dir / f"group_scores_update_{step}.npy", self.last_group_scores)
        window_to_group_path = out_dir / "window_to_group.npy"
        if not window_to_group_path.exists():
            np.save(window_to_group_path, self.window_to_group)

        meta_path = out_dir / "weight_update_history.json"
        record = {
            "step": int(step),
            **self.stats(),
        }
        history = []
        if meta_path.exists():
            history = json.loads(meta_path.read_text())
        history.append(record)
        meta_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
