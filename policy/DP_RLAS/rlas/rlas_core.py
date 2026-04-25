import json
from pathlib import Path

import numpy as np


class RLASState:
    def __init__(
        self,
        num_anchors: int,
        temperature: float = 1.5,
        epsilon_mix: float = 0.2,
        ema_beta: float = 0.5,
        max_weight: float = 5.0,
    ):
        self.num_anchors = int(num_anchors)
        self.temperature = float(temperature)
        self.epsilon_mix = float(epsilon_mix)
        self.ema_beta = float(ema_beta)
        self.max_weight = float(max_weight)

        self.baseline_losses = None
        self.weights = np.ones((self.num_anchors,), dtype=np.float32)
        self.update_count = 0
        self.last_update_step = None

    @property
    def initialized(self):
        return self.baseline_losses is not None

    def initialize(self, losses: np.ndarray, step: int):
        losses = np.asarray(losses, dtype=np.float32)
        self._check_losses(losses)
        self.baseline_losses = losses.copy()
        self.weights = np.ones_like(losses, dtype=np.float32)
        self.update_count = 0
        self.last_update_step = int(step)
        return self.weights.copy()

    def update(self, current_losses: np.ndarray, step: int):
        if self.baseline_losses is None:
            return self.initialize(current_losses, step)

        current_losses = np.asarray(current_losses, dtype=np.float32)
        self._check_losses(current_losses)

        reducible = current_losses - self.baseline_losses
        positive = np.maximum(reducible, 0.0)

        if float(positive.std()) > 1e-8:
            scores = (positive - float(positive.mean())) / float(positive.std())
        else:
            scores = np.zeros_like(positive, dtype=np.float32)

        logits = scores / self.temperature
        logits = logits - float(np.max(logits))
        prob = np.exp(logits)
        prob = prob / float(prob.sum())

        uniform = np.ones_like(prob) / len(prob)
        mixed = (1.0 - self.epsilon_mix) * prob + self.epsilon_mix * uniform

        weights = mixed / float(mixed.mean())
        weights = np.clip(weights, 1e-8, self.max_weight)
        weights = weights / float(weights.mean())
        weights = weights.astype(np.float32)

        self.baseline_losses = (
            self.ema_beta * self.baseline_losses + (1.0 - self.ema_beta) * current_losses
        ).astype(np.float32)

        self.weights = weights
        self.update_count += 1
        self.last_update_step = int(step)
        return self.weights.copy()

    def _check_losses(self, losses):
        if losses.shape != (self.num_anchors,):
            raise ValueError(f"loss shape mismatch: expected {(self.num_anchors,)}, got {losses.shape}")
        if not np.all(np.isfinite(losses)):
            raise ValueError("losses contain non-finite values")

    def stats(self):
        w = self.weights
        return {
            "rlas_update_count": int(self.update_count),
            "rlas_weight_mean": float(w.mean()),
            "rlas_weight_std": float(w.std()),
            "rlas_weight_min": float(w.min()),
            "rlas_weight_max": float(w.max()),
            "rlas_last_update_step": None if self.last_update_step is None else int(self.last_update_step),
        }

    def save_snapshot(self, out_dir, step: int, current_losses=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / f"weights_update_{step}.npy", self.weights)
        if self.baseline_losses is not None:
            np.save(out_dir / f"baseline_update_{step}.npy", self.baseline_losses)
        if current_losses is not None:
            np.save(out_dir / f"current_losses_update_{step}.npy", current_losses)

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
