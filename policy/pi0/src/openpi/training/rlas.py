from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


class DynamicWeightedRandomSampler(torch.utils.data.Sampler[int]):
    """
    ACT-style dynamic sampler.

    active=False:
        warmup 前均匀无放回 shuffle，等价普通 DataLoader。
    active=True:
        RLAS 更新权重后，按当前权重 full weighted replacement。
    """

    def __init__(self, data_size: int, seed: int = 0):
        self.data_size = int(data_size)
        self.rng = np.random.default_rng(int(seed))
        self.weights = np.ones((self.data_size, ), dtype=np.float64)
        self.prob = self.weights / self.weights.sum()
        self.active = False

    def update_weights(self, weights: np.ndarray) -> None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (self.data_size, ):
            raise ValueError(f"weights shape mismatch: expected {(self.data_size,)}, got {weights.shape}")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights contain non-finite values")
        if np.any(weights <= 0):
            raise ValueError("weights must be positive")

        self.weights = weights.copy()
        self.prob = self.weights / float(self.weights.sum())
        self.active = True

    def __iter__(self) -> Iterator[int]:
        if not self.active:
            idx = self.rng.permutation(self.data_size)
        else:
            idx = self.rng.choice(
                self.data_size,
                size=self.data_size,
                replace=True,
                p=self.prob,
            )
        return iter(idx.astype(np.int64).tolist())

    def __len__(self) -> int:
        return self.data_size


class RLASState:
    """
    ACT-style RLAS core:
    reducible = current_loss - baseline_loss
    positive = max(reducible, 0)
    weights = softmax(positive^alpha / temperature)
    weights = (1 - epsilon) * weights + epsilon * uniform
    baseline = ema_beta * baseline + (1 - ema_beta) * current_loss
    """

    def __init__(
        self,
        num_anchors: int,
        temperature: float = 1.0,
        epsilon_mix: float = 0.1,
        ema_beta: float = 0.5,
        alpha: float = 1.0,
    ):
        self.num_anchors = int(num_anchors)
        self.temperature = float(temperature)
        self.epsilon_mix = float(epsilon_mix)
        self.ema_beta = float(ema_beta)
        self.alpha = float(alpha)

        self.baseline_losses: np.ndarray | None = None
        self.weights = np.ones((self.num_anchors, ), dtype=np.float32)
        self.update_count = 0
        self.last_update_step: int | None = None

        self.last_reducible: np.ndarray | None = None
        self.last_positive: np.ndarray | None = None

    @property
    def initialized(self) -> bool:
        return self.baseline_losses is not None

    def initialize(self, losses: np.ndarray, step: int) -> np.ndarray:
        losses = np.asarray(losses, dtype=np.float32)
        self._check_losses(losses)

        self.baseline_losses = losses.copy()
        self.weights = np.ones_like(losses, dtype=np.float32)
        self.update_count = 0
        self.last_update_step = int(step)

        self.last_reducible = np.zeros_like(losses, dtype=np.float32)
        self.last_positive = np.zeros_like(losses, dtype=np.float32)

        return self.weights.copy()

    def update(self, current_losses: np.ndarray, step: int) -> np.ndarray:
        if self.baseline_losses is None:
            return self.initialize(current_losses, step)

        current_losses = np.asarray(current_losses, dtype=np.float32)
        self._check_losses(current_losses)

        reducible = current_losses - self.baseline_losses
        positive = np.maximum(reducible, 0.0)

        if self.alpha != 1.0:
            score = np.power(positive, self.alpha)
        else:
            score = positive

        logits = score / max(self.temperature, 1e-12)
        logits = logits - float(np.max(logits))
        prob = np.exp(logits)
        prob = prob / float(np.sum(prob))

        uniform = np.ones_like(prob) / len(prob)
        mixed = (1.0 - self.epsilon_mix) * prob + self.epsilon_mix * uniform

        # 用均值归一化权重，方便日志解释：1.0 = 等价均匀采样
        weights = mixed / float(np.mean(mixed))
        weights = weights.astype(np.float32)

        self.baseline_losses = (
            self.ema_beta * self.baseline_losses + (1.0 - self.ema_beta) * current_losses
        ).astype(np.float32)

        self.weights = weights
        self.last_reducible = reducible.astype(np.float32)
        self.last_positive = positive.astype(np.float32)
        self.update_count += 1
        self.last_update_step = int(step)

        return self.weights.copy()

    def _check_losses(self, losses: np.ndarray) -> None:
        if losses.shape != (self.num_anchors, ):
            raise ValueError(f"loss shape mismatch: expected {(self.num_anchors,)}, got {losses.shape}")
        if not np.all(np.isfinite(losses)):
            raise ValueError("losses contain non-finite values")

    def stats(self) -> dict:
        w = self.weights
        reducible = self.last_reducible
        positive = self.last_positive

        if reducible is None:
            reducible = np.zeros_like(w)
        if positive is None:
            positive = np.zeros_like(w)

        return {
            "rlas_update_count": int(self.update_count),
            "rlas_last_update_step": None if self.last_update_step is None else int(self.last_update_step),
            "rlas_weight_mean": float(w.mean()),
            "rlas_weight_std": float(w.std()),
            "rlas_weight_min": float(w.min()),
            "rlas_weight_max": float(w.max()),
            "reducible_mean": float(reducible.mean()),
            "reducible_std": float(reducible.std()),
            "reducible_min": float(reducible.min()),
            "reducible_max": float(reducible.max()),
            "positive_ratio": float((positive > 0).mean()),
            "positive_mean": float(positive.mean()),
            "positive_max": float(positive.max()),
        }

    def save_snapshot(self, out_dir: str | Path, step: int, current_losses: np.ndarray | None = None) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / f"weights_update_{step}.npy", self.weights)

        if self.baseline_losses is not None:
            np.save(out_dir / f"baseline_update_{step}.npy", self.baseline_losses)

        if current_losses is not None:
            np.save(out_dir / f"current_losses_update_{step}.npy", current_losses)

        if self.last_reducible is not None:
            np.save(out_dir / f"reducible_update_{step}.npy", self.last_reducible)

        meta_path = out_dir / "weight_update_history.json"
        record = {
            "step": int(step),
            **self.stats(),
        }

        history = []
        if meta_path.exists():
            history = json.loads(meta_path.read_text(encoding="utf-8"))

        history.append(record)
        meta_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
