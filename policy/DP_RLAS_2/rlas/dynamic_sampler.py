import numpy as np


class DynamicWeightedBatchSampler:
    def __init__(
        self,
        data_size: int,
        batch_size: int,
        seed: int = 0,
        drop_last: bool = True,
    ):
        assert drop_last
        self.data_size = int(data_size)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.num_batch = self.data_size // self.batch_size
        self.discard = self.data_size - self.batch_size * self.num_batch
        self.rng = np.random.default_rng(seed)

        self.weights = np.ones((self.data_size,), dtype=np.float64)
        self.prob = self.weights / float(self.weights.sum())
        self.active = False

    def update_weights(self, weights):
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (self.data_size,):
            raise ValueError(f"weights shape mismatch: expected {(self.data_size,)}, got {weights.shape}")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights contain non-finite values")
        if np.any(weights <= 0):
            raise ValueError("weights must be positive")

        self.weights = weights.copy()
        self.prob = self.weights / float(self.weights.sum())
        self.active = True

    def __iter__(self):
        if not self.active:
            perm = self.rng.permutation(self.data_size)
            if self.discard > 0:
                perm = perm[:-self.discard]
            perm = perm.reshape(self.num_batch, self.batch_size)
            for i in range(self.num_batch):
                yield perm[i].astype(np.int64)
            return

        num_samples = self.num_batch * self.batch_size
        idx = self.rng.choice(
            self.data_size,
            size=num_samples,
            replace=True,
            p=self.prob,
        )
        idx = idx.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield idx[i].astype(np.int64)

    def __len__(self):
        return self.num_batch


DynamicMixedPriorityBatchSampler = DynamicWeightedBatchSampler
