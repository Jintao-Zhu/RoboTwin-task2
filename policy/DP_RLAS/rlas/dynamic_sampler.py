import numpy as np


class DynamicMixedPriorityBatchSampler:
    def __init__(
        self,
        data_size: int,
        batch_size: int,
        seed: int = 0,
        priority_fraction: float = 0.2,
        drop_last: bool = True,
    ):
        assert drop_last
        self.data_size = int(data_size)
        self.batch_size = int(batch_size)
        self.priority_fraction = float(priority_fraction)
        self.drop_last = bool(drop_last)
        self.num_batch = self.data_size // self.batch_size
        self.discard = self.data_size - self.batch_size * self.num_batch
        self.rng = np.random.default_rng(seed)

        self.weights = np.ones((self.data_size,), dtype=np.float64)
        self.prob = self.weights / float(self.weights.sum())
        self.active = False

        self.num_priority = max(1, int(round(self.batch_size * self.priority_fraction)))
        self.num_uniform = self.batch_size - self.num_priority
        if self.num_uniform <= 0:
            raise ValueError("priority_fraction too large")

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

        uniform_perm = self.rng.permutation(self.data_size)
        cursor = 0

        for _ in range(self.num_batch):
            need = self.num_uniform
            chunks = []
            while need > 0:
                remain = self.data_size - cursor
                if remain <= 0:
                    uniform_perm = self.rng.permutation(self.data_size)
                    cursor = 0
                    remain = self.data_size
                take = min(need, remain)
                chunks.append(uniform_perm[cursor:cursor + take])
                cursor += take
                need -= take

            uniform_idx = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
            priority_idx = self.rng.choice(
                self.data_size,
                size=self.num_priority,
                replace=True,
                p=self.prob,
            )

            batch = np.concatenate([uniform_idx, priority_idx], axis=0)
            self.rng.shuffle(batch)
            yield batch.astype(np.int64)

    def __len__(self):
        return self.num_batch
