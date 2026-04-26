import numpy as np


class DynamicGroupWeightedBatchSampler:
    def __init__(
        self,
        data_size: int,
        batch_size: int,
        group_to_indices: list,
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

        self.group_to_indices = [np.asarray(v, dtype=np.int64) for v in group_to_indices]
        self.num_groups = len(self.group_to_indices)
        if self.num_groups <= 0:
            raise ValueError("group_to_indices must not be empty")
        for gid, idxs in enumerate(self.group_to_indices):
            if idxs.ndim != 1 or idxs.size == 0:
                raise ValueError(f"group {gid} must be non-empty 1D indices")
            if np.any(idxs < 0) or np.any(idxs >= self.data_size):
                raise ValueError(f"group {gid} contains out-of-range index")

        self.group_weights = np.ones((self.num_groups,), dtype=np.float64)
        self.group_prob = self.group_weights / float(self.group_weights.sum())
        self.active = False

    def update_weights(self, group_weights):
        group_weights = np.asarray(group_weights, dtype=np.float64)
        if group_weights.shape != (self.num_groups,):
            raise ValueError(f"weights shape mismatch: expected {(self.num_groups,)}, got {group_weights.shape}")
        if not np.all(np.isfinite(group_weights)):
            raise ValueError("weights contain non-finite values")
        if np.any(group_weights <= 0):
            raise ValueError("weights must be positive")

        self.group_weights = group_weights.copy()
        self.group_prob = self.group_weights / float(self.group_weights.sum())
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
        sampled_groups = self.rng.choice(
            self.num_groups,
            size=num_samples,
            replace=True,
            p=self.group_prob,
        )
        idx = np.empty((num_samples,), dtype=np.int64)
        for i, gid in enumerate(sampled_groups):
            candidates = self.group_to_indices[gid]
            idx[i] = int(self.rng.choice(candidates))

        idx = idx.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield idx[i].astype(np.int64)

    def __len__(self):
        return self.num_batch


DynamicWeightedBatchSampler = DynamicGroupWeightedBatchSampler
DynamicMixedPriorityBatchSampler = DynamicGroupWeightedBatchSampler
