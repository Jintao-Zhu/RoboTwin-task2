import numpy as np
import torch


class DPAnchorScorer:
    def __init__(self, dataset, device, num_train_timesteps: int, seed: int = 12345):
        self.dataset = dataset
        self.device = torch.device(device)
        self.num_anchors = len(dataset)
        self.seed = int(seed)

        self.batch_size = int(dataset.batch_size)
        self.horizon = int(dataset.horizon)
        self.action_dim = int(dataset.replay_buffer["action"].shape[-1])

        rng = torch.Generator(device="cpu")
        rng.manual_seed(self.seed)

        self.noise_bank = torch.randn(
            self.num_anchors,
            self.horizon,
            self.action_dim,
            generator=rng,
            dtype=torch.float32,
        )
        self.timestep_bank = torch.randint(
            low=0,
            high=int(num_train_timesteps),
            size=(self.num_anchors,),
            generator=rng,
            dtype=torch.long,
        )

    @torch.no_grad()
    def score_all(self, policy):
        was_training = policy.training
        policy.eval()

        losses = np.empty((self.num_anchors,), dtype=np.float32)

        for start in range(0, self.num_anchors, self.batch_size):
            end = min(start + self.batch_size, self.num_anchors)
            real_count = end - start

            idx = np.arange(start, end, dtype=np.int64)

            # Dataset batched __getitem__ requires len(idx)==dataset.batch_size.
            if real_count < self.batch_size:
                pad = np.full((self.batch_size - real_count,), end - 1, dtype=np.int64)
                idx_padded = np.concatenate([idx, pad], axis=0)
            else:
                idx_padded = idx

            raw_batch = self.dataset[idx_padded]
            batch = self.dataset.postprocess(raw_batch, self.device)

            noise = self.noise_bank[idx_padded].to(self.device, non_blocking=True)
            timesteps = self.timestep_bank[idx_padded].to(self.device, non_blocking=True)

            loss = policy.compute_loss_per_sample(
                batch,
                noise=noise,
                timesteps=timesteps,
            )
            loss_np = loss.detach().float().cpu().numpy()

            losses[start:end] = loss_np[:real_count]

        if was_training:
            policy.train()

        return losses
