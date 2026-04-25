if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy

import tqdm, random
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from rlas.rlas_core import RLASState
from rlas.dp_anchor_scorer import DPAnchorScorer
from rlas.dynamic_sampler import DynamicMixedPriorityBatchSampler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class RobotWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        seed = cfg.training.seed
        head_camera_type = cfg.head_camera_type
        target_updates = cfg.training.target_updates
        save_every_updates = cfg.training.save_every_updates
        save_init_checkpoint = cfg.training.save_init_checkpoint
        staged_mode = target_updates is not None

        stage_ckpt_dir = None
        if staged_mode:
            if save_every_updates is None or int(save_every_updates) <= 0:
                raise ValueError("training.save_every_updates must be a positive integer when training.target_updates is set.")
            task_name = str(cfg.task.name)
            task_config = str(cfg.setting)
            expert_data_num = str(cfg.expert_data_num)
            stage_ckpt_dir = pathlib.Path("checkpoints") / task_name / f"{task_config}-u{int(target_updates)}-{expert_data_num}-seed{seed}"
            stage_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        rlas_cfg = cfg.training.get("rlas", None)
        rlas_enabled = bool(rlas_cfg is not None and bool(rlas_cfg.get("enabled", False)))

        if rlas_enabled:
            sampling_plan_path = cfg.task.dataset.get("sampling_plan_path", None)
            if sampling_plan_path is not None:
                raise ValueError(
                    "RLAS must run on native DP full train windows. "
                    "Do not enable task.dataset.sampling_plan_path together with training.rlas.enabled=True."
                )

        rlas_sampler = None
        if rlas_enabled:
            rlas_sampler = DynamicMixedPriorityBatchSampler(
                data_size=len(dataset),
                batch_size=cfg.dataloader.batch_size,
                seed=seed,
                priority_fraction=float(rlas_cfg.priority_fraction),
                drop_last=True,
            )

            def collate(x):
                assert len(x) == 1
                return x[0]

            train_dataloader = DataLoader(
                dataset,
                collate_fn=collate,
                sampler=rlas_sampler,
                num_workers=cfg.dataloader.num_workers,
                pin_memory=False,
                persistent_workers=cfg.dataloader.persistent_workers,
            )
        else:
            train_dataloader = create_dataloader(dataset, **cfg.dataloader)

        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = create_dataloader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=int(target_updates) if staged_mode else (
                (len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every
            ),
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)
        env_runner = None

        # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(save_dir=os.path.join(self.output_dir, "checkpoints"),
                                             **cfg.checkpoint.topk)

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        rlas_state = None
        rlas_scorer = None
        rlas_snapshot_dir = None
        if rlas_enabled:
            rlas_state = RLASState(
                num_anchors=len(dataset),
                temperature=float(rlas_cfg.temperature),
                epsilon_mix=float(rlas_cfg.epsilon_mix),
                ema_beta=float(rlas_cfg.ema_beta),
                max_weight=float(rlas_cfg.max_weight),
            )
            rlas_scorer = DPAnchorScorer(
                dataset=dataset,
                device=device,
                num_train_timesteps=self.model.noise_scheduler.config.num_train_timesteps,
                seed=int(rlas_cfg.scoring_seed),
            )
            if staged_mode:
                rlas_snapshot_dir = stage_ckpt_dir / str(rlas_cfg.snapshot_dirname)
            else:
                rlas_snapshot_dir = pathlib.Path(self.output_dir) / str(rlas_cfg.snapshot_dirname)
            rlas_snapshot_dir.mkdir(parents=True, exist_ok=True)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")

        with JsonLogger(log_path) as json_logger:
            if staged_mode and save_init_checkpoint and self.global_step == 0:
                init_ckpt_path = stage_ckpt_dir / f"policy_update_0_seed_{seed}.ckpt"
                self.save_checkpoint(path=init_ckpt_path, use_thread=False)

            self.optimizer.zero_grad()
            stop_training = staged_mode and (self.global_step >= int(target_updates))

            for local_epoch_idx in range(cfg.training.num_epochs):
                if stop_training:
                    break
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Training epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        if staged_mode and self.global_step >= int(target_updates):
                            stop_training = True
                            break
                        batch = dataset.postprocess(batch, device)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        should_step = ((batch_idx + 1) % cfg.training.gradient_accumulate_every == 0) or is_last_batch
                        rlas_log = {}

                        # step optimizer and update-based counters
                        if should_step:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                            self.global_step += 1

                            if rlas_enabled:
                                warmup = int(rlas_cfg.warmup_updates)
                                interval = int(rlas_cfg.update_interval)

                                should_rlas_update = False
                                if self.global_step == warmup:
                                    should_rlas_update = True
                                elif self.global_step > warmup and ((self.global_step - warmup) % interval == 0):
                                    should_rlas_update = True

                                if should_rlas_update:
                                    print(f"[RLAS] scoring all anchors at update {self.global_step} ...")
                                    current_losses = rlas_scorer.score_all(self.model)
                                    if not rlas_state.initialized:
                                        new_weights = rlas_state.initialize(current_losses, step=self.global_step)
                                        print(f"[RLAS] initialized baseline losses at update {self.global_step}")
                                    else:
                                        new_weights = rlas_state.update(current_losses, step=self.global_step)
                                        print(f"[RLAS] updated weights at update {self.global_step}")

                                    rlas_sampler.update_weights(new_weights)
                                    rlas_state.save_snapshot(
                                        rlas_snapshot_dir,
                                        step=self.global_step,
                                        current_losses=current_losses,
                                    )
                                    rlas_log.update(rlas_state.stats())

                            # update ema on optimizer step
                            if cfg.training.use_ema:
                                ema.step(self.model)

                            if staged_mode and (self.global_step % int(save_every_updates) == 0):
                                update_ckpt_path = stage_ckpt_dir / f"policy_update_{self.global_step}_seed_{seed}.ckpt"
                                self.save_checkpoint(path=update_ckpt_path, use_thread=False)

                            if staged_mode and self.global_step >= int(target_updates):
                                stop_training = True

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }
                        if rlas_log:
                            step_log.update(rlas_log)

                        if (not is_last_batch) and (not stop_training):
                            # log of last step is combined with validation and rollout
                            json_logger.log(step_log)

                        if (cfg.training.max_train_steps
                                is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                            break
                        if stop_training:
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                if len(train_losses) > 0:
                    train_loss = np.mean(train_losses)
                    step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     # log all
                #     step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                                val_dataloader,
                                desc=f"Validation epoch {self.epoch}",
                                leave=False,
                                mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dataset.postprocess(batch, device)
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps
                                        is not None) and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log["val_loss"] = val_loss

                # run diffusion sampling on a training batch
                if ((self.epoch % cfg.training.sample_every) == 0) and (train_sampling_batch is not None):
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = batch["obs"]
                        gt_action = batch["action"]

                        result = policy.predict_action(obs_dict)
                        pred_action = result["action_pred"]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # checkpoint
                if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    save_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
                    self.save_checkpoint(f"checkpoints/{save_name}-{seed}/{self.epoch + 1}.ckpt")  # TODO

                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                step_log.setdefault("global_step", self.global_step)
                step_log.setdefault("epoch", self.epoch)
                json_logger.log(step_log)
                self.epoch += 1

            if staged_mode:
                self.save_checkpoint(path=stage_ckpt_dir / "policy_last.ckpt", use_thread=False)


class BatchSampler:

    def __init__(
        self,
        data_size: int,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = True,
    ):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None

    def __iter__(self):
        if self.shuffle:
            perm = self.rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[:-self.discard]
        perm = perm.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch


class WeightedBatchSampler:

    def __init__(
        self,
        data_size: int,
        batch_size: int,
        weights,
        seed: int = 0,
        drop_last: bool = True,
    ):
        assert drop_last
        self.data_size = int(data_size)
        self.batch_size = int(batch_size)
        self.num_batch = self.data_size // self.batch_size
        self.total_samples = self.num_batch * self.batch_size
        self.rng = np.random.default_rng(seed)

        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (self.data_size,):
            raise ValueError(f"weights shape mismatch: expected ({self.data_size},), got {weights.shape}")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights contain non-finite values.")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            raise ValueError("weights sum must be > 0.")
        self.prob = weights / weight_sum

    def __iter__(self):
        sampled = self.rng.choice(self.data_size, size=self.total_samples, replace=True, p=self.prob)
        sampled = sampled.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield sampled[i]

    def __len__(self):
        return self.num_batch


class MixedPriorityBatchSampler:

    def __init__(
        self,
        data_size: int,
        batch_size: int,
        weights: np.ndarray,
        seed: int,
        priority_fraction: float = 0.2,
        drop_last: bool = True,
    ):
        self.data_size = int(data_size)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.rng = np.random.default_rng(seed)

        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (self.data_size,):
            raise ValueError(f"weights shape mismatch: expected ({self.data_size},), got {weights.shape}")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights contain non-finite values.")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            raise ValueError("weights sum must be > 0.")
        self.prob = weights / weight_sum

        pf = float(priority_fraction)
        if pf <= 0.0 or pf >= 1.0:
            raise ValueError("priority_fraction must be in (0, 1).")
        self.num_priority = max(1, int(round(self.batch_size * pf)))
        self.num_uniform = self.batch_size - self.num_priority
        if self.num_uniform <= 0:
            raise ValueError("priority_fraction too large for current batch_size.")

        if self.drop_last:
            self.num_batch = self.data_size // self.batch_size
        else:
            self.num_batch = int(np.ceil(self.data_size / float(self.batch_size)))

    def __iter__(self):
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


def create_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int = 0,
):
    sample_weights = None
    if hasattr(dataset, "get_train_sample_weights"):
        sample_weights = dataset.get_train_sample_weights()

    if shuffle and sample_weights is not None:
        batch_sampler = MixedPriorityBatchSampler(
            data_size=len(dataset),
            batch_size=batch_size,
            weights=sample_weights,
            seed=seed,
            priority_fraction=0.2,
            drop_last=True,
        )
    else:
        batch_sampler = BatchSampler(len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=True)

    def collate(x):
        assert len(x) == 1
        return x[0]

    dataloader = DataLoader(
        dataset,
        collate_fn=collate,
        sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
    )
    return dataloader


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = RobotWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
