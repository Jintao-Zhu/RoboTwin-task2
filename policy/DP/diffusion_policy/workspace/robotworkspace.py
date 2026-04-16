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
        target_updates = cfg.training.get("target_updates", None)
        save_every_updates = cfg.training.get("save_every_updates", None)
        save_init_checkpoint = bool(cfg.training.get("save_init_checkpoint", True))
        if target_updates is not None:
            target_updates = int(target_updates)
            if target_updates < 0:
                raise ValueError("training.target_updates must be >= 0")
        if save_every_updates is not None:
            save_every_updates = int(save_every_updates)
            if save_every_updates <= 0:
                raise ValueError("training.save_every_updates must be > 0")

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
        train_dataloader = create_dataloader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = create_dataloader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        default_num_training_steps = (len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every
        scheduler_num_training_steps = default_num_training_steps
        if target_updates is not None:
            scheduler_num_training_steps = max(1, target_updates)
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=scheduler_num_training_steps,
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

        stage_ckpt_dir = None
        if target_updates is not None:
            task_name = cfg.task_name
            task_config = cfg.setting if cfg.setting is not None else "default"
            expert_data_num = cfg.expert_data_num if cfg.expert_data_num is not None else "na"
            stage_ckpt_dir = pathlib.Path("checkpoints").joinpath(
                str(task_name),
                f"{task_config}-u{target_updates}-{expert_data_num}-seed{seed}",
            )
            stage_ckpt_dir.mkdir(parents=True, exist_ok=True)

        def save_stage_update_checkpoint(update_step):
            if stage_ckpt_dir is None:
                return None
            ckpt_path = stage_ckpt_dir / f"policy_update_{update_step}_seed_{seed}.ckpt"
            self.save_checkpoint(path=ckpt_path, use_thread=False)
            return ckpt_path

        def save_stage_last_checkpoint():
            if stage_ckpt_dir is None:
                return None
            ckpt_path = stage_ckpt_dir / "policy_last.ckpt"
            self.save_checkpoint(path=ckpt_path, use_thread=False)
            return ckpt_path

        if target_updates is not None and save_init_checkpoint and self.global_step == 0:
            save_stage_update_checkpoint(0)

        self.optimizer.zero_grad()
        stop_training = target_updates is not None and self.global_step >= target_updates

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")

        with JsonLogger(log_path) as json_logger:
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
                        batch = dataset.postprocess(batch, device)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        should_step = ((batch_idx + 1) % cfg.training.gradient_accumulate_every == 0)
                        if should_step:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                            self.global_step += 1

                            # update ema after optimizer update
                            if cfg.training.use_ema:
                                ema.step(self.model)

                            if (target_updates is not None and save_every_updates is not None
                                    and self.global_step % save_every_updates == 0):
                                save_stage_update_checkpoint(self.global_step)

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

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            json_logger.log(step_log)

                        if (cfg.training.max_train_steps
                                is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                        if target_updates is not None and self.global_step >= target_updates:
                            stop_training = True
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
                if (not stop_training) and ((self.epoch % cfg.training.val_every) == 0):
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
                if (not stop_training) and ((self.epoch % cfg.training.sample_every) == 0):
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
                if target_updates is None and ((self.epoch + 1) % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    dataset_cfg = self.cfg.task.dataset
                    save_base = pathlib.Path(dataset_cfg.zarr_path).stem
                    # 根据标签来源决定保存目录名称
                    if getattr(dataset_cfg, "mix_expert_action", False):
                        if getattr(dataset_cfg, "add_expert_noise", False):
                            mode_tag = "with_rdt_mix_noise"
                        else:
                            mode_tag = "with_rdt_mix"
                    elif getattr(dataset_cfg, "use_expert_action", False):
                        if getattr(dataset_cfg, "add_expert_noise", False):
                            mode_tag = "expert_noise"
                        else:
                            mode_tag = "expert_only"
                    else:
                        mode_tag = "with_rdt_only"
                    save_name = f"{save_base}_{mode_tag}"
                    self.save_checkpoint(f"checkpoints/{save_name}/{self.epoch + 1}.ckpt")

                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                json_logger.log(step_log)
                self.epoch += 1

        if target_updates is not None:
            if (save_every_updates is None) or (self.global_step % save_every_updates != 0):
                save_stage_update_checkpoint(self.global_step)
            save_stage_last_checkpoint()


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
