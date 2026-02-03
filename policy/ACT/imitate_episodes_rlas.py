"""
RLAS 增强版训练脚本

本文件是 imitate_episodes.py 的 RLAS 增强版本。
为了保持代码整洁和最小侵入，RLAS 相关逻辑都封装在本文件中，
原始 imitate_episodes.py 保持不变。

使用方法：
    # 启用 RLAS 训练
    python imitate_episodes_rlas.py \
        --task_name sim-stack_blocks_two-demo_clean-200 \
        --ckpt_dir ./act_ckpt/... \
        --policy_class ACT \
        --sampling_plan_path ./sampling_plans/... \
        --enable_rlas \
        --rlas_warmup_updates 10000 \
        --rlas_update_interval 10000 \
        --target_updates 120000 \
        ...
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import math
import torch
import numpy as np
import pickle
import argparse
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data_with_rlas_support, set_seed, compute_dict_mean, detach_dict
from act_policy import ACTPolicy, CNNMLPPolicy

# RLAS 模块
from rlas import RLASConfig, RLASState, rlas_weight_update_hook, add_rlas_args, setup_rlas_visualization


def main(args):
    set_seed(1)
    
    # 基础参数
    is_eval = args["eval"]
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_epochs = args["num_epochs"]

    # 任务配置
    is_sim = task_name[:4] == "sim-"
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    
    dataset_dir = task_config["dataset_dir"]
    num_episodes = task_config["num_episodes"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]

    # 策略配置
    state_dim = args.get("state_dim", 14)
    lr_backbone = 1e-5
    backbone = "resnet18"
    
    if policy_class == "ACT":
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": 4,
            "dec_layers": 7,
            "nheads": 8,
            "camera_names": camera_names,
        }
    else:
        raise NotImplementedError(f"RLAS 当前只支持 ACT 策略，不支持 {policy_class}")

    # RLAS 配置
    rlas_config = RLASConfig.from_args(args)
    
    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args.get("temporal_agg", False),
        "camera_names": camera_names,
        "target_updates": args.get("target_updates"),
        "save_every_updates": args.get("save_every_updates"),
        "log_every_updates": args.get("log_every_updates"),
        "eval_every_updates": args.get("eval_every_updates"),
        "rlas_config": rlas_config,
        "rlas_disable_viz": args.get("rlas_disable_viz", False),
    }

    if is_eval:
        print("RLAS 训练脚本不支持 --eval 模式，请使用原始 imitate_episodes.py")
        exit(1)

    # 加载数据（RLAS 版本）
    train_dataloader, val_dataloader, stats, is_sim, train_dataset, dynamic_sampler = \
        load_data_with_rlas_support(
            dataset_dir,
            num_episodes,
            camera_names,
            batch_size_train,
            batch_size_val,
            sampling_plan_path=args.get("sampling_plan_path"),
            plan_num_samples=int(args.get("plan_num_samples") or 0),
            plan_sampler_seed=int(args.get("plan_sampler_seed") or 0),
            plan_no_replacement=bool(args.get("plan_no_replacement")),
            plan_verify_sha256=(not bool(args.get("plan_no_verify_sha256"))),
            enable_rlas=rlas_config.enable,
        )

    # 验证 RLAS 配置
    if rlas_config.enable:
        if args.get("sampling_plan_path") is None:
            raise ValueError("启用 RLAS (--enable_rlas) 需要同时指定 --sampling_plan_path")
        if dynamic_sampler is None:
            raise ValueError("RLAS 模式下 dynamic_sampler 不应为 None")
        print(f"\n[RLAS] 配置：{rlas_config.to_dict()}\n")

    # 保存数据集统计
    os.makedirs(ckpt_dir, exist_ok=True)
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    
    # 保存 RLAS 配置
    if rlas_config.enable:
        rlas_config_path = os.path.join(ckpt_dir, "rlas_config.json")
        with open(rlas_config_path, "w", encoding="utf-8") as f:
            json.dump(rlas_config.to_dict(), f, indent=2)

    # 训练
    best_ckpt_info = train_bc_rlas(
        train_dataloader, val_dataloader, config,
        train_dataset=train_dataset,
        dynamic_sampler=dynamic_sampler,
    )
    
    best_step, min_val_loss, best_state_dict = best_ckpt_info

    # 保存最佳 checkpoint
    ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ update {best_step}")


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data = image_data.cuda()
    qpos_data = qpos_data.cuda()
    action_data = action_data.cuda()
    is_pad = is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc_rlas(train_dataloader, val_dataloader, config, train_dataset=None, dynamic_sampler=None):
    """
    RLAS 增强版训练函数
    
    与原始 train_bc 的区别：
    1. 在 warmup 结束后计算 baseline losses
    2. 每隔 update_interval 更新采样权重
    3. 使用 EMA 更新 baseline
    """
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    target_updates = config.get("target_updates")
    save_every_updates = config.get("save_every_updates")
    log_every_updates = config.get("log_every_updates")
    eval_every_updates = config.get("eval_every_updates")
    rlas_config = config.get("rlas_config", RLASConfig())

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    # RLAS 状态
    rlas_state = RLASState(config=rlas_config) if rlas_config.enable else None
    
    # 初始化 RLAS 可视化（默认启用）
    if rlas_state is not None:
        enable_viz = not config.get("rlas_disable_viz", False)
        setup_rlas_visualization(rlas_state, ckpt_dir, enable=enable_viz)

    def run_validation(step_tag: str):
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for data in val_dataloader:
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
        print(f"{step_tag} Val loss: {epoch_summary['loss']:.5f}")
        return epoch_summary

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    # 必须使用 update-based 模式
    if target_updates is None:
        raise ValueError("RLAS 训练必须使用 --target_updates 参数指定训练预算")

    target_updates = int(target_updates)
    steps_per_epoch = max(1, len(train_dataloader))
    needed_epochs = math.ceil(target_updates / steps_per_epoch)
    num_epochs = needed_epochs

    # 默认值设置
    if not eval_every_updates or int(eval_every_updates) <= 0:
        eval_every_updates = max(100, steps_per_epoch * 20)
    else:
        eval_every_updates = int(eval_every_updates)

    if not save_every_updates or int(save_every_updates) <= 0:
        save_every_updates = max(5000, eval_every_updates)
    else:
        save_every_updates = int(save_every_updates)

    if not log_every_updates or int(log_every_updates) <= 0:
        log_every_updates = min(50, eval_every_updates)
    else:
        log_every_updates = int(log_every_updates)

    print(
        f"[train] target_updates={target_updates}, steps_per_epoch={steps_per_epoch}, "
        f"max_epochs={num_epochs}, eval_every={eval_every_updates}, "
        f"save_every={save_every_updates}, log_every={log_every_updates}"
    )
    if rlas_config.enable:
        print(
            f"[RLAS] warmup={rlas_config.warmup_updates}, "
            f"update_interval={rlas_config.update_interval}, "
            f"temperature={rlas_config.temperature}, epsilon_mix={rlas_config.epsilon_mix}"
        )

    update = 0
    val_update_steps = []

    # 初始验证
    epoch_summary = run_validation(step_tag=f"[update {update}]")
    validation_history.append(epoch_summary)
    val_update_steps.append(update)
    epoch_val_loss = float(epoch_summary["loss"])
    if epoch_val_loss < min_val_loss:
        min_val_loss = epoch_val_loss
        best_ckpt_info = (update, min_val_loss, deepcopy(policy.state_dict()))

    policy.train()
    optimizer.zero_grad()
    window_losses = []

    for epoch in tqdm(range(num_epochs), desc="Training"):
        for data in train_dataloader:
            forward_dict = forward_pass(data, policy)
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            update += 1
            train_history.append(detach_dict(forward_dict))
            window_losses.append(float(loss.detach().cpu()))

            # === RLAS 权重更新钩子 ===
            if rlas_config.enable and rlas_state is not None and dynamic_sampler is not None:
                rlas_weight_update_hook(
                    policy=policy,
                    train_dataset=train_dataset,
                    sampler=dynamic_sampler,
                    rlas_state=rlas_state,
                    current_update=update,
                    ckpt_dir=ckpt_dir,
                    device="cuda",
                )

            # 日志
            if log_every_updates and (update % log_every_updates == 0) and window_losses:
                avg_loss = sum(window_losses) / len(window_losses)
                rlas_info = ""
                if rlas_config.enable and rlas_state is not None:
                    rlas_info = f" | RLAS updates: {rlas_state.total_weight_updates}"
                print(f"[update {update}] Train loss(avg): {avg_loss:.5f}{rlas_info}")
                window_losses = []

            # 保存
            if save_every_updates and (update % save_every_updates == 0):
                ckpt_path = os.path.join(ckpt_dir, f"policy_update_{update}_seed_{seed}.ckpt")
                torch.save(policy.state_dict(), ckpt_path)
                
                # 同时保存 RLAS 状态
                if rlas_config.enable and rlas_state is not None:
                    rlas_state.save_snapshot(ckpt_dir, update)

            # 验证
            if eval_every_updates and (update % eval_every_updates == 0):
                epoch_summary = run_validation(step_tag=f"[update {update}]")
                validation_history.append(epoch_summary)
                val_update_steps.append(update)
                epoch_val_loss = float(epoch_summary["loss"])
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (update, min_val_loss, deepcopy(policy.state_dict()))
                policy.train()

            if update >= target_updates:
                break
        if update >= target_updates:
            break

    # 最终验证
    if val_update_steps and val_update_steps[-1] != update:
        epoch_summary = run_validation(step_tag=f"[update {update}]")
        validation_history.append(epoch_summary)
        val_update_steps.append(update)
        epoch_val_loss = float(epoch_summary["loss"])
        if epoch_val_loss < min_val_loss:
            min_val_loss = epoch_val_loss
            best_ckpt_info = (update, min_val_loss, deepcopy(policy.state_dict()))

    # 保存最终模型
    ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)

    # 保存 RLAS 最终状态
    if rlas_config.enable and rlas_state is not None:
        rlas_state.save_snapshot(ckpt_dir, update)
        rlas_state_path = os.path.join(ckpt_dir, "rlas_final_state.json")
        with open(rlas_state_path, "w", encoding="utf-8") as f:
            json.dump(rlas_state.state_dict(), f, indent=2, default=str)

    best_update, min_val_loss, best_state_dict = best_ckpt_info
    print(f"Training finished: Seed {seed}, val loss {min_val_loss:.6f} at update {best_update}")

    return best_ckpt_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLAS 增强版 ACT 训练脚本")
    
    # 基础参数
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--policy_class", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    
    # update-based 训练
    parser.add_argument("--target_updates", type=int, required=True)
    parser.add_argument("--eval_every_updates", type=int, default=0)
    parser.add_argument("--save_every_updates", type=int, default=0)
    parser.add_argument("--log_every_updates", type=int, default=0)
    
    # Sampling plan
    parser.add_argument("--sampling_plan_path", type=str, default=None)
    parser.add_argument("--plan_num_samples", type=int, default=0)
    parser.add_argument("--plan_sampler_seed", type=int, default=0)
    parser.add_argument("--plan_no_replacement", action="store_true")
    parser.add_argument("--plan_no_verify_sha256", action="store_true")
    
    # ACT 特定参数
    parser.add_argument("--kl_weight", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--state_dim", type=int, default=14)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--temporal_agg", action="store_true")
    
    # RLAS 参数
    add_rlas_args(parser)
    
    main(vars(parser.parse_args()))
