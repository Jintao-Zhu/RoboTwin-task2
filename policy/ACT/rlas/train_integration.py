"""
RLAS 训练集成模块

本文件提供了将 RLAS 集成到 ACT 训练流程的辅助函数。
设计原则：最小侵入，所有 RLAS 逻辑都封装在本模块中，
只需在 imitate_episodes.py 中调用几个函数即可完成集成。
"""

import os
import numpy as np
import torch
from typing import Optional, Tuple
from torch.utils.data import DataLoader

from .config import RLASConfig
from .dynamic_sampler import DynamicWeightedSampler
from .anchor_scorer import compute_anchor_losses
from .rlas_core import RLASState


def setup_rlas_visualization(
    rlas_state: RLASState,
    ckpt_dir: str,
    enable: bool = True,
) -> None:
    """
    初始化 RLAS 可视化模块
    
    参数：
        rlas_state: RLASState 实例
        ckpt_dir: checkpoint 目录
        enable: 是否启用可视化（默认 True）
    """
    if enable:
        viz_dir = os.path.join(ckpt_dir, "rlas_viz")
        rlas_state.enable_visualization(output_dir=viz_dir, enabled=True)
    else:
        rlas_state._enable_visualization = False
        print("[RLAS] 可视化已禁用")


def create_rlas_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_samples: int,
    replacement: bool = True,
    seed: int = 0,
    num_workers: int = 1,
    prefetch_factor: int = 1,
) -> Tuple[DataLoader, DynamicWeightedSampler]:
    """
    创建支持 RLAS 动态权重更新的 DataLoader
    
    参数：
        dataset: EpisodicDataset（anchor 模式）
        batch_size: 批量大小
        num_samples: 每个 epoch 采样数
        replacement: 是否有放回采样
        seed: 随机种子
        num_workers: DataLoader 工作进程数
        prefetch_factor: 预取因子
        
    返回：
        dataloader: 配置好的 DataLoader
        sampler: DynamicWeightedSampler 实例（用于后续权重更新）
    """
    num_anchors = len(dataset)
    
    sampler = DynamicWeightedSampler(
        num_anchors=num_anchors,
        num_samples=num_samples,
        initial_weights=None,  # 初始均匀权重
        replacement=replacement,
        seed=seed,
    )
    
    loader_kwargs = {"pin_memory": True, "num_workers": num_workers}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        **loader_kwargs,
    )
    
    return dataloader, sampler


def rlas_weight_update_hook(
    policy: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    sampler: DynamicWeightedSampler,
    rlas_state: RLASState,
    current_update: int,
    ckpt_dir: str,
    device: str = "cuda",
) -> bool:
    """
    RLAS 权重更新钩子函数
    
    在训练循环中调用此函数检查是否需要更新权重。
    如果需要更新，会自动计算新权重并更新 sampler。
    
    参数：
        policy: ACT 策略模型
        train_dataset: 训练集（anchor 模式的 EpisodicDataset）
        sampler: DynamicWeightedSampler 实例
        rlas_state: RLASState 状态管理器
        current_update: 当前更新步数
        ckpt_dir: checkpoint 保存目录
        device: 计算设备
        
    返回：
        updated: 是否进行了权重更新
    """
    if not rlas_state.config.enable:
        return False
    
    if not rlas_state.should_update_weights(current_update):
        return False
    
    print(f"\n[RLAS] 开始计算 anchor losses @ update {current_update}...")
    
    # 计算当前模型在所有 anchor 上的 loss
    current_losses = compute_anchor_losses(
        model=policy,
        dataset=train_dataset,
        batch_size=rlas_state.config.scoring_batch_size,
        device=device,
        show_progress=True,
    )
    
    # 更新权重
    new_weights = rlas_state.compute_and_update_weights(
        current_losses=current_losses,
        current_update=current_update,
    )
    
    # 更新 sampler
    sampler.update_weights(new_weights)
    
    # 保存快照
    rlas_state.save_snapshot(ckpt_dir, current_update)
    
    print(f"[RLAS] 权重更新完成，sampler 已同步\n")
    
    return True


def add_rlas_args(parser) -> None:
    """
    向 ArgumentParser 添加 RLAS 相关参数
    
    参数：
        parser: argparse.ArgumentParser 实例
    """
    rlas_group = parser.add_argument_group("RLAS (Reducible Loss for Anchor Selection)")
    
    rlas_group.add_argument(
        "--enable_rlas",
        action="store_true",
        help="启用 RLAS 动态采样权重（需要同时使用 --sampling_plan_path）",
    )
    rlas_group.add_argument(
        "--rlas_warmup_updates",
        type=int,
        default=10000,
        help="RLAS warmup 步数：在此之后开始计算 baseline（默认 10000）",
    )
    rlas_group.add_argument(
        "--rlas_update_interval",
        type=int,
        default=10000,
        help="RLAS 权重更新间隔（默认 10000）",
    )
    rlas_group.add_argument(
        "--rlas_temperature",
        type=float,
        default=1.0,
        help="RLAS softmax 温度（默认 1.0）",
    )
    rlas_group.add_argument(
        "--rlas_epsilon_mix",
        type=float,
        default=0.1,
        help="RLAS 与均匀分布混合比例（默认 0.1）",
    )
    rlas_group.add_argument(
        "--rlas_ema_beta",
        type=float,
        default=0.5,
        help="RLAS baseline EMA 更新速率（默认 0.5）",
    )
    rlas_group.add_argument(
        "--rlas_alpha",
        type=float,
        default=1.0,
        help="RLAS 权重幂次参数（默认 1.0）",
    )
    rlas_group.add_argument(
        "--rlas_scoring_batch_size",
        type=int,
        default=64,
        help="RLAS 计算 anchor loss 时的 batch size（默认 64）",
    )
    rlas_group.add_argument(
        "--rlas_save_snapshots",
        action="store_true",
        default=True,
        help="保存 RLAS 权重快照（默认启用）",
    )
    rlas_group.add_argument(
        "--rlas_enable_viz",
        action="store_true",
        default=True,
        help="启用 RLAS 可视化记录（默认启用）",
    )
    rlas_group.add_argument(
        "--rlas_disable_viz",
        action="store_true",
        default=False,
        help="禁用 RLAS 可视化记录",
    )
