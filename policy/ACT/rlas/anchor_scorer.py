"""
Anchor Loss 计算模块

本文件实现了高效批量计算所有 anchor 的 loss。
这是 RLAS 的关键组件，用于计算每个训练片段的当前损失，
进而计算 reducible loss 并更新采样权重。

设计要点：
1. 使用 torch.no_grad() 避免不必要的梯度计算
2. 可配置 batch_size 以平衡速度和显存
3. 只计算 action loss（L1），不包括 KL loss
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Callable, Optional
from tqdm import tqdm


def compute_anchor_losses(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    batch_size: int = 64,
    device: str = "cuda",
    show_progress: bool = True,
) -> np.ndarray:
    """
    计算所有 anchor 的 action loss（L1 loss）
    
    参数：
        model: ACT 策略模型（应处于 eval 模式）
        dataset: EpisodicDataset（anchor 模式，即 dataset.anchors is not None）
        batch_size: 批量大小（越大越快，但需要更多显存）
        device: 计算设备 ("cuda" 或 "cpu")
        show_progress: 是否显示进度条
        
    返回：
        losses: shape=(num_anchors,) 的 numpy 数组，每个元素是对应 anchor 的 L1 loss
        
    注意：
        - 此函数会将模型设为 eval 模式，计算完成后恢复为 train 模式
        - 使用 torch.no_grad() 避免梯度计算
    """
    was_training = model.training
    model.eval()
    
    # 创建顺序 DataLoader（不 shuffle，保证索引对应）
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 避免多进程问题
        pin_memory=True,
    )
    
    all_losses = []
    
    iterator = tqdm(loader, desc="计算 anchor losses") if show_progress else loader
    
    with torch.no_grad():
        for batch in iterator:
            image_data, qpos_data, action_data, is_pad = batch
            image_data = image_data.to(device)
            qpos_data = qpos_data.to(device)
            action_data = action_data.to(device)
            is_pad = is_pad.to(device)
            
            # 调用模型获取 per-sample loss
            per_sample_losses = _compute_per_sample_l1(
                model, qpos_data, image_data, action_data, is_pad
            )
            all_losses.append(per_sample_losses.cpu().numpy())
    
    if was_training:
        model.train()
    
    return np.concatenate(all_losses)


def _compute_per_sample_l1(
    model: torch.nn.Module,
    qpos: torch.Tensor,
    image: torch.Tensor,
    actions: torch.Tensor,
    is_pad: torch.Tensor,
) -> torch.Tensor:
    """
    计算每个样本的 L1 loss（不聚合为 batch mean）
    
    参数：
        model: ACT 策略模型
        qpos: 机器人状态 (B, state_dim)
        image: 图像数据 (B, num_cameras, C, H, W)
        actions: 动作序列 (B, T, action_dim)
        is_pad: padding mask (B, T)
        
    返回：
        per_sample_losses: (B,) 每个样本的 L1 loss
    """
    import torchvision.transforms as transforms
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    image = normalize(image)
    
    # 截断到 model.num_queries
    num_queries = model.model.num_queries
    actions = actions[:, :num_queries]
    is_pad = is_pad[:, :num_queries]
    
    # 前向传播
    env_state = None
    a_hat, is_pad_hat, (mu, logvar) = model.model(
        qpos, image, env_state, actions, is_pad
    )
    
    # 计算 per-sample L1 loss（不使用 reduction="mean"）
    # all_l1: (B, T, action_dim)
    all_l1 = torch.nn.functional.l1_loss(actions, a_hat, reduction="none")
    
    # 应用 padding mask：~is_pad 表示非 padding 部分
    # masked_l1: (B, T, action_dim)
    masked_l1 = all_l1 * (~is_pad.unsqueeze(-1))
    
    # 对每个样本求平均：(B,)
    # 分母是非 padding 的元素数
    num_valid = (~is_pad).float().sum(dim=1, keepdim=True) * actions.shape[-1]
    num_valid = num_valid.clamp(min=1)  # 避免除零
    per_sample_losses = masked_l1.sum(dim=(1, 2)) / num_valid.squeeze(-1)
    
    return per_sample_losses


def compute_anchor_losses_batched(
    forward_fn: Callable,
    dataset: torch.utils.data.Dataset,
    batch_size: int = 64,
    device: str = "cuda",
    show_progress: bool = True,
) -> np.ndarray:
    """
    使用自定义 forward 函数计算 anchor losses
    
    这是一个更灵活的接口，允许用户提供自己的前向函数。
    
    参数：
        forward_fn: 前向函数，签名为 (batch) -> per_sample_losses: Tensor (B,)
        dataset: EpisodicDataset
        batch_size: 批量大小
        device: 计算设备
        show_progress: 是否显示进度条
        
    返回：
        losses: shape=(num_anchors,) 的 numpy 数组
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    all_losses = []
    iterator = tqdm(loader, desc="计算 anchor losses") if show_progress else loader
    
    with torch.no_grad():
        for batch in iterator:
            per_sample_losses = forward_fn(batch)
            all_losses.append(per_sample_losses.cpu().numpy())
    
    return np.concatenate(all_losses)
