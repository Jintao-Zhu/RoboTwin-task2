"""
RLAS (Reducible Loss for Anchor Selection) 模块

本模块实现了基于 RHO-LOSS 论文思想的动态样本权重调整策略。
核心思想：优先训练"可学习、值得学习、尚未学会"的样本。

主要组件：
- config.py: RLAS 超参数配置
- dynamic_sampler.py: 支持运行时权重更新的采样器
- anchor_scorer.py: 高效计算每个 anchor 的 loss
- rlas_core.py: RLAS 核心算法（权重计算、EMA 更新等）

使用方法：
1. 在训练脚本中启用 --enable_rlas
2. 设置 warmup、update_interval 等超参数
3. 训练过程中会自动动态调整采样权重
"""

from .config import RLASConfig
from .dynamic_sampler import DynamicWeightedSampler
from .anchor_scorer import compute_anchor_losses, compute_anchor_losses_batched
from .rlas_core import (
    compute_reducible_losses,
    losses_to_weights,
    ema_update,
    RLASState,
)
from .train_integration import (
    create_rlas_dataloader,
    rlas_weight_update_hook,
    add_rlas_args,
    setup_rlas_visualization,
)
from .visualizer import RLASVisualizer

__all__ = [
    "RLASConfig",
    "DynamicWeightedSampler",
    "compute_anchor_losses",
    "compute_anchor_losses_batched",
    "compute_reducible_losses",
    "losses_to_weights",
    "ema_update",
    "RLASState",
    "create_rlas_dataloader",
    "rlas_weight_update_hook",
    "add_rlas_args",
    "setup_rlas_visualization",
    "RLASVisualizer",
]
