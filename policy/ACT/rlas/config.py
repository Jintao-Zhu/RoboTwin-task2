"""
RLAS 配置模块

本文件定义了 RLAS 的所有超参数配置。
超参数设计基于 RHO-LOSS 论文的最佳实践。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RLASConfig:
    """
    RLAS 超参数配置类
    
    属性说明：
    - enable: 是否启用 RLAS（默认 False，与 baseline 保持兼容）
    - warmup_updates: 热身阶段更新数，在此之后计算 baseline loss
    - update_interval: 权重更新间隔（每隔多少 updates 重新计算权重）
    - temperature: softmax 温度参数，控制权重分布的平滑程度
        - temperature 越大，权重分布越均匀
        - temperature 越小，高 reducible loss 的样本权重越突出
    - epsilon_mix: 与均匀分布混合的比例 [0, 1]
        - 最终权重 = (1 - epsilon_mix) * rlas_weights + epsilon_mix * uniform_weights
        - 确保所有样本都有机会被采样，避免完全忽略某些样本
    - ema_beta: baseline EMA 更新速率 [0, 1]
        - baseline_new = ema_beta * baseline_old + (1 - ema_beta) * current
        - ema_beta 越大，baseline 更新越慢
    - alpha: 权重幂次参数（可选，用于调整权重的锐度）
    - scoring_batch_size: 计算 anchor loss 时的 batch size（越大越快，但需要更多显存）
    """
    
    enable: bool = False
    warmup_updates: int = 10000
    update_interval: int = 10000
    temperature: float = 1.0
    epsilon_mix: float = 0.1
    ema_beta: float = 0.5
    alpha: float = 1.0
    scoring_batch_size: int = 64
    
    # 可选：保存权重快照便于分析
    save_weight_snapshots: bool = True
    
    def __post_init__(self):
        """参数校验"""
        assert self.warmup_updates >= 0, f"warmup_updates 必须 >= 0, 当前值: {self.warmup_updates}"
        assert self.update_interval > 0, f"update_interval 必须 > 0, 当前值: {self.update_interval}"
        assert self.temperature > 0, f"temperature 必须 > 0, 当前值: {self.temperature}"
        assert 0 <= self.epsilon_mix <= 1, f"epsilon_mix 必须在 [0, 1] 范围内, 当前值: {self.epsilon_mix}"
        assert 0 <= self.ema_beta <= 1, f"ema_beta 必须在 [0, 1] 范围内, 当前值: {self.ema_beta}"
        assert self.alpha > 0, f"alpha 必须 > 0, 当前值: {self.alpha}"
        assert self.scoring_batch_size > 0, f"scoring_batch_size 必须 > 0, 当前值: {self.scoring_batch_size}"

    @classmethod
    def from_args(cls, args: dict) -> "RLASConfig":
        """
        从命令行参数字典创建配置
        
        参数：
            args: 包含 RLAS 相关参数的字典
            
        返回：
            RLASConfig 实例
        """
        return cls(
            enable=bool(args.get("enable_rlas", False)),
            warmup_updates=int(args.get("rlas_warmup_updates", 10000)),
            update_interval=int(args.get("rlas_update_interval", 10000)),
            temperature=float(args.get("rlas_temperature", 1.0)),
            epsilon_mix=float(args.get("rlas_epsilon_mix", 0.1)),
            ema_beta=float(args.get("rlas_ema_beta", 0.5)),
            alpha=float(args.get("rlas_alpha", 1.0)),
            scoring_batch_size=int(args.get("rlas_scoring_batch_size", 64)),
            save_weight_snapshots=bool(args.get("rlas_save_snapshots", True)),
        )

    def to_dict(self) -> dict:
        """转换为字典，便于序列化"""
        return {
            "enable": self.enable,
            "warmup_updates": self.warmup_updates,
            "update_interval": self.update_interval,
            "temperature": self.temperature,
            "epsilon_mix": self.epsilon_mix,
            "ema_beta": self.ema_beta,
            "alpha": self.alpha,
            "scoring_batch_size": self.scoring_batch_size,
            "save_weight_snapshots": self.save_weight_snapshots,
        }
