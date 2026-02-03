"""
RLAS 核心算法模块

本文件实现了 RLAS 的核心算法逻辑：
1. Reducible Loss 计算
2. Loss 到采样权重的转换
3. EMA 更新 baseline
4. RLASState 状态管理

理论背景（RHO-LOSS, ICML 2022）：
- Reducible Loss = Current Loss - Baseline Loss
- 优先训练 reducible loss 高的样本（"还没学会但应该学会"的样本）
- 使用 softmax 将 reducible loss 转换为采样概率
"""

import numpy as np
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, TYPE_CHECKING
from .config import RLASConfig

# 延迟导入以避免循环依赖
if TYPE_CHECKING:
    from .visualizer import RLASVisualizer


def compute_reducible_losses(
    current_losses: np.ndarray,
    baseline_losses: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    计算 Reducible Loss
    
    公式：reducible_loss = current_loss - baseline_loss
    
    直觉解释：
    - reducible > 0：当前模型在这个样本上比 baseline 差 → 还没学会，值得优先学
    - reducible <= 0：当前模型已经比 baseline 好 → 已经学会，不需要优先学
    
    参数：
        current_losses: 当前模型在每个 anchor 上的 loss，shape=(num_anchors,)
        baseline_losses: baseline 模型在每个 anchor 上的 loss，shape=(num_anchors,)
        epsilon: 数值稳定性常数（避免除零等问题）
        
    返回：
        reducible_losses: shape=(num_anchors,) 的 reducible loss 数组
    """
    assert len(current_losses) == len(baseline_losses), \
        f"current_losses ({len(current_losses)}) 和 baseline_losses ({len(baseline_losses)}) 长度不匹配"
    
    reducible = current_losses - baseline_losses
    
    return reducible


def losses_to_weights(
    reducible_losses: np.ndarray,
    temperature: float = 1.0,
    epsilon_mix: float = 0.1,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    将 Reducible Loss 转换为采样权重
    
    转换步骤：
    1. 对 reducible loss 应用 alpha 幂次（可选）
    2. 使用 softmax 转换为概率分布
    3. 与均匀分布混合（epsilon mixing）
    
    参数：
        reducible_losses: shape=(num_anchors,) 的 reducible loss 数组
        temperature: softmax 温度
            - temperature 越大，权重分布越均匀
            - temperature 越小，高 reducible loss 的样本权重越突出
        epsilon_mix: 与均匀分布混合的比例 [0, 1]
            - 最终权重 = (1 - epsilon_mix) * softmax_weights + epsilon_mix * uniform
            - 确保所有样本都有机会被采样
        alpha: 权重幂次参数（在 softmax 之前对 reducible loss 应用）
            - alpha > 1：增强高 reducible loss 样本的权重
            - alpha < 1：减弱差异
            
    返回：
        weights: shape=(num_anchors,) 的采样权重（已归一化为概率分布）
    """
    n = len(reducible_losses)
    
    # 处理 reducible loss（只关注正值，即"还没学会"的样本）
    # 负值（已学会）设为 0，不优先采样但也不排除
    positive_reducible = np.maximum(reducible_losses, 0)
    
    # 应用 alpha 幂次
    if alpha != 1.0:
        positive_reducible = np.power(positive_reducible + 1e-8, alpha)
    
    # 数值稳定的 softmax
    scaled = positive_reducible / (temperature + 1e-8)
    scaled = scaled - np.max(scaled)  # 防止 exp 溢出
    exp_vals = np.exp(scaled)
    softmax_weights = exp_vals / (exp_vals.sum() + 1e-8)
    
    # Epsilon mixing：与均匀分布混合
    uniform = np.ones(n, dtype=np.float64) / n
    weights = (1 - epsilon_mix) * softmax_weights + epsilon_mix * uniform
    
    # 最终归一化（确保是有效概率分布）
    weights = weights / (weights.sum() + 1e-8)
    
    return weights


def ema_update(
    baseline_losses: np.ndarray,
    current_losses: np.ndarray,
    beta: float = 0.5,
) -> np.ndarray:
    """
    使用 EMA（指数移动平均）更新 baseline loss
    
    公式：baseline_new = beta * baseline_old + (1 - beta) * current
    
    参数：
        baseline_losses: 旧的 baseline loss，shape=(num_anchors,)
        current_losses: 当前模型的 loss，shape=(num_anchors,)
        beta: EMA 系数 [0, 1]
            - beta 越大，baseline 更新越慢（更保守）
            - beta 越小，baseline 更新越快（更激进）
            
    返回：
        new_baseline: 更新后的 baseline loss，shape=(num_anchors,)
    """
    return beta * baseline_losses + (1 - beta) * current_losses


@dataclass
class RLASState:
    """
    RLAS 运行时状态管理类
    
    用于在训练过程中跟踪 RLAS 的状态，包括：
    - baseline_losses: 用于计算 reducible loss 的参考 loss
    - current_weights: 当前的采样权重
    - 统计信息和历史记录
    - 可视化器（可选，默认开启）
    """
    
    config: RLASConfig
    baseline_losses: Optional[np.ndarray] = None
    current_weights: Optional[np.ndarray] = None
    weight_update_history: list = field(default_factory=list)
    loss_history: list = field(default_factory=list)  # 用于绘制 loss 曲线
    
    # 统计信息
    total_weight_updates: int = 0
    last_update_step: int = -1
    
    # 可视化器（延迟初始化）
    _visualizer: Optional[Any] = field(default=None, repr=False)
    _enable_visualization: bool = True
    _visualization_dir: Optional[str] = None
    
    def is_warmup_complete(self, current_update: int) -> bool:
        """检查 warmup 阶段是否完成"""
        return current_update >= self.config.warmup_updates
    
    def enable_visualization(self, output_dir: str, enabled: bool = True) -> None:
        """
        启用或禁用可视化功能
        
        参数：
            output_dir: 可视化输出目录（如 ckpt_dir/rlas_viz）
            enabled: 是否启用可视化
        """
        self._enable_visualization = enabled
        self._visualization_dir = output_dir
        
        if enabled and self._visualizer is None:
            from .visualizer import RLASVisualizer
            os.makedirs(output_dir, exist_ok=True)
            self._visualizer = RLASVisualizer(output_dir=output_dir)
            print(f"[RLAS] 可视化已启用，输出目录: {output_dir}")
    
    def should_update_weights(self, current_update: int) -> bool:
        """检查当前步是否应该更新权重"""
        if not self.is_warmup_complete(current_update):
            return False
        
        # warmup 结束后的第一次更新
        if self.baseline_losses is None:
            return True
        
        # 按照 update_interval 周期更新
        steps_since_last = current_update - self.last_update_step
        return steps_since_last >= self.config.update_interval
    
    def compute_and_update_weights(
        self,
        current_losses: np.ndarray,
        current_update: int,
    ) -> np.ndarray:
        """
        计算并更新采样权重
        
        参数：
            current_losses: 当前模型在所有 anchor 上的 loss
            current_update: 当前的更新步数
            
        返回：
            new_weights: 新的采样权重
        """
        reducible_losses = None  # 用于可视化
        
        if self.baseline_losses is None:
            # 第一次计算：使用当前 loss 作为 baseline
            self.baseline_losses = current_losses.copy()
            # 初始权重为均匀分布
            n = len(current_losses)
            self.current_weights = np.ones(n, dtype=np.float64) / n
            print(f"[RLAS] Warmup 完成 @ update {current_update}，初始化 baseline losses")
            
            # 记录初始状态
            self.loss_history.append({
                "update": current_update,
                "current_loss_mean": float(current_losses.mean()),
                "baseline_loss_mean": float(self.baseline_losses.mean()),
                "reducible_loss_mean": 0.0,
            })
        else:
            # 计算 reducible loss
            reducible_losses = compute_reducible_losses(
                current_losses, self.baseline_losses
            )
            
            # 转换为权重
            self.current_weights = losses_to_weights(
                reducible_losses,
                temperature=self.config.temperature,
                epsilon_mix=self.config.epsilon_mix,
                alpha=self.config.alpha,
            )
            
            # EMA 更新 baseline
            self.baseline_losses = ema_update(
                self.baseline_losses,
                current_losses,
                beta=self.config.ema_beta,
            )
            
            # 记录统计信息
            stats = {
                "update": current_update,
                "reducible_mean": float(reducible_losses.mean()),
                "reducible_std": float(reducible_losses.std()),
                "reducible_max": float(reducible_losses.max()),
                "reducible_min": float(reducible_losses.min()),
                "positive_ratio": float((reducible_losses > 0).mean()),
                "weight_entropy": float(-np.sum(self.current_weights * np.log(self.current_weights + 1e-10))),
            }
            self.weight_update_history.append(stats)
            
            # 记录 loss 历史（用于绘图）
            self.loss_history.append({
                "update": current_update,
                "current_loss_mean": float(current_losses.mean()),
                "baseline_loss_mean": float(self.baseline_losses.mean()),
                "reducible_loss_mean": float(reducible_losses.mean()),
            })
            
            print(
                f"[RLAS] 权重更新 @ update {current_update}: "
                f"reducible_mean={stats['reducible_mean']:.6f}, "
                f"positive_ratio={stats['positive_ratio']:.2%}, "
                f"weight_entropy={stats['weight_entropy']:.4f}"
            )
            
            # 生成可视化报告（如果启用）
            if self._enable_visualization and self._visualizer is not None:
                self._generate_visualization(
                    current_losses=current_losses,
                    reducible_losses=reducible_losses,
                    current_update=current_update,
                )
        
        self.total_weight_updates += 1
        self.last_update_step = current_update
        
        return self.current_weights
    
    def _generate_visualization(
        self,
        current_losses: np.ndarray,
        reducible_losses: np.ndarray,
        current_update: int,
    ) -> None:
        """
        生成权重更新的可视化报告
        
        参数：
            current_losses: 当前 loss
            reducible_losses: reducible loss
            current_update: 当前更新步数
        """
        try:
            # 构造统计信息字典
            stats = {
                "reducible_mean": float(reducible_losses.mean()),
                "reducible_std": float(reducible_losses.std()),
                "positive_ratio": float((reducible_losses > 0).mean()),
                "weight_entropy": float(-np.sum(self.current_weights * np.log(self.current_weights + 1e-10))),
            }
            
            self._visualizer.record_update(
                update=current_update,
                weights=self.current_weights,
                current_losses=current_losses,
                baseline_losses=self.baseline_losses,
                reducible_losses=reducible_losses,
                stats=stats,
            )
        except Exception as e:
            print(f"[RLAS] 可视化生成警告: {e}")
    
    def save_snapshot(self, ckpt_dir: str, current_update: int) -> None:
        """
        保存当前状态快照
        
        参数：
            ckpt_dir: checkpoint 目录
            current_update: 当前更新步数
        """
        if not self.config.save_weight_snapshots:
            return
        
        rlas_dir = os.path.join(ckpt_dir, "rlas_snapshots")
        os.makedirs(rlas_dir, exist_ok=True)
        
        # 保存权重
        if self.current_weights is not None:
            np.save(
                os.path.join(rlas_dir, f"weights_update_{current_update}.npy"),
                self.current_weights,
            )
        
        # 保存 baseline
        if self.baseline_losses is not None:
            np.save(
                os.path.join(rlas_dir, f"baseline_update_{current_update}.npy"),
                self.baseline_losses,
            )
        
        # 保存历史统计
        history_path = os.path.join(rlas_dir, "weight_update_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.weight_update_history, f, indent=2)
    
    def state_dict(self) -> Dict[str, Any]:
        """获取可序列化的状态字典"""
        return {
            "config": self.config.to_dict(),
            "baseline_losses": self.baseline_losses.tolist() if self.baseline_losses is not None else None,
            "current_weights": self.current_weights.tolist() if self.current_weights is not None else None,
            "total_weight_updates": self.total_weight_updates,
            "last_update_step": self.last_update_step,
            "weight_update_history": self.weight_update_history,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """从状态字典恢复"""
        if state_dict.get("baseline_losses") is not None:
            self.baseline_losses = np.array(state_dict["baseline_losses"])
        if state_dict.get("current_weights") is not None:
            self.current_weights = np.array(state_dict["current_weights"])
        self.total_weight_updates = state_dict.get("total_weight_updates", 0)
        self.last_update_step = state_dict.get("last_update_step", -1)
        self.weight_update_history = state_dict.get("weight_update_history", [])
