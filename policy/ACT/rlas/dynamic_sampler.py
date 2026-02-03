"""
动态权重采样器模块

本文件实现了支持运行时权重更新的 PyTorch Sampler。
与标准的 WeightedRandomSampler 不同，DynamicWeightedSampler 允许
在训练过程中动态更新采样权重，这是 RLAS 算法的核心需求。
"""

import numpy as np
import torch
from torch.utils.data import Sampler
from typing import Iterator, Optional, List
import threading


class DynamicWeightedSampler(Sampler[int]):
    """
    支持运行时权重更新的加权随机采样器
    
    与 PyTorch 的 WeightedRandomSampler 相比，本采样器支持：
    1. 通过 update_weights() 方法在训练过程中动态更新权重
    2. 线程安全的权重更新
    3. 权重归一化自动处理
    
    使用场景：
    - RLAS 训练中，每隔一定步数根据 reducible loss 更新采样权重
    
    参数：
        num_anchors: anchor 总数（即训练集中可采样的片段数）
        num_samples: 每个 epoch 采样多少个样本
        initial_weights: 初始权重，如果为 None 则使用均匀权重
        replacement: 是否有放回采样
        seed: 随机种子，保证可复现
    """
    
    def __init__(
        self,
        num_anchors: int,
        num_samples: int,
        initial_weights: Optional[np.ndarray] = None,
        replacement: bool = True,
        seed: int = 0,
    ):
        # 注意：不调用父类 __init__，因为 Sampler 基类的 __init__ 在不同版本有不同行为
        # 我们完全自己管理状态
        
        assert num_anchors > 0, f"num_anchors 必须 > 0, 当前值: {num_anchors}"
        assert num_samples > 0, f"num_samples 必须 > 0, 当前值: {num_samples}"
        
        self.num_anchors = num_anchors
        self.num_samples = num_samples
        self.replacement = replacement
        self.seed = seed
        
        # 初始化权重（线程安全访问需要锁）
        self._lock = threading.Lock()
        if initial_weights is not None:
            self._weights = np.array(initial_weights, dtype=np.float64)
            assert len(self._weights) == num_anchors, \
                f"initial_weights 长度 ({len(self._weights)}) 必须等于 num_anchors ({num_anchors})"
        else:
            # 默认均匀权重
            self._weights = np.ones(num_anchors, dtype=np.float64)
        
        # 归一化权重
        self._normalize_weights()
        
        # 随机数生成器
        self._rng = np.random.default_rng(seed)
        
        # 记录权重更新次数
        self._update_count = 0
    
    def _normalize_weights(self):
        """归一化权重为概率分布"""
        total = self._weights.sum()
        if total > 0:
            self._weights = self._weights / total
        else:
            # 如果所有权重都是 0，退化为均匀分布
            self._weights = np.ones(self.num_anchors, dtype=np.float64) / self.num_anchors
    
    def update_weights(self, new_weights: np.ndarray) -> None:
        """
        更新采样权重
        
        参数：
            new_weights: 新的权重数组，长度必须等于 num_anchors
                        不需要预先归一化，函数会自动处理
        
        注意：
            - 此方法是线程安全的
            - 更新后的权重会在下一次 __iter__ 调用时生效
        """
        assert len(new_weights) == self.num_anchors, \
            f"new_weights 长度 ({len(new_weights)}) 必须等于 num_anchors ({self.num_anchors})"
        
        with self._lock:
            self._weights = np.array(new_weights, dtype=np.float64)
            self._normalize_weights()
            self._update_count += 1
    
    def get_weights(self) -> np.ndarray:
        """
        获取当前权重（归一化后的）
        
        返回：
            当前权重的副本
        """
        with self._lock:
            return self._weights.copy()
    
    def get_update_count(self) -> int:
        """获取权重更新次数"""
        return self._update_count
    
    def __iter__(self) -> Iterator[int]:
        """
        返回采样索引的迭代器
        
        每次调用时使用当前权重进行采样
        """
        with self._lock:
            weights = self._weights.copy()
        
        if self.replacement:
            # 有放回采样
            indices = self._rng.choice(
                self.num_anchors,
                size=self.num_samples,
                replace=True,
                p=weights,
            )
        else:
            # 无放回采样
            # 注意：如果 num_samples > num_anchors，这会失败
            if self.num_samples > self.num_anchors:
                raise ValueError(
                    f"无放回采样时，num_samples ({self.num_samples}) 不能大于 num_anchors ({self.num_anchors})"
                )
            indices = self._rng.choice(
                self.num_anchors,
                size=self.num_samples,
                replace=False,
                p=weights,
            )
        
        yield from indices.tolist()
    
    def __len__(self) -> int:
        """返回每个 epoch 的样本数"""
        return self.num_samples
    
    def state_dict(self) -> dict:
        """
        获取采样器状态，便于保存和恢复
        
        返回：
            包含所有状态的字典
        """
        with self._lock:
            return {
                "num_anchors": self.num_anchors,
                "num_samples": self.num_samples,
                "weights": self._weights.copy(),
                "replacement": self.replacement,
                "seed": self.seed,
                "update_count": self._update_count,
                # 注意：rng 状态不容易序列化，重新加载后会从 seed 重建
            }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """
        从状态字典恢复采样器状态
        
        参数：
            state_dict: 之前通过 state_dict() 获取的字典
        """
        with self._lock:
            self._weights = np.array(state_dict["weights"], dtype=np.float64)
            self._update_count = state_dict.get("update_count", 0)
