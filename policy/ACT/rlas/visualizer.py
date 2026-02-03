"""
RLAS 可视化模块

本文件提供 RLAS 训练过程中的可视化功能：
1. 样本权重分布可视化
2. Reducible Loss 分布可视化
3. Loss 下降曲线
4. 权重熵变化曲线

所有图像保存到 ckpt_dir 同级的 rlas_visualizations/ 文件夹中。
"""

import os
import numpy as np
import json
from typing import Optional, List, Dict, Any

# 使用 Agg 后端避免显示问题
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class RLASVisualizer:
    """
    RLAS 可视化器
    
    自动记录并可视化 RLAS 训练过程中的关键信息：
    - 每次权重更新时的样本分布
    - Loss 下降曲线
    - 权重熵变化
    
    使用方法：
        visualizer = RLASVisualizer(output_dir)
        visualizer.record_update(update, weights, losses, reducible, stats)
        visualizer.save_all_plots()  # 训练结束时调用
    """
    
    def __init__(self, output_dir: str, enabled: bool = True):
        """
        初始化可视化器
        
        参数：
            output_dir: 输出目录（通常与 log 目录同级）
            enabled: 是否启用可视化（默认启用）
        """
        self.output_dir = output_dir
        self.enabled = enabled
        
        if self.enabled:
            os.makedirs(output_dir, exist_ok=True)
        
        # 记录历史数据
        self.update_steps: List[int] = []
        self.weight_entropies: List[float] = []
        self.reducible_means: List[float] = []
        self.reducible_stds: List[float] = []
        self.positive_ratios: List[float] = []
        self.current_loss_means: List[float] = []
        self.baseline_loss_means: List[float] = []
        
        # 训练 loss 历史（用于绘制下降曲线）
        self.train_loss_steps: List[int] = []
        self.train_loss_values: List[float] = []
        
    def record_train_loss(self, step: int, loss: float) -> None:
        """
        记录训练 loss（用于绘制下降曲线）
        
        参数：
            step: 当前训练步数
            loss: 当前 batch 的 loss
        """
        if not self.enabled:
            return
        self.train_loss_steps.append(step)
        self.train_loss_values.append(loss)
    
    def record_update(
        self,
        update: int,
        weights: np.ndarray,
        current_losses: np.ndarray,
        baseline_losses: np.ndarray,
        reducible_losses: np.ndarray,
        stats: Dict[str, Any],
    ) -> None:
        """
        记录一次权重更新的信息并生成可视化
        
        参数：
            update: 当前更新步数
            weights: 新的采样权重
            current_losses: 当前模型的 loss
            baseline_losses: baseline loss
            reducible_losses: reducible loss
            stats: 统计信息字典
        """
        if not self.enabled:
            return
        
        # 记录历史
        self.update_steps.append(update)
        self.weight_entropies.append(stats.get("weight_entropy", 0))
        self.reducible_means.append(stats.get("reducible_mean", 0))
        self.reducible_stds.append(stats.get("reducible_std", 0))
        self.positive_ratios.append(stats.get("positive_ratio", 0))
        self.current_loss_means.append(float(current_losses.mean()))
        self.baseline_loss_means.append(float(baseline_losses.mean()))
        
        # 生成当前更新的可视化
        self._plot_weight_distribution(update, weights)
        self._plot_loss_distribution(update, current_losses, baseline_losses, reducible_losses)
        self._plot_combined_dashboard(update, weights, current_losses, baseline_losses, reducible_losses, stats)
    
    def _plot_weight_distribution(self, update: int, weights: np.ndarray) -> None:
        """
        绘制样本权重分布直方图
        
        图像说明：
        - X轴：采样权重值
        - Y轴：样本数量
        - 红色虚线：均匀分布的权重值 (1/N)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n = len(weights)
        uniform_weight = 1.0 / n
        
        # 直方图
        ax.hist(weights, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(uniform_weight, color='red', linestyle='--', linewidth=2, 
                   label=f'Uniform = {uniform_weight:.2e}')
        
        # 标注
        ax.set_xlabel('Sample Weight', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'RLAS Weight Distribution @ Update {update}', fontsize=14)
        ax.legend(fontsize=10)
        
        # 添加统计信息
        stats_text = (
            f'N = {n}\n'
            f'Max = {weights.max():.2e}\n'
            f'Min = {weights.min():.2e}\n'
            f'Std = {weights.std():.2e}\n'
            f'Max/Uniform = {weights.max() / uniform_weight:.1f}x'
        )
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'weight_dist_update_{update}.png'), dpi=150)
        plt.close(fig)
    
    def _plot_loss_distribution(
        self,
        update: int,
        current_losses: np.ndarray,
        baseline_losses: np.ndarray,
        reducible_losses: np.ndarray,
    ) -> None:
        """
        绘制 Loss 分布对比图
        
        图像说明（三个子图）：
        1. Current Loss vs Baseline Loss 直方图
        2. Reducible Loss 直方图
        3. Current vs Baseline 散点图
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 子图1：Current vs Baseline Loss 直方图
        ax1 = axes[0]
        ax1.hist(baseline_losses, bins=50, alpha=0.5, label='Baseline Loss', color='blue')
        ax1.hist(current_losses, bins=50, alpha=0.5, label='Current Loss', color='orange')
        ax1.set_xlabel('Loss Value', fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('Loss Distribution Comparison', fontsize=12)
        ax1.legend(fontsize=9)
        
        # 子图2：Reducible Loss 直方图
        ax2 = axes[1]
        colors = ['green' if r > 0 else 'red' for r in reducible_losses]
        ax2.hist(reducible_losses[reducible_losses > 0], bins=30, alpha=0.7, 
                 label=f'Positive ({(reducible_losses > 0).sum()})', color='green')
        ax2.hist(reducible_losses[reducible_losses <= 0], bins=30, alpha=0.7,
                 label=f'Negative ({(reducible_losses <= 0).sum()})', color='red')
        ax2.axvline(0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Reducible Loss', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('Reducible Loss Distribution', fontsize=12)
        ax2.legend(fontsize=9)
        
        # 子图3：散点图
        ax3 = axes[2]
        # 随机采样最多 2000 个点避免过慢
        n = len(current_losses)
        if n > 2000:
            idx = np.random.choice(n, 2000, replace=False)
        else:
            idx = np.arange(n)
        
        scatter = ax3.scatter(baseline_losses[idx], current_losses[idx], 
                             c=reducible_losses[idx], cmap='RdYlGn_r', 
                             alpha=0.5, s=10)
        # 对角线（current = baseline）
        lims = [min(baseline_losses.min(), current_losses.min()),
                max(baseline_losses.max(), current_losses.max())]
        ax3.plot(lims, lims, 'k--', alpha=0.5, label='Current = Baseline')
        ax3.set_xlabel('Baseline Loss', fontsize=11)
        ax3.set_ylabel('Current Loss', fontsize=11)
        ax3.set_title('Current vs Baseline Loss', fontsize=12)
        plt.colorbar(scatter, ax=ax3, label='Reducible Loss')
        ax3.legend(fontsize=9)
        
        plt.suptitle(f'Loss Analysis @ Update {update}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'loss_dist_update_{update}.png'), dpi=150)
        plt.close(fig)
    
    def _plot_combined_dashboard(
        self,
        update: int,
        weights: np.ndarray,
        current_losses: np.ndarray,
        baseline_losses: np.ndarray,
        reducible_losses: np.ndarray,
        stats: Dict[str, Any],
    ) -> None:
        """
        绘制综合仪表盘（单张图包含所有关键信息）
        
        布局：
        +-------------------+-------------------+
        |   Weight Dist     |   Reducible Dist  |
        +-------------------+-------------------+
        |   Weight Heatmap  |   Stats Summary   |
        +-------------------+-------------------+
        """
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
        
        n = len(weights)
        uniform_weight = 1.0 / n
        
        # 左上：权重分布
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(weights, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(uniform_weight, color='red', linestyle='--', linewidth=2, label='Uniform')
        ax1.set_xlabel('Weight')
        ax1.set_ylabel('Count')
        ax1.set_title('Sample Weight Distribution')
        ax1.legend()
        
        # 右上：Reducible Loss 分布
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(reducible_losses[reducible_losses > 0], bins=30, alpha=0.7, 
                 label='Positive (learnable)', color='green')
        ax2.hist(reducible_losses[reducible_losses <= 0], bins=30, alpha=0.7,
                 label='Negative (learned)', color='red')
        ax2.axvline(0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Reducible Loss')
        ax2.set_ylabel('Count')
        ax2.set_title('Reducible Loss Distribution')
        ax2.legend()
        
        # 左下：权重热力图（按 episode 排列）
        ax3 = fig.add_subplot(gs[1, 0])
        # 将权重 reshape 成近似方形便于可视化
        side = int(np.ceil(np.sqrt(n)))
        padded_weights = np.zeros(side * side)
        padded_weights[:n] = weights
        weight_grid = padded_weights.reshape(side, side)
        im = ax3.imshow(weight_grid, cmap='hot', aspect='auto')
        ax3.set_title('Weight Heatmap (by anchor index)')
        ax3.set_xlabel('Index (col)')
        ax3.set_ylabel('Index (row)')
        plt.colorbar(im, ax=ax3, label='Weight')
        
        # 右下：统计摘要
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        summary_text = f"""
RLAS Status @ Update {update}
{'='*40}

[Weight Statistics]
  * Num Samples (N): {n:,}
  * Weight Max: {weights.max():.4e}
  * Weight Min: {weights.min():.4e}
  * Weight Std: {weights.std():.4e}
  * Max/Uniform: {weights.max() / uniform_weight:.1f}x
  * Entropy: {stats.get('weight_entropy', 0):.4f}
    (Higher=More uniform, Max={np.log(n):.2f})

[Reducible Loss Statistics]
  * Mean: {stats.get('reducible_mean', 0):.6f}
  * Std: {stats.get('reducible_std', 0):.6f}
  * Max: {stats.get('reducible_max', 0):.6f}
  * Min: {stats.get('reducible_min', 0):.6f}
  * Positive Ratio (Learnable): {stats.get('positive_ratio', 0)*100:.1f}%

[Loss Statistics]
  * Current Loss Mean: {current_losses.mean():.6f}
  * Baseline Loss Mean: {baseline_losses.mean():.6f}
  * Loss Reduction: {baseline_losses.mean() - current_losses.mean():.6f}
"""
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle(f'RLAS Dashboard @ Update {update}', fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(self.output_dir, f'dashboard_update_{update}.png'), dpi=150)
        plt.close(fig)
    
    def save_training_curves(self) -> None:
        """
        保存训练曲线图（在训练结束时调用）
        
        包含：
        1. Training Loss 下降曲线
        2. 权重熵变化曲线
        3. Reducible Loss 均值变化曲线
        4. Positive Ratio 变化曲线
        """
        if not self.enabled or len(self.update_steps) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 子图1：Training Loss 下降曲线
        ax1 = axes[0, 0]
        if len(self.train_loss_steps) > 0:
            # 平滑处理：取滑动平均
            window = min(100, len(self.train_loss_values) // 10 + 1)
            if window > 1:
                smoothed = np.convolve(self.train_loss_values, 
                                       np.ones(window)/window, mode='valid')
                smoothed_steps = self.train_loss_steps[window-1:]
                ax1.plot(smoothed_steps, smoothed, 'b-', alpha=0.8, label='Smoothed Loss')
            ax1.plot(self.train_loss_steps, self.train_loss_values, 'b-', alpha=0.2, label='Raw Loss')
            # 标记权重更新时刻
            for step in self.update_steps:
                ax1.axvline(step, color='red', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2：权重熵变化
        ax2 = axes[0, 1]
        ax2.plot(self.update_steps, self.weight_entropies, 'go-', markersize=6)
        if len(self.update_steps) > 0:
            # 添加最大熵参考线
            n_samples = len(self.current_loss_means)  # 近似样本数
            if n_samples > 0:
                # 注：这里用一个估计值
                ax2.axhline(np.log(18000), color='red', linestyle='--', 
                           alpha=0.5, label='Max Entropy (uniform)')
        ax2.set_xlabel('Update Step')
        ax2.set_ylabel('Weight Entropy')
        ax2.set_title('Weight Distribution Entropy Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 子图3：Reducible Loss 均值变化
        ax3 = axes[1, 0]
        ax3.errorbar(self.update_steps, self.reducible_means, 
                     yerr=self.reducible_stds, fmt='o-', capsize=3,
                     color='purple', alpha=0.7)
        ax3.axhline(0, color='black', linestyle='-', linewidth=1)
        ax3.set_xlabel('Update Step')
        ax3.set_ylabel('Reducible Loss')
        ax3.set_title('Mean Reducible Loss Over Time')
        ax3.grid(True, alpha=0.3)
        
        # 子图4：Positive Ratio 变化
        ax4 = axes[1, 1]
        ax4.plot(self.update_steps, [p * 100 for p in self.positive_ratios], 
                 'ro-', markersize=6)
        ax4.set_xlabel('Update Step')
        ax4.set_ylabel('Positive Ratio (%)')
        ax4.set_title('Ratio of Learnable Samples Over Time')
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('RLAS Training Curves', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'), dpi=150)
        plt.close(fig)
        
        # 保存 Loss 对比曲线
        self._save_loss_comparison_curve()
        
        print(f"[RLAS Visualizer] 训练曲线已保存到: {self.output_dir}")
    
    def _save_loss_comparison_curve(self) -> None:
        """
        保存 Current Loss vs Baseline Loss 对比曲线
        """
        if len(self.update_steps) < 2:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.update_steps, self.current_loss_means, 'b-o', 
                label='Current Loss (mean)', markersize=6)
        ax.plot(self.update_steps, self.baseline_loss_means, 'r--s', 
                label='Baseline Loss (mean)', markersize=6)
        
        # 填充差异区域
        ax.fill_between(self.update_steps, self.baseline_loss_means, 
                        self.current_loss_means, alpha=0.3, color='green',
                        where=[c < b for c, b in zip(self.current_loss_means, self.baseline_loss_means)],
                        label='Improvement')
        
        ax.set_xlabel('Update Step', fontsize=12)
        ax.set_ylabel('Mean Loss', fontsize=12)
        ax.set_title('Current vs Baseline Loss Over Training', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'loss_comparison_curve.png'), dpi=150)
        plt.close(fig)
    
    def save_summary_json(self) -> None:
        """保存所有统计数据为 JSON"""
        if not self.enabled:
            return
        
        summary = {
            "update_steps": self.update_steps,
            "weight_entropies": self.weight_entropies,
            "reducible_means": self.reducible_means,
            "reducible_stds": self.reducible_stds,
            "positive_ratios": self.positive_ratios,
            "current_loss_means": self.current_loss_means,
            "baseline_loss_means": self.baseline_loss_means,
        }
        
        with open(os.path.join(self.output_dir, "rlas_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    
    def finalize(self) -> None:
        """
        训练结束时调用，生成所有汇总图表
        """
        if not self.enabled:
            return
        
        self.save_training_curves()
        self.save_summary_json()
        print(f"[RLAS Visualizer] 所有可视化已保存到: {self.output_dir}")
