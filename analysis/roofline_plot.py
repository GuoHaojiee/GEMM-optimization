#!/usr/bin/env python3
"""
roofline_plot.py — 专用 Roofline 模型绘图工具

功能：
  1. 从 CSV 读取实测 GFLOPS 数据
  2. 根据理论分析设置各版本的算术强度
  3. 绘制 Roofline 天花板线 + 各版本实测点
  4. 自动判断各版本是 Memory Bound 还是 Compute Bound

用法：
  python3 analysis/roofline_plot.py                     # 使用默认 CSV
  python3 analysis/roofline_plot.py --cpu results/cpu_results.csv
  python3 analysis/roofline_plot.py --size 2048         # 只画特定矩阵大小
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

# ============================================================
# 硬件参数（请根据你的机器修改）
# ============================================================

HARDWARE = {
    'cpu': {
        'name': 'CPU',
        'peak_gflops': 100.0,    # GFLOPS，修改为你的 CPU 实际值
        'peak_bw_gbs': 50.0,     # GB/s，修改为你的内存带宽实际值
    },
    'gpu': {
        'name': 'GPU',
        'peak_gflops': 10000.0,  # GFLOPS，修改为你的 GPU 实际值
        'peak_bw_gbs': 500.0,    # GB/s，修改为你的 GPU 显存带宽实际值
    }
}

# ============================================================
# 各版本的理论算术强度（FLOP/Byte）
# 这些值需要根据理论分析填写，参见 notes/04_roofline_model.md
# ============================================================

AI_MAP = {
    # CPU 版本
    'V1_Naive_ijk':     0.25,   # 2MNK / (4 * 4MNK) ≈ 0.125, 实际更低因为 Cache miss
    'V2_ikj':           0.25,   # 访存模式改善但每元素读取次数不变
    'V3_Tiling':        16.0,   # AI = BLOCK_SIZE/4 = 64/4
    'V4_AVX2':          0.5,    # 比 V2 略高（向量化减少了指令开销）
    'V5_AVX2+Tiling':   16.0,   # 同 V3，加 SIMD
    'V6_OMP+AVX2+Tile': 16.0,   # 同 V5

    # CUDA 版本
    'V7_CUDA_Naive':    0.25,   # 无数据复用
    'V8_Shared_Mem':    4.0,    # AI = TILE_SIZE/4 = 16/4
    'V9_Register':      8.0,    # 更高寄存器复用
    'V10_Vectorized':   10.0,   # 向量化加载
    'V11_cuBLAS':       25.0,   # 接近峰值
}

# ============================================================
# 核心绘图函数
# ============================================================

def plot_roofline_chart(df, hw_params, target_size=None, output_path='results/roofline_detailed.png'):
    """
    绘制详细的 Roofline 图

    参数：
        df:          包含 version, M, gflops 的 DataFrame
        hw_params:   硬件参数字典 {peak_gflops, peak_bw_gbs, name}
        target_size: 使用哪个矩阵大小的结果（None 表示最大）
        output_path: 输出图片路径
    """
    peak_g  = hw_params['peak_gflops']
    peak_bw = hw_params['peak_bw_gbs']
    ridge   = peak_g / peak_bw  # 拐点（FLOP/Byte）

    fig, ax = plt.subplots(figsize=(12, 8))

    # ---- 绘制 Roofline 主线 ----
    ai_range = np.logspace(-2, 2.5, 500)
    roofline = np.minimum(peak_g, ai_range * peak_bw)
    ax.loglog(ai_range, roofline, 'k-', linewidth=2.5, zorder=5, label='Roofline')

    # ---- 延伸线（虚线）----
    # 内存带宽延长线
    ax.loglog(ai_range, ai_range * peak_bw, 'b--', alpha=0.3, linewidth=1,
              label=f'内存带宽 {peak_bw:.0f} GB/s')
    # 算力延长线
    ax.axhline(peak_g, color='r', linestyle='--', alpha=0.3, linewidth=1,
               label=f'峰值算力 {peak_g:.0f} GFLOPS')

    # ---- 拐点标注 ----
    ax.scatter([ridge], [peak_g], s=120, color='black', zorder=10, marker='*')
    ax.annotate(f'拐点\nAI={ridge:.1f} F/B\n峰值={peak_g:.0f} GFLOPS',
                xy=(ridge, peak_g),
                xytext=(ridge * 2, peak_g * 0.6),
                fontsize=10, color='black',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # ---- Memory Bound / Compute Bound 区域标注 ----
    ax.fill_betweenx([1, peak_g], 1e-2, ridge, alpha=0.05, color='blue')
    ax.fill_betweenx([1, peak_g], ridge, 1e3, alpha=0.05, color='red')
    ax.text(0.05, peak_g * 0.3, 'Memory\nBound', fontsize=11,
            color='blue', alpha=0.6, ha='center')
    ax.text(ridge * 5, peak_g * 0.5, 'Compute\nBound', fontsize=11,
            color='red', alpha=0.6, ha='center')

    # ---- 绘制各版本实测点 ----
    if df is not None and not df.empty:
        if target_size is None:
            target_size = df['M'].max()

        sub = df[df['M'] == target_size].copy()
        sub = sub.drop_duplicates(subset='version')

        colors = plt.cm.Set1(np.linspace(0, 0.8, len(sub)))

        for i, (_, row) in enumerate(sub.iterrows()):
            ver    = row['version']
            gflops = row['gflops']
            ai     = AI_MAP.get(ver, 1.0)
            bound  = "M" if ai < ridge else "C"  # Memory or Compute bound

            ax.scatter(ai, gflops, s=150, color=colors[i], zorder=8,
                       marker='o', edgecolors='black', linewidth=0.5)

            # 标注版本名和 bound 类型
            ax.annotate(f'{ver}\n({gflops:.0f} GFLOPS, {bound}-bound)',
                        xy=(ai, gflops),
                        xytext=(0, 12), textcoords='offset points',
                        fontsize=8, color=colors[i],
                        ha='center',
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])

            # 绘制"效率箭头"：从实测点到 Roofline 的距离
            roofline_at_ai = min(peak_g, ai * peak_bw)
            efficiency = gflops / roofline_at_ai * 100
            ax.annotate('',
                        xy=(ai, roofline_at_ai),
                        xytext=(ai, gflops),
                        arrowprops=dict(arrowstyle='->', color=colors[i],
                                        lw=1, linestyle='dotted', alpha=0.5))

    # ---- 坐标轴设置 ----
    ax.set_xlabel('算术强度 Arithmetic Intensity (FLOP/Byte)', fontsize=12)
    ax.set_ylabel('性能 Performance (GFLOPS)', fontsize=12)
    ax.set_title(f"{hw_params['name']} Roofline Model（M=N=K={target_size}）",
                 fontsize=14, fontweight='bold')

    ax.set_xlim(5e-3, 200)
    ax.set_ylim(0.1, peak_g * 1.5)

    # 自定义刻度标签
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{x:.0f}' if x >= 1 else f'{x:.2f}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f'{y:.0f}'))

    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')

    # 添加说明文字
    fig.text(0.01, 0.01,
             '● M-bound: 受内存带宽限制，AI < 拐点  '
             '● C-bound: 受算力限制，AI > 拐点',
             fontsize=9, alpha=0.7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"保存：{output_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Roofline 模型绘图')
    parser.add_argument('--cpu', default='results/cpu_results.csv',
                        help='CPU 结果 CSV 路径')
    parser.add_argument('--cuda', default='results/cuda_results.csv',
                        help='CUDA 结果 CSV 路径')
    parser.add_argument('--size', type=int, default=None,
                        help='使用哪个矩阵大小的结果（默认最大）')
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)

    # CPU Roofline
    if os.path.exists(args.cpu):
        cpu_df = pd.read_csv(args.cpu)
        if 'threads' in cpu_df.columns:
            cpu_df = cpu_df[cpu_df['threads'] == 1]
        plot_roofline_chart(cpu_df, HARDWARE['cpu'], args.size,
                            'results/roofline_cpu.png')

    # GPU Roofline
    if os.path.exists(args.cuda):
        cuda_df = pd.read_csv(args.cuda)
        plot_roofline_chart(cuda_df, HARDWARE['gpu'], args.size,
                            'results/roofline_gpu.png')


if __name__ == '__main__':
    main()
