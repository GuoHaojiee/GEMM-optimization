#!/usr/bin/env python3
"""
plot_results.py — GEMM 优化结果可视化（生成 4 张图）

图1：各版本各矩阵大小的 GFLOPS 柱状图
图2：Roofline 图（各版本实测点 + 理论天花板线）
图3：CUDA 带宽利用率柱状图
图4：Scalability 折线图（矩阵大小 vs GFLOPS）

运行方式：
  python3 analysis/plot_results.py
  python3 analysis/plot_results.py --cpu-only  # 只有 CPU 结果时
"""

import sys
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 使用非交互式后端（无需 display，适合服务器）
matplotlib.use('Agg')

# ============================================================
# 配置参数（根据你的机器修改）
# ============================================================

# 硬件参数（与 include/roofline.h 保持一致）
CPU_PEAK_GFLOPS = 100.0   # CPU 峰值算力（GFLOPS）
CPU_PEAK_BW_GBS = 50.0    # CPU 内存带宽（GB/s）
GPU_PEAK_GFLOPS = 10000.0  # GPU 峰值算力（GFLOPS）
GPU_PEAK_BW_GBS = 500.0   # GPU 内存带宽（GB/s）

# 各版本的理论算术强度（FLOP/Byte）
# 计算公式参见 include/roofline.h 的注释
CPU_AI = {
    'V1_Naive_ijk':     0.25,   # 无 Cache 复用
    'V2_ikj':           0.25,   # 同 V1（访存模式改善但 AI 不变）
    'V3_Tiling':        16.0,   # AI ≈ BLOCK_SIZE/4 = 64/4
    'V4_AVX2':          0.25,   # 只有向量化，无 tiling
    'V5_AVX2+Tiling':   16.0,   # 同 V3
    'V6_OMP+AVX2+Tile': 16.0,   # 同 V3
}

CUDA_AI = {
    'V7_CUDA_Naive':    0.25,   # 无 Shared Memory
    'V8_Shared_Mem':    4.0,    # AI ≈ TILE_SIZE/4 = 16/4
    'V9_Register':      8.0,    # 更高寄存器复用
    'V10_Vectorized':   10.0,   # 向量化加载进一步提升
    'V11_cuBLAS':       25.0,   # 接近 compute bound
}

# 颜色方案
VERSION_COLORS = {
    'V1_Naive_ijk':     '#e74c3c',
    'V2_ikj':           '#e67e22',
    'V3_Tiling':        '#f1c40f',
    'V4_AVX2':          '#2ecc71',
    'V5_AVX2+Tiling':   '#1abc9c',
    'V6_OMP+AVX2+Tile': '#3498db',
    'V7_CUDA_Naive':    '#e74c3c',
    'V8_Shared_Mem':    '#e67e22',
    'V9_Register':      '#2ecc71',
    'V10_Vectorized':   '#1abc9c',
    'V11_cuBLAS':       '#9b59b6',
}

# ============================================================
# 数据加载
# ============================================================

def load_csv(path):
    if not os.path.exists(path):
        print(f"[WARNING] 文件不存在：{path}")
        return None
    df = pd.read_csv(path)
    print(f"加载 {path}，共 {len(df)} 行")
    return df


# ============================================================
# 图1：GFLOPS 柱状图
# ============================================================

def plot_gflops_bar(df, title, output_path):
    """
    分组柱状图：X 轴为矩阵大小，不同颜色为不同版本
    """
    if df is None or df.empty:
        print(f"跳过图 {output_path}：无数据")
        return

    # 取线程数为 1 的记录（若有 threads 列）
    if 'threads' in df.columns:
        df = df[df['threads'] == 1].copy()

    sizes = sorted(df['M'].unique())
    versions = df['version'].unique().tolist()

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(sizes))
    n_vers = len(versions)
    width = 0.8 / n_vers

    for i, ver in enumerate(versions):
        sub = df[df['version'] == ver]
        gflops_vals = [sub[sub['M'] == s]['gflops'].values[0]
                       if len(sub[sub['M'] == s]) > 0 else 0
                       for s in sizes]
        color = VERSION_COLORS.get(ver, f'C{i}')
        bars = ax.bar(x + i * width - 0.4 + width/2, gflops_vals,
                      width, label=ver, color=color, alpha=0.85)

        # 在柱子上方标注数值
        for bar, val in zip(bars, gflops_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xlabel('矩阵大小 (M=N=K)', fontsize=12)
    ax.set_ylabel('GFLOPS', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"保存：{output_path}")


# ============================================================
# 图2：Roofline 图
# ============================================================

def plot_roofline(cpu_df, cuda_df, output_path):
    """
    Roofline 图：X 轴为算术强度（log），Y 轴为 GFLOPS（log）
    包含两条 roofline 线（CPU 和 GPU）和各版本的实测点
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, df, peak_g, peak_bw, ai_map, label in [
        (axes[0], cpu_df, CPU_PEAK_GFLOPS, CPU_PEAK_BW_GBS, CPU_AI, "CPU Roofline"),
        (axes[1], cuda_df, GPU_PEAK_GFLOPS, GPU_PEAK_BW_GBS, CUDA_AI, "GPU Roofline"),
    ]:
        # ---- 绘制 Roofline 天花板线 ----
        ai_range = np.logspace(-2, 3, 200)  # 算术强度范围 0.01 - 1000 FLOP/Byte
        # Roofline = min(计算天花板, 带宽天花板)
        roofline = np.minimum(peak_g, ai_range * peak_bw)
        ax.loglog(ai_range, roofline, 'k-', linewidth=2, label='Roofline')

        # 标注拐点
        ridge = peak_g / peak_bw
        ax.axvline(ridge, color='gray', linestyle='--', alpha=0.5)
        ax.text(ridge * 1.1, peak_g * 0.5,
                f'拐点\n{ridge:.1f} F/B', fontsize=9, color='gray')

        # 标注两段
        ax.text(0.015, peak_bw * 0.015,
                '内存 Bound', fontsize=9, color='blue', rotation=35)
        ax.text(ridge * 2, peak_g * 0.6,
                f'计算 Bound\n峰值={peak_g:.0f} GFLOPS', fontsize=9, color='red')

        # ---- 绘制各版本实测点 ----
        if df is not None and not df.empty:
            # 取最大矩阵的结果（最能代表稳态性能）
            max_size = df['M'].max()
            sub = df[df['M'] == max_size]

            for _, row in sub.iterrows():
                ver = row['version']
                ai = ai_map.get(ver, 1.0)
                gflops = row['gflops']
                color = VERSION_COLORS.get(ver, 'gray')

                ax.scatter(ai, gflops, s=100, color=color, zorder=5, alpha=0.9)
                ax.annotate(ver.replace('_', '\n'), (ai, gflops),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=7, color=color)

        ax.set_xlabel('算术强度 (FLOP/Byte)', fontsize=11)
        ax.set_ylabel('性能 (GFLOPS)', fontsize=11)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

    fig.suptitle('Roofline Model 分析', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"保存：{output_path}")


# ============================================================
# 图3：CUDA 带宽利用率柱状图
# ============================================================

def plot_bandwidth(cuda_df, output_path):
    """
    CUDA 各版本的内存带宽利用率（GB/s）和占峰值比例
    """
    if cuda_df is None or cuda_df.empty:
        print(f"跳过图 {output_path}：无 CUDA 数据")
        return

    # 取最大矩阵的结果
    max_size = cuda_df['M'].max()
    sub = cuda_df[cuda_df['M'] == max_size].copy()

    if 'bandwidth_gb_s' not in sub.columns:
        print(f"跳过图 {output_path}：缺少 bandwidth_gb_s 列")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    versions = sub['version'].tolist()
    bw_vals  = sub['bandwidth_gb_s'].tolist()
    pct_vals = [v / GPU_PEAK_BW_GBS * 100 for v in bw_vals]
    colors   = [VERSION_COLORS.get(v, 'gray') for v in versions]

    x = np.arange(len(versions))

    # 左图：带宽绝对值
    ax1.bar(x, bw_vals, color=colors, alpha=0.85)
    ax1.axhline(GPU_PEAK_BW_GBS, color='red', linestyle='--',
                label=f'峰值带宽 {GPU_PEAK_BW_GBS} GB/s')
    ax1.set_xticks(x)
    ax1.set_xticklabels(versions, rotation=30, ha='right', fontsize=9)
    ax1.set_ylabel('内存带宽 (GB/s)', fontsize=11)
    ax1.set_title(f'CUDA 内存带宽（M=N=K={max_size}）', fontsize=12)
    ax1.legend(); ax1.grid(axis='y', alpha=0.3)

    # 右图：带宽占峰值百分比
    ax2.bar(x, pct_vals, color=colors, alpha=0.85)
    ax2.axhline(100, color='red', linestyle='--', label='峰值带宽 100%')
    ax2.set_xticks(x)
    ax2.set_xticklabels(versions, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('峰值带宽利用率 (%)', fontsize=11)
    ax2.set_title('带宽效率', fontsize=12)
    ax2.legend(); ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"保存：{output_path}")


# ============================================================
# 图4：Scalability 折线图
# ============================================================

def plot_scalability(df, title, output_path):
    """
    折线图：X 轴为矩阵大小，Y 轴为 GFLOPS
    展示各版本随矩阵规模增大的扩展性
    """
    if df is None or df.empty:
        print(f"跳过图 {output_path}：无数据")
        return

    if 'threads' in df.columns:
        df = df[df['threads'] == 1].copy()

    versions = df['version'].unique().tolist()
    sizes    = sorted(df['M'].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, ver in enumerate(versions):
        sub = df[df['version'] == ver].sort_values('M')
        gflops = sub['gflops'].tolist()
        sizes_ver = sub['M'].tolist()
        color = VERSION_COLORS.get(ver, f'C{i}')
        ax.plot(sizes_ver, gflops, 'o-', color=color, label=ver,
                linewidth=2, markersize=6)

    ax.set_xlabel('矩阵大小 (M=N=K)', fontsize=12)
    ax.set_ylabel('GFLOPS', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"保存：{output_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='GEMM 结果可视化')
    parser.add_argument('--cpu-only', action='store_true', help='只生成 CPU 图表')
    parser.add_argument('--gpu-only', action='store_true', help='只生成 GPU 图表')
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs('results', exist_ok=True)

    # 加载数据
    cpu_df  = load_csv('results/cpu_results.csv')
    cuda_df = None if args.cpu_only else load_csv('results/cuda_results.csv')

    print("\n生成图表...")

    # 图1a：CPU GFLOPS 柱状图
    if not args.gpu_only:
        plot_gflops_bar(cpu_df, 'CPU GEMM 各版本性能（GFLOPS）',
                        'results/cpu_gflops.png')

    # 图1b：GPU GFLOPS 柱状图
    if not args.cpu_only:
        plot_gflops_bar(cuda_df, 'CUDA GEMM 各版本性能（GFLOPS）',
                        'results/cuda_gflops.png')

    # 图2：Roofline
    plot_roofline(cpu_df, cuda_df, 'results/roofline.png')

    # 图3：CUDA 带宽
    if not args.cpu_only:
        plot_bandwidth(cuda_df, 'results/cuda_bandwidth.png')

    # 图4a：CPU Scalability
    if not args.gpu_only:
        plot_scalability(cpu_df, 'CPU GEMM 扩展性（矩阵大小 vs GFLOPS）',
                         'results/cpu_scalability.png')

    # 图4b：GPU Scalability
    if not args.cpu_only:
        plot_scalability(cuda_df, 'CUDA GEMM 扩展性（矩阵大小 vs GFLOPS）',
                         'results/cuda_scalability.png')

    print("\n所有图表已保存到 results/ 目录")


if __name__ == '__main__':
    main()
