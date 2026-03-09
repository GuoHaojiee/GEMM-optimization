#!/bin/bash
# ============================================================
# profile_cpu.sh — CPU 性能分析（perf stat）
# ============================================================
#
# 用法：
#   chmod +x scripts/profile_cpu.sh
#   ./scripts/profile_cpu.sh              # 默认大小 1024
#   ./scripts/profile_cpu.sh 2048         # 指定矩阵大小
#   sudo ./scripts/profile_cpu.sh 1024    # 某些指标需要 root
#
# 需要：
#   - perf 工具（linux-tools-$(uname -r) 包）
#   - benchmark_cpu 已编译
#
# 安装 perf：
#   Ubuntu/Debian: sudo apt-get install linux-tools-generic
#   查看可用事件：perf list
# ============================================================

SIZE=${1:-1024}
BINARY="./build/benchmark_cpu"

if [ ! -f "$BINARY" ]; then
    echo "错误：未找到 $BINARY，请先运行 scripts/run_all.sh 编译"
    exit 1
fi

echo "============================================================"
echo "CPU 性能分析：矩阵大小 M=N=K=$SIZE"
echo "============================================================"

# ============================================================
# perf stat 指标说明
# ============================================================
# cache-references     : L3 Cache 访问总次数（包括 hit 和 miss）
# cache-misses         : L3 Cache miss 次数（越少越好）
#
# L1-dcache-loads      : L1 数据 Cache 读取次数（命中+未命中）
# L1-dcache-load-misses: L1 数据 Cache 读取 miss 次数
#   → miss率 = L1-dcache-load-misses / L1-dcache-loads
#   → V1（ijk）的 miss 率远高于 V2（ikj）
#
# LLC-loads            : Last Level Cache（通常是 L3）读取次数
# LLC-load-misses      : L3 miss 次数（触发 DRAM 访问）
#   → LLC miss 代表实际的 DRAM 访问，直接影响内存带宽消耗
#
# instructions         : 执行的指令总数
# cycles               : CPU 周期数
#   → IPC（Instructions Per Cycle）= instructions / cycles
#   → 高 IPC 说明流水线利用充分，没有太多 stall
#
# fp_arith_inst_retired.256b_packed_single: 执行的 AVX2 FP 指令数（256 位单精度）
#   → 用于验证 V4-V6 是否真正使用了 AVX2（非 0 说明向量化有效）
#   → 需要 root 权限
# ============================================================

echo ""
echo "【L1/L3 Cache 效率分析】"
perf stat -e \
    cache-references,\
    cache-misses,\
    L1-dcache-loads,\
    L1-dcache-load-misses,\
    LLC-loads,\
    LLC-load-misses,\
    instructions,\
    cycles \
    "$BINARY" --size "$SIZE" --threads 1 2>&1

echo ""
echo "【解读指南】"
echo "  L1 miss 率 = L1-dcache-load-misses / L1-dcache-loads"
echo "  期望：V1 miss率 >> V2 miss率（因为 B 矩阵列访问 vs 行访问）"
echo "  LLC miss 率 = LLC-load-misses / LLC-loads"
echo "  期望：V3（分块）的 LLC miss 远少于 V2（无分块）"
echo ""
echo "  IPC = instructions / cycles"
echo "  期望：V4/V5（AVX2）的 IPC 比 V1 高（每条指令计算 8 个数据）"

# ============================================================
# 额外：用 perf record + perf report 做热点分析
# ============================================================
echo ""
echo "【可选：热点函数分析】"
echo "运行以下命令查看热点："
echo ""
echo "  sudo perf record -g $BINARY --size $SIZE"
echo "  sudo perf report"
echo ""
echo "【可选：Cache 友好度可视化】"
echo "  sudo perf mem record $BINARY --size $SIZE"
echo "  sudo perf mem report"
