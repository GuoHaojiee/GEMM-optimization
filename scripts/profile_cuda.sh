#!/bin/bash
# ============================================================
# profile_cuda.sh — CUDA Kernel 性能分析（Nsight Compute）
# ============================================================
#
# 用法：
#   chmod +x scripts/profile_cuda.sh
#   ./scripts/profile_cuda.sh              # 默认分析 V9，矩阵 2048
#   ./scripts/profile_cuda.sh 9 2048       # 分析 V9，大小 2048
#   ./scripts/profile_cuda.sh 7 1024       # 分析 V7，大小 1024
#
# 需要：
#   - ncu（Nsight Compute CLI）：CUDA Toolkit 自带
#   - 通常需要 sudo 或设置 NV_PERF_PARANOID_PERMISSIONS
#
# 设置权限（无需 sudo）：
#   sudo sh -c "echo 0 > /proc/sys/kernel/perf_event_paranoid"
#   sudo sh -c "echo 0 > /proc/sys/kernel/kptr_restrict"
# ============================================================

VERSION=${1:-9}
SIZE=${2:-2048}
BINARY="./build/benchmark_cuda"
OUTPUT_DIR="results/profiles"

if [ ! -f "$BINARY" ]; then
    echo "错误：未找到 $BINARY，请先编译 CUDA 版本"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "CUDA V${VERSION} 性能分析：矩阵大小 M=N=K=$SIZE"
echo "============================================================"

# ============================================================
# 核心指标说明
# ============================================================
# dram__bytes_read.sum
#   DRAM（显存）读取字节数总量
#   越接近理论最小值（3 × M×N × sizeof(float)）说明 Cache/Shared 复用越好
#   Naive 版本可能是理论值的 16+ 倍（因为无合并访问）
#
# dram__bytes_write.sum
#   DRAM 写入字节数（通常等于 M×N × sizeof(float)，即 C 矩阵大小）
#
# sm__throughput.avg.pct_of_peak_sustained_active
#   SM（流式多处理器）利用率，百分比
#   100% = SM 全速运行
#   典型值：V7 约 30%, V9 约 70%, cuBLAS 约 90%
#
# l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum
#   L1 Texture Cache 的 Global Memory 加载字节数
#   如果远大于 dram__bytes_read，说明 L1 Cache 命中率高（好事）
#   如果接近 dram__bytes_read，说明 L1 几乎没有复用（Naive 版本的特征）
#
# launch__occupancy
#   实际 Occupancy（活跃 Warp 数 / SM 最大支持 Warp 数）
#   高 Occupancy 有助于隐藏内存延迟（用其他 Warp 的计算填充等待时间）
#   但高 Occupancy 不是万能的：若瓶颈是计算，提高 Occupancy 没有帮助
#
# smsp__sass_average_data_bytes_per_sector_mem_shared.pct
#   Shared Memory 的 Bank 效率
#   100% = 无 Bank Conflict
#   低于 100% = 有 Bank Conflict，影响 Shared Memory 带宽
# ============================================================

echo ""
echo "【完整性能分析（核心指标）】"
ncu \
    --target-processes all \
    --metrics \
        dram__bytes_read.sum,\
        dram__bytes_write.sum,\
        sm__throughput.avg.pct_of_peak_sustained_active,\
        l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
        launch__occupancy,\
        smsp__sass_average_data_bytes_per_sector_mem_shared.pct,\
        sm__sass_thread_inst_executed_op_ffma_pred_on.sum \
    --export "$OUTPUT_DIR/v${VERSION}_${SIZE}.ncu-rep" \
    "$BINARY" --size "$SIZE" --version "$VERSION"

echo ""
echo "【性能摘要】"
echo "分析报告已保存到：$OUTPUT_DIR/v${VERSION}_${SIZE}.ncu-rep"
echo ""
echo "用 Nsight Compute GUI 查看详细报告："
echo "  ncu-ui $OUTPUT_DIR/v${VERSION}_${SIZE}.ncu-rep"

# ============================================================
# 快速分析：带宽利用率
# ============================================================
echo ""
echo "【快速带宽分析（仅显存带宽）】"
ncu \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum \
    "$BINARY" --size "$SIZE" --version "$VERSION" 2>&1 | grep -E "dram|Kernel"

# ============================================================
# 可选：完整 Roofline 分析
# ============================================================
echo ""
echo "【可选：Roofline 分析（耗时较长）】"
echo "运行以下命令获取完整 Roofline 分析："
echo ""
echo "  ncu --set roofline $BINARY --size $SIZE --version $VERSION"
echo ""
echo "或在 Nsight Compute GUI 中选择 'Roofline' 分析集"
