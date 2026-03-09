#!/bin/bash
# ============================================================
# run_all.sh — 一键编译 + 运行所有 benchmark + 生成图表
# ============================================================
#
# 用法：
#   chmod +x scripts/run_all.sh
#   ./scripts/run_all.sh                    # 默认大小 512 1024 2048 4096
#   ./scripts/run_all.sh --cpu-only         # 只运行 CPU benchmark
#   ./scripts/run_all.sh --size 1024 2048   # 指定矩阵大小
#
# 前置条件：
#   - cmake >= 3.18
#   - g++ with AVX2 support
#   - CUDA Toolkit（可选，无 CUDA 时跳过 GPU benchmark）
#   - Python 3 + pandas + matplotlib（用于绘图）
# ============================================================

set -e  # 遇到错误立即退出
set -u  # 使用未定义变量时报错

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# 解析参数
CPU_ONLY=false
SIZES="512 1024 2048 4096"
THREADS="1 4 8"

for arg in "$@"; do
    case $arg in
        --cpu-only) CPU_ONLY=true ;;
        --size)     shift; SIZES="$*"; break ;;
    esac
done

cd "$PROJECT_ROOT"

# ============================================================
# 步骤1：创建必要目录
# ============================================================
info "创建目录..."
mkdir -p build results

# ============================================================
# 步骤2：CMake 配置
# ============================================================
info "配置 CMake..."
cd build

# 传入 CUDA 架构（根据你的 GPU 修改）
# sm_75: RTX 20系列, T4
# sm_80: A100
# sm_86: RTX 30系列
# sm_89: RTX 40系列
# 查询命令：nvidia-smi --query-gpu=compute_cap --format=csv,noheader
CUDA_ARCH=${CUDA_ARCH:-"75"}

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ARCH="sm_${CUDA_ARCH}" \
    2>&1 | tail -20

# ============================================================
# 步骤3：编译
# ============================================================
info "编译（使用 $(nproc) 个并行任务）..."
make -j$(nproc)
success "编译完成"

cd "$PROJECT_ROOT"

# ============================================================
# 步骤4：运行 CPU Benchmark
# ============================================================
info "运行 CPU Benchmark..."
info "矩阵大小：$SIZES"
info "线程数：$THREADS"

./build/benchmark_cpu \
    --size $SIZES \
    --threads $THREADS

success "CPU Benchmark 完成，结果写入 results/cpu_results.csv"

# ============================================================
# 步骤5：运行 CUDA Benchmark（如果可用）
# ============================================================
if [ "$CPU_ONLY" = false ] && [ -f "./build/benchmark_cuda" ]; then
    info "运行 CUDA Benchmark..."
    ./build/benchmark_cuda --size $SIZES
    success "CUDA Benchmark 完成，结果写入 results/cuda_results.csv"
else
    if [ "$CPU_ONLY" = true ]; then
        info "跳过 CUDA Benchmark（--cpu-only 模式）"
    else
        info "未找到 benchmark_cuda（未安装 CUDA，跳过）"
    fi
fi

# ============================================================
# 步骤6：生成图表
# ============================================================
if command -v python3 &> /dev/null; then
    info "安装 Python 依赖..."
    pip3 install -q -r analysis/requirements.txt 2>/dev/null || true

    info "生成图表..."
    if [ "$CPU_ONLY" = true ]; then
        python3 analysis/plot_results.py --cpu-only
    else
        python3 analysis/plot_results.py
    fi
    python3 analysis/roofline_plot.py
    success "图表已保存到 results/ 目录"
else
    info "未找到 python3，跳过图表生成"
fi

# ============================================================
# 汇总
# ============================================================
echo ""
echo "============================================================"
success "所有任务完成！"
echo "结果文件："
ls -la results/ 2>/dev/null || echo "  (results/ 目录为空)"
echo "============================================================"
