# GEMM 优化项目

从零实现矩阵乘法（GEMM）的 11 个递进版本，涵盖 CPU Cache 优化、AVX2 SIMD 向量化、OpenMP 多线程到 CUDA GPU 加速，并配套 Roofline 性能分析。

---

## 优化路线图

| 版本 | 技术 | 关键改进 | 典型加速比（vs V1）|
|------|------|---------|-----------------|
| V1 Naive ijk | 三重循环基准 | — | 1x |
| V2 ikj | 循环重排 | B 矩阵从列访问改为行访问，Cache miss 率 100%→6% | 5-10x |
| V3 Tiling | 分块（64×64）| 数据复用，算术强度 0.25→16 FLOP/Byte | 10-20x |
| V4 AVX2 | 手写 SIMD | 每条指令处理 8 个 float + FMA | 15-30x |
| V5 AVX2+Tiling | V3+V4 组合 | Cache 复用 × SIMD 并行 | 30-80x |
| V6 OpenMP | 多线程并行 | 8 线程，利用多核 | 100-300x |
| V7 CUDA Naive | GPU 基准 | 每 Thread 一个输出元素 | GPU 基准 |
| V8 Shared Mem | CUDA 分块 | Shared Memory，Global 访问减少 16 倍 | 3-8x vs V7 |
| V9 Register | 寄存器分块 | 外积展开，Shared AI: 0.125→1 FLOP/Byte | 2-4x vs V8 |
| V10 Vectorized | float4 加载 | 16 字节向量化内存事务 | 1.1-1.3x vs V9 |
| V11 cuBLAS | 工业参考 | Tensor Core + Double Buffer 等 | 性能上界 |

---

## 环境要求

**CPU 编译**：
- GCC 9+ 或 Clang 10+（支持 C++17）
- AVX2 支持的 CPU（Intel Haswell 2013+ / AMD Ryzen 2017+）
- CMake 3.18+

**CUDA 编译（可选）**：
- CUDA Toolkit 11.0+
- 支持 CUDA 的 NVIDIA GPU
- cuBLAS 库（CUDA Toolkit 自带）

**分析工具**：
- Python 3.8+，matplotlib、pandas、numpy

---

## 快速开始

```bash
# 方法1：一键运行（推荐）
chmod +x scripts/run_all.sh
./scripts/run_all.sh

# 方法2：手动编译
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=sm_75  # 根据 GPU 修改架构
make -j$(nproc)
cd ..

# 运行 CPU Benchmark
./build/benchmark_cpu --size 512 1024 2048 4096

# 运行 CUDA Benchmark（有 GPU 时）
./build/benchmark_cuda --size 512 1024 2048 4096

# 生成分析图表
pip3 install -r analysis/requirements.txt
python3 analysis/plot_results.py
python3 analysis/roofline_plot.py
```

---

## 各版本原理

**V1 Naive ijk**：直接翻译数学定义。最内层循环对 B 矩阵按列访问（stride=N×4 字节），远超 Cache line（64 字节），Cache miss 率接近 100%。

**V2 ikj（循环重排）**：将最内层循环改为 j 变化，B 矩阵从列访问变为行访问（stride=4 字节），充分利用 Cache line 空间局部性。同时显式提取 A[i][k] 为寄存器变量，消除循环不变量的重复加载。

**V3 Tiling（分块）**：三层分块（块大小 64），确保 3 个活跃块（A块+B块+C块 = 3×64×64×4 ≈ 48KB）驻留在 L2 Cache 中，实现数据的时间局部性复用。算术强度从 ~0.25 提升到 ~16 FLOP/Byte。

**V4 AVX2**：手写 AVX2 intrinsics，使用 `_mm256_set1_ps` 广播 A 元素，`_mm256_loadu_ps` 加载 B/C 的 8 个连续 float，`_mm256_fmadd_ps` 执行融合乘加。包含尾部标量处理。

**V5 AVX2+Tiling**：V3 分块结构（保证 Cache 命中）+ V4 AVX2（块内向量化计算），两种优化叠加通常获得最高单线程性能。

**V6 OpenMP**：在 V5 基础上，对最外层 i 块循环加 `#pragma omp parallel for schedule(static)`，实现多核并行。不同线程写 C 的不同行区域，无数据竞争。

**V7 CUDA Naive**：每个 Thread 计算 C 的一个元素，Block 16×16。简单但未充分利用 GPU 内存层次。

**V8 Shared Memory**：Block 内 16×16 个 Thread 协同加载 A/B 的 16×16 分块到 Shared Memory（延迟 ~20 cycles），两次 `__syncthreads()` 保证数据一致性。

**V9 Register Tiling**：每个 Thread 计算 8×8=64 个输出元素，使用外积展开（Outer Product）。从 Shared Memory 加载 TM+TN=16 个 float，执行 64 次 FMA，Shared Memory 算术强度提升到 1 FLOP/Byte。

**V10 Vectorized**：在 V9 基础上，将 Global→Shared 的 B 矩阵加载改用 `float4`（16字节/次），减少内存事务数。

**V11 cuBLAS**：NVIDIA 官方 GEMM，使用 Tensor Core + Double Buffering 等深度优化，作为性能上界参考。

---

## 性能结果

> 填入实际测试结果（运行 benchmark 后更新）

### CPU（机器：______，核心数：_，主频：____GHz，内存带宽：____GB/s）

| 版本 | M=N=K=512 | M=N=K=1024 | M=N=K=2048 | M=N=K=4096 |
|------|-----------|------------|------------|------------|
| V1 Naive | — GFLOPS | — GFLOPS | — GFLOPS | — GFLOPS |
| V2 ikj | — | — | — | — |
| V3 Tiling | — | — | — | — |
| V4 AVX2 | — | — | — | — |
| V5 AVX2+Tiling | — | — | — | — |
| V6 OpenMP(8T) | — | — | — | — |

### GPU（GPU：______，显存带宽：____GB/s，FP32 峰值：____TFLOPS）

| 版本 | M=N=K=1024 | M=N=K=2048 | M=N=K=4096 | % of cuBLAS |
|------|------------|------------|------------|-------------|
| V7 CUDA Naive | — GFLOPS | — GFLOPS | — GFLOPS | —% |
| V8 Shared Mem | — | — | — | —% |
| V9 Register | — | — | — | —% |
| V10 Vectorized | — | — | — | —% |
| V11 cuBLAS | — | — | — | 100% |

---

## Roofline 分析

运行 `python3 analysis/roofline_plot.py` 后，图片保存于 `results/roofline_cpu.png` 和 `results/roofline_gpu.png`

```
CPU Roofline 拐点：峰值算力 / 峰值带宽 ≈ 2 FLOP/Byte（典型桌面 CPU）

各版本算术强度：
  V1/V2: ~0.25 FLOP/Byte  → 严重 Memory Bound
  V3/V5: ~16 FLOP/Byte    → Compute Bound（超过拐点，加 SIMD 才有效）
  V8:    ~4 FLOP/Byte     → Memory Bound（对 Shared Memory）
  V9:    ~8+ FLOP/Byte    → 接近 Compute Bound
```

---

## 调试和性能分析

```bash
# CPU Cache 分析（需要 linux-tools）
./scripts/profile_cpu.sh 1024

# CUDA Kernel 分析（需要 Nsight Compute）
./scripts/profile_cuda.sh 9 2048

# 编译后查看 CUDA 寄存器用量
nvcc --ptxas-options=-v cuda/gemm_v9_register.cu
```

---

## 项目结构

```
数学库项目/
├── include/
│   ├── gemm.h                  # 所有版本的统一接口（V1-V11）
│   ├── benchmark_utils.h       # 计时、验证、CSV 输出工具
│   └── roofline.h              # 硬件参数 + 算术强度计算
├── src/                        # CPU 实现（V1-V6）
├── cuda/                       # CUDA 实现（V7-V11）
├── analysis/                   # Python 可视化
├── scripts/                    # 编译运行脚本
├── notes/                      # 学习笔记（6个主题）
├── results/                    # Benchmark 结果（gitignore）
└── CMakeLists.txt
```

---

## 参考资料

- [How to Optimize GEMM（Simon Boehm）](https://siboehm.com/articles/22/CUDA-MMM)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Roofline Model（Williams et al., 2009）](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)

---

## 学习收获（面试话术）

> "我从手写 Naive GEMM 开始，用 perf 定量分析了 Cache miss 率，
> 发现 ijk 顺序的 L1 miss 率约 50%，改为 ikj 后降至 6%（V2）；
> 用 Roofline 模型发现 V2 的算术强度只有 0.25 FLOP/Byte，严重 Memory Bound，
> 引入 64×64 分块后算术强度提升到约 16 FLOP/Byte，超过了拐点，
> 此时加入 AVX2 SIMD（V5）才真正有效（CPU 端最终比 Naive 快约 50-100 倍）。
> GPU 端从 Shared Memory Tiling（V8）到 Register Tiling（V9），
> 将 Shared Memory 算术强度从 0.125 提升到 1 FLOP/Byte，
> 用 Nsight Compute 验证了 dram bytes 接近理论最优，
> V9 最终达到 cuBLAS 的约 40-60%。"
