# GEMM 优化项目 — Claude Code 提示词（修订版）

我需要你帮我构建一个完整的、用于求职高性能计算岗位的 GEMM 优化项目。
这个项目将作为我简历上的核心项目，代码质量和注释质量要达到
"能在面试中讲清楚每一行"的标准。

---

## 项目目标

实现单精度矩阵乘法（SGEMM）C = A × B 的逐级性能优化，
从 Naive 三重循环到接近 cuBLAS 的水平。

**基本约定（全项目统一）：**
- 数据类型：float（单精度，FP32）
- 存储格式：Row-major（行优先），一维数组 `float* A` 表示 M×K 矩阵，`A[i*K+j]`
- 矩阵维度：M×K 的 A 乘以 K×N 的 B，得到 M×N 的 C
- 不要求 M/N/K 是 2 的幂或特定块大小的倍数，所有版本必须正确处理边界情况
- 正确性验证：使用 **逐元素相对误差** `|C_test[i] - C_ref[i]| / max(|C_ref[i]|, 1e-7) < 1e-3`，
  而非绝对误差（因为 float 累加误差与矩阵大小正相关）

---

## 必须实现的版本（按顺序）

### CPU 部分（单线程 → 多线程 → SIMD）

**V1 — Naive ijk 三重循环**
- 最朴素的 i-j-k 循环实现
- 作为正确性参考基准和性能下限
- 注释要点：分析该顺序下对 A/B/C 三个矩阵的访存模式（stride-1 vs stride-N）

**V2 — 循环重排 ikj**
- 将内层循环改为 k-j 顺序
- 注释要点：画出 ijk vs ikj 的内存访问示意（哪个方向是 stride-1），
  解释 Cache line 的空间局部性如何因此改善

**V3 — 分块 Tiling**
- 对 i/j/k 三个维度分块，块大小作为 `constexpr int BLOCK_SIZE = 64` 可调
- 注释要点：解释为什么块大小要匹配 L1 Cache 容量（3 个 BLOCK×BLOCK 的 float 子矩阵要能放入 L1），
  给出计算公式 `3 × BS × BS × 4 bytes ≤ L1 size`
- 正确处理维度不是 BLOCK_SIZE 整数倍的情况

**V4 — AVX2 SIMD 向量化**
- 使用 `_mm256_loadu_ps` / `_mm256_storeu_ps` / `_mm256_fmadd_ps` 等 intrinsics
- 内循环一次处理 8 个 float（256bit / 32bit = 8）
- 注释要点：
  - `_mm256_set1_ps(scalar)` 是广播（broadcast），将标量复制到 8 个 lane
  - `_mm256_fmadd_ps(a,b,c)` = a*b+c 单条指令完成，吞吐量优于分开的 mul+add
  - 为什么用 `loadu`（非对齐加载）而非 `load`（对齐加载），以及何时可以改用对齐版本
  - 处理 N 不是 8 的倍数的尾部元素（scalar fallback 或 masked store）
- 结合 ikj 循环顺序以确保内层循环方向与 SIMD 加载方向一致

**V5 — AVX2 + Tiling**
- V3 和 V4 的组合：分块后在块内使用 AVX2 intrinsics
- 这是单线程 CPU 最优版本

**V6 — OpenMP 多线程并行**
- 在 V5 基础上对最外层循环（i 方向分块循环）加 `#pragma omp parallel for`
- 注释要点：
  - 为什么并行化最外层而非内层（减少线程创建开销，保持 Cache 友好性）
  - `schedule(static)` vs `schedule(dynamic)` 的选择依据
  - False sharing 问题：每个线程写 C 的不同行，天然不会冲突
  - 线程数与物理核心数的关系、超线程的影响

### CUDA 部分

**V7 — CUDA Naive（Global Memory Only）**
- 每个 Thread 计算 C 的一个元素，直接从 Global Memory 读取 A 和 B
- Grid/Block 配置：二维 block（16×16），二维 grid 向上取整覆盖 M×N
- 注释要点：分析该实现对 Global Memory 的访问总量（2MNK 次 float 读取），
  解释为什么极其低效

**V8 — CUDA Shared Memory Tiling**
- 16×16 或 32×32 分块，协同加载（cooperative loading）A 和 B 的子矩阵到 Shared Memory
- **两次 `__syncthreads()` 的位置和原因必须详细注释：**
  1. 加载完成后同步 — 确保所有线程的数据都已写入 Shared Memory 再开始计算
  2. 计算完成后同步 — 确保所有线程用完当前子矩阵后再加载下一块（防止覆盖正在使用的数据）
- 注释要点：
  - Shared Memory 的容量限制如何决定块大小（48KB per SM，每个 block 分到多少）
  - Bank conflict 的概念和本实现中为什么没有 bank conflict（连续线程访问连续地址）
  - Occupancy（占用率）的基本概念

**V9 — CUDA Register Tiling（Thread-level Tiling / 寄存器分块）**
- 每个线程计算 C 的一个 TM×TN 的小块（如 8×8），而非单个元素
- A 和 B 的数据先从 Shared Memory 加载到寄存器（`float regA[TM], regB[TN]`），再做外积累加
- 注释要点：
  - 为什么这是从 Shared Memory 到 cuBLAS 之间最关键的优化
    （大幅减少 Shared Memory 读取次数：每次从 smem 加载被 TM+TN 个寄存器复用 TM*TN 次）
  - 数据复用比的计算：每个线程加载 TM+TN 个 float，计算 TM×TN 次 FMA
  - 寄存器压力与 Occupancy 的权衡：TM/TN 越大寄存器越多，但 Occupancy 可能下降
  - 如何选择 TM/TN 的值（常见选择：4×4, 8×8）

**V10 — CUDA 向量化 Global Memory 加载**
- 在 V9 基础上，将 Global Memory → Shared Memory 的加载改用 `float4`（128bit）
- 一条指令加载 4 个 float，减少内存事务数
- 注释要点：
  - `float4` 加载要求地址 16 字节对齐
  - 加载后如何拆分到 Shared Memory 的正确位置
  - 与 V9 的性能差距分析（主要提升 memory bandwidth utilization）

**V11 — cuBLAS 对比基准**
- 调用 `cublasSgemm` 作为性能上限参考
- 记录性能数字，不需要自己实现
- 注释：cuBLAS 内部使用了哪些我们没覆盖的优化
  （double buffering / software pipelining / warp-level MMA / auto-tuning）

---

## 性能分析（Profiling）

### Roofline Model 分析

每个版本需要计算：
- **理论算术强度**（Arithmetic Intensity）= `2MNK FLOPs / 实际内存传输字节数`
  - V1 Naive：A 的每个元素被读 N 次，B 的每个元素被读 M 次 → AI ≈ 2MNK / (4*(MK*N + KN*M)) 很低
  - V8 Shared Memory：A 和 B 各从 Global Memory 读一次 → AI ≈ 2MNK / (4*2*MKN/TILE) 提升
  - 每个版本的 AI 需要推导并注释在代码中
- **实测性能**（GFLOPS）= 2MNK / (time_in_seconds) / 1e9
- **硬件参数**（在 `roofline.h` 中定义，用户根据自己的硬件手动填写）：
  - CPU：峰值 GFLOPS = 核心数 × 频率 × FMA吞吐 × SIMD宽度，内存带宽 GB/s
  - GPU：峰值 GFLOPS（查 spec），HBM/GDDR 带宽 GB/s（查 spec）
  - 提供注释指导用户如何查到这些数字

Python 脚本绘制 Roofline 图时，在图上标注每个版本的实测点，用不同颜色和标记区分 CPU/CUDA 版本。

### CPU Profiling

- **首选方案**：提供 `scripts/profile_cpu.sh`，使用 `perf stat` 统计：
  ```
  perf stat -e cache-references,cache-misses,L1-dcache-load-misses,\
  L1-dcache-loads,LLC-load-misses,LLC-loads,instructions,cycles \
  ./benchmark_cpu --size 1024
  ```
- 输出解读注释：Cache miss rate = misses/loads，CPI = cycles/instructions
- 不依赖 PAPI（大多数系统没有预装），perf 即可满足需求

### CUDA Profiling

- 使用 **CUDA Event** 计时（`cudaEventCreate`, `cudaEventRecord`, `cudaEventElapsedTime`），不用 CPU 端计时
- 提供 `scripts/profile_cuda.sh`，封装 `ncu`（Nsight Compute）命令：
  ```
  ncu --metrics \
    dram__bytes_read.sum,dram__bytes_write.sum,\
    sm__throughput.avg.pct_of_peak_sustained_active,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
    launch__occupancy \
    ./benchmark_cuda --size 2048 --version 9
  ```
- 在代码注释中说明每个 metric 的含义和如何解读
- 自动计算：实测带宽 = (bytes_read + bytes_write) / kernel_time，
  带宽利用率 = 实测带宽 / GPU 峰值带宽

---

## Benchmark 框架

### CPU Benchmark（`src/benchmark_cpu.cpp`）
- 命令行：`./benchmark_cpu --size 512 1024 2048 4096 --threads 1 4 8`
- 流程：分配矩阵 → 随机初始化 → Naive 计算参考结果 → 逐版本运行 → 验证 → 计时
- Warm-up 3 次，正式测试 5 次取中位数（比平均数更稳定）
- 输出表格（对齐打印到终端）+ 写入 `results/cpu_results.csv`
- CSV 列：`version,M,N,K,threads,time_ms,gflops,speedup_vs_naive`

### CUDA Benchmark（`cuda/benchmark_cuda.cu`）
- 命令行：`./benchmark_cuda --size 512 1024 2048 4096`
- 使用 CUDA Event 计时
- 先在 GPU 上用 Naive kernel 计算参考结果（避免 CPU→GPU 传输参考矩阵的开销）
- 输出表格 + 写入 `results/cuda_results.csv`
- CSV 列：`version,M,N,K,time_ms,gflops,bandwidth_gb_s,pct_of_cublas`

### Python 可视化（`analysis/plot_results.py`）
读取上述 CSV 文件，生成以下图表（保存为 PNG，DPI=150）：
1. **GFLOPS 柱状图**：X 轴为版本名，不同矩阵大小用不同颜色分组
2. **Roofline 图**：X 轴为算术强度 (FLOP/Byte)，Y 轴为 GFLOPS，绘制 roofline 天花板线，标注每个版本的实测点
3. **CUDA 带宽利用率柱状图**：每个 CUDA 版本的实测带宽占峰值带宽的百分比
4. **Scalability 折线图**：X 轴为矩阵大小，Y 轴为 GFLOPS，每条线代表一个版本

---

## 代码注释标准

每个优化版本的函数头部必须包含一个注释块：

```cpp
/**
 * 【版本】V3 — 分块 Tiling
 * 【原理】将大矩阵分成 BLOCK_SIZE × BLOCK_SIZE 的子矩阵，
 *         使 3 个子矩阵同时驻留在 L1 Cache 中，减少对主存的访问。
 * 【为什么快】
 *   - Naive: 每个 B 元素被从内存加载 M 次
 *   - Tiling: 每个 B 元素被加载到 Cache 后复用 BLOCK_SIZE 次
 *   - 数据复用率提升 BLOCK_SIZE 倍
 * 【硬件视角】
 *   L1 Cache 通常 32-48KB。
 *   BLOCK_SIZE=64 时，3 个子矩阵 = 3×64×64×4B = 48KB，刚好放入 L1。
 *   BLOCK_SIZE=32 时，3×32×32×4B = 12KB，更保守但也更通用。
 * 【面试要点】
 *   - Q: 块大小怎么选？ A: 受 L1 大小约束，经验值 32-64
 *   - Q: 为什么不能无限大？ A: 超过 Cache 容量就会频繁 evict，反而变慢
 *   - Q: Tiling 对 TLB miss 有帮助吗？ A: 有，减少了页面跨度
 */
```

函数体内，每个不显而易见的操作都需要行内注释。

---

## 项目目录结构

```
gemm-optimization/
├── include/
│   ├── gemm.h                  # 统一函数接口声明（所有版本用同一签名）
│   ├── benchmark_utils.h       # 计时、验证、CSV输出工具
│   └── roofline.h              # 硬件参数定义 + 算术强度计算
├── src/
│   ├── gemm_v1_naive.cpp           # ijk 基准
│   ├── gemm_v2_ikj.cpp             # 循环重排
│   ├── gemm_v3_tiling.cpp          # 分块
│   ├── gemm_v4_avx2.cpp            # AVX2 SIMD
│   ├── gemm_v5_avx2_tiling.cpp     # AVX2 + 分块
│   ├── gemm_v6_openmp.cpp          # OpenMP 多线程 + AVX2 + 分块
│   └── benchmark_cpu.cpp           # CPU 测试主程序
├── cuda/
│   ├── gemm_v7_naive.cu            # Global Memory only
│   ├── gemm_v8_shared.cu           # Shared Memory Tiling
│   ├── gemm_v9_register.cu         # Register Tiling（关键版本）
│   ├── gemm_v10_vectorized.cu      # + float4 向量化加载
│   ├── cublas_reference.cu         # cuBLAS 封装
│   └── benchmark_cuda.cu           # CUDA 测试主程序
├── analysis/
│   ├── plot_results.py             # 读取 CSV 绘制 4 张性能图
│   ├── roofline_plot.py            # 单独的 Roofline 绘图脚本
│   └── requirements.txt           # matplotlib numpy pandas
├── scripts/
│   ├── run_all.sh                  # 一键编译 + 运行所有 benchmark
│   ├── profile_cpu.sh              # perf stat 分析
│   └── profile_cuda.sh             # ncu 分析
├── notes/
│   ├── 01_cache_and_tiling.md      # Cache 原理 + Tiling 分析
│   ├── 02_simd_avx2.md             # SIMD 指令集 + intrinsics 详解
│   ├── 03_cuda_memory_hierarchy.md # CUDA 内存模型（Global/Shared/Register/L1/L2）
│   ├── 04_roofline_model.md        # Roofline 分析方法 + 如何计算 AI
│   ├── 05_register_tiling.md       # 寄存器分块的原理（这是最难也最重要的部分）
│   └── 06_interview_qa.md          # 面试题库（20+题）
├── results/                        # benchmark 输出（gitignore）
├── .gitignore
├── CMakeLists.txt                  # CMake 构建系统
└── README.md
```

---

## notes/06_interview_qa.md 要求

整理 **至少 20 题**，格式如下：

```
### Q1: 为什么 ikj 循环顺序比 ijk 快？

**回答：**
ijk 顺序中，内层循环遍历 k，每次迭代访问 B[k][j] 时 k 在变化，
而 B 是行优先存储，所以 B[k][j] 到 B[k+1][j] 跨了整行（stride = N），
几乎每次都是 Cache miss。
ikj 顺序中，内层循环遍历 j，B[k][j] 到 B[k][j+1] 是连续地址（stride = 1），
一条 Cache line（64B）加载后可复用 16 个 float，空间局部性极好。

**核心关键词：** Cache line, spatial locality, stride-1 access, row-major
**延伸追问：** 如果 B 是列优先（column-major）存储，哪种循环顺序更快？
```

必须覆盖的主题（每个至少 2-3 题）：
- **Cache 原理**：Cache line 大小、associativity、eviction policy、false sharing
- **SIMD 向量化**：FMA 优势、对齐加载、尾部处理、auto-vectorization vs intrinsics
- **CUDA 内存层次**：Global/Shared/Register/L1/L2 的延迟和带宽差异、coalescing、bank conflict
- **CUDA 执行模型**：warp、SIMT、occupancy、warp divergence
- **Register Tiling**：为什么是 GEMM 最重要的优化、数据复用比如何计算
- **Roofline Model**：compute bound vs memory bound 判断方法、如何读图、优化方向建议
- **通用优化方法论**：Amdahl's law、profiling 驱动优化、避免过早优化

---

## README.md 要求

1. **项目简介**：一句话说清楚做了什么
2. **优化路线图**：用简洁的表格列出所有版本及其核心优化手段
3. **环境要求**：
   - GCC 9+ with AVX2/FMA support
   - CUDA Toolkit 11.0+，GPU Compute Capability ≥ 7.5（Turing 及以上）
   - CMake 3.18+
   - Python 3.8+，`pip install -r analysis/requirements.txt`
4. **编译与运行**：完整的 cmake 命令序列
5. **各版本优化原理**：每个版本 2-3 句话总结
6. **性能结果**：预留表格模板，用户填入实测数据
7. **Roofline 分析结论**：预留位置
8. **学到了什么**（可直接用于面试自我介绍中的项目部分）

---

## 编译要求

### CMakeLists.txt
- 使用 CMake 3.18+，`enable_language(CUDA)` 原生支持
- CPU 目标：`benchmark_cpu`，编译选项 `-O3 -march=native -mavx2 -mfma -fopenmp`
- CUDA 目标：`benchmark_cuda`，`-arch=sm_75`（可通过 CMake 变量覆盖）
- 自动检测 CUDA 是否可用，如果不可用只编译 CPU 部分（用 `if(CMAKE_CUDA_COMPILER)` 判断）
- 链接 cuBLAS：`target_link_libraries(benchmark_cuda PRIVATE cublas)`

---

## 代码风格要求

1. 不要为了简洁牺牲可读性，复杂操作拆开写
2. 每个优化版本独立文件，不要合并
3. 所有版本共用同一函数签名：
   `void gemm_vN(const float* A, const float* B, float* C, int M, int N, int K);`
4. CUDA 版本的 host wrapper 也遵循上述签名（内部完成设备内存分配和传输）
5. notes/ 的文档按照"假设读者有 C++ 基础但不了解 HPC"的标准撰写
6. 代码中的英文注释和中文注释均可，但保持每个文件内一致

---

## 提醒

- 请先创建完整的目录结构和所有文件
- 每个 .cpp / .cu 文件都要能独立阅读、独立理解
- 最重要的版本是 V9（Register Tiling）和 V6（OpenMP），请重点投入注释
- notes/05_register_tiling.md 要写得像教程，从"为什么 Shared Memory 版本还不够快"讲起
