# 03 CUDA 内存层次结构详解

## 1. GPU 内存层次（从慢到快）

```
层次           每 SM 容量   延迟（cycles）   带宽估算（全 GPU）  作用域
─────────────────────────────────────────────────────────────────────────
Global Memory  几十 GB      400-800          500-2000 GB/s   所有 Thread
L2 Cache       几 MB        ~150             -               所有 Thread
L1/Shared Mem  ~48KB        ~20-40           ~19 TB/s        Block 内共享
Register       ~256KB/SM    1                极高            每个 Thread
```

注：Shared Memory 和 L1 Cache 共享物理资源，可通过 `cudaFuncSetCacheConfig` 配置比例。

---

## 2. Global Memory

### 特征

- 位于 GPU 板载 DRAM（GDDR6 / HBM2e）
- 延迟最高（400-800 cycles），但带宽也最高（PCIe 以内的 GPU 内访问）
- 对齐访问（coalesced）时带宽利用率高

### 内存合并访问（Coalescing）★★★

**最重要的 CUDA 优化原则**

```
合并访问（好）：
  Warp 内 32 个 Thread，Thread i 访问 addr + i
  ─────────────────────────────────────────────
  T0   T1   T2   T3   ...  T31
  addr  addr+1 addr+2 ...  addr+31   ← 连续 128 字节 = 4 个 Cache line
  硬件将 32 次访问合并为 1-4 次内存事务 ✓

非合并访问（坏）：
  Thread i 访问 addr + i * stride（stride > 1）
  ─────────────────────────────────────────────
  T0        T1        T2       T31
  addr    addr+128  addr+256  addr+3968   ← 每个 Thread 在不同 Cache line
  硬件需要 32 次独立的内存事务 ✗（理论上慢 32 倍！）
```

**GEMM 中的 Coalescing 分析**：

| 版本 | 访问 A | 访问 B | Coalescing |
|------|--------|--------|-----------|
| V7 Naive | 非合并（同 Warp 访问不同行）| 合并（同一行）| 部分 |
| V8 Shared | 合并加载（协同加载 A/B 行）| 合并加载 | 完全合并 |

---

## 3. Shared Memory

### 特征

- 片上 SRAM，每个 SM 约 48KB（Ampere GPU 可达 100KB）
- 延迟约 20-40 cycles（比 Global Memory 快 10-20 倍）
- Block 内所有 Thread 共享，需要用 `__syncthreads()` 协调

### Bank Conflict（存储体冲突）

Shared Memory 被分为 32 个 **Bank**（对应 Warp 的 32 个 Thread）：
- Bank i 存放地址 4×i, 4×(i+32), 4×(i+64), ... 的数据
- 同一 Warp 的多个 Thread 访问**同一 Bank 的不同地址**→ Bank Conflict（串行化）

```
无 Bank Conflict（理想）：
  Thread 0 → Bank 0（addr 0）
  Thread 1 → Bank 1（addr 4）
  Thread 2 → Bank 2（addr 8）
  ...（连续访问，每个 Bank 只被 1 个 Thread 访问）
  → 1 个 Shared Memory 事务

2-way Bank Conflict：
  Thread 0 → Bank 0（addr 0）
  Thread 16 → Bank 0（addr 128）  ← 与 Thread 0 同 Bank！
  其余 Thread → 不同 Bank
  → 需要 2 个 Shared Memory 事务（性能下降 2 倍）
```

**GEMM V8 中的 Bank Conflict 分析**：

```cpp
__shared__ float Bs[TILE_SIZE][TILE_SIZE];  // TILE_SIZE=16
// 访问 Bs[k][tx]：k 固定，tx 变化
// tx = 0..15，每个 Thread 访问 Bs[k][0], Bs[k][1], ..., Bs[k][15]
// 地址连续，映射到不同的 Bank（0..15）→ 无冲突 ✓
```

---

## 4. Register（寄存器）

### 特征

- 每个 Thread 独享，延迟 1 cycle
- 每个 SM 约 65536 个（32 位）寄存器
- 每个 Thread 最多使用约 255 个寄存器
- **寄存器溢出（Register Spilling）**：超出限制时数据被放到 Local Memory（实际上是 Global Memory，很慢！）

### 与 Occupancy 的关系

```
Occupancy = 实际活跃 Warp 数 / SM 最大支持 Warp 数

影响因素：
  寄存器用量（registers/thread）：
    每 Thread 用 32 reg → 1 SM 可容纳 65536/32 = 2048 Thread = 64 Warp
    每 Thread 用 64 reg → 1 SM 可容纳 1024 Thread = 32 Warp（Occupancy 降低）

  Shared Memory 用量（smem/block）：
    TILE_SIZE=16：Bs+As = 2×16×16×4 = 2KB/block
    每 SM 48KB：48/2 = 24 个 Block 可同时在线
    若 Block 有 256 Thread → 24×256/2048 = 3072/2048 → Occupancy 100%（受其他因素限制）

查看寄存器用量：
  nvcc --ptxas-options=-v kernel.cu
  输出类似：Used 32 registers, 2048+0 bytes smem
```

### 寄存器 vs Occupancy 的权衡

- V9 Register Tiling：每 Thread 使用约 64 个寄存器（accum[8][8]=64，加上临时变量）
- 这可能降低 Occupancy（每 Thread 更多寄存器 → SM 能容纳更少的活跃 Warp）
- 但每 Thread 计算量增大（64 个输出 vs V8 的 1 个）
- 权衡结果：V9 通常比 V8 快（计算密度提升超过 Occupancy 下降的影响）

---

## 5. L1/L2 Cache

### L1 Cache

- 与 Shared Memory 共享物理 SRAM（可配置比例）
- 默认配置：约 32KB L1 + 16KB Shared，或 48KB Shared
- 透明缓存：Global Memory 的访问自动经过 L1（只读数据效果更好）
- 使用 `__ldg`（Load through texture cache）可获得只读 Cache 优化：
  ```cpp
  float val = __ldg(&A[idx]);  // 通过只读 Cache 加载，适合不会被写的数据
  ```

### L2 Cache

- GPU 级别共享（所有 SM 共享）
- 容量约 2-40MB（不同 GPU 差异很大）
- 延迟约 150 cycles（比 L1 慢，比 DRAM 快）

---

## 6. 内存访问模式总结

### 常见问题和对策

| 问题 | 症状（ncu 指标）| 解决方案 |
|------|----------------|----------|
| 非合并访问 | dram__bytes >> 理论值 | 改变访问模式（如转置矩阵）|
| Bank Conflict | smsp__pct < 100% | 调整 Shared Memory 布局（加 padding）|
| 低 Occupancy | launch__occupancy 低 | 减少寄存器或 smem 用量，增大 Block |
| 寄存器溢出 | 程序突然变慢 | 减少 TM/TN，或用 __launch_bounds__ 限制寄存器 |

### 用 Nsight Compute 分析

```bash
# 查看显存带宽利用率
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum \
              dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./benchmark_cuda --size 2048 --version 8

# 查看 Shared Memory Bank Conflict
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
    ./benchmark_cuda --size 2048 --version 8

# 查看 Occupancy
ncu --metrics launch__occupancy ./benchmark_cuda --size 2048 --version 9
```

---

## 7. CUDA 执行模型补充

### Warp 和 SIMT

- GPU 以 **Warp**（32 个 Thread）为调度单位
- SIMT（Single Instruction Multiple Thread）：Warp 内 32 个 Thread 执行相同指令
- 区别于 SIMD：SIMT 的 Thread 可以有各自的寄存器和程序计数器（理论上可以分叉）

### Warp Divergence

```cpp
// 坏的代码（Warp Divergence）：
if (threadIdx.x % 2 == 0) {
    // 偶数 Thread 走这里
} else {
    // 奇数 Thread 走这里
}
// Warp 内的 32 个 Thread 有 16 个走 if，16 个走 else
// GPU 需要执行两次：先执行偶数 Thread 的路径（奇数 Thread 被屏蔽），
// 再执行奇数 Thread 的路径（偶数 Thread 被屏蔽）
// 总时间 ≈ 2x（最坏情况）

// GEMM 中的边界检查会引入 Divergence：
if (row < M && col < N) { ... }
// 在矩阵边缘的 Block，部分 Thread 满足条件，部分不满足 → Divergence
// 但这只发生在边界 Block（M 和 N 的最后一行/列的 Block），影响较小
```

---

## 8. 面试话术

> "CUDA 的内存层次与 CPU 类似但有所不同。
> Global Memory 容量最大但延迟最高（~800 cycles），
> Shared Memory 是片上 SRAM，延迟只有 ~20 cycles 但每 Block 只有 48KB。
> 关键优化原则有两条：
> 第一，确保 Global Memory 的合并访问（Coalescing），
> 让同一 Warp 的 32 个 Thread 访问连续的内存地址，
> 将 32 次独立事务合并为 1-4 次，提升带宽利用率；
> 第二，用 Shared Memory 缓存高频访问的数据，
> 通过 __syncthreads() 协调 Block 内的协同加载，
> 将 Global Memory 访问次数减少 TILE_SIZE 倍。"
