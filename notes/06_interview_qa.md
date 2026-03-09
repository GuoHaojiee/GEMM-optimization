# 06 面试题库（20+题）

---

## Cache 原理（3 题）

### Q1: 为什么 ikj 循环顺序比 ijk 快？

**回答：**

在 ijk 顺序中，最内层循环 k 变化，此时访问 B[k][j]：
- j 固定，k 递增，访问 B 矩阵的一列
- 相邻元素的内存地址差 N×4 字节（stride=N）
- 当 N=1024，步长=4KB，远超 Cache line（64 字节）
- 每次访问必然加载新的 Cache line → Cache miss 率接近 100%

改为 ikj 顺序后，最内层循环 j 变化，此时访问 B[k][j]：
- k 固定，j 递增，访问 B 矩阵的一行
- 相邻元素的内存地址差 4 字节（stride=1）
- 每 16 个 float 只需加载 1 个 Cache line，命中率 15/16 ≈ 94%
- 同时，A[i][k] 提取为寄存器变量（循环不变量），消除重复内存访问

**量化效果**：对 N=1024 的矩阵，典型加速 5-10 倍。

**核心关键词：** Cache line（64字节）, spatial locality（空间局部性）, stride-1 访问, row-major 存储

**延伸追问：**
- 还有其他 6 种循环顺序（kij, kji, jki, jik），哪种最好？→ ikj 和 kij 都是最优的（最内层 j 变化，C 和 B 都是行访问）
- Cache line 大小为什么是 64 字节？→ 权衡延迟（太大浪费带宽）和局部性（太小无法复用），64 字节是工业实践中的最优选择

---

### Q2: Cache 的 Set-Associative 结构是什么？Conflict Miss 怎么发生的？

**回答：**

Cache 被分为多个 Set，每个 Set 包含 N 个 Way（N-way 组相联）。
内存地址通过 hash 映射到某个 Set，该 Set 内的 N 个 Way 可以缓存不同的内存块。

**Conflict Miss 发生**：
当两个频繁访问的内存地址映射到同一个 Set 时，若 Set 已满，新数据必须驱逐旧数据。
若这两个地址交替访问，就会发生 Cache thrashing（反复互相驱逐）。

**GEMM 中的实例**：
如果矩阵的行步长（N×sizeof(float)）是 L1 Cache 大小的整数倍，
A 矩阵和 B 矩阵的某些行可能映射到相同的 Set，产生 Conflict Miss。

**解决方案**：
1. 分块（Tiling）：减少每次加载的数据量，降低 Set 竞争
2. Padding：给矩阵行添加几个字节的填充，使步长错开，避免地址对齐冲突

**核心关键词：** Set-Associative, LRU 驱逐策略, Conflict Miss, Thrashing, Padding

**延伸追问：**
- 为什么不用 Fully Associative Cache？→ 比较器成本随 Way 数指数增长，N=1024 时硬件代价过高
- Direct-Mapped Cache（1-way）的优缺点？→ 最快（无需比较）但 Conflict Miss 最多

---

### Q3: 什么是 False Sharing？如何在 GEMM 的多线程版本中避免？

**回答：**

False Sharing：两个 CPU Core 修改的是同一 Cache line 中的**不同变量**（不是同一变量），但 Cache 一致性协议（MESI）仍然把它当作冲突处理，导致 Cache line 反复在 Core 之间"弹跳"。

**具体机制**（MESI 协议）：
- Core 0 修改变量 A（位于 Cache line X）→ Cache line X 变为 Modified 状态
- Core 1 修改变量 B（也在 Cache line X）→ Core 0 的 Cache line X 被 Invalidated
- Core 0 下次读 A 时，需要重新从 L3/内存加载 → 虽然 A 根本没被 Core 1 修改！

**在 GEMM V6 中的分析**：
- V6 按 i 块并行，不同线程写 C 的不同行
- 当行大小 N×sizeof(float) ≥ 64 字节（N≥16 时），不同行不在同一 Cache line
- → 基本无 False Sharing（N 通常远大于 16）
- 需要注意：若并行化 j 块，不同线程写同一行的不同 j 段，可能发生 False Sharing

**核心关键词：** Cache line（64 字节）, MESI 协议, Cache Invalidation, 伪共享

**延伸追问：**
- 如何验证是否有 False Sharing？→ 用 perf c2c 工具，查看 cache-to-cache transfer 次数
- 如何在数据结构设计中避免 False Sharing？→ 每个线程的独立变量用 `alignas(64)` 对齐到 Cache line

---

## SIMD 向量化（3 题）

### Q4: FMA（Fused Multiply-Add）与 mul+add 的区别是什么？

**回答：**

**指令数**：FMA 一条指令完成 a×b+c，比 mul+add 少一条指令，减少指令发射压力。

**精度更高**：
- `mul + add`：先执行 `a*b`，中间结果被截断为 float 精度，再与 c 相加
- `fmadd`：中间结果保持更高精度（硬件内部以 80 位或 double 精度计算），最后才截断

**性能等价**：在 Haswell+ CPU 上，FMA 的延迟与 mul 相同（约 4-5 cycles），
但每个时钟周期可以发射 2 条 FMA，即 2×8=16 GFLOPS/GHz（AVX2）。

**编译要求**：必须用 `-mfma` 编译选项才能生成 `vfmadd` 系列指令。

**核心关键词：** Fused Multiply-Add, 指令级并行, 浮点精度, 延迟/吞吐

**延伸追问：**
- AVX2 一个时钟周期最多做几个 GFLOPS？→ 2条FMA × 8float × 2FLOPs = 每周期32 FLOPs，3.5GHz CPU = 3.5×32=112 GFLOPS（单核峰值）
- 为什么 FMA 的精度比 mul+add 高？→ 中间乘积保持完整精度不截断，避免了两次舍入误差

---

### Q5: 对齐加载（_mm256_load_ps）和非对齐加载（_mm256_loadu_ps）的区别是什么？

**回答：**

**硬件要求**：
- `_mm256_load_ps`（对齐）：要求内存地址是 32 字节对齐（地址 % 32 == 0）
- `_mm256_loadu_ps`（非对齐）：无对齐要求

**性能差异（现代 CPU）**：
- Sandy Bridge 时代：非对齐加载比对齐慢约 1 cycle
- Haswell（2013）以后：几乎无差异（硬件自动处理非对齐）
- 例外：跨越 Cache line 边界（cross cache line boundary）的加载有微小惩罚

**使用建议**：
1. 大多数场景：直接用 `loadu`，简单安全
2. 追求极致性能：用 `aligned_alloc(32, size)` 分配 32 字节对齐内存，再用 `load`
3. 调试辅助：使用 `load` 时若地址未对齐会触发 SIGBUS，帮助发现对齐问题

**核心关键词：** 32 字节对齐（256 位 = 32 字节）, aligned_alloc, Cross Cache Line, SIGBUS

**延伸追问：**
- 什么时候内存会自然对齐？→ 全局/静态数组默认对齐；`malloc` 通常 16 字节对齐；`posix_memalign(32, ...)` 可指定
- AVX-512 的对齐要求是多少？→ 64 字节对齐

---

### Q6: 为什么有时候编译器的自动向量化不如手写 SIMD 快？

**回答：**

编译器自动向量化的局限：

1. **Aliasing 保守**：若两个指针可能指向同一内存（aliasing），编译器不敢向量化。
   解决：添加 `__restrict__` 告知编译器无 aliasing。

2. **未知的运行时值**：循环边界、步长等在编译时未知，编译器生成 scalar fallback 路径。
   手写时我们知道步长恒为 1，直接写向量化代码。

3. **指令集保守**：`-O3` 不一定开启 AVX2，需要 `-march=native` 或 `-mavx2`。

4. **尾部处理**：编译器可能生成额外的 scalar 代码来处理边界情况，影响热路径。

5. **内联失败**：若函数被分开编译，编译器无法跨函数向量化。

**实测**（GEMM V2 vs V4）：
- g++ -O3（不加 march=native）：通常只生成 SSE 128 位指令（4 float）
- 手写 AVX2：确保 256 位（8 float）+ FMA，额外提速 1.5-2x

**核心关键词：** __restrict__, -march=native, Auto-vectorization, Scalar Fallback

**延伸追问：**
- 如何验证代码是否被向量化？→ godbolt.org 查看汇编，找 ymm 寄存器或 vmovups
- -ffast-math 对向量化的影响？→ 允许编译器重排浮点运算（不满足结合律），可以向量化更多代码，但可能改变数值结果

---

## CUDA 内存层次（3 题）

### Q7: 什么是 Memory Coalescing？如何在 GEMM 中实现？

**回答：**

**定义**：CUDA 中，当同一 Warp（32个 Thread）的所有 Thread 访问连续的 Global Memory 地址时，硬件将这 32 次访问合并（coalesce）为少数几次内存事务（通常 1-4 次 128 字节的事务）。

**条件**：Thread i 访问地址 base + i（连续访问），且起始地址是 128 字节对齐。

**V7 Naive 的访问分析**：
- 同一 Warp 的 Thread（tx=0..31，ty 相同）
- 访问 B[k][tx]：B 的同一行的 32 个连续元素 → **合并** ✓
- 访问 A[ty][k]：ty 不同（不同行），stride=K → 实际上同一 Warp 内 ty 相同... 分析需仔细

**V8 Shared Memory 中如何实现 Coalescing**：
- 协同加载 A 块：Thread(tx,ty) 加载 A[block_row+ty][tile*T+tx]，同一 Warp 的 Thread 加载 A 的同一行的连续元素 → **合并** ✓
- 协同加载 B 块：同理，行访问 → **合并** ✓
- 这正是 V8 比 V7 快的关键原因之一

**核心关键词：** Warp（32 Thread）, 128 字节内存事务, Aligned Access, Transaction

**延伸追问：**
- 非合并访问最坏情况下慢多少？→ 32 次独立事务 vs 合并的 1-4 次，最坏慢 8-32 倍
- 如何判断是否合并访问？→ Nsight Compute 的 `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` 指标，值越小越合并

---

### Q8: 什么是 CUDA Shared Memory 的 Bank Conflict？如何避免？

**回答：**

**Bank 结构**：Shared Memory 被划分为 32 个 Bank，Bank i 存储地址 4×(32n+i) 的 4 字节数据。
一次访问可以同时服务 32 个不同 Bank（一个 Warp 的 32 个 Thread）。

**Bank Conflict**：同一 Warp 的多个 Thread 访问**同一 Bank 的不同地址**时，这些访问被串行化。

**发生条件**：Thread i 访问地址 A[i × stride]，当 stride % 32 != 0 且 stride != 0 时可能发生冲突（stride 是 Bank 数 32 的因子时最严重）。

**示例**：
```cpp
// 无 Bank Conflict（连续访问，每个 Bank 被 1 个 Thread 访问）
shared[threadIdx.x]

// 有 Bank Conflict（stride=2，偶数 Thread 都访问偶数 Bank，奇数 Thread 访问奇数 Bank）
shared[threadIdx.x * 2]  // stride=2，同一 Bank 被 2 个 Thread 访问 → 2-way conflict

// 最坏情况（stride=32，所有 Thread 访问同一 Bank）
shared[threadIdx.x * 32]  // 32-way conflict，性能下降 32 倍！
```

**避免方法**：
1. 调整数据布局，使访问 stride 不是 Bank 数的因子
2. **Padding**：给 Shared Memory 数组加额外一列
   ```cpp
   __shared__ float A[BK][BM + 1];  // +1 是关键，打破对齐冲突
   ```
3. 改变 Thread 访问的索引公式

**核心关键词：** 32 个 Bank，4 字节粒度，串行化，Padding

**延伸追问：**
- 如何用 Nsight Compute 检测 Bank Conflict？→ `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` 指标
- 广播（多个 Thread 访问同一 Bank 的同一地址）是否算 Conflict？→ 不算！广播由硬件高效处理，不串行化

---

### Q9: Global Memory、Shared Memory、Register 各适合存什么数据？

**回答：**

| 内存类型 | 延迟 | 容量 | 作用域 | 最适合存放 |
|---------|------|------|--------|-----------|
| Global Memory | ~800 cycles | 几十 GB | 全局 | 输入矩阵 A、B，输出矩阵 C（整体） |
| Shared Memory | ~20 cycles | 48KB/Block | Block 内共享 | 被 Block 内多 Thread 重复访问的数据块（如 GEMM 的 A/B 分块） |
| Register | 1 cycle | ~256 个/Thread | 每个 Thread | 循环变量、累加器、频繁使用的临时值 |
| L1 Cache | ~20 cycles | 32KB | 自动管理 | 只读数据（用 __ldg 或 const __restrict__ 提示 GPU 使用只读 Cache）|

**GEMM 中的具体运用（V9）**：
- A、B 全矩阵：Global Memory
- A/B 的分块（BM×BK 和 BK×BN）：Shared Memory（被 Block 内 64 个 Thread 共享）
- accum[TM][TN]（每 Thread 的 C 子矩阵累加器）：Register（64 个浮点寄存器）
- regA[TM]、regB[TN]（每次外积的临时数据）：Register

**核心关键词：** 数据局部性, 数据复用, 访问粒度匹配

**延伸追问：**
- Local Memory 是什么？→ 寄存器溢出时，数据被放到 Local Memory（实质是 Global Memory 的特定区域，很慢）
- 如何知道是否发生了 Register Spilling？→ nvcc --ptxas-options=-v，查看 "lmem" 条目（local memory 用量）

---

## CUDA 执行模型（3 题）

### Q10: Warp 是什么？SIMT 与 SIMD 的区别是什么？

**回答：**

**Warp**：GPU 调度的基本单位，每个 Warp 由 32 个 Thread 组成，它们同时执行相同的指令。

**SIMT（Single Instruction Multiple Thread）**：
- 32 个 Thread 执行相同指令，但每个 Thread 有自己的寄存器和程序计数器
- 程序员以单 Thread 视角编程，GPU 自动并行化
- 支持条件分支（但产生 Warp Divergence）

**SIMD（Single Instruction Multiple Data）**：
- 程序员显式操作向量寄存器（如 `__m256`）
- 必须手动处理不同数据的不同操作（masking）
- 性能可预测性更好，但编程更复杂

**关键区别**：
- SIMT 对程序员透明（抽象层次更高）
- SIMD 对程序员显式（控制更精细）
- 现代 GPU 本质上是 SIMD 硬件，SIMT 是软件抽象层

**核心关键词：** Warp（32 Thread）, SIMT, 程序计数器, 调度粒度

**延伸追问：**
- 同一 Block 内的 Thread 之间可以通信吗？→ 可以，通过 Shared Memory + __syncthreads()
- 不同 Block 之间呢？→ 不能直接通信，需要写回 Global Memory 后由另一个 Kernel 读取

---

### Q11: 什么是 Warp Divergence？在 GEMM 中如何分析其影响？

**回答：**

**定义**：Warp 内的不同 Thread 因条件分支走不同路径，GPU 需要串行执行两个路径（用掩码屏蔽不参与的 Thread），总耗时 = 两个路径的耗时之和（最坏情况）。

**机制**：
```cpp
if (threadIdx.x < 16) {
    // 前 16 个 Thread 走这里（后 16 个被屏蔽）
} else {
    // 后 16 个 Thread 走这里（前 16 个被屏蔽）
}
// 总耗时 ≈ 2x 无分支的时间
```

**GEMM 中的 Divergence 来源**：
边界检查：
```cpp
if (row < M && col < N) { ... }
```
对于位于矩阵边角的 Block（M 或 N 不是 TILE_SIZE 的倍数），Block 内部分 Thread 满足条件，部分不满足，产生 Divergence。

**影响评估**：
- 大多数 Block 在矩阵内部，无 Divergence
- 只有边界 Block 有 Divergence（M/TILE × N/TILE 个 Block 中，约 2 × (M/TILE + N/TILE) 个是边界 Block）
- 对于 M=N=2048，TILE=16：约 256 个 Block 中只有 ~32 个边界 Block → 影响约 12%

**优化方向**：确保 M、N 是 TILE_SIZE 的倍数（Padding），完全消除边界 Divergence。

**核心关键词：** 分支谓词（Branch Predication），掩码（Masking），性能开销，边界处理

---

### Q12: 什么是 Occupancy？高 Occupancy 一定带来高性能吗？

**回答：**

**定义**：Occupancy = 每个 SM 实际活跃的 Warp 数 / SM 最大支持的 Warp 数（通常 64 Warp）。

**为什么 Occupancy 重要**：
当一个 Warp 等待内存访问时（延迟 ~400 cycles），SM 会切换到其他 Warp 执行。
高 Occupancy 意味着有更多备用 Warp 可以用来隐藏延迟（Latency Hiding）。

**影响 Occupancy 的因素**：
- 寄存器用量（最主要）：每 Thread 32 寄存器 → Occupancy 100%，64 寄存器 → 50%
- Shared Memory 用量：每 Block 16KB → Occupancy 受 Shared Memory 上限制约
- Block 大小：Block 太小 → SM 内活跃 Block 数多但每个 Block 的 Warp 少

**高 Occupancy 不一定高性能的情况**：

1. **Compute Bound 时**：SM 已经满负荷计算，增加 Warp 没有额外计算资源，Occupancy 不重要

2. **Shared Memory / L1 Cache 反而更重要**：减少寄存器用量提高 Occupancy，但可能导致寄存器溢出（更慢）

3. **Register Tiling（V9）的权衡**：每 Thread 64 个累加器寄存器 → Occupancy 降低，但计算密度大幅提升 → 整体性能更好

**规律**：对于 Memory Bound 的 Kernel，高 Occupancy 有助于隐藏延迟；对于 Compute Bound 的 Kernel，高 Occupancy 不再有效。

**核心关键词：** Latency Hiding, SM, Register 预算, 配置空间

---

## Register Tiling（3 题）

### Q13: 为什么 Register Tiling 比 Shared Memory Tiling 快？

**回答：**

Register Tiling 通过让每个 Thread 计算更多输出元素，提高了对 Shared Memory 数据的复用比，从而降低了 Shared Memory 带宽需求。

**量化对比**（以 TM=TN=8 为例）：

| 指标 | V8 Shared | V9 Register |
|------|-----------|-------------|
| 每 Thread 计算输出元素数 | 1 | 64（TM×TN） |
| 每次外积的 Smem 读取 | 2T float | TM+TN=16 float |
| 每次 Smem 读取的 FMA 数 | 1 | 4（64/16） |
| AI 相对 Smem（FLOP/Byte）| 0.125 | 1.0 |

更高的 AI 意味着：相同的 Shared Memory 带宽，V9 能完成 8 倍更多的计算。

此外，V9 减少了 __syncthreads() 的相对开销：
- V8：每 T=16 次 FMA 一次 __syncthreads()
- V9：每 BK=8 次外积（即 64×8=512 次 FMA）一次 __syncthreads()

**核心关键词：** 数据复用比, 算术强度（相对 Shared Memory）, 同步开销

**延伸追问：**
- TM=TN=4 和 TM=TN=8 哪个更好？→ 取决于寄存器压力，通常 8 更好，但需要实测
- 还有什么可以进一步优化？→ Double Buffering（加载和计算 overlap），Tensor Core 利用

---

### Q14: 外积展开（Outer Product）的具体计算流程是什么？

**回答：**

外积是 Register Tiling 的核心操作：

```
对每个 K 方向的元素 k（0..BK-1）：
  1. 从 Shared Memory As 加载 A 的列切片：regA[m] = As[k][thread_row + m]  (m=0..TM-1)
  2. 从 Shared Memory Bs 加载 B 的行切片：regB[n] = Bs[k][thread_col + n]  (n=0..TN-1)
  3. 外积更新：accum[m][n] += regA[m] * regB[n]  (所有 m, n 的组合)

以 TM=TN=2 为例（简化）：
  regA = [a0, a1]（从 As[k][0..1] 加载）
  regB = [b0, b1]（从 Bs[k][0..1] 加载）

  外积：
    accum[0][0] += a0 * b0
    accum[0][1] += a0 * b1  ← a0 被复用！
    accum[1][0] += a1 * b0  ← b0 被复用！
    accum[1][1] += a1 * b1

  4 次 FMA，读取 4 个 float，算术强度 = 4 / (4×4) = 0.25 FLOP/Byte
  （TM=TN=8：64 次 FMA，读取 16 float，AI = 1 FLOP/Byte）
```

**为什么叫"外积"**：
- regA 是一个列向量（TM 个元素）
- regB 是一个行向量（TN 个元素）
- 两者的"外积"（向量张量积）= TM×TN 矩阵
- 这正是 accum 矩阵在一次 k 迭代中的更新量

**核心关键词：** 向量外积, 寄存器复用, TM×TN FMA

---

### Q15: 如何决定 Register Tiling 的 BM/BN/BK/TM/TN 参数？

**回答：**

**约束条件**：

1. **Shared Memory 上限**（每 Block 约 48KB）：
   `As[BK][BM] + Bs[BK][BN] ≤ 48KB`
   `(BK×BM + BK×BN) × 4 ≤ 48×1024`
   BM=BN=64, BK=8：(8×64 + 8×64) × 4 = 4KB ✓（远小于上限）

2. **寄存器上限**（每 Thread 约 255 个）：
   `accum[TM][TN] + regA[TM] + regB[TN] + 其他 ≤ 255`
   TM=TN=8：64 + 8 + 8 + 约 20 = ~100 寄存器 ✓

3. **Block 大小限制**（最多 1024 Thread）：
   `(BM/TM) × (BN/TN) ≤ 1024`
   64/8 × 64/8 = 8×8 = 64 ✓

**优化目标**：
- 最大化算术强度：AI = TM×TN×BK×2 / ((BM+BN)×BK×4) = TM×TN / ((BM/BN_unit+1)×4)
  → 增大 TM, TN（但受寄存器限制）
- 维持足够的 Occupancy（用于隐藏 Global Memory 延迟）
  → 不能让寄存器用量过多（降低 Occupancy）

**实践经验**：
- 常用配置：BM=BN=64, BK=8, TM=TN=8 → 良好的均衡
- 更激进：BM=BN=128, TM=TN=8 → 更好的 AI，但需要更多 Shared Memory
- 进阶：BM=BN=128, TM=TN=4 → 适合寄存器压力较大时

**核心关键词：** 约束优化, 寄存器预算, Shared Memory 预算, Occupancy 均衡

---

## Roofline 模型（2 题）

### Q16: 如何用 Roofline 模型判断一个 GEMM 版本的优化方向？

**回答：**

步骤：
1. 计算该版本的**理论算术强度** AI（FLOP/Byte）
2. 查看实测 GFLOPS
3. 计算 Roofline 上界 = min(峰值算力, AI × 峰值带宽)
4. 比较实测 vs 上界，判断瓶颈和效率

**以 V1 Naive 为例**：
- AI ≈ 0.25 FLOP/Byte
- 峰值带宽 = 50 GB/s，带宽上界 = 0.25 × 50 = 12.5 GFLOPS
- 峰值算力 = 100 GFLOPS
- Roofline = min(100, 12.5) = 12.5 GFLOPS → **Memory Bound**
- 若实测 2 GFLOPS → 只达到上界的 16%
- **优化方向**：提高 AI（分块，增加数据复用）

**以 V3 Tiling 为例**：
- AI ≈ 16 FLOP/Byte
- 带宽上界 = 16 × 50 = 800 GFLOPS > 峰值算力 100 GFLOPS
- Roofline = 100 GFLOPS → **Compute Bound**
- 若实测 60 GFLOPS → 达到上界的 60%
- **优化方向**：加 SIMD（V5），提高每周期 FLOPs；提升指令发射效率

**核心结论**：
- Memory Bound → 增加算术强度（分块、Shared Memory）
- Compute Bound → 增加 SIMD 宽度（AVX2、Tensor Core）
- 既不在 Memory 线也不在 Compute 线 → 有其他瓶颈（如 Branch Divergence、同步开销）

**核心关键词：** 算术强度, 拐点（Ridge Point）, Memory/Compute Bound 判断

---

### Q17: 算术强度的计算中，"内存访问量"应该怎么定义？

**回答：**

**正确定义**：算术强度中的 Memory Traffic = 实际从 DRAM（或目标存储层）读/写的字节数（而非代码中的 load/store 次数，因为 Cache 会拦截部分访问）。

**最优情况（每个元素只读写一次）**：
```
AI_optimal = 2MNK / [(MK + KN + MN) × sizeof(float)]
（方阵 n×n：AI = 2n³ / (3n² × 4) = n/6 FLOP/Byte）
n=1024 → AI=171（很高！）→ 实际达不到，因为 Cache 不够大
```

**实际情况（考虑 Cache 效果）**：
- V1（无任何复用）：每个矩阵元素从 DRAM 读取约 N 次 → AI ≈ 0.25 FLOP/Byte
- V3（L1 Cache 足够容纳 3 个 Block）：每个元素只从 DRAM 读一次 → AI ≈ T/4 FLOP/Byte
- V8 CUDA（Shared Memory 可容纳 2 个 T×T 块）：AI ≈ T/4 FLOP/Byte

**关键**：AI 依赖于使用的存储层次：
- 相对 DRAM：V3 和 V8 的 AI 类似（分块后都约 T/4）
- 相对 Shared Memory：V9 的 AI 更高（TM×TN/(TM+TN)/4 = 1 FLOP/Byte）

**核心关键词：** DRAM Traffic, Cache 效果, 有效算术强度

---

## 通用方法论（3 题）

### Q18: Amdahl's Law（阿姆达尔定律）如何指导 GEMM 优化？

**回答：**

**Amdahl's Law**：若程序中可并行化的比例为 p，使用 n 个并行单元，最大加速比为：
```
S = 1 / [(1-p) + p/n]
```

当 n→∞，最大加速比 = 1/(1-p)，即串行部分决定了性能上界。

**在 GEMM 优化中的应用**：

1. **CPU 多线程（V6）**：
   - GEMM 的计算部分（矩阵乘法）完全可并行化（p ≈ 1）
   - 非并行部分：内存分配、初始化、同步开销
   - 实际效率：线程数翻倍，性能提升通常约 1.7-1.9x（不是 2x）
   - 原因：内存带宽成为新的瓶颈（共享带宽）

2. **优化优先级**：
   - 应先优化热点路径（占总时间 90%+ 的计算核心）
   - 如果三重循环占 99% 的时间，优化内存分配（占 1%）不值得花大力气

3. **指导意义**：
   - GEMM 非常适合并行化（几乎 100% 的计算是独立的）
   - 但带宽是"串行瓶颈"——多个核心共享带宽，实际扩展性有限

**核心关键词：** 并行化比例, 串行瓶颈, 加速比上界, 带宽共享

**延伸追问：**
- Gustafson's Law（古斯塔夫森定律）与 Amdahl 的区别？→ Amdahl 分析固定问题规模的加速比，Gustafson 分析固定时间内可解决的问题规模（更适合实际高性能计算）

---

### Q19: 如何系统地对一个程序进行性能优化？（优化方法论）

**回答：**

**原则：先测量，再优化（Measure before Optimize）**

**系统化步骤**：

1. **建立基准（Baseline）**：
   - 实现最简单的版本（如 V1 Naive）
   - 测量准确的基准性能（避免热身偏差，用中位数）

2. **定位瓶颈（Profiling）**：
   - CPU：`perf stat` 查看 Cache miss、IPC
   - CUDA：Nsight Compute 查看 SM 利用率、带宽、Occupancy
   - **不能靠猜，猜错了优化效果为零**

3. **理论分析**：
   - 用 Roofline 模型判断是 Memory Bound 还是 Compute Bound
   - 计算算术强度，确定瓶颈类型

4. **针对瓶颈优化**：
   - Memory Bound → 提高数据复用（Tiling、Shared Memory）
   - Compute Bound → 提高指令级并行（SIMD、FMA、展开）
   - 带宽受限 → 减少冗余访问，提高 Coalescing

5. **验证效果**：
   - 每次优化后重新 benchmark，确认有效
   - 同时验证正确性（浮点误差）

6. **迭代**：
   - 每次解决一个瓶颈，直到下一个瓶颈出现

**本项目的优化路径**：
V1（基准）→ V2（访存模式）→ V3（Cache 复用）→ V4（SIMD）→ V5（组合）→ V6（多线程）
→ V7（GPU 基准）→ V8（Shared Memory）→ V9（Register Tiling）→ V10（向量化加载）

**核心关键词：** 先测量, Bottleneck Driven, Roofline 模型, Profiling Tools

---

### Q20: 这个项目中，你最重要的学习和发现是什么？

**回答（示例）**：

"这个项目最重要的收获是**定量分析**的重要性，而不是凭直觉猜测瓶颈。

具体例子：在做 V3（分块）之前，我用 perf 测量了 V2 的 LLC miss 率，发现大矩阵（N=2048）时 LLC miss 率高达 30%，这证明数据集超过了 L3 Cache。这才让我有信心加入分块优化，而不是盲目尝试。

另一个发现：Roofline 模型预测 V3 之后算力才是瓶颈（AI 从 0.25 提升到 16 FLOP/Byte，超过了 CPU 的拐点约 2 FLOP/Byte），这告诉我在 V3 之前加 AVX2（V4）是低效的，应该先分块再向量化——这与 V5 同时做两件事的设计一致。

在 CUDA 方向，Register Tiling（V9）让我理解了一个关键概念：即使是 Shared Memory 也可以成为瓶颈（对 V8 来说），真正的优化是让计算密度永远高于任何一级存储层次的带宽。"

**核心关键词：** 数据驱动优化, 定量分析, Roofline, 优化的层次性

---

### Q21: 如何解释 benchmark 中的性能抖动（Jitter）？

**回答：**

**常见原因**：

1. **CPU 频率动态调整（Turbo Boost）**：
   - 初次运行时 CPU 从节能频率（1-2 GHz）提升到 Turbo 频率（4-5 GHz），需要 100ms+
   - 解决：用 3-5 次 warm-up（本项目就是这样做的）

2. **OS 调度干扰**：
   - 其他进程偶尔抢占 CPU 时间片，导致偶发性延迟
   - 解决：取多次运行的**中位数**（而非均值），排除异常值

3. **DRAM 刷新延迟（Refresh Stall）**：
   - DDR4 内存约每 64ms 刷新一次，刷新期间（约 1-4μs）无法访问
   - 解决：多次测量，统计分析

4. **Cache 状态不一致**：
   - 首次运行 Cache 是"冷"的（Cold Cache），后续运行是"热"的
   - 解决：warm-up 后正式测量，确保 Cache 处于稳定状态

**本项目的处理方式**：3 次 warm-up + 5 次正式测量 + 取中位数，可以有效消除大部分抖动。

**核心关键词：** Turbo Boost, Cache Warm-up, 中位数 vs 均值, 置信区间

---

### Q22: 如果矩阵 M、N、K 大小不同（非方阵），需要注意什么？

**回答：**

本项目使用统一签名 `(A, B, C, M, N, K)`，A 是 M×K，B 是 K×N，C 是 M×N。

**边界处理**（最常见问题）：
- 分块后，最后一个块的大小可能小于 BLOCK_SIZE
- 必须用 `std::min(i + BLOCK_SIZE, M)` 计算实际块大小，不能盲目用 BLOCK_SIZE
- CUDA 中需要边界检查：`if (row < M && col < N)`

**性能影响**：
- 非方阵时，某一维度很小（如 M=16, K=N=1024）会导致 Block/Thread 利用率低
- CUDA 中 Grid 大小 = ceil(M/BM) × ceil(N/BN)，M 很小时 Grid 行数少，SM 利用率低
- 解决：对非方阵使用不同的 Block 大小或特殊的 kernel

**cuBLAS 的处理**：
- cuBLAS 对各种矩阵形状都有优化，内部会根据 M/N/K 的大小选择最优 Kernel
- 我们的实现对非方阵的效率通常比 cuBLAS 差更多

**核心关键词：** 边界检查, std::min, Grid/Block 利用率, 非方阵优化
