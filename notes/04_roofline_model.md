# 04 Roofline 模型

## 1. 什么是 Roofline 模型

Roofline 模型（屋顶线模型）是一个直观的性能分析框架，用于判断程序的性能瓶颈：

**核心公式**：

```
实际可达性能（GFLOPS）≤ min(峰值算力, 算术强度 × 峰值带宽)
                            ────────────────────────────────
                            这条线就是"屋顶"
```

其中：
- **算术强度（Arithmetic Intensity, AI）**：程序执行的浮点操作数 / 内存访问字节数（FLOP/Byte）
- **峰值算力**：CPU/GPU 的最高浮点运算速率（GFLOPS/TFLOPS）
- **峰值带宽**：内存最高读写速率（GB/s）

---

## 2. Roofline 图解读

```
GFLOPS（对数轴）
     │
峰值算力 ─────────────────────────────────────────────── 算力天花板
     │                                              /
     │                                           /
     │                                        /
     │                                     /
     │                                  /  Memory Bound 区域
     │           Compute Bound 区域  /
     │                             /
     │                          /
     │                       /
     │                    /
     └──────────────────┼──────────────────────── AI（FLOP/Byte，对数轴）
                      拐点
                  AI = 峰值算力 / 峰值带宽
```

**解读规则**：
- 如果程序的 `(AI, GFLOPS)` 点落在**斜线段上**→ Memory Bound（带宽限制）
- 如果程序的 `(AI, GFLOPS)` 点落在**水平段上**→ Compute Bound（算力限制）
- 点距离屋顶线越近，优化越好；距离越远，还有提升空间

---

## 3. 各 GEMM 版本的算术强度推导

### 通用公式

GEMM（M×K 乘以 K×N）的浮点操作数：
```
FLOPs = 2 × M × N × K
（每个 C[i][j] 需要 K 次乘法 + K 次加法）
```

### V1/V2 — 无数据复用

**假设**：无 Cache 复用，每次访问 A/B 都从 DRAM 加载。

```
访问 A：每个 C[i][j] 需要读 A 的第 i 行（K 个 float）
         C 有 M×N 个元素，但 A 的每行被读 N 次（计算 C 的第 i 行的所有元素）
         实际：A 的每个元素被读 N 次 → 总读取量 = M × K × N × 4 字节

访问 B：每个 C[i][j] 需要读 B 的第 j 列（K 个 float）
         B 的每列被读 M 次 → 总读取量 = K × N × M × 4 字节

访问 C：每个元素读写各 K 次（累加 K 次） → 2 × M × N × K × 4 字节

总 Memory Traffic ≈ (MKN + KNM + 2MNK) × 4 = 4MNK × 4 字节

AI = 2MNK / (16MNK) = 1/8 FLOP/Byte ≈ 0.125 FLOP/Byte
```

实际测量中，因为 L1 Cache 会捕捉部分访问，有效 AI 约 0.25 FLOP/Byte。

**结论**：V1/V2 严重 Memory Bound。

### V3/V5 — 分块后（Block Size=T）

**假设**：三个 T×T 的块都驻留在 Cache 中，每个元素只从 DRAM 读取一次。

```
每个矩阵元素只加载一次：
  A 总读取量：M × K × 4 字节
  B 总读取量：K × N × 4 字节
  C 总读取量：M × N × 4 字节（最后写一次，忽略读操作）

总 Memory Traffic = (MK + KN + MN) × 4 字节

对于方阵（M=N=K=n）：
Total = 3n² × 4 = 12n² 字节
FLOPs = 2n³

AI = 2n³ / (12n²) = n/6 ≈ n/6 FLOP/Byte

当 n=1024：AI ≈ 170 FLOP/Byte（Compute Bound！）
```

但实际上，分块后有效 AI ≈ T/4 FLOP/Byte（T 为块大小），因为：
- 每次加载一个 T×T 的 A 块，这个块被 N/T 个 B 块复用 T 次
- 有效复用因子 = T
- AI ≈ 2T²×4 / (2T²×4/T × 4) ... （推导略，结果约 T/4）

T=64：AI ≈ 16 FLOP/Byte（比 V1 提升 64 倍！）

### V8 — CUDA Shared Memory（TILE_SIZE=T）

从 Global Memory 的视角：
```
每个 K-tile 将 T×T 的 A 块和 B 块加载到 Shared Memory
A 块大小：T × T × 4 字节，被 T 个 Thread（行方向）复用 T 次（不同 k）？

更简单的推导：
每个 A 元素从 Global Memory 读取 N/T 次（每个 K-tile）
         但由于 Shared Memory，Block 内 T 个 Thread 共享同一 A 块
         每个 A 元素被 Block 内 T 个 Thread 复用 → T 倍复用
总读取量 = M×K×N/(T×T) × T × 4 = M×K×N/T × 4 字节（简化）

AI ≈ 2MNK / (M×K×N/T × 4 × 2) = T/4 FLOP/Byte

T=16：AI ≈ 4 FLOP/Byte
```

### V9 — Register Tiling

在 V8 的基础上，寄存器进一步复用 Shared Memory 数据：
- 从 Shared Memory 读 TM+TN = 16 个 float
- 计算 TM×TN = 64 次 FMA
- AI（相对 Shared Memory）= 64 / (16×4) = 1 FLOP/Byte（已经很高）

从 Global Memory 视角（与 V8 类似，但 BK 更大）：
- 典型值 AI ≈ 8-16 FLOP/Byte（比 V8 高）

---

## 4. 如何读图和判断优化方向

### 看图示例

```
实测结果（CPU，峰值算力=100 GFLOPS，带宽=50 GB/s，拐点=2 FLOP/Byte）：

  AI         GFLOPS     位置分析
  ─────────────────────────────────────
  V1: 0.25   2          在斜线下方 → Memory Bound 且效率低（只达理论上界的16%）
  V2: 0.25   8          在斜线下方 → 仍然 Memory Bound（比 V1 好，但 AI 未变）
  V3: 16     45         在水平段下方 → Compute Bound（达到理论上界的45%）
  V5: 16     80         接近算力天花板 → 接近最优！
```

### 优化方向判断

1. **Memory Bound（点在斜线段，未达到斜线）**：
   - 提高算术强度（分块、使用 Shared Memory）
   - 改善内存访问模式（Coalescing、减少 stride 访问）

2. **Memory Bound（点在斜线上）**：
   - 已经充分利用带宽，需要提高 AI（分块等）

3. **Compute Bound（点在水平段，未达到）**：
   - 减少指令开销（循环展开）
   - 使用 SIMD/Tensor Core
   - 提高 ILP（指令级并行）

4. **已在屋顶线上**：
   - 已达最优！（此时需要换更快的硬件）

---

## 5. 实测 vs 理论的差距

实际性能通常低于理论 Roofline，原因：

1. **Cache 争用**：多个数据争用同一 Cache set → Conflict Miss
2. **流水线气泡**：数据依赖导致 stall
3. **地址计算开销**：乘法、加法指令占用执行单元
4. **TLB Miss**：虚拟地址翻译开销（大矩阵时）
5. **DRAM 非理想行为**：Bank conflict、刷新延迟等

典型效率（实测/理论）：
- V3 Tiling：30-60% 的 Roofline
- V5 AVX2+Tiling：50-80%
- cuBLAS：80-95%

---

## 6. 面试话术

> "Roofline 模型告诉我们程序的理论性能上界是算力和带宽的最小值。
> 通过计算每个版本的算术强度（FLOPs/Byte），
> 我可以判断优化方向：
> V1/V2 的算术强度约 0.25 FLOP/Byte，远低于拐点，是 Memory Bound，
> 需要提高数据复用；
> V3 分块后 AI 提升到约 16 FLOP/Byte，超过了拐点（约 2 FLOP/Byte），
> 成为 Compute Bound，这时加 SIMD（V5）才能有效提升性能。
> 如果仍在 Memory Bound 阶段就添加 SIMD，收益会很小。"
