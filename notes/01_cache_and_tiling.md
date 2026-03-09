# 01 Cache 原理与 Tiling 优化

## 1. 存储层次结构

```
层次          大小           延迟（cycles）   带宽估算
────────────────────────────────────────────────────────
寄存器        ~KB            1               极高（无内存访问）
L1 Cache      32KB           4               ~1 TB/s
L2 Cache      256KB-1MB      12              ~400 GB/s
L3 Cache      8-64MB         40-50           ~200 GB/s
主内存 DRAM   几十 GB        200-300         ~40-60 GB/s
```

**关键比例**：一次 L3 Cache miss 的代价 ≈ 50 次 L1 命中的代价

→ **优化核心：减少 Cache miss，尤其是 LLC（L3）miss**

---

## 2. Cache Line

### 基本概念

- CPU 以 **64 字节**为单位从内存加载数据（一个 Cache line）
- 一个 Cache line = 16 个 float（64 / 4 = 16）
- 访问某个 float 时，与其相邻的 15 个 float 也被自动加载到 Cache

```
内存布局（float 数组）：
[f0][f1][f2][f3][f4][f5][f6][f7][f8][f9][f10][f11][f12][f13][f14][f15] ← 1 Cache line
[f16]...
         ↑
  访问 f0，整个 Cache line（f0~f15）都被加载
  下次访问 f1~f15 → Cache hit！只需 4 cycles
```

### Spatial Locality（空间局部性）

- 顺序访问 → 每 16 个 float 只 miss 1 次 → miss 率 1/16 ≈ 6%
- 列访问（步长 N×4 字节）→ 每次访问都是新 Cache line → miss 率 100%

### Cache 的组织方式

**Set-Associative Cache**（面试常考）：
- Cache 被分成若干 **Set**（组）
- 每个 Set 包含若干 **Way**（路）
- 内存地址通过 hash 映射到某个 Set
- 该 Set 内有多个 Way 可以存放不同内存块

```
L1 Cache（32KB, 8-way 64 组）：
  地址 → 取中间 6 位作为 Set 索引（0..63）
  Set 内有 8 个 Way，存放 8 条 Cache line（8×64=512字节）
  若同一 Set 内要存放第 9 条 → 必须驱逐一条（通常用 LRU 策略）
```

**Conflict Miss（冲突未命中）**：
- 两个内存地址映射到同一 Set，但 Set 已满
- 例：两个大数组访问时，可能频繁相互驱逐（thrashing）
- Tiling 就是为了避免这种情况

---

## 3. GEMM 的 Cache 访问分析

### ijk 顺序（V1 Naive）

```
for i:
  for j:
    for k:  ← 最内层，k 变化
      C[i][j] += A[i][k] * B[k][j]
```

| 操作数 | 访问模式 | 步长 | Cache 行为 |
|--------|----------|------|------------|
| A[i][k] | i 固定，k 递增 → **行访问** | 4 字节 | 连续 ✓，每 16 个 float 1 次 miss |
| B[k][j] | j 固定，k 递增 → **列访问** | N×4 字节 | 跳跃 ✗，N=1024 时步长=4KB，必然 miss |
| C[i][j] | i、j 均固定 → **同一地址** | 0 | 理论上寄存器复用，但受 aliasing 限制 |

**B 矩阵的 Cache miss**：N=1024 时，访问 B 的一列需要跨越 1024 × 4 = 4096 字节，远超 L1 大小（32KB），几乎每次都 miss。

### ikj 顺序（V2）

```
for i:
  for k:
    a_ik = A[i][k]  ← 提升为寄存器
    for j:  ← 最内层，j 变化
      C[i][j] += a_ik * B[k][j]
```

| 操作数 | 访问模式 | 步长 | Cache 行为 |
|--------|----------|------|------------|
| a_ik | 寄存器 | - | 最快 ✓✓ |
| B[k][j] | k 固定，j 递增 → **行访问** | 4 字节 | 连续 ✓ |
| C[i][j] | i 固定，j 递增 → **行访问** | 4 字节 | 连续 ✓ |

**三个操作数都 Cache 友好**，Cache miss 率从 ~100% 降至 ~6%。

---

## 4. Tiling 原理

### 为什么还需要 Tiling？

V2（ikj）改善了局部性，但当矩阵很大时（N=2048，矩阵占 2048×2048×4 = 16MB），A 的一行和 B 的一行加起来就超过 L1 Cache。

对同一行 A，遍历所有 j 时，B 的数据可能在 L2/L3 之间反复移动。

### Tiling 思路

把大矩阵切成 T×T 的小块，确保三个活跃块（A块、B块、C块）同时驻留在 L1 Cache：

```
大矩阵（N=2048）：
  A 一行大小 = 2048 × 4 = 8KB > L1

  ┌─────────────────────────────────────┐
  │           A（2048×2048）            │
  └─────────────────────────────────────┘
  →分块→

  A[i..i+T, k..k+T]：T×T×4 字节
  B[k..k+T, j..j+T]：T×T×4 字节
  C[i..i+T, j..j+T]：T×T×4 字节
  合计：3T²×4 字节 ≤ L1（32KB）
  → T ≤ 52，选 T=32 或 64
```

### 块大小计算

```
3 × T² × sizeof(float) ≤ L1_size
T² ≤ L1_size / (3 × 4) = 32768 / 12 ≈ 2730
T ≤ 52

实践选择：
  T=32：3 × 32² × 4 = 12KB（L1 占用率 37%，非常安全）
  T=64：3 × 64² × 4 = 48KB（略超 L1，但适合 L2=256KB）

建议：先用 T=64 测试，再根据 perf cache-miss 结果调整
```

### Tiling 的算术强度分析（Roofline 模型角度）

**不分块时的有效内存访问量**：
- A 的每一列元素被 N 个不同的 C 块需要，但可能在 Cache 中被驱逐
- 有效算术强度 ≈ 0.25 FLOP/Byte（Memory Bound）

**分块后（块大小 T）**：
- A 的 T×K 块加载一次后，被 N/T 个 B 块复用（若 Cache 足够）
- B 的 K×T 块加载一次后，被 M/T 个 A 块复用
- 每个矩阵元素只从 DRAM 加载一次（理想情况）
- 有效算术强度 ≈ 2MNK / [(MK + KN + MN) × 4] ≈ T/4 FLOP/Byte
- T=64 → AI ≈ 16 FLOP/Byte（提升 64 倍！）

---

## 5. 用 perf 验证

```bash
# 对比 V1 和 V2 的 L1 Cache miss 率
perf stat -e L1-dcache-load-misses,L1-dcache-loads ./benchmark_cpu --size 1024

# 典型结果：
#   V1 (ijk)：L1 miss 率 ≈ 20-30%
#   V2 (ikj)：L1 miss 率 ≈ 2-5%

# 对比 V2 和 V3 的 LLC miss 率（L3 miss）
perf stat -e LLC-load-misses,LLC-loads ./benchmark_cpu --size 2048
#   V2：LLC miss 率较高（无分块，数据反复被驱逐）
#   V3：LLC miss 率显著降低（分块后数据复用）
```

---

## 6. 各版本 Cache 性能对比表（估算）

| 版本 | L1 miss 率 | LLC miss 率 | 说明 |
|------|-----------|-------------|------|
| V1 ijk | ~50% | ~25% | B 列访问，灾难性 miss |
| V2 ikj | ~6% | ~10% | 行访问，Cache line 充分利用 |
| V3 Tiling | ~6% | ~2% | 分块后数据驻留 Cache |
| V5 AVX2+Tiling | ~6% | ~2% | 同 V3，加上 SIMD 并行 |

---

## 7. 面试要点总结

**Q: 什么是 Cache line？为什么重要？**
> Cache line 是 CPU 从内存加载数据的最小单位，通常 64 字节 = 16 个 float。
> 访问一个 float 时，整个 Cache line 都被加载。
> 顺序访问可以复用同一 Cache line 中的其他 float（命中），
> 而跳跃访问每次都需要新 Cache line（miss）。
> L1 miss 代价约 50x，因此优化 Cache 利用率是高性能计算的核心。

**Q: Tiling 的核心思想是什么？**
> 把大矩阵切成小块，让三个活跃数据块（A块、B块、C块）同时驻留在 L1 Cache 中，
> 从而实现数据的时间局部性复用。
> 在 Roofline 模型中，Tiling 将算术强度从 ~0.25 提升到 ~T/4 FLOP/Byte，
> 可能将程序从 Memory Bound 转变为 Compute Bound。
