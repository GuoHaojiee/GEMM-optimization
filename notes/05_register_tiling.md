# 05 Register Tiling（寄存器分块）教程

## 为什么 Shared Memory 版本（V8）还不够快？

### V8 的性能瓶颈

V8 中每个 Thread 只计算 C 的 **1 个元素**。

对每个 K-tile（TILE_SIZE=T=16），每个 Thread 的操作：

```
加载 As 中的 1 行（T=16 个 float 从 Shared Memory 读）
加载 Bs 中的 1 列（T=16 个 float 从 Shared Memory 读）
计算 T=16 次 FMA

Shared Memory 读：32 个 float（128 字节）
FMA 计算：16 次
算术强度（相对 Shared Memory）= 16 / (32×4) = 0.125 FLOP/Byte
```

**问题**：Shared Memory 带宽虽然比 Global Memory 高 10-20x，但 0.125 FLOP/Byte 的算术强度意味着每从 Shared Memory 读 1 字节，只能做 0.125 次 FMA。

GPU 的算力远超 Shared Memory 带宽（即使是 Shared Memory），这意味着 V8 仍然受带宽限制——只不过是 **Shared Memory 带宽**限制。

### 量化分析

以 A100 GPU 为例：
- FP32 峰值算力：312 TFLOPS（使用 Tensor Core）/ 19.5 TFLOPS（不用）
- Shared Memory 带宽：~19 TB/s（单 SM）× 108 SM = ~2000 TB/s（全 GPU）

计算所需带宽 = 19.5 TFLOPS / 0.125 FLOP/Byte = 156 TB/s

全 GPU Shared Memory 带宽约 2000 TB/s，看似足够，但实际上每个 SM 的 Shared Memory 带宽有限，加上 Bank Conflict 等问题，V8 的实际性能通常只有 cuBLAS 的 20-40%。

---

## Register Tiling 的核心思路

**关键洞察**：让每个 Thread 计算更多的输出元素（不只是 1 个），从而复用 Shared Memory 中的数据。

### 外积展开（Outer Product）

假设每个 Thread 计算 TM×TN（如 8×8=64）个输出元素。

对每个 BK 步骤（k 维度的一个切片）：

```
从 Shared Memory 读：
  regA[TM] = A 的一列切片（TM 个元素）
  regB[TN] = B 的一行切片（TN 个元素）
  总读取：TM + TN 个 float

计算：
  外积 regA ⊗ regB → 更新 accum[TM][TN]（TM×TN 次 FMA）
  TM=TN=8：8+8=16 个 float，64 次 FMA
```

算术强度（相对 Shared Memory）= TM×TN / (TM+TN) / sizeof(float)
= 64 / (16 × 4) = **1 FLOP/Byte**（比 V8 的 0.125 提升 8 倍！）

---

## 数据复用比的详细推导

### 为什么叫"外积"？

```
列向量 regA（TM=8 个元素）：
  [a0]
  [a1]
  [a2]
  ...
  [a7]

行向量 regB（TN=8 个元素）：
  [b0, b1, b2, ..., b7]

外积 = regA × regBᵀ = TM×TN 矩阵：
  [a0×b0,  a0×b1,  ..., a0×b7 ]
  [a1×b0,  a1×b1,  ..., a1×b7 ]
  ...
  [a7×b0,  a7×b1,  ..., a7×b7 ]
  = 64 次乘法！
```

**每个 regA[m] 被复用 TN=8 次**（与 regB 的每个元素各一次）
**每个 regB[n] 被复用 TM=8 次**（与 regA 的每个元素各一次）

这就是"复用比 TM"或"复用比 TN"的含义。

### 与 V8 的对比

| 版本 | 每Thread输出元素 | Smem 读取 | FMA | AI（Smem）|
|------|----------------|-----------|-----|----------|
| V8   | 1              | 2T=32 float | T=16 次 | 0.125 |
| V9   | TM×TN=64       | TM+TN=16 float | 64 次 | 1.0 |

→ V9 在相同的 Shared Memory 读取量下，完成了 4 倍的计算！

---

## 代码逐行解读

### 参数配置

```cpp
constexpr int BM = 64;  // 每个 Block 负责 C 的 64 行
constexpr int BN = 64;  // 每个 Block 负责 C 的 64 列
constexpr int BK = 8;   // K 方向每次处理 8 列/行
constexpr int TM = 8;   // 每个 Thread 计算 C 的 8 行
constexpr int TN = 8;   // 每个 Thread 计算 C 的 8 列
```

参数的含义关系：
- Block 内 Thread 数 = BM/TM × BN/TN = 8 × 8 = 64
- Shared Memory：As[BK][BM]=512 float=2KB，Bs[BK][BN]=512 float=2KB，共 4KB
- 每个 Thread 的累加器：accum[TM][TN]=64 个寄存器

### 线程坐标映射

```cpp
int tx = threadIdx.x;  // 0..BN/TN-1 = 0..7（决定 Thread 负责哪些 N 方向元素）
int ty = threadIdx.y;  // 0..BM/TM-1 = 0..7（决定 Thread 负责哪些 M 方向元素）

int block_row = blockIdx.y * BM;  // Block 负责的 C 的起始行
int block_col = blockIdx.x * BN;  // Block 负责的 C 的起始列

int thread_row = ty * TM;  // Thread 在 Block 内负责的起始行（相对）
int thread_col = tx * TN;  // Thread 在 Block 内负责的起始列（相对）
```

### 协同加载到 Shared Memory

```cpp
// 64 个 Thread 合作加载 As[8][64] = 512 个 float
// 每个 Thread 加载 512/64 = 8 个 float
for (int load_idx = thread_idx;   // 线性化的加载索引
     load_idx < BK * BM;
     load_idx += THREADS) {       // 步长 = Thread 总数

    int as_k = load_idx / BM;    // 确定加载 As 的哪一行（k 方向）
    int as_m = load_idx % BM;    // 确定加载 As 的哪一列（m 方向）

    // 对应 A 矩阵的全局坐标
    int global_row = block_row + as_m;
    int global_col = k_base + as_k;

    As[as_k][as_m] = A[global_row * K + global_col];  // Global → Shared
}
__syncthreads();  // 必须等所有 Thread 完成加载！
```

### 外积计算

```cpp
float regA[TM];  // 8 个寄存器
float regB[TN];  // 8 个寄存器
// accum[TM][TN]：64 个寄存器（在 K-tile 循环外声明，持续累加）

for (int k = 0; k < BK; k++) {   // BK=8 次迭代
    // 从 Shared Memory 加载 A 的一列切片到寄存器
    for (int m = 0; m < TM; m++)
        regA[m] = As[k][thread_row + m];  // 8 次 Shared Memory 读

    // 从 Shared Memory 加载 B 的一行切片到寄存器
    for (int n = 0; n < TN; n++)
        regB[n] = Bs[k][thread_col + n];  // 8 次 Shared Memory 读

    // 外积：TM × TN = 64 次 FMA（全部从寄存器读）
    for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++)
            accum[m][n] += regA[m] * regB[n];  // 寄存器 × 寄存器！
}
```

**关键**：最内层的 `accum[m][n] += regA[m] * regB[n]` 完全在寄存器中操作，延迟 1 cycle。

### BK 方向的循环

对每个 K-tile（外层 `k_tile` 循环），都执行一次上述流程：
1. 加载 A 的 BM×BK 块（从 Global 到 Shared）
2. 加载 B 的 BK×BN 块（从 Global 到 Shared）
3. BK 次外积更新（累加到 accum）

经过 K/BK 次 K-tile 循环后，accum 中保存了完整的 C 子矩阵。

---

## TM/TN 选择的权衡

| TM=TN | 寄存器数(accum) | 算术强度 | Block线程数 | Occupancy |
|-------|----------------|---------|------------|----------|
| 4     | 16             | 0.25    | 256        | 高       |
| 8     | 64             | 1.0     | 64         | 中       |
| 16    | 256            | 4.0     | 16         | 低       |

**实践经验**：
- TM=TN=8 通常是最优的均衡点
- TM/TN 更大时，寄存器溢出（spilling）风险增加，性能反而下降
- 可以通过 `nvcc --ptxas-options=-v` 检查寄存器用量

---

## 进一步优化方向

### Double Buffering（双缓冲，cuBLAS 的关键优化）

```cpp
// 使用两套 Shared Memory Buffer，交替使用
__shared__ float As[2][BK][BM];  // 双缓冲
__shared__ float Bs[2][BK][BN];

int cur = 0;  // 当前使用的 buffer
// 预加载第一个 tile
load_tile(As[cur], Bs[cur], ...);
__syncthreads();

for (int k_tile = 0; k_tile < num_tiles - 1; k_tile++) {
    int next = 1 - cur;
    // 异步预加载下一个 tile（使用 cp.async 指令，Ampere+ GPU）
    async_load_tile(As[next], Bs[next], ...);

    // 同时计算当前 tile
    compute_outer_product(As[cur], Bs[cur], accum, ...);

    __syncthreads();  // 等待异步加载完成（或用 cp.async.wait_group）
    cur = next;
}
// 处理最后一个 tile
compute_outer_product(As[cur], Bs[cur], accum, ...);
```

Double Buffering 使 Global Memory 加载与 Shared Memory 计算重叠，隐藏加载延迟。

---

## 面试话术

> "Register Tiling（V9）是从 V8 Shared Memory 版本的一个关键升级。
> V8 中每个 Thread 只计算 1 个输出元素，
> Shared Memory 的算术强度仅为 0.125 FLOP/Byte，
> 仍然受 Shared Memory 带宽限制。
>
> V9 的核心思想是让每个 Thread 计算 TM×TN=8×8=64 个输出元素，
> 通过外积展开（Outer Product）方式：
> 每次从 Shared Memory 加载 TM+TN=16 个 float，
> 计算 TM×TN=64 次 FMA，
> 算术强度提升到 1 FLOP/Byte（比 V8 高 8 倍）。
>
> 代价是每个 Thread 需要更多寄存器（64 个 float 的累加器），
> 这会降低 SM 的 Occupancy，
> 但计算密度的提升通常超过 Occupancy 下降的影响，
> 最终比 V8 快 2-4 倍。"
