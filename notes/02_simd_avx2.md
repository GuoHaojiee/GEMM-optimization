# 02 SIMD 向量化与 AVX2 详解

## 1. SIMD 基础

SIMD（Single Instruction Multiple Data）：一条指令同时处理多个数据。

```
标量（Scalar）：
  a0*b0, a1*b1, a2*b2, a3*b3, a4*b4, a5*b5, a6*b6, a7*b7  ← 8 条指令

AVX2 向量（SIMD）：
  [a0,a1,a2,a3,a4,a5,a6,a7] × [b0,b1,b2,b3,b4,b5,b6,b7]    ← 1 条指令
  ↑ 8 个 float 同时计算，理论吞吐是标量的 8 倍
```

### 各代 SIMD 指令集

| 指令集 | 寄存器宽度 | float 个数 | 需要 CPU |
|--------|-----------|-----------|---------|
| SSE    | 128 位    | 4         | Pentium III+ |
| SSE4.2 | 128 位    | 4         | Intel Nehalem+ |
| AVX    | 256 位    | 8         | Intel Sandy Bridge+ (2011) |
| AVX2   | 256 位    | 8         | Intel Haswell+ (2013) |
| AVX-512| 512 位    | 16        | Intel Skylake-X+ (部分 CPU 有) |

**编译选项**：`g++ -mavx2 -mfma` 开启 AVX2 + FMA 支持。

---

## 2. AVX2 关键 Intrinsics

### 加载（Load）

```cpp
#include <immintrin.h>

// 非对齐加载（最常用，ptr 无需对齐）
__m256 v = _mm256_loadu_ps(ptr);
// 汇编：vmovups ymm0, [ptr]
// 适用场景：大多数情况，代价很小（现代 CPU 非对齐加载几乎无惩罚）

// 对齐加载（ptr 必须 32 字节对齐）
__m256 v = _mm256_load_ps(ptr);
// 汇编：vmovaps ymm0, [ptr]
// 性能：几乎与 loadu 相同（Haswell 之后对齐惩罚消除）
// 优点：不对齐时会触发异常（帮助调试对齐问题）

// 广播（把一个标量复制到所有 8 个位置）
__m256 v = _mm256_set1_ps(scalar);
// 汇编：vbroadcastss ymm0, [scalar_addr]
// 适用：GEMM 中广播 A[i][k] 到整个 SIMD 寄存器

// 从内存广播（更高效，直接从内存广播，无需先加载到标量寄存器）
__m256 v = _mm256_broadcast_ss(&scalar);
```

### 存储（Store）

```cpp
// 非对齐存储
_mm256_storeu_ps(ptr, v);
// 汇编：vmovups [ptr], ymm0

// 对齐存储（ptr 必须 32 字节对齐）
_mm256_store_ps(ptr, v);
// 汇编：vmovaps [ptr], ymm0
```

### 算术运算

```cpp
// 加法
__m256 r = _mm256_add_ps(a, b);      // a + b（逐元素）

// 乘法
__m256 r = _mm256_mul_ps(a, b);      // a * b（逐元素）

// FMA（融合乘加）★★★ 最重要
__m256 r = _mm256_fmadd_ps(a, b, c); // a*b + c（一条指令！）
// 汇编：vfmadd231ps ymm2, ymm0, ymm1
// 为什么重要：
//   1. 一条指令完成两个操作（比 mul+add 少一条指令）
//   2. 中间结果不截断（精度更高）
//   3. 现代 CPU 的 FMA 单元与 MUL 单元相同（无额外延迟）
// 需要：-mfma 编译选项

// 其他常用
__m256 r = _mm256_sub_ps(a, b);      // a - b
__m256 r = _mm256_div_ps(a, b);      // a / b（很慢，避免在热点路径使用）
__m256 r = _mm256_sqrt_ps(a);        // sqrt(a)（每元素）
```

### 水平操作（较慢，避免在内层循环使用）

```cpp
// 水平加法（hadd）：相邻元素两两相加
// 用途：将 8 个 float 的 SIMD 寄存器规约成 1 个 float
// 实现：需要 2 次 hadd + extractf128 + haddps（较慢）
__m256 h = _mm256_hadd_ps(a, b);

// 更快的水平规约（手写）：
inline float hsum_avx(__m256 v) {
    // v = [a0,a1,a2,a3,a4,a5,a6,a7]
    __m128 lo  = _mm256_castps256_ps128(v);       // [a0,a1,a2,a3]
    __m128 hi  = _mm256_extractf128_ps(v, 1);     // [a4,a5,a6,a7]
    __m128 sum = _mm_add_ps(lo, hi);              // [a0+a4, a1+a5, ...]
    __m128 shuf = _mm_movehdup_ps(sum);
    __m128 s2   = _mm_add_ps(sum, shuf);          // a0+a4+a1+a5, ...
    __m128 s3   = _mm_movehl_ps(shuf, s2);
    return _mm_cvtss_f32(_mm_add_ss(s2, s3));     // 最终求和
}
```

---

## 3. 对齐 vs 非对齐加载的性能差异

### 历史背景

早期 CPU（SSE 时代）：非对齐加载（loadu）比对齐加载（load）慢约 2-3 倍。

现代 CPU（Haswell 以后）：两者几乎没有性能差异（当地址自然对齐时）。

### 当前建议

1. **默认使用 loadu/storeu**：代码更简单，性能几乎相同。

2. **若追求极致性能**：用 `posix_memalign` 或 C++17 `std::aligned_alloc` 分配 32 字节对齐的内存，再用 `load/store`。

```cpp
// 分配 32 字节对齐内存
float* A = (float*)aligned_alloc(32, M * K * sizeof(float));

// 或使用 new（C++17 直接支持对齐）
// std::allocator_traits...

// 然后可以用对齐加载
__m256 v = _mm256_load_ps(&A[i]);  // 确保 i % 8 == 0（即起始地址 32 字节对齐）
```

3. **何时对齐很重要**：
   - 使用 `store` 而非 `storeu` 时，若地址未对齐会触发 SIGBUS（帮助发现对齐 bug）
   - 在某些老系统上（虚拟化环境），非对齐访问仍有惩罚

---

## 4. 尾部处理策略

当 N 不是 8 的倍数时，最后 `N % 8` 个元素需要特殊处理。

### 策略1：标量尾部（本项目采用）

```cpp
int j = 0;
for (; j <= N - 8; j += 8) {
    // AVX2 处理 8 个元素
    __m256 c = _mm256_loadu_ps(&C[i*N+j]);
    __m256 b = _mm256_loadu_ps(&B[k*N+j]);
    c = _mm256_fmadd_ps(a_vec, b, c);
    _mm256_storeu_ps(&C[i*N+j], c);
}
for (; j < N; j++) {
    // 标量处理剩余元素
    C[i*N+j] += a_scalar * B[k*N+j];
}
```

优点：简单、正确
缺点：最后不到 8 个元素用标量，损失一些性能（当 N 远大于 8 时影响很小）

### 策略2：Masking（AVX-512 更适合，AVX2 较复杂）

```cpp
// AVX2 中的 masking 比较繁琐
// 通常只在 N % 8 != 0 的比例很大时才值得
__m256i mask = ...;  // 根据剩余元素数量构造掩码
__m256 partial = _mm256_maskload_ps(&C[i*N+j], mask);
// ... 处理 ...
_mm256_maskstore_ps(&C[i*N+j], mask, result);
```

### 策略3：Padding（简化代码，但增加内存）

```cpp
// 将矩阵宽度 Pad 到 8 的倍数（填充 0）
int N_pad = (N + 7) / 8 * 8;
// 分配 N_pad × sizeof(float) 的内存
// 填充部分初始化为 0
// 然后整个 j 循环都可以用 AVX2，无需处理尾部
```

---

## 5. 自动向量化 vs 手写 Intrinsics

### 编译器自动向量化的限制

**情况1：Aliasing 问题**

```cpp
// 编译器不敢向量化（A 和 C 可能指向同一内存）
void f(float* A, float* C, int n) {
    for (int i = 0; i < n; i++) C[i] = A[i] * 2;
}

// 加 __restrict__ 告诉编译器无 aliasing → 可以向量化
void f(float* __restrict__ A, float* __restrict__ C, int n) {
    for (int i = 0; i < n; i++) C[i] = A[i] * 2;
}
```

**情况2：依赖关系（Reduction）**

```cpp
float sum = 0;
for (int i = 0; i < n; i++) sum += A[i];  // 编译器可能向量化（reduction）
// 但结果可能与串行不同（浮点加法不满足结合律）
// 需要 -ffast-math 才会自动向量化 reduction
```

**情况3：复杂下标**

```cpp
// 编译器难以向量化（B 的列访问）
for (int k = 0; k < K; k++)
    C[i*N+j] += A[i*K+k] * B[k*N+j];  // B 按列访问，stride=N
```

### 验证方法：godbolt.org

1. 打开 https://godbolt.org
2. 粘贴代码，选择 `x86-64 gcc 13.2` + `-O3 -march=native -mavx2 -mfma`
3. 在汇编输出中查找：
   - `ymm` 寄存器 → AVX2 向量化 ✓
   - `vfmadd231ps` → FMA 指令 ✓
   - `xmm` 寄存器 → 只有 SSE（128位），未完全向量化 ✗
   - `mulss/addss` → 标量（完全未向量化）✗✗

### 手写 vs 自动：实际对比

| 场景 | 自动向量化 | 手写 AVX2 | 差距 |
|------|-----------|-----------|------|
| 简单 element-wise | 通常生成 AVX2 | 相同 | 几乎无差距 |
| 有 __restrict__ | 通常生成 AVX2 | 相同 | 几乎无差距 |
| 无 __restrict__ | 保守，可能 SSE | 强制 AVX2 | 2x |
| 复杂下标（GEMM） | 可能失败 | 有效 | 2-4x |
| Tiling + SIMD 组合 | 可能找不到机会 | 有效 | 2-8x |

---

## 6. GEMM 中的 AVX2 实现要点

```cpp
// V4 核心循环（ikj + AVX2）
for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
        // 广播 A[i][k]：一次内存读，广播到 8 个位置
        __m256 a_broadcast = _mm256_set1_ps(A[i*K+k]);

        int j = 0;
        for (; j <= N - 8; j += 8) {
            // 加载 C[i][j..j+7]（连续 8 个 float）
            __m256 c_vec = _mm256_loadu_ps(&C[i*N+j]);
            // 加载 B[k][j..j+7]（连续 8 个 float）
            __m256 b_vec = _mm256_loadu_ps(&B[k*N+j]);
            // FMA: c += a * b（8 个 float 同时计算）
            c_vec = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);
            // 写回
            _mm256_storeu_ps(&C[i*N+j], c_vec);
        }
        // 标量尾部
        for (; j < N; j++) C[i*N+j] += A[i*K+k] * B[k*N+j];
    }
}
```

**计算量分析**：
- 每次 `j` 循环迭代：1 次 loadu（B）+ 1 次 loadu（C）+ 1 次 fmadd + 1 次 storeu = 8 个 FMA
- 理论吞吐：每周期 2 个 FMA 指令 × 8 float = 每周期 16 个 FP 操作
- 每 GHz = 16 GFLOPS（加上流水线重叠，现代 CPU 可达此理论值）

---

## 7. 面试话术

> "我用 godbolt 分析了编译器自动向量化的汇编，发现由于缺少 `__restrict__` 修饰，
> 编译器只生成了保守的 SSE 128 位指令，每次只处理 4 个 float。
> 通过手写 AVX2 intrinsics，强制使用 256 位 FMA 指令，
> 将每次运算的数据宽度从 4 增加到 8 个 float，
> 结合 FMA 融合乘加（一条指令完成乘法和加法），
> 比编译器自动向量化额外提升了约 1.8 倍。"
