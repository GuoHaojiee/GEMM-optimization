# 知识点2：SIMD向量化与编译优化

## 什么是SIMD？

```
标量计算（每次处理1个）：
  a[0]*b[0], a[1]*b[1], a[2]*b[2], ..., a[7]*b[7]  ← 8条指令

AVX2向量计算（每次处理8个）：
  [a0,a1,a2,a3,a4,a5,a6,a7] × [b0,b1,b2,b3,b4,b5,b6,b7]  ← 1条指令
```

- SSE: 128位寄存器 = 4个float
- AVX: 256位寄存器 = 8个float  
- AVX-512: 512位寄存器 = 16个float（服务器CPU）

---

## 编译器自动向量化的局限

```cpp
// 编译器「能」自动向量化这个：
for (int i = 0; i < n; i++)
    c[i] = a[i] * b[i];

// 编译器「不敢」自动向量化这个（不确定指针是否有别名）：
void f(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++)
        c[i] = a[i] * b[i];  // a和c可能指向同一块内存？
}

// 加上 __restrict__ 告诉编译器没有别名，可以向量化：
void f(float* __restrict__ a, float* __restrict__ b,
       float* __restrict__ c, int n) { ... }
```

---

## 如何用godbolt.org验证向量化

1. 打开 https://godbolt.org
2. 左边粘贴代码
3. 右边选择 `x86-64 gcc` + 参数 `-O3 -march=native`
4. 查看汇编中的指令前缀：
   - `vmovss` / `vfmaddss` → 标量（没有向量化）
   - `vmovups` / `vfmaddps` → 128位向量（SSE）
   - `vmovups` + ymm寄存器 → 256位向量（AVX2）✓

```asm
# 这是好的（AVX2向量化）：
vmovups  ymm0, YMMWORD PTR [rsi+rax]   # 加载8个float
vfmadd231ps ymm1, ymm0, YMMWORD PTR ... # 8个FMA同时执行

# 这是不好的（只有标量）：
movss    xmm0, DWORD PTR [rsi+rax*4]   # 只加载1个float
fmul     xmm0, xmm1
```

---

## 关键AVX2 Intrinsics速查

```cpp
#include <immintrin.h>

// 加载
__m256 v = _mm256_loadu_ps(ptr);     // 非对齐加载（常用）
__m256 v = _mm256_load_ps(ptr);      // 对齐加载（ptr必须32字节对齐，更快）

// 广播（把一个值复制到8个位置）
__m256 v = _mm256_set1_ps(scalar);

// 运算
__m256 r = _mm256_add_ps(a, b);      // a + b
__m256 r = _mm256_mul_ps(a, b);      // a * b
__m256 r = _mm256_fmadd_ps(a, b, c); // a*b + c （FMA，需要-mfma编译选项）

// 存储
_mm256_storeu_ps(ptr, v);            // 非对齐存储
_mm256_store_ps(ptr, v);             // 对齐存储

// 水平求和（把8个float加成1个，比较慢，避免在内层循环用）
// 方法：用两次hadd再配合extractf128
```

---

## 编译选项对性能的影响

```bash
# 基础优化
g++ -O0  # 无优化，debug用
g++ -O2  # 标准优化
g++ -O3  # 激进优化（开启自动向量化、循环展开等）

# 针对当前CPU的优化（生产中用，不可移植）
g++ -O3 -march=native  # 使用当前CPU支持的最高指令集
g++ -O3 -march=native -mtune=native  # 同时优化指令调度

# 手动指定指令集
g++ -O3 -mavx2 -mfma  # 开启AVX2和FMA支持

# 其他有用选项
-funroll-loops         # 循环展开
-ffast-math            # 宽松浮点（允许重新排序，损失少量精度换性能）
-fno-omit-frame-pointer # debug时保留栈帧（perf分析用）
```

---

## 面试话术

> "我用godbolt对比了开-O3前后的汇编，
>  发现编译器在这段代码上保守地使用了SSE 128位指令，
>  原因是它无法确认输入指针不互相重叠。
>  我添加了__restrict__修饰符，
>  并在循环边界处理好了余数，
>  手写AVX2 intrinsics后强制使用256位FMA指令，
>  比编译器自动版本额外提升了约1.8倍。"
