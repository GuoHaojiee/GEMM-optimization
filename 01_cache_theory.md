# 知识点1：Cache原理与内存访问优化

## 为什么Cache这么重要？

```
存储层次       大小        延迟（cycles）   带宽
──────────────────────────────────────────────────
Register       ~KB         1               极高
L1 Cache       32KB        4               ~1 TB/s
L2 Cache       256KB       12              ~400 GB/s
L3 Cache       8MB         40              ~200 GB/s
主内存(DRAM)   几十GB      200+            ~50 GB/s
```

一次L1 Cache miss的代价 ≈ 50次L1命中的代价  
→ **优化的核心就是减少Cache miss**

---

## Cache Line（缓存行）

- CPU以64字节为单位从内存加载数据（一个Cache Line）
- 访问一个float时，连带前后共16个float都被加载到Cache
- 如果下次访问这16个float中的任何一个，都是Cache命中

```
内存：[f0][f1][f2]...[f15][f16]...[f31]...
            ↑一次Cache Line加载
访问f0 → f0~f15都进Cache → 访问f1~f15都是命中
```

---

## 矩阵乘法的Cache分析

```
C(NxN) = A(NxN) × B(NxN)，N=1024

ijk顺序访问B：
  B[0][j], B[1][j], B[2][j], ...  ← 列访问，步长=N*4=4KB
  每次访问都跨越1个Cache Line  → Cache miss率接近100%
  
ikj顺序访问B：
  B[k][0], B[k][1], B[k][2], ...  ← 行访问，步长=4B
  每16次访问只有1次Cache miss（命中率15/16≈94%）
```

---

## Tiling的数学分析

**不分块时的内存访问量：**
- 计算C[i][j]需要访问A的第i行（N次）+ B的第j列（N次）
- C的N²个元素，总访问量 ≈ 2N³次

**分块后（块大小T）：**
- 每个T×T的C块需要A和B各T×T的块，这两块总大小=2T²×4字节
- 如果2T²×4 < L1 Cache大小（32KB），则每次加载后可被完全复用T次
- 总主内存访问量 ≈ 2N³/T次（减少了T倍！）

---

## 验证方法：perf stat

```bash
# 统计Cache miss次数
perf stat -e cache-misses,cache-references,instructions ./benchmark

# 对比ijk vs ikj的cache-miss差异
# ijk版本的cache-miss会比ikj多很多倍
```

---

## 面试时的表达方式

> "ijk顺序中，最内层循环对B矩阵进行列访问，
>  步长为N×4字节，远大于Cache Line的64字节，
>  导致每次访问都是Cache miss。
>  改为ikj顺序后，最内层循环变为行访问，
>  步长为4字节，充分利用了Cache Line的空间局部性，
>  Cache miss率从接近100%降至约6%。"
