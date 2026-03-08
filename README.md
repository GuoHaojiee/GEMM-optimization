# GEMM 优化项目：从零到接近cuBLAS

## 项目结构

```
gemm_project/
├── include/
│   └── gemm.h          # 统一头文件
├── src/
│   ├── gemm_naive.cpp      # 版本1：朴素实现
│   ├── gemm_optimized.cpp  # 版本2：循环重排 + 分块
│   ├── gemm_simd.cpp       # 版本3：AVX2手写向量化
│   └── benchmark.cpp       # 统一性能测试入口
├── cuda/
│   ├── gemm_cuda_naive.cu  # 版本4：CUDA朴素实现
│   └── gemm_cuda_shared.cu # 版本5：Shared Memory优化
├── notes/
│   ├── 01_cache_theory.md  # 知识点：Cache原理
│   ├── 02_simd_theory.md   # 知识点：SIMD向量化
│   └── 03_cuda_theory.md   # 知识点：CUDA优化
└── Makefile
```

## 编译方法

```bash
# 编译CPU版本
make cpu

# 编译CUDA版本（需要CUDA环境）
make cuda

# 运行所有benchmark
make bench

# 用godbolt查看汇编（重要！）
g++ -O3 -march=native -S src/gemm_naive.cpp -o /tmp/naive.s
```

## 预期性能收益（参考）

| 版本 | 相对Naive的提升 | 关键技术 |
|------|---------------|---------|
| Naive (ijk) | 1x | 基准 |
| 循环重排 (ikj) | 3~5x | Cache友好 |
| 分块 Tiling | 5~10x | L1/L2 Cache复用 |
| AVX2 SIMD | 10~20x | 256bit向量指令 |
| CUDA Naive | 50~100x | GPU并行 |
| CUDA Shared Mem | 200~500x | 减少显存访问 |

---

## 面试时能讲的话

> "我从手写Naive GEMM开始，通过分析cache miss和内存访问模式，
> 逐步引入循环重排、分块优化和AVX2向量化，CPU端较baseline提升约XX倍。
> GPU端用CUDA Shared Memory优化，将全局内存访问降低16倍，
> 最终达到cuBLAS的约70%性能，并用Nsight Compute定位了剩余瓶颈。"
