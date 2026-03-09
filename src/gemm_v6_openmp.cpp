// ============================================================
// gemm_v6_openmp.cpp — V6 OpenMP 多线程 + AVX2 + Tiling
// ============================================================
//
// 【版本】V6 — OpenMP 并行化 + AVX2 + Tiling
// 【编译】需要：-fopenmp -mavx2 -mfma -O3
//
// 【并行化策略】
//   在 V5（AVX2 + Tiling）的基础上，在最外层 i 块循环上加并行
//   #pragma omp parallel for schedule(static)
//
// 【为什么在外层 i 块并行？】
//   1. 减少并行化开销：
//      OpenMP 线程创建/同步有开销（约 1-10 微秒）
//      并行化粒度越粗，开销占比越小
//      外层循环迭代次数 = M/BLOCK_SIZE，每次迭代工作量大
//
//   2. 数据独立性：
//      不同 i 块计算的是 C 的不同行 → 完全独立，无数据依赖
//      读 A 的不同行 → 独立（只读，不需要互斥）
//      读 B 的所有列 → 共享读，无竞争（只读操作是线程安全的）
//      写 C 的不同行 → 独立，无需加锁！
//
//   3. False Sharing 分析：
//      False Sharing：多个线程写入同一个 Cache line 中的不同元素
//      本实现的情况：不同线程写 C 的不同行（不同的 i 块）
//      C 的行大小 = N × sizeof(float)，当 N ≥ 16 时，不同行不在同一 Cache line
//      → 基本无 False Sharing 问题（N 通常远大于 16）
//
// 【schedule(static) vs schedule(dynamic) 分析】
//   schedule(static)：
//     - 编译时将迭代均分给所有线程（每线程约 M/BLOCK_SIZE/nthreads 次迭代）
//     - 优点：零运行时调度开销，线程局部性好（每个线程始终处理同一段 C 的行）
//     - 适用场景：每次迭代工作量相同（GEMM 中各块大小相近），推荐使用
//
//   schedule(dynamic, chunk_size)：
//     - 运行时动态分配，先完成的线程领取新任务
//     - 优点：负载均衡更好（适合不规则工作量）
//     - 缺点：有调度开销和线程间同步
//     - GEMM 中工作量均匀，不需要 dynamic
//
// 【线程数设置方式】
//   方法1：代码内设置 omp_set_num_threads(nthreads)
//   方法2：环境变量 OMP_NUM_THREADS=8 ./benchmark_cpu
//   方法3：让 OpenMP 自动使用 omp_get_max_threads()（通常等于核心数）
//
//   本实现使用方法1（命令行参数控制），便于 benchmark 对比
//
// 【超线程（HyperThreading）的影响】
//   物理核 vs 逻辑核：8核16线程的CPU
//   GEMM 是计算密集型，超线程增益有限（两个逻辑核共享同一物理核的执行单元）
//   通常将线程数设为物理核数效果最好
//   实测对比：threads=8 vs threads=16，性能差异可能很小
//
// 【NUMA（非一致性内存访问）考虑】
//   多路服务器上，每个 CPU Socket 有自己的内存控制器
//   访问远端内存（另一个 Socket 的内存）比本地慢 2-3x
//   解决方案：用 numa_alloc_local 分配线程本地内存
//   桌面 CPU（单 Socket）通常不需要考虑这个问题
//
// 【性能基准（仅供参考）】
//   M=N=K=2048，8线程：约 200-500 GFLOPS（比 V5 快 4-8x）
// ============================================================

#include "../include/gemm.h"
#include <immintrin.h>  // AVX2
#include <omp.h>        // OpenMP
#include <cstring>
#include <algorithm>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

void gemm_v6(const float* A, const float* B, float* C, int M, int N, int K) {
    // 清零 C（单线程执行，避免多线程 memset 竞争）
    memset(C, 0, (size_t)M * N * sizeof(float));

    // ====== 外层并行化：按 i 块并行 ======
    //
    // #pragma omp parallel for：并行化紧接着的 for 循环
    // schedule(static)：均匀静态分配，每个线程处理连续的 i 块
    //
    // 数据共享属性（OpenMP 默认规则）：
    //   循环变量 i：private（每个线程有独立副本）
    //   A, B, C, M, N, K：shared（所有线程共享指针/值）
    //   i_end：firstprivate 或在循环体内声明（本代码中是局部变量 → private）
    //
    // 线程安全性：
    //   读 A：共享只读 ✓
    //   读 B：共享只读 ✓（B 被所有线程并发读取，但只读操作是安全的）
    //   写 C：不同 i 块 → 不同地址范围 → 无数据竞争 ✓
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i += BLOCK_SIZE) {
        int i_end = std::min(i + BLOCK_SIZE, M);

        // k、j 块循环：每个线程独立执行，无同步需要
        for (int k = 0; k < K; k += BLOCK_SIZE) {
            int k_end = std::min(k + BLOCK_SIZE, K);

            for (int j = 0; j < N; j += BLOCK_SIZE) {
                int j_end = std::min(j + BLOCK_SIZE, N);

                // 块内 AVX2 向量化计算（与 V5 完全相同）
                for (int ii = i; ii < i_end; ii++) {
                    for (int kk = k; kk < k_end; kk++) {
                        __m256 ymm_a = _mm256_set1_ps(A[ii * K + kk]);

                        int jj = j;
                        for (; jj <= j_end - 8; jj += 8) {
                            __m256 ymm_c = _mm256_loadu_ps(&C[ii * N + jj]);
                            __m256 ymm_b = _mm256_loadu_ps(&B[kk * N + jj]);
                            ymm_c = _mm256_fmadd_ps(ymm_a, ymm_b, ymm_c);
                            _mm256_storeu_ps(&C[ii * N + jj], ymm_c);
                        }

                        // 尾部标量处理
                        float a_scalar = A[ii * K + kk];
                        for (; jj < j_end; jj++) {
                            C[ii * N + jj] += a_scalar * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
    // OpenMP parallel for 结束时有隐式 barrier（等待所有线程完成）
}
