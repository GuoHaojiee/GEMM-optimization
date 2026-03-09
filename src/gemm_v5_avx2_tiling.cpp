// ============================================================
// gemm_v5_avx2_tiling.cpp — V5 AVX2 + 分块（向量化+Cache复用组合）
// ============================================================
//
// 【版本】V5 — AVX2 SIMD + Tiling（组合优化）
// 【编译】需要：-mavx2 -mfma -O3
//
// 【优化思路】
//   V3（分块）解决了 Cache miss 问题，但块内计算是标量
//   V4（AVX2）实现了向量化，但未优化 Cache miss
//
//   V5 = V3 + V4 的组合：
//   - 外层：分块确保工作集在 Cache 中（时间局部性）
//   - 内层：AVX2 向量化确保高计算吞吐（SIMD 并行）
//
// 【块大小选择】
//   同 V3：3 × BLOCK_SIZE² × 4 ≤ L1/L2 Cache
//   这里仍选 64，实际调优时建议测试 32/64/128
//
// 【性能分析】
//   相比 V3（分块）：内层 AVX2 提升约 2-4x
//   相比 V4（AVX2）：分块提升约 2-5x（取决于矩阵大小）
//   V5 通常是 CPU 端最快的单线程版本（V6 加了多线程）
//
//   典型 GFLOPS（M=N=K=1024）：约 50-150 GFLOPS
//
// 【编译器会自动做 V5 级别的优化吗？】
//   使用 -O3 -march=native 时，GCC/Clang 可能做到类似优化，但：
//   1. 编译器的 tiling 块大小可能不是最优的
//   2. 需要 __restrict__ 才能激进向量化
//   3. 手写版本可以更精细控制
//   手写版本通常比编译器自动版本快 1.5-3x
//
// 【面试话术】
//   "V5 是 Cache 优化和 SIMD 优化的结合。分块解决了内存带宽瓶颈，
//    使有效算术强度从 ~0.25 提升到 ~16 FLOP/Byte；
//    AVX2 向量化使每个 FMA 操作处理 8 个数据而非 1 个，
//    两者叠加后比 Naive 版本快约 50-100x。"
// ============================================================

#include "../include/gemm.h"
#include <immintrin.h>  // AVX2
#include <cstring>
#include <algorithm>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

void gemm_v5(const float* A, const float* B, float* C, int M, int N, int K) {
    // 清零 C
    memset(C, 0, (size_t)M * N * sizeof(float));

    // ====== 外层分块：确保活跃数据块适合 Cache ======
    //
    // 外层循环顺序：i-k-j
    //   选择这个顺序的原因：
    //   - 固定 i、k 块后，内层 j 循环可以顺序写 C 和顺序读 B
    //   - A 块（i×k）在 j 块遍历时保持不变，充分复用
    for (int i = 0; i < M; i += BLOCK_SIZE) {
        int i_end = std::min(i + BLOCK_SIZE, M);

        for (int k = 0; k < K; k += BLOCK_SIZE) {
            int k_end = std::min(k + BLOCK_SIZE, K);

            for (int j = 0; j < N; j += BLOCK_SIZE) {
                int j_end = std::min(j + BLOCK_SIZE, N);

                // ====== 内层计算：块内使用 AVX2 向量化 ======
                //
                // 此时处理的是 C[i..i_end, j..j_end] += A[i..i_end, k..k_end] * B[k..k_end, j..j_end]
                // 这三个小块的大小约为 BLOCK_SIZE² × 4 字节
                // 全部驻留在 Cache 中
                for (int ii = i; ii < i_end; ii++) {
                    for (int kk = k; kk < k_end; kk++) {
                        // 广播 A[ii][kk]（块内 ikj 循环中的循环不变量）
                        __m256 ymm_a = _mm256_set1_ps(A[ii * K + kk]);

                        // j 维度向量化：处理 j_end - j 个元素
                        // 注意：这里 j 的起点不一定是 0，
                        // 所以 jj 是绝对列索引，对应内存偏移 jj
                        int jj = j;
                        for (; jj <= j_end - 8; jj += 8) {
                            // 向量化 FMA：C[ii][jj..jj+7] += A[ii][kk] * B[kk][jj..jj+7]
                            __m256 ymm_c = _mm256_loadu_ps(&C[ii * N + jj]);
                            __m256 ymm_b = _mm256_loadu_ps(&B[kk * N + jj]);
                            ymm_c = _mm256_fmadd_ps(ymm_a, ymm_b, ymm_c);
                            _mm256_storeu_ps(&C[ii * N + jj], ymm_c);
                        }

                        // 块内尾部处理（当块大小不是 8 的倍数，或最后一个块较小时）
                        float a_scalar = A[ii * K + kk];
                        for (; jj < j_end; jj++) {
                            C[ii * N + jj] += a_scalar * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
}
