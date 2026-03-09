// ============================================================
// gemm_v1_naive.cpp — V1 Naive ijk 三重循环（基准版本）
// ============================================================
//
// 【版本】V1 — Naive ijk
// 【原理】最直接的三重循环实现，直接翻译数学定义
//         C[i][j] = Σ(k=0..K-1) A[i][k] * B[k][j]
//
// 【访存分析（最内层循环 k 变化时）】
//   A[i][k]：i 固定，k 递增 → 访问 A 的一行 → stride=1 ✓ 连续
//   B[k][j]：j 固定，k 递增 → 访问 B 的一列 → stride=N ✗ 大步长跳跃
//   C[i][j]：i、j 都固定  → 固定元素，每次累加 ✓ 寄存器复用
//
//   B 矩阵的访问步长 = N × sizeof(float) = N × 4 字节
//   当 N=1024 时，步长 = 4096 字节 = 64 个 Cache Line！
//   → 每次访问 B[k][j] 几乎必然是 Cache miss
//
// 【Cache miss 估算】
//   B 矩阵总大小 K×N×4 字节
//   每访问一列需要 K 次 Cache miss（每个都是新 Cache Line）
//   总 Cache miss 次数 ≈ M×N×K/1（因为列访问无复用）
//
// 【面试要点】
//   Q: 为什么 ijk 比 ikj 慢？
//   A: ijk 中最内层循环按列访问 B，步长为 N*4 字节，每次访问
//      都跨越多个 Cache Line，Cache miss 率接近 100%；
//      而 ikj 中最内层按行访问 B，stride=1，充分利用 Cache line 空间局部性。
//
//   Q: 这段代码编译后会自动向量化吗？
//   A: 通常不会，因为 j 在内层固定，编译器难以向量化列访问的 B[k][j]。
//
// 【性能基准（仅供参考，实际取决于机器）】
//   M=N=K=1024：约 0.5-2 GFLOPS（严重受内存带宽限制）
// ============================================================

#include "../include/gemm.h"
#include <cstring>  // memset

void gemm_v1(const float* A, const float* B, float* C, int M, int N, int K) {
    // 清零输出矩阵 C（M×N 个 float）
    // 必须清零：后续用 += 累加，不清零会产生错误结果
    memset(C, 0, (size_t)M * N * sizeof(float));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                // 访存模式详解：
                //   A[i*K+k]：i 固定，k 变化 → 访问 A 的第 i 行
                //              内存地址连续，每 16 个元素共享一个 Cache line ✓
                //
                //   B[k*N+j]：j 固定，k 变化 → 访问 B 的第 j 列
                //              相邻元素地址差 N*4 字节（远大于 64 字节）
                //              每次访问都是不同的 Cache line → 几乎必然 Cache miss ✗
                //
                //   C[i*N+j]：i、j 均固定 → 同一元素反复读写
                //              理论上编译器会将其提升到寄存器中（但不一定，
                //              因为 C 和 A/B 可能 alias）
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}
