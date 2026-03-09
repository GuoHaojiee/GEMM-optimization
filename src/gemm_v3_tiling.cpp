// ============================================================
// gemm_v3_tiling.cpp — V3 分块 Tiling（Cache 复用版本）
// ============================================================
//
// 【版本】V3 — Blocking / Tiling
// 【优化思路】
//   V2（ikj）在 N 较大时（如 2048），即使访问模式 Cache 友好，
//   但 A 行（1×K×4 字节）和 B 行（K×N 小块）也可能超过 L1/L2 Cache，
//   导致后续迭代时数据被驱逐（evict），需要重新从 L3 或内存加载。
//
//   分块（Tiling）的核心思想：
//   把大矩阵切成小块，确保三个"活跃"数据块（A块、B块、C块）
//   同时驻留在 L1 Cache 中，实现数据的时间局部性（temporal locality）复用。
//
// 【块大小选择】
//   条件：3个块同时放入 L1 Cache
//   公式：3 × BLOCK_SIZE × BLOCK_SIZE × sizeof(float) ≤ L1 Cache 大小
//   L1=32KB: 3 × BS² × 4 ≤ 32768 → BS² ≤ 2730 → BS ≤ 52
//   实践中选 32 或 64（2 的幂次，硬件更友好）
//
//   这里选 BLOCK_SIZE=64：
//   3 × 64 × 64 × 4 = 49152 字节 ≈ 48KB
//   略大于 L1（32KB），但适合 L2（256KB），实际性能仍很好
//   （L2 带宽比 L3 快 3-5x）
//
// 【算术强度分析（Roofline 角度）】
//   不分块（V2）：每个数据元素可能被多次从内存加载
//     A 的每行被 N 次访问，但可能在 Cache 中丢失重新加载
//     有效算术强度 ≈ 0.25 FLOP/Byte（内存 bound）
//
//   分块后（块大小 T=64）：
//     A 的 T×K 块被 B 的 T 列复用 T 次 → 复用因子 T
//     B 的 K×T 块被 A 的 T 行复用 T 次
//     有效算术强度 ≈ T/4 = 16 FLOP/Byte（显著提升！）
//
// 【循环顺序：为什么外层用 i-k-j，内层用 ikj】
//   外层：i→k→j 顺序确保 A块 和 B块 在内层计算时保持在 Cache 中
//     - i 块：确定要计算 C 的哪些行（i..i+BLOCK_SIZE）
//     - k 块：A 的列范围和 B 的行范围（k..k+BLOCK_SIZE）
//     - j 块：B 的列范围（j..j+BLOCK_SIZE）
//   内层：ikj 顺序（同 V2），确保最内层 j 循环是连续访问
//
// 【边界处理】
//   M/N/K 不要求是 BLOCK_SIZE 的倍数
//   使用 std::min(x + BLOCK_SIZE, limit) 确保不越界
//
// 【面试要点】
//   Q: 分块大小如何选择？
//   A: 目标是让三个 active tile（A块+B块+C块）同时放入 L1 Cache：
//      3 × T² × 4 ≤ L1_size，T=sqrt(L1_size/12)
//      L1=32KB → T≤52，通常选 32 或 64
//
//   Q: 分块对 Roofline 模型的影响？
//   A: 分块将算术强度从 ~0.25 提升到 ~T/4 FLOP/Byte，
//      T=64 时约 16 FLOP/Byte，可能从 Memory Bound 转变为 Compute Bound
// ============================================================

#include "../include/gemm.h"
#include <cstring>   // memset
#include <algorithm> // std::min

// 分块大小（可通过编译选项覆盖）
// 选择 64：3 × 64² × 4 = 49152 字节，适合大多数 CPU 的 L2 Cache
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

void gemm_v3(const float* A, const float* B, float* C, int M, int N, int K) {
    // 清零 C 矩阵
    memset(C, 0, (size_t)M * N * sizeof(float));

    // ====== 外层三重循环：遍历块 ======
    //
    // i 块：处理 C 的第 i..i_end 行（同时也是 A 的行范围）
    for (int i = 0; i < M; i += BLOCK_SIZE) {
        int i_end = std::min(i + BLOCK_SIZE, M);  // 边界安全

        // k 块：A 的列范围 = B 的行范围（沿 K 维度分块）
        // 确定了哪块 A 和哪块 B 在内层循环中被重复使用
        for (int k = 0; k < K; k += BLOCK_SIZE) {
            int k_end = std::min(k + BLOCK_SIZE, K);

            // j 块：处理 C 的第 j..j_end 列（同时也是 B 的列范围）
            for (int j = 0; j < N; j += BLOCK_SIZE) {
                int j_end = std::min(j + BLOCK_SIZE, N);

                // ====== 内层三重循环：在块内计算 ======
                //
                // 此时 A[i..i_end, k..k_end] 和 B[k..k_end, j..j_end]
                // 的大小均约为 BLOCK_SIZE×BLOCK_SIZE×4 字节
                // 两块合计约 2×64×64×4 = 32KB，加上 C 块约 48KB
                // → 全部驻留在 L2 Cache 中
                //
                // 内层使用 ikj 顺序（同 V2），确保 j 循环是连续访问
                for (int ii = i; ii < i_end; ii++) {
                    for (int kk = k; kk < k_end; kk++) {
                        // 提取 A[ii][kk] 为寄存器变量（循环不变量）
                        float a_ik = A[ii * K + kk];

                        for (int jj = j; jj < j_end; jj++) {
                            // B[kk][jj]：kk 固定，jj 递增 → 行访问 ✓
                            // C[ii][jj]：ii 固定，jj 递增 → 行访问 ✓
                            // a_ik：寄存器 ✓✓
                            C[ii * N + jj] += a_ik * B[kk * N + jj];
                        }
                    }
                }
                // 每次完成一个 (i,k,j) 块的计算后，
                // 进入下一个 j 块时需要重新加载 B 块（A块可能仍在 Cache）
                // 这比 V2 中矩阵可能完全不在 Cache 中要好得多
            }
        }
    }
}
