// ============================================================
// gemm_v2_ikj.cpp — V2 循环重排 ikj（Cache 友好版本）
// ============================================================
//
// 【版本】V2 — ikj 循环顺序
// 【优化思路】
//   把 V1 的 ijk 改为 ikj 顺序，将最内层循环改为 j 变化。
//   核心洞察：最内层循环决定访存模式，j 在最内层时三个操作数都是行访问。
//
// 【访存分析（最内层循环 j 变化时）】
//   A[i][k]：i、k 均固定 → 同一常量，编译器提升为寄存器变量 ✓✓
//   B[k][j]：k 固定，j 递增 → 访问 B 的第 k 行 → stride=1 ✓ 连续
//   C[i][j]：i 固定，j 递增 → 访问 C 的第 i 行 → stride=1 ✓ 连续
//
//   三个操作数全部是 stride-1 访问！Cache line 空间局部性最大化。
//   每 16 个 float 只需 1 次 Cache miss（一条 Cache line = 64 字节 = 16 float）
//   → Cache miss 率从 ~100% 降至 ~6%（1/16）
//
// 【寄存器提升（Register Promotion）】
//   A[i*K+k] 在内层循环中 i、k 均不变，是循环不变量（loop invariant）
//   代码中显式将其提取为 float a_ik：
//     - 明确告知编译器这是常量，无需反复从内存读取
//     - 即使没有 __restrict__，编译器也能安全提升
//     - 实际测试：开启 -O3 后编译器通常会自动做这个优化
//
// 【与 V1 的 Cache miss 对比】
//   假设 N=K=1024，矩阵大小 4MB，L3 Cache 8MB
//   V1：B 矩阵每次列访问 → M×N×K 次 Cache miss（每次都 miss）
//   V2：B/C 行访问 → M×K × (N/16) 次 Cache miss（每 16 个元素只 miss 1 次）
//   改进约 16x（Cache line 中 float 个数）
//
// 【编译器自动向量化】
//   ikj 顺序的最内层循环是 j 变化，访问连续内存，是编译器最容易向量化的模式。
//   用 g++ -O3 -march=native 编译时，通常会自动生成 AVX2 向量指令。
//   可以用 godbolt.org 查看汇编验证：找 "vmovups" 或 "vfmadd" 开头的指令。
//
// 【性能基准（仅供参考）】
//   M=N=K=1024：约 5-20 GFLOPS（比 V1 快 5-10x）
// ============================================================

#include "../include/gemm.h"
#include <cstring>  // memset

void gemm_v2(const float* A, const float* B, float* C, int M, int N, int K) {
    // 清零输出矩阵
    memset(C, 0, (size_t)M * N * sizeof(float));

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            // 【寄存器提升】
            // 将 A[i][k] 提取为局部变量 a_ik
            // 在下面的 j 循环中 i、k 均不变，a_ik 是常量
            //
            // 优化效果：
            //   原来：每次循环读 A[i*K+k]（可能是 Cache line 中的数据，但仍需地址计算）
            //   现在：直接读寄存器，延迟 1 cycle（vs 内存访问 4+ cycles）
            //
            // 与 __restrict__ 的关系：
            //   若不加 __restrict__，编译器担心 C 和 A 有别名，无法确认 A[i*K+k]
            //   在 j 循环中不变，可能不敢自动提升。显式提取消除了这个顾虑。
            float a_ik = A[i * K + k];

            for (int j = 0; j < N; j++) {
                // 访存模式：
                //   a_ik：寄存器 ✓✓  延迟 ~1 cycle
                //   B[k*N+j]：k 固定，j 递增 → B 的第 k 行 → stride=1 ✓
                //   C[i*N+j]：i 固定，j 递增 → C 的第 i 行 → stride=1 ✓
                //
                // 编译器能否向量化？可以！
                //   a_ik 是常量（可用 _mm256_set1_ps 广播）
                //   B 和 C 的访问都是连续内存（可用 _mm256_loadu_ps）
                //   这正是 V4 手写 AVX2 的原型
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}
