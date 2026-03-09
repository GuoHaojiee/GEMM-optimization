// ============================================================
// gemm_v4_avx2.cpp — V4 AVX2 手写 SIMD 向量化
// ============================================================
//
// 【版本】V4 — AVX2 SIMD (ikj + 向量化)
// 【编译】需要：-mavx2 -mfma
//
// 【为什么手写 SIMD？】
//   编译器（g++ -O3）在很多情况下能自动向量化 V2（ikj 顺序），
//   但存在以下局限：
//   1. Aliasing 问题：若没有 __restrict__，编译器担心指针重叠，
//      不敢激进向量化（会插入 scalar fallback 路径）
//   2. 对齐未知：编译器保守地使用 vmovups（非对齐）而非 vmovaps（对齐）
//   3. 寄存器分配：编译器生成的代码可能无法最优利用寄存器
//
//   手写 AVX2 好处：
//   - 强制使用 256 位（8 float）而非 128 位 SSE 指令
//   - 明确使用 FMA（_mm256_fmadd_ps），比 mul + add 节省一条指令
//   - 完全控制尾部处理和内存对齐策略
//
// 【AVX2 关键 Intrinsics 说明】
//   _mm256_set1_ps(x)     : 广播标量 → [x,x,x,x,x,x,x,x]（8个float）
//                           编译为：vbroadcastss ymm0, [mem]
//
//   _mm256_loadu_ps(ptr)  : 从内存加载 8 个 float（不要求对齐）
//                           编译为：vmovups ymm0, [ptr]
//                           vs _mm256_load_ps: 要求 32 字节对齐，稍快
//                           何时用 load：知道数据是 32 字节对齐时（用 aligned_alloc）
//
//   _mm256_fmadd_ps(a,b,c): a*b + c，融合乘加（Fused Multiply-Add）
//                           一条指令完成两个操作，比 mul+add 精度更高（中间结果不截断）
//                           编译为：vfmadd231ps ymm2, ymm0, ymm1
//                           需要：硬件支持 FMA（Intel Haswell+ / AMD Ryzen+）
//
//   _mm256_storeu_ps(ptr,v): 将 8 float 写回内存（不要求对齐）
//                            编译为：vmovups [ptr], ymm0
//
// 【尾部处理（Tail Handling）】
//   当 N 不是 8 的倍数时，最后 N%8 个元素不能用 AVX2 处理
//   必须用标量代码处理尾部，否则会越界访问！
//   这是手写 SIMD 代码最容易出 bug 的地方。
//
// 【性能基准（仅供参考）】
//   M=N=K=1024：约 30-80 GFLOPS（比 V2 快 2-4x）
// ============================================================

#include "../include/gemm.h"
#include <immintrin.h>  // AVX2 Intrinsics 头文件（包含 _mm256_* 系列函数）
#include <cstring>      // memset

void gemm_v4(const float* A, const float* B, float* C, int M, int N, int K) {
    // 清零 C 矩阵
    memset(C, 0, (size_t)M * N * sizeof(float));

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            // 【寄存器广播】
            // 将标量 A[i][k] 广播到 256 位寄存器的 8 个位置
            // 内存只读一次，然后供 j 循环的每次迭代复用
            //
            // 内存中：A[i*K+k] = 单个 float，例如 0.5
            // 广播后：ymm_a = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            __m256 ymm_a = _mm256_set1_ps(A[i * K + k]);

            // 【向量化内层 j 循环】
            // 每次迭代处理 8 个连续的 float（j, j+1, ..., j+7）
            // 必须确保 j+7 < N，否则越界！（用 j <= N-8 保证）
            int j = 0;
            for (; j <= N - 8; j += 8) {
                // 步骤1：从内存加载 C[i][j..j+7]（8个连续 float）
                // 使用 loadu（非对齐），因为我们无法保证 j 是 8 的倍数对齐
                // 当确认 C 的起始地址是 32 字节对齐，且 j 是 8 的倍数时，
                // 可改用 _mm256_load_ps（速度略快）
                __m256 ymm_c = _mm256_loadu_ps(&C[i * N + j]);

                // 步骤2：从内存加载 B[k][j..j+7]（8个连续 float）
                // k 固定，j 递增 → 行访问 → stride=1 → 连续内存 ✓
                __m256 ymm_b = _mm256_loadu_ps(&B[k * N + j]);

                // 步骤3：融合乘加 FMA
                // ymm_c = ymm_a * ymm_b + ymm_c
                // 即：C[i][j..j+7] += A[i][k] * B[k][j..j+7]
                //
                // 为什么用 fmadd 而不是 mul + add？
                //   - 一条指令完成两个操作，减少指令数
                //   - 中间结果保持更高精度（不截断）
                //   - 现代 CPU 上 FMA 延迟仅比 mul 多约 1 cycle，但吞吐相同
                ymm_c = _mm256_fmadd_ps(ymm_a, ymm_b, ymm_c);

                // 步骤4：将结果写回内存
                // 同样使用非对齐写（storeu）
                _mm256_storeu_ps(&C[i * N + j], ymm_c);
            }

            // 【尾部处理（Tail Loop）】
            // 处理 N % 8 个剩余元素（当 N 不是 8 的倍数时）
            // 退回到标量计算，确保正确性
            //
            // 注意：a_scalar 和 ymm_a 使用的是同一个 A[i][k] 的值
            // 只是格式不同（标量 vs 向量寄存器）
            float a_scalar = A[i * K + k];
            for (; j < N; j++) {
                C[i * N + j] += a_scalar * B[k * N + j];
            }
        }
    }
}
