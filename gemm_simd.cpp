// ============================================================
// 版本4：AVX2 SIMD 手写向量化
// ============================================================
// 背景：为什么需要手写SIMD？
//
// 编译器的自动向量化有局限：
//   1. 编译器不确定指针是否对齐（aliasing问题）
//   2. 编译器不知道数组是否16/32字节对齐
//   3. 某些复杂的内存访问模式编译器无法向量化
//   → 手写SIMD可以在编译器做不到的地方接管控制权
//
// AVX2基础知识：
//   SIMD = Single Instruction Multiple Data
//   AVX2寄存器：256位宽 = 可同时处理 8个float（32位）
//   相当于一条指令做了8次乘法/加法
//   理论上比标量代码快8倍
//
// 关键指令（面试时能说出来很加分）：
//   _mm256_load_ps    : 从对齐内存加载8个float到寄存器
//   _mm256_loadu_ps   : 从非对齐内存加载（比load慢一点）
//   _mm256_set1_ps    : 把一个float广播到8个位置
//   _mm256_fmadd_ps   : a*b+c 融合乘加（一条指令！）
//   _mm256_store_ps   : 把寄存器内容写回内存
//   _mm256_add_ps     : 8个float同时相加
//
// 编译时需要开启AVX2支持：
//   g++ -O3 -mavx2 -mfma gemm_simd.cpp
// ============================================================

#include "gemm.h"
#include <immintrin.h>  // AVX2头文件

void gemm_simd_avx2(const float* A, const float* B, float* C, int n) {
    memset(C, 0, n * n * sizeof(float));

    // 采用分块+SIMD结合的方案
    // 内层j循环每次处理8个元素（AVX2一次处理8个float）
    const int TILE_I = 32;
    const int TILE_K = 32;

    for (int i = 0; i < n; i += TILE_I) {
        for (int k = 0; k < n; k += TILE_K) {
            int i_end = std::min(i + TILE_I, n);
            int k_end = std::min(k + TILE_K, n);

            for (int ii = i; ii < i_end; ii++) {
                for (int kk = k; kk < k_end; kk++) {

                    // 把 A[ii][kk] 广播到一个256位寄存器的8个位置
                    // 例如 a_val = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                    __m256 a_val = _mm256_set1_ps(A[ii * n + kk]);

                    // j循环：每次处理8个float
                    int j = 0;
                    for (; j <= n - 8; j += 8) {
                        // 加载 C[ii][j..j+7] 到寄存器
                        __m256 c_vec = _mm256_loadu_ps(&C[ii * n + j]);

                        // 加载 B[kk][j..j+7] 到寄存器
                        __m256 b_vec = _mm256_loadu_ps(&B[kk * n + j]);

                        // 执行 c_vec = a_val * b_vec + c_vec
                        // FMA: Fused Multiply-Add，一条指令完成，比分开算精度更好
                        c_vec = _mm256_fmadd_ps(a_val, b_vec, c_vec);

                        // 写回内存
                        _mm256_storeu_ps(&C[ii * n + j], c_vec);
                    }

                    // 处理n不是8的倍数时的余数部分（标量处理）
                    float a_scalar = A[ii * n + kk];
                    for (; j < n; j++) {
                        C[ii * n + j] += a_scalar * B[kk * n + j];
                    }
                }
            }
        }
    }
}

// ============================================================
// 对比实验：让编译器自动向量化 vs 手写SIMD
// ============================================================
// 运行方式：
//   1. 用godbolt.org分别编译 gemm_ikj 和 gemm_simd_avx2
//   2. 查看汇编，对比指令类型
//   3. 在benchmark中对比两者性能
//
// 预期结果：
//   - ikj + -O3: 编译器可能生成128位SSE指令（vmovss, vfmadd）
//   - 手写AVX2:  确定使用256位指令（vmovups, vfmadd256）
//   - 手写版本通常比编译器自动版本快1.5~2倍
//
// 面试话术：
//   "我发现编译器在处理非对齐内存时保守地使用了128位指令，
//    通过手写AVX2 intrinsics强制使用256位指令，
//    结合FMA融合乘加，相比编译器自动向量化额外提升了约X倍。"
// ============================================================
