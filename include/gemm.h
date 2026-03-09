#pragma once
// ============================================================
// gemm.h — 所有 GEMM 版本的统一接口声明
// ============================================================
// 矩阵规格（Row-major 行主序存储）：
//   A: M×K，A[i][k] = A[i*K + k]
//   B: K×N，B[k][j] = B[k*N + j]
//   C: M×N，C[i][j] = C[i*N + j]
//
// 统一函数签名（全项目所有版本）：
//   void gemm_vN(const float* A, const float* B, float* C, int M, int N, int K);
//
// 设计原则：
//   - 所有版本语义相同：C = A * B（先清零C再累加）
//   - 边界安全：M/N/K 不要求是块大小的倍数
//   - 线程安全：无全局状态，可并发调用
// ============================================================

#include <algorithm>  // std::min
#include <cstring>    // memset

// ============================================================
// CPU 版本（V1-V6）
// ============================================================

/**
 * V1 — Naive ijk 三重循环（基准版本）
 * 访存特征：B 矩阵列访问，大量 Cache miss
 * 时间复杂度：O(MNK)
 */
void gemm_v1(const float* A, const float* B, float* C, int M, int N, int K);

/**
 * V2 — 循环重排 ikj（Cache 友好版本）
 * 优化：最内层循环改为 j，B[k][j] 变为行访问，stride=1
 * 预期加速：相比 V1 提升 3~10x（取决于矩阵大小）
 */
void gemm_v2(const float* A, const float* B, float* C, int M, int N, int K);

/**
 * V3 — 分块 Tiling（Cache 复用版本）
 * 优化：三层分块，确保工作集适合 L1/L2 Cache
 * 块大小：BLOCK_SIZE=64，3块 = 3×64×64×4 = 48KB ≈ L1 大小
 */
void gemm_v3(const float* A, const float* B, float* C, int M, int N, int K);

/**
 * V4 — AVX2 手写 SIMD（向量化版本）
 * 优化：使用 _mm256_fmadd_ps，一次处理 8 个 float
 * 需要编译选项：-mavx2 -mfma
 */
void gemm_v4(const float* A, const float* B, float* C, int M, int N, int K);

/**
 * V5 — AVX2 + Tiling（向量化+分块组合版本）
 * 优化：V3 的分块结构 + V4 的 AVX2 内层计算
 * 通常是 CPU 端最快的版本
 */
void gemm_v5(const float* A, const float* B, float* C, int M, int N, int K);

/**
 * V6 — OpenMP + AVX2 + Tiling（多线程并行版本）
 * 优化：外层循环并行化，多核利用
 * 需要编译选项：-fopenmp
 */
void gemm_v6(const float* A, const float* B, float* C, int M, int N, int K);

// ============================================================
// CUDA 版本（V7-V11）— 声明在此，实现在 cuda/ 目录
// ============================================================

#ifdef __CUDACC__

/**
 * V7 — CUDA Naive（GPU 基准版本）
 * 每个 Thread 计算 C 的一个元素，Block 16×16
 * 问题：B 矩阵列访问，无法合并（non-coalesced）
 */
void gemm_v7(const float* A, const float* B, float* C, int M, int N, int K);

/**
 * V8 — CUDA Shared Memory Tiling
 * 协同加载 A/B 分块到 Shared Memory，TILE_SIZE=16
 * Global Memory 访问减少 TILE_SIZE 倍
 */
void gemm_v8(const float* A, const float* B, float* C, int M, int N, int K);

/**
 * V9 — CUDA Register Tiling（寄存器分块，最重要的 CUDA 优化）
 * 参数：BM=64, BN=64, BK=8, TM=8, TN=8
 * 每个 Thread 计算 TM×TN=64 个输出元素，最大化寄存器复用
 */
void gemm_v9(const float* A, const float* B, float* C, int M, int N, int K);

/**
 * V10 — CUDA Vectorized Load（向量化内存访问）
 * 在 V9 基础上，Global→Shared 加载改用 float4（16字节/次）
 * 减少内存事务数，提升带宽利用率
 */
void gemm_v10(const float* A, const float* B, float* C, int M, int N, int K);

/**
 * V11 — cuBLAS 参考实现
 * 工业级优化（double buffering, tensor core 等），作为性能上界
 */
void gemm_v11(const float* A, const float* B, float* C, int M, int N, int K);

#endif  // __CUDACC__

// ============================================================
// 常用宏定义
// ============================================================

// 块大小（CPU 分块版本）
// 3 × 64 × 64 × 4 = 49152 字节 ≈ L1 Cache 大小
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif
