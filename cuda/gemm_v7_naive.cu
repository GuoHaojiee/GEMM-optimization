// ============================================================
// gemm_v7_naive.cu — V7 CUDA Naive（GPU 基准版本）
// ============================================================
//
// 【版本】V7 — CUDA Naive
// 【Thread 配置】Block: 16×16（256 threads/block），Grid: ceil(N/16) × ceil(M/16)
//
// 【核心思路】
//   最直接的 GPU 并行化：每个 Thread 计算 C 的一个元素
//   Thread(tx, ty) 对应 C[row][col]，其中 row = blockIdx.y*16+ty, col = blockIdx.x*16+tx
//
// 【访存分析（性能瓶颈）】
//   每个 Thread 需要读取：
//     A 的一整行：A[row][0..K-1]（K 次 Global Memory 读）
//     B 的一整列：B[0..K-1][col]（K 次 Global Memory 读）
//
//   同一 Warp（32个Thread，同一行ty，tx=0..31）的访存模式：
//     访问 B[k][0], B[k][1], ..., B[k][31]  → 连续 → 合并访问（Coalesced）✓
//     访问 A[0][k], A[1][k], ..., A[31][k]  → 步长=K → 非合并（Non-coalesced）✗
//
//   注意：CUDA 中 Warp 内 Thread 按 tx 方向排列
//   同一 Warp = 同一 blockIdx，ty 相同，tx 连续（0-31 是同一 Warp 的前半或后半）
//   → 实际上 A 的访问也是合并的（不同 Thread 访问同一 k 下不同 row，但由于
//     A 是 M×K 的矩阵，同一 Warp 的 Thread 有相同的 tx，访问的是不同行 A[row][k]，
//     步长为 K，不合并）
//
// 【Global Memory 访问量】
//   每个 Thread 读 A: K 次，读 B: K 次
//   总读取量 = M × N × 2K × 4 字节（远超矩阵实际大小！因为没有数据复用）
//
// 【Roofline 算术强度】
//   FLOPs = 2 × M × N × K
//   Memory = 4 × (M×K×N + K×N×M) × 4 字节（A 读 N 次，B 读 M 次）
//   AI ≈ 0.25 FLOP/Byte → 严重 Memory Bound
//
// 【Host Wrapper 说明】
//   gemm_v7() 是供外部调用的接口，内部完成：
//   1. cudaMalloc 分配 GPU 内存
//   2. cudaMemcpy 从 CPU 拷贝到 GPU
//   3. 启动 Kernel
//   4. cudaMemcpy 从 GPU 拷回 CPU
//   5. cudaFree 释放 GPU 内存
//   这样调用者无需关心 GPU 内存管理
// ============================================================

#include <cuda_runtime.h>
#include <cstdio>
#include "../include/benchmark_utils.h"  // CUDA_CHECK 宏

// ============================================================
// CUDA Kernel：每个 Thread 计算 C 的一个元素
// ============================================================

__global__ void gemm_v7_kernel(
    const float* __restrict__ A,   // M×K，行主序
    const float* __restrict__ B,   // K×N，行主序
    float* __restrict__ C,         // M×N，行主序
    int M, int N, int K)
{
    // 当前 Thread 对应 C 矩阵的行列号
    // blockDim.x = blockDim.y = BLOCK_SIZE = 16
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 行

    // 边界检查：Grid 大小向上取整，可能产生超出矩阵边界的 Thread
    if (row >= M || col >= N) return;

    // 计算 C[row][col] = Σ(k=0..K-1) A[row][k] * B[k][col]
    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
        // A[row][k]：row 由 blockIdx.y 决定（不同 Warp 访问不同行）
        //            k 在循环中递增 → 同一 Thread 按行访问 A ✓
        //            但同一 Warp 的 Thread（ty 相同，tx=0..31）访问的是 A[row][k]
        //            全部 ty 相同 → 相同的 row → 重复访问同一地址 → 广播（broadcast）
        //            （硬件会优化为一次加载后广播给 16 个 Thread）
        //
        // B[k][col]：k 固定，col 变化（tx=0..15） → B 的第 k 行的 16 个连续元素
        //            同一 Warp 访问 B[k][tx+offset]，连续内存 → 合并访问 ✓
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

// ============================================================
// Host Wrapper 函数
// ============================================================

void gemm_v7(const float* A, const float* B, float* C, int M, int N, int K) {
    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_B = (size_t)K * N * sizeof(float);
    size_t bytes_C = (size_t)M * N * sizeof(float);

    // 分配 GPU 显存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    // 从 CPU 拷贝数据到 GPU（Host to Device）
    CUDA_CHECK(cudaMemcpy(d_A, A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, bytes_B, cudaMemcpyHostToDevice));

    // 启动 Kernel
    // Block 大小：16×16 = 256 threads（接近每个 SM 最大 1024 threads 的 1/4）
    // Grid 大小：向上取整，确保覆盖所有 M×N 的元素
    const int BLOCK = 16;
    dim3 blockDim(BLOCK, BLOCK);
    dim3 gridDim((N + BLOCK - 1) / BLOCK,   // x 方向：列
                 (M + BLOCK - 1) / BLOCK);   // y 方向：行

    gemm_v7_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());    // 检查 Kernel 启动错误
    CUDA_CHECK(cudaDeviceSynchronize()); // 等待 GPU 完成

    // 从 GPU 拷贝结果回 CPU（Device to Host）
    CUDA_CHECK(cudaMemcpy(C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    // 释放显存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================
// Kernel 版本（供 benchmark_cuda.cu 直接调用，避免重复内存分配）
// ============================================================
void gemm_v7_device(const float* d_A, const float* d_B, float* d_C,
                    int M, int N, int K) {
    const int BLOCK = 16;
    dim3 blockDim(BLOCK, BLOCK);
    dim3 gridDim((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);
    gemm_v7_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}
