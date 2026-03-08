// ============================================================
// CUDA版本1：Naive GEMM
// ============================================================
// 知识点回顾（面试必备）：
//
// GPU线程层次结构：
//   Grid → Block → Thread
//   一个Grid包含多个Block（最多2^31个）
//   一个Block包含多个Thread（最多1024个）
//   同一Block内的Thread共享Shared Memory和L1 Cache
//
// 内存层次结构（从慢到快）：
//   Global Memory: 显卡上的DRAM，几十GB，延迟~800cycles ← 最慢
//   L2 Cache:      全GPU共享，延迟~200cycles
//   Shared Memory: 每个Block独享，~48KB，延迟~20cycles  ← 快
//   Register:      每个Thread独享，延迟1cycle           ← 最快
//
// Naive版本的问题：
//   每个Thread独立计算C[i][j]的一个元素
//   需要读A的一整行 + B的一整列
//   相邻Thread读取B的相邻列，这些列在内存中不连续
//   → 无法合并内存访问（Coalescing），效率低
// ============================================================

#include <cuda_runtime.h>
#include <stdio.h>

#define N 512
#define BLOCK_SIZE 16  // 每个Block是16x16=256个Thread

// CUDA错误检查宏（实际开发必备）
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================
// Kernel：每个Thread计算C矩阵的一个元素
// ============================================================
__global__ void gemm_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int n)
{
    // 计算当前Thread负责的行列号
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) return;  // 边界检查

    float sum = 0.0f;
    for (int k = 0; k < n; k++) {
        // A[row][k]: row固定，k递增 → 同一Warp的Thread访问不同row
        //            → 访问模式：A[0][k], A[1][k], ... A[31][k] → 列访问 → 非合并
        // B[k][col]: col不同的Thread访问B的同一行的不同列
        //            → 访问模式：B[k][0], B[k][1], ... B[k][31] → 行访问 → 合并 ✓
        sum += A[row * n + k] * B[k * n + col];
    }

    C[row * n + col] = sum;
}

void run_gemm_naive(const float* h_A, const float* h_B, float* h_C, int n) {
    float *d_A, *d_B, *d_C;
    size_t bytes = n * n * sizeof(float);

    // 分配显存
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // 拷贝数据到GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // 启动配置：Grid大小 = ceil(N/BLOCK_SIZE) x ceil(N/BLOCK_SIZE)
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // 启动Kernel
    gemm_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());  // 等待GPU完成

    // 拷贝结果回CPU
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // 释放显存
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
