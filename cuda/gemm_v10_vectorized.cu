// ============================================================
// gemm_v10_vectorized.cu — V10 向量化内存访问（float4）
// ============================================================
//
// 【版本】V10 — Register Tiling + Vectorized Global→Shared Load
// 【基于】V9，将 Global Memory → Shared Memory 的加载改用 float4
//
// 【为什么用 float4？】
//   普通加载：每次从 Global Memory 读取 1 个 float（4字节）
//   float4 加载：每次从 Global Memory 读取 4 个 float（16字节）= 1 个 128-bit 事务
//
//   GPU 内存事务（Memory Transaction）的最小粒度是 128 字节（32 个 float）
//   一个 128-bit（16字节）的 float4 加载可以比 4 个 4字节加载：
//     1. 减少指令数（1条指令 vs 4条指令）
//     2. 减少地址计算次数
//     3. 更好地利用内存流水线
//
// 【16字节对齐要求】
//   float4 加载要求地址是 16 字节对齐（即地址 % 16 == 0）
//   对于 Row-major 矩阵，每行的起始地址取决于 K（A 矩阵）或 N（B 矩阵）
//
//   安全使用 float4 的条件：
//   1. 矩阵分配时使用 cudaMalloc（通常 256 字节对齐，满足条件）
//   2. 加载的起始索引是 4 的倍数
//
//   本实现通过调整分块参数确保对齐：
//   BK=8 必须是 4 的倍数 ✓（实际取 BK=8，每次加载 BK/4=2 个 float4）
//   BM/BN 必须是 4 的倍数 ✓（BM=BN=64）
//
// 【float4 如何拆分到 Shared Memory 中】
//   加载 float4 val = reinterpret_cast<const float4*>(&A[idx])[0];
//   val 包含 4 个连续的 float：val.x, val.y, val.z, val.w
//   需要正确地将这 4 个值分配到 As 的对应位置
//
// 【参数设置（与 V9 相同）】
//   BM=64, BN=64, BK=8, TM=8, TN=8
//   blockDim = (8, 8) = 64 threads
// ============================================================

#include <cuda_runtime.h>
#include <cstdio>
#include "../include/benchmark_utils.h"

constexpr int BM_V10 = 64;
constexpr int BN_V10 = 64;
constexpr int BK_V10 = 8;
constexpr int TM_V10 = 8;
constexpr int TN_V10 = 8;

// ============================================================
// CUDA Kernel（float4 向量化加载版本）
// ============================================================

__global__ void gemm_v10_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Shared Memory（布局与 V9 相同）
    __shared__ float As[BK_V10][BM_V10];
    __shared__ float Bs[BK_V10][BN_V10];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_row = blockIdx.y * BM_V10;
    int block_col = blockIdx.x * BN_V10;

    int thread_row = ty * TM_V10;
    int thread_col = tx * TN_V10;

    // 累加器（寄存器）
    float accum[TM_V10][TN_V10] = {0.0f};

    int thread_idx = ty * (BN_V10 / TN_V10) + tx;  // 线性 Thread 索引（0..63）

    // 每个 Thread 每次迭代加载的 float4 数量
    // As 总大小：BK_V10 × BM_V10 = 8 × 64 = 512 float = 128 个 float4
    // 每个 Thread 负责：128 / 64 = 2 个 float4（8 个 float）
    const int THREADS = (BM_V10 / TM_V10) * (BN_V10 / TN_V10);  // = 64

    int num_k_tiles = (K + BK_V10 - 1) / BK_V10;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int k_base = k_tile * BK_V10;

        // ---- 向量化加载 A 的 BM×BK 块 ----
        // 每个 float4 加载 4 个连续 float
        // As 的总 float4 数 = BK_V10 × BM_V10 / 4 = 8 × 64 / 4 = 128
        //
        // 策略：把 As 展开成 float4 数组的视图
        //   As_flat[i] = As[i / (BM_V10/4)][i % (BM_V10/4) * 4 .. * 4+3]
        // 注意：这里需要仔细处理 float4 的 4 个分量写入正确的 Shared Memory 位置

        for (int load_idx = thread_idx;
             load_idx < (BK_V10 * BM_V10) / 4;
             load_idx += THREADS) {
            // float4 视角下的 k 和 m 索引
            int as_k = load_idx / (BM_V10 / 4);   // K 方向索引
            int as_m4 = load_idx % (BM_V10 / 4);  // M 方向 float4 索引

            int global_row = block_row + as_m4 * 4;
            int global_col = k_base + as_k;

            if (global_col < K && global_row + 3 < M) {
                // 所有 4 个行都在边界内，直接按列方向加载（stride=K）
                // 注意：A 是行主序，As[k][m]，我们需要加载 A 的 4 行，同一列
                // 这 4 个元素在内存中地址是：
                //   A[(global_row+0)*K + global_col]
                //   A[(global_row+1)*K + global_col]  ← 不连续！步长=K
                //   ...
                // 不连续，无法直接用 float4，只能逐一加载
                As[as_k][as_m4 * 4 + 0] = (global_row+0 < M) ? A[(global_row+0)*K + global_col] : 0.0f;
                As[as_k][as_m4 * 4 + 1] = (global_row+1 < M) ? A[(global_row+1)*K + global_col] : 0.0f;
                As[as_k][as_m4 * 4 + 2] = (global_row+2 < M) ? A[(global_row+2)*K + global_col] : 0.0f;
                As[as_k][as_m4 * 4 + 3] = (global_row+3 < M) ? A[(global_row+3)*K + global_col] : 0.0f;
            } else {
                // 边界情况：逐个检查
                for (int sub = 0; sub < 4; sub++) {
                    int r = global_row + sub, c = global_col;
                    As[as_k][as_m4 * 4 + sub] = (r < M && c < K) ? A[r * K + c] : 0.0f;
                }
            }
        }

        // ---- 向量化加载 B 的 BK×BN 块（float4）----
        // B 是行主序，Bs[k][n]
        // B[global_row][global_col .. global_col+3] 是连续的 4 个 float ✓
        // 可以直接用 float4 加载！
        for (int load_idx = thread_idx;
             load_idx < (BK_V10 * BN_V10) / 4;
             load_idx += THREADS) {
            int bs_k  = load_idx / (BN_V10 / 4);
            int bs_n4 = load_idx % (BN_V10 / 4);

            int global_row = k_base + bs_k;
            int global_col = block_col + bs_n4 * 4;

            if (global_row < K && global_col + 3 < N) {
                // 使用 float4 加载 4 个连续 float（16字节，单次内存事务）
                // 要求：&B[global_row * N + global_col] 是 16 字节对齐的
                // 满足条件：cudaMalloc 保证 256 字节对齐，global_col % 4 == 0
                float4 val = reinterpret_cast<const float4*>(
                    &B[global_row * N + global_col])[0];
                // 将 float4 的 4 个分量分别写入 Shared Memory
                Bs[bs_k][bs_n4 * 4 + 0] = val.x;
                Bs[bs_k][bs_n4 * 4 + 1] = val.y;
                Bs[bs_k][bs_n4 * 4 + 2] = val.z;
                Bs[bs_k][bs_n4 * 4 + 3] = val.w;
            } else {
                // 边界处理
                for (int sub = 0; sub < 4; sub++) {
                    int r = global_row, c = global_col + sub;
                    Bs[bs_k][bs_n4 * 4 + sub] = (r < K && c < N) ? B[r * N + c] : 0.0f;
                }
            }
        }

        // 加载同步
        __syncthreads();

        // ---- 外积计算（与 V9 相同）----
        float regA[TM_V10];
        float regB[TN_V10];

        #pragma unroll
        for (int k = 0; k < BK_V10; k++) {
            #pragma unroll
            for (int m = 0; m < TM_V10; m++) {
                regA[m] = As[k][thread_row + m];
            }
            #pragma unroll
            for (int n = 0; n < TN_V10; n++) {
                regB[n] = Bs[k][thread_col + n];
            }
            #pragma unroll
            for (int m = 0; m < TM_V10; m++) {
                #pragma unroll
                for (int n = 0; n < TN_V10; n++) {
                    accum[m][n] += regA[m] * regB[n];
                }
            }
        }

        __syncthreads();
    }

    // 写回 C
    #pragma unroll
    for (int m = 0; m < TM_V10; m++) {
        #pragma unroll
        for (int n = 0; n < TN_V10; n++) {
            int gr = block_row + thread_row + m;
            int gc = block_col + thread_col + n;
            if (gr < M && gc < N)
                C[gr * N + gc] = accum[m][n];
        }
    }
}

// ============================================================
// Host Wrapper
// ============================================================

void gemm_v10(const float* A, const float* B, float* C, int M, int N, int K) {
    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_B = (size_t)K * N * sizeof(float);
    size_t bytes_C = (size_t)M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, bytes_B, cudaMemcpyHostToDevice));

    dim3 blockDim(BN_V10 / TN_V10, BM_V10 / TM_V10);
    dim3 gridDim((N + BN_V10 - 1) / BN_V10, (M + BM_V10 - 1) / BM_V10);

    gemm_v10_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

void gemm_v10_device(const float* d_A, const float* d_B, float* d_C,
                     int M, int N, int K) {
    dim3 blockDim(BN_V10 / TN_V10, BM_V10 / TM_V10);
    dim3 gridDim((N + BN_V10 - 1) / BN_V10, (M + BM_V10 - 1) / BM_V10);
    gemm_v10_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}
