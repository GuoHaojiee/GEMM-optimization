// ============================================================
// gemm_v8_shared.cu — V8 CUDA Shared Memory Tiling
// ============================================================
//
// 【版本】V8 — Shared Memory 分块（TILE_SIZE=16）
// 【Thread 配置】Block: 16×16，Grid: ceil(N/16) × ceil(M/16)
//
// 【核心思路：为什么 Shared Memory 能提速？】
//   V7（Naive）中，所有数据访问都走 Global Memory（延迟 ~400-800 cycles）
//   Shared Memory 与 L1 Cache 共享物理资源：
//     - 容量：每个 SM 约 48KB（可配置）
//     - 延迟：~20-40 cycles（比 Global Memory 快 20x）
//     - 带宽：理论上约 19 TB/s（全 GPU，比 Global Memory 带宽大 10x）
//
//   Tiling 策略：
//   1. 把 A 和 B 沿 K 维度分成 TILE_SIZE 宽的切片
//   2. 每次处理一个 K-tile：
//      a. 协同加载：Block 内 16×16 个 Thread 协作，将 A 的 16×16 块
//         和 B 的 16×16 块从 Global Memory 搬到 Shared Memory
//      b. 计算：在 Shared Memory 中做 16×16 次 FMA（延迟仅 ~20 cycles）
//   3. 滑动 K 方向，重复上述过程
//
// 【__syncthreads() 的两次使用（面试重点！）】
//   第一次 __syncthreads()（加载后）：
//     目的：等待 Block 内所有 Thread 完成加载
//     原因：Thread 0 可能很快加载完 As[0][0]，就要开始计算
//           但 Thread 255 还没加载完 As[15][15]
//           如果没有同步，Thread 0 读到的 As[15][15] 可能是未初始化的值
//     结果：确保 As 和 Bs 被完整填充后，所有 Thread 才开始计算
//
//   第二次 __syncthreads()（计算后）：
//     目的：防止"快"的 Thread 在"慢"的 Thread 还在读当前 tile 时，
//           抢先进入下一轮循环，覆盖 Shared Memory 中的数据
//     原因：Shared Memory 被所有 Thread 共享，下一轮循环会覆盖 As/Bs
//           若没有同步，慢线程可能读到下一 tile 的数据（数据污染）
//     结果：确保所有 Thread 都完成计算后，才开始加载下一个 K-tile
//
//   常见错误：只用一次 __syncthreads()，或把位置放错
//
// 【内存合并访问（Coalescing）】
//   协同加载 A[row][tile*T + tx]：
//     同一 Warp（ty 相同，tx=0..15）访问 A 的第 row 行的连续 16 个元素 ✓
//   协同加载 B[(tile*T + ty)][col]：
//     同一 Warp（ty 相同，tx=0..15）访问 B 的第 (tile*T+ty) 行的连续 16 个元素 ✓
//   → 加载阶段内存合并！
//
// 【性能提升量化】
//   Global Memory 访问减少因子 = TILE_SIZE = 16
//   原来：每个元素被读取 N/TILE 次（每轮 tile 各独立读取）
//   实际：由于 Shared Memory 的复用，每个 A/B 元素只被读取一次（理想情况）
//   Naive vs Shared：通常提速 3-8x
//
// 【Shared Memory 容量限制】
//   As + Bs = 2 × TILE_SIZE × TILE_SIZE × 4 字节
//   TILE_SIZE=16: 2 × 16 × 16 × 4 = 2 KB（远小于 48 KB 上限）
//   TILE_SIZE=32: 2 × 32 × 32 × 4 = 8 KB（仍然可行）
// ============================================================

#include <cuda_runtime.h>
#include <cstdio>
#include "../include/benchmark_utils.h"

#define TILE_SIZE 16  // 分块大小（同时也是 Block 的边长）

// ============================================================
// CUDA Kernel
// ============================================================

__global__ void gemm_v8_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // 声明 Shared Memory
    // __shared__ 关键字：分配在每个 SM 的 on-chip SRAM 上
    // Block 内所有 Thread 共享这两个数组
    __shared__ float As[TILE_SIZE][TILE_SIZE];  // A 的分块缓存
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];  // B 的分块缓存

    int tx = threadIdx.x;  // Thread 在 Block 内的 x 坐标（列方向）
    int ty = threadIdx.y;  // Thread 在 Block 内的 y 坐标（行方向）

    // 当前 Thread 对应 C 矩阵的全局行列号
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // 累加器（存放 C[row][col] 的部分和）
    // 在 K 方向滑动过程中逐步累加
    float sum = 0.0f;

    // 遍历 K 方向的所有 tile
    // num_tiles = ceil(K / TILE_SIZE)，向上取整处理 K 不是 TILE_SIZE 倍数的情况
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < num_tiles; tile++) {

        // ============================================================
        // 阶段1：协同加载 Global Memory → Shared Memory
        // ============================================================
        // 每个 Thread 负责加载 As 和 Bs 中的一个元素
        // 16×16 个 Thread 恰好覆盖 16×16 的矩阵块

        // 加载 As[ty][tx] = A[row][tile*TILE_SIZE + tx]
        // 即：A 矩阵的第 row 行，第 (tile*TILE_SIZE + tx) 列
        int a_col = tile * TILE_SIZE + tx;
        if (row < M && a_col < K)
            As[ty][tx] = A[row * K + a_col];
        else
            As[ty][tx] = 0.0f;  // 边界填零（等效于 padding）

        // 加载 Bs[ty][tx] = B[tile*TILE_SIZE + ty][col]
        // 即：B 矩阵的第 (tile*TILE_SIZE + ty) 行，第 col 列
        int b_row = tile * TILE_SIZE + ty;
        if (b_row < K && col < N)
            Bs[ty][tx] = B[b_row * N + col];
        else
            Bs[ty][tx] = 0.0f;

        // ============================================================
        // 第一次 __syncthreads()：加载同步屏障
        // ============================================================
        // 作用：等待 Block 内所有 Thread 完成上面的 Global→Shared 加载
        //
        // 为什么必须在这里同步？
        //   Thread(0,0) 可能很快完成加载，立即开始下面的计算循环
        //   而 Thread(15,15) 可能还在等待 Global Memory 响应
        //   若没有这个 syncthreads，Thread(0,0) 读到的 As[*][15] 和 Bs[15][*]
        //   可能是 Thread(15,*) 还没写入的垃圾数据
        //
        // 代价：Block 内所有 Thread 都要在此等待最慢的那个 Thread
        // 典型延迟：几十到几百 cycles（相比数据加载延迟，这个代价可以接受）
        __syncthreads();

        // ============================================================
        // 阶段2：在 Shared Memory 中计算（延迟仅 ~20 cycles）
        // ============================================================
        // 计算 C[row][col] 的部分和（当前 tile 的贡献）
        // TILE_SIZE 次乘加，全部读取 Shared Memory（无 Global Memory 访问）
        //
        // #pragma unroll：提示编译器展开这个定长循环
        // 效果：消除循环控制开销（比较、跳转指令），提升指令级并行
        // TILE_SIZE=16 时展开为 16 条独立的 FMA 指令
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            // As[ty][k]：当前 Thread 所在行（ty）的第 k 个元素
            // Bs[k][tx]：当前 Thread 所在列（tx）的第 k 个元素
            // 读 Shared Memory：延迟 ~20 cycles vs Global Memory ~400-800 cycles
            sum += As[ty][k] * Bs[k][tx];
        }

        // ============================================================
        // 第二次 __syncthreads()：计算同步屏障
        // ============================================================
        // 作用：等待 Block 内所有 Thread 完成当前 tile 的计算
        //
        // 为什么必须在这里同步？
        //   在下一轮循环开始时，Block 内的 Thread 会开始覆盖 As 和 Bs
        //   （写入下一个 K-tile 的数据）
        //   若某个 Thread 已进入下一轮的加载阶段，而另一个 Thread 还在
        //   读取当前 tile 的 As/Bs，就会发生数据竞争（Data Race）
        //   快的 Thread 会覆盖慢的 Thread 还在读取的 Shared Memory！
        //
        // 没有这个 syncthreads 的后果：
        //   间歇性的计算错误（非确定性 bug，极难复现和调试）
        __syncthreads();
    }

    // 写回全局内存
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ============================================================
// Host Wrapper
// ============================================================

void gemm_v8(const float* A, const float* B, float* C, int M, int N, int K) {
    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_B = (size_t)K * N * sizeof(float);
    size_t bytes_C = (size_t)M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, bytes_B, cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    gemm_v8_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

void gemm_v8_device(const float* d_A, const float* d_B, float* d_C,
                    int M, int N, int K) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_v8_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}
