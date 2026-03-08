// ============================================================
// CUDA版本2：Shared Memory 优化（核心优化，面试重点！）
// ============================================================
// 优化思路：
//   Naive版本中，计算C[i][j]需要读A的第i行和B的第j列
//   如果相邻Thread计算同一行A的不同C元素，A的数据可以共享
//
// Tiling策略（和CPU分块思路相同，但利用的是Shared Memory）：
//   1. 把A和B分成 TILE x TILE 的小块
//   2. 每个Block负责计算C的一个TILE x TILE 的块
//   3. 协同加载：Block内所有Thread一起把A块和B块搬到Shared Memory
//   4. 在Shared Memory中做计算（延迟只有~20 cycles）
//   5. 滑动到下一个块，重复
//
// 性能分析：
//   Naive: 每个Thread读A和B各n次 → 2*n次Global Memory访问
//   优化后: 每TILE次计算只需要一次Global Memory访问
//           Global Memory访问次数减少 TILE 倍
//   TILE=16: 减少16倍Global Memory访问 → 显著提升
//
// 内存合并访问（Coalescing）：
//   同一Warp（32个Thread）同时访问连续内存 → 硬件合并为1次事务
//   本实现中协同加载时确保了合并访问
// ============================================================

#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16  // Shared Memory块大小（受限于每个Block的Shared Memory容量）
                      // TILE_SIZE=16: 每个A块+B块 = 2*16*16*4 = 2KB，远小于48KB上限

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

__global__ void gemm_shared_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int n)
{
    // 声明Shared Memory（__shared__关键字）
    // 这块内存Block内所有Thread共享
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // 当前Thread在Block内的位置
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 当前Thread负责的全局行列号
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // 遍历所有K方向的分块
    // 每次迭代：加载一个A块和B块到Shared Memory，然后做TILE_SIZE次乘加
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; tile++) {

        // ---- 协同加载阶段 ----
        // 每个Thread负责加载A块和B块中的一个元素
        // 这是Block内所有256个Thread同时在做的事

        // 加载A[row][tile*TILE+tx]
        if (row < n && (tile * TILE_SIZE + tx) < n)
            As[ty][tx] = A[row * n + tile * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        // 加载B[tile*TILE+ty][col]
        if ((tile * TILE_SIZE + ty) < n && col < n)
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * n + col];
        else
            Bs[ty][tx] = 0.0f;

        // ---- 同步屏障 ----
        // 必须等所有Thread都完成加载，再开始计算
        // 否则可能读到其他Thread还没加载完的数据
        __syncthreads();

        // ---- 计算阶段 ----
        // 使用Shared Memory中的数据做TILE_SIZE次乘加
        // 延迟~20 cycles，远低于Global Memory的~800 cycles
        #pragma unroll  // 提示编译器展开循环，减少循环控制开销
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // ---- 再次同步 ----
        // 等所有Thread完成计算，再进入下一轮加载
        // 防止快的Thread在慢的Thread还在计算时覆盖Shared Memory
        __syncthreads();
    }

    // 写回结果
    if (row < n && col < n)
        C[row * n + col] = sum;
}

// ============================================================
// 使用 Nsight Compute 分析性能瓶颈（面试加分项）
// ============================================================
// 命令：ncu --metrics sm__througput.avg,
//             l1tex__t_bytes.sum,
//             dram__bytes.sum
//       ./gemm_cuda
//
// 关键指标含义：
//   sm__throughput     : SM利用率，越高越好
//   l1tex__t_bytes     : L1 Cache传输量
//   dram__bytes        : 实际访问显存量（越小越好）
//   achieved_occupancy : 活跃Warp比例，越高越好
//
// 面试话术：
//   "用Nsight Compute分析后发现，Naive版本的dram__bytes
//    是理论值的16倍（因为无法合并访问），
//    优化后降低到接近理论值，带宽利用率从12%提升到76%"
// ============================================================

void run_gemm_shared(const float* h_A, const float* h_B, float* h_C, int n) {
    float *d_A, *d_B, *d_C;
    size_t bytes = n * n * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE,
                 (n + TILE_SIZE - 1) / TILE_SIZE);

    gemm_shared_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
