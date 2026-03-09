// ============================================================
// gemm_v9_register.cu — V9 Register Tiling（寄存器分块，最重要的 CUDA 优化）
// ============================================================
//
// 【版本】V9 — Register Tiling / Thread-Level Tiling
//
// 【参数配置】
//   BM = 64：每个 Block 负责 C 矩阵的 64 行
//   BN = 64：每个 Block 负责 C 矩阵的 64 列
//   BK = 8： K 方向每次处理 8 列（决定 Shared Memory 大小和循环次数）
//   TM = 8： 每个 Thread 负责 C 的 8 行
//   TN = 8： 每个 Thread 负责 C 的 8 列
//
// 【线程配置】
//   blockDim = (BN/TN, BM/TM) = (8, 8) = 64 个 Thread
//   每个 Thread 计算 TM×TN = 8×8 = 64 个输出元素
//   gridDim  = (ceil(N/BN), ceil(M/BM))
//
// ============================================================
//
// 【为什么 V8（Shared Memory）还不够快？】
//
//   V8 中每个 Thread 只计算 C 的 1 个元素
//   对应的 Shared Memory 访问：
//     从 As 读 TILE_SIZE 次（A 的一行）
//     从 Bs 读 TILE_SIZE 次（B 的一列）
//     合计：2 × TILE_SIZE 次 Shared Memory 读
//     计算：TILE_SIZE 次 FMA
//     算术强度（相对 Shared Memory）= TILE_SIZE / (2×TILE_SIZE×4) = 1/8 FLOP/Byte
//
//   Shared Memory 带宽虽然高，但算术强度仍然不够！
//   GPU 的计算能力远超内存带宽（即使是 Shared Memory）
//
// ============================================================
//
// 【Register Tiling 的核心思路：外积展开（Outer Product）】
//
//   每个 Thread 负责 TM×TN 的 C 子矩阵
//   对每个 BK 步骤（k 维度分块）：
//     从 As 加载 TM 个元素（A 的 TM 行 × 1 列）到寄存器 regA[TM]
//     从 Bs 加载 TN 个元素（B 的 1 行 × TN 列）到寄存器 regB[TN]
//     做外积：regA[TM] ⊗ regB[TN] → 更新 accum[TM][TN]（TM×TN 次 FMA）
//
//   关键！数据复用比分析：
//     加载数据量：TM + TN = 8 + 8 = 16 个 float（从 Shared Memory 读）
//     计算量：TM × TN = 64 次 FMA
//     算术强度（相对 Shared Memory）= 64 / (16×4) = 1 FLOP/Byte
//     比 V8 的 1/8 提升了 8 倍！
//
//   另一个角度理解复用比：
//     regA[i] 被使用了 TN=8 次（与 regB 的每个元素各做一次 FMA）
//     regB[j] 被使用了 TM=8 次（与 regA 的每个元素各做一次 FMA）
//     这就是"外积"的含义：列向量 × 行向量 → 矩阵
//
// ============================================================
//
// 【Shared Memory 布局和访问模式】
//
//   As[BK][BM] = [8][64]：K 方向在外，M 方向在内
//   Bs[BK][BN] = [8][64]：K 方向在外，N 方向在内
//
//   选择这种布局的原因：
//     加载 A 时：As[k][m] 按 m 方向连续 → 合并访问 ✓
//     加载 B 时：Bs[k][n] 按 n 方向连续 → 合并访问 ✓
//     读取时：同一 Thread 按 k 方向顺序读 → 合并（连续地址）✓
//
// ============================================================
//
// 【TM/TN 选择的权衡】
//   TM/TN 越大：
//     ✓ 算术强度越高（TM×TN / (TM+TN) 随 TM/TN 增大而增大）
//     ✓ 寄存器复用更好
//     ✗ 每个 Thread 需要更多寄存器（accum[TM][TN] = TM×TN 个 float）
//       寄存器用多了 → Occupancy 下降（每个 SM 的寄存器有限）
//       Occupancy 低 → 无法隐藏内存延迟
//     ✗ Block 内 Thread 数减少（BM/TM × BN/TN），降低并行度
//
//   常见的权衡点：TM=TN=4 或 8（目前 cuBLAS 通常用更大的 tile）
//
// ============================================================
//
// 【Bank Conflict 分析】
//   Shared Memory 有 32 个 Bank，连续的 4 字节地址映射到连续的 Bank
//   当同一 Warp 的多个 Thread 访问同一 Bank 的不同地址时，发生 Bank Conflict
//
//   As[BK][BM]：同一 Warp 的 Thread（同 ty，不同 tx）访问 As[k][tx*TM + m]
//               若 TM 为奇数 → 可能 Bank Conflict
//               TM=8，BM=64：As 每行 64 元素，每 Thread 读 8 个，
//               不同 Thread 访问不同地址段 → 无冲突（stride=8，Bank 分布均匀）
// ============================================================

#include <cuda_runtime.h>
#include <cstdio>
#include "../include/benchmark_utils.h"

// ============================================================
// 超参数定义
// ============================================================

constexpr int BM = 64;  // Block 负责 C 的行数（Block M-tile size）
constexpr int BN = 64;  // Block 负责 C 的列数（Block N-tile size）
constexpr int BK = 8;   // K 方向分块大小（控制 Shared Memory 大小）
constexpr int TM = 8;   // 每个 Thread 负责 C 的行数（Thread M-tile size）
constexpr int TN = 8;   // 每个 Thread 负责 C 的列数（Thread N-tile size）

// 由上面参数导出：
// blockDim = (BN/TN, BM/TM) = (8, 8) = 64 threads
// As 大小：BK × BM = 8 × 64 = 512 float = 2KB
// Bs 大小：BK × BN = 8 × 64 = 512 float = 2KB
// 合计：4KB Shared Memory per block

// ============================================================
// CUDA Kernel
// ============================================================

__global__ void gemm_v9_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // ---- Shared Memory 声明 ----
    // As[BK][BM]：A 的 BK×BM 子块
    //   布局：k 方向为外层，m 方向为内层
    //   存储 A 矩阵中当前 Block 负责的 BM 行，宽度 BK
    __shared__ float As[BK][BM];

    // Bs[BK][BN]：B 的 BK×BN 子块
    __shared__ float Bs[BK][BN];

    // ---- Thread 在 Block 内的位置 ----
    int tx = threadIdx.x;  // x 方向：0..BN/TN-1 = 0..7
    int ty = threadIdx.y;  // y 方向：0..BM/TM-1 = 0..7

    // ---- 当前 Block 负责的 C 的起始行列 ----
    int block_row = blockIdx.y * BM;  // Block 起始行
    int block_col = blockIdx.x * BN;  // Block 起始列

    // ---- 每个 Thread 负责的 C 子块的起始行列 ----
    // Thread(tx, ty) 负责 C 的 [block_row + ty*TM .. block_row + (ty+1)*TM)
    //                          × [block_col + tx*TN .. block_col + (tx+1)*TN)
    int thread_row = ty * TM;  // 在 Block 内的行偏移
    int thread_col = tx * TN;  // 在 Block 内的列偏移

    // ---- 寄存器：每个 Thread 的累加器 ----
    // accum[TM][TN] = 8×8 = 64 个 float 寄存器
    // 这些值在 K 方向迭代过程中逐步累加
    float accum[TM][TN] = {0.0f};  // 初始化为 0

    // ---- 用于加载 A/B 到 Shared Memory 的辅助变量 ----
    // Block 内共有 BM/TM × BN/TN = 8×8 = 64 个 Thread
    // As 大小 BK×BM = 8×64 = 512 个 float，每个 Thread 加载 512/64 = 8 个
    // Bs 大小 BK×BN = 8×64 = 512 个 float，每个 Thread 加载 8 个
    int thread_idx = ty * (BN / TN) + tx;  // Thread 在 Block 内的线性索引（0..63）

    // ============================================================
    // 主循环：沿 K 维度滑动
    // ============================================================
    int num_k_tiles = (K + BK - 1) / BK;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int k_base = k_tile * BK;  // 当前 K-tile 的全局起始位置

        // ---- 协同加载 A 的 BM×BK 块到 As ----
        // 每个 Thread 加载 BM*BK/num_threads = 512/64 = 8 个元素
        // 采用线性化分配：Thread i 负责 As 的第 i, i+num_threads, ... 个元素
        for (int load_idx = thread_idx;
             load_idx < BK * BM;
             load_idx += (BM / TM) * (BN / TN)) {
            // 将线性索引转换为 As[k][m] 坐标
            int as_k = load_idx / BM;  // K 方向索引（0..BK-1）
            int as_m = load_idx % BM;  // M 方向索引（0..BM-1）

            // 计算 A 的全局坐标
            int global_row = block_row + as_m;
            int global_col = k_base + as_k;

            // 边界检查后加载
            if (global_row < M && global_col < K)
                As[as_k][as_m] = A[global_row * K + global_col];
            else
                As[as_k][as_m] = 0.0f;
        }

        // ---- 协同加载 B 的 BK×BN 块到 Bs ----
        for (int load_idx = thread_idx;
             load_idx < BK * BN;
             load_idx += (BM / TM) * (BN / TN)) {
            int bs_k = load_idx / BN;  // K 方向索引
            int bs_n = load_idx % BN;  // N 方向索引

            int global_row = k_base + bs_k;
            int global_col = block_col + bs_n;

            if (global_row < K && global_col < N)
                Bs[bs_k][bs_n] = B[global_row * N + global_col];
            else
                Bs[bs_k][bs_n] = 0.0f;
        }

        // 加载同步：等待所有 Thread 完成 Shared Memory 填充
        __syncthreads();

        // ============================================================
        // 核心计算：外积展开（Outer Product）
        // ============================================================
        // 对当前 BK 内的每个 k 值，做一次外积更新
        //
        // 外积的含义：
        //   regA[TM] = A 的一列切片（TM 个元素）
        //   regB[TN] = B 的一行切片（TN 个元素）
        //   外积 = TM×TN 个 FMA：accum[m][n] += regA[m] * regB[n]
        //
        // 数据复用：
        //   regA[m] 被复用 TN=8 次（与所有 TN 个 regB[n] 各做一次乘法）
        //   regB[n] 被复用 TM=8 次（与所有 TM 个 regA[m] 各做一次乘法）
        //   从 Shared Memory 读 TM+TN=16 float → 计算 TM×TN=64 次 FMA
        //   寄存器级算术强度 = 64/(16×4) = 1 FLOP/Byte（比 V8 高 8x）

        float regA[TM];  // A 的列切片（当前 Thread 负责的 TM 行）
        float regB[TN];  // B 的行切片（当前 Thread 负责的 TN 列）

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // 从 Shared Memory 加载 A 的切片（TM 个连续元素）
            // As[k][thread_row .. thread_row+TM-1]
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                regA[m] = As[k][thread_row + m];
            }

            // 从 Shared Memory 加载 B 的切片（TN 个连续元素）
            // Bs[k][thread_col .. thread_col+TN-1]
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                regB[n] = Bs[k][thread_col + n];
            }

            // 外积更新：TM × TN 次 FMA
            // accum[m][n] += regA[m] * regB[n]
            // 所有 accum 都在寄存器中，无内存访问！
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    accum[m][n] += regA[m] * regB[n];
                }
            }
        }

        // 计算同步：防止下一轮循环覆盖当前 tile 还在使用的 Shared Memory
        __syncthreads();
    }

    // ============================================================
    // 写回全局内存：将 accum[TM][TN] 写入 C 的对应位置
    // ============================================================
    // 每个 Thread 写 TM×TN = 64 个元素
    // 不同 Thread 写 C 的不同区域，无数据竞争
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int global_row = block_row + thread_row + m;
            int global_col = block_col + thread_col + n;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = accum[m][n];
            }
        }
    }
}

// ============================================================
// Host Wrapper
// ============================================================

void gemm_v9(const float* A, const float* B, float* C, int M, int N, int K) {
    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_B = (size_t)K * N * sizeof(float);
    size_t bytes_C = (size_t)M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, bytes_B, cudaMemcpyHostToDevice));

    // blockDim = (BN/TN, BM/TM) = (8, 8) = 64 threads
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    gemm_v9_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

void gemm_v9_device(const float* d_A, const float* d_B, float* d_C,
                    int M, int N, int K) {
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_v9_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}
