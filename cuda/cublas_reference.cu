// ============================================================
// cublas_reference.cu — V11 cuBLAS 参考实现（性能上界）
// ============================================================
//
// 【版本】V11 — cuBLAS SGEMM（工业级参考）
//
// 【cuBLAS 简介】
//   cuBLAS 是 NVIDIA 提供的官方 GPU BLAS（Basic Linear Algebra Subprograms）库
//   经过 NVIDIA 工程师多年深度优化，代表了同类 GPU 上的接近峰值性能
//   通常作为我们自己实现的性能上界（ceiling）
//
// 【cuBLAS 内部使用的优化技术（部分公开）】
//   1. Double Buffering（双缓冲）：
//      在计算当前 tile 的同时，异步预加载下一个 tile 到 Shared Memory
//      利用 cp.async 指令（Ampere+ GPU）隐藏 Global Memory 延迟
//
//   2. Warp-Level Matrix Multiply Accumulate（WMMA）：
//      在 Volta/Turing/Ampere GPU 上使用 Tensor Core
//      Tensor Core 每个时钟周期可做 4×4×4 的 FP16 矩阵乘（或 16×16×16）
//      比普通 CUDA Core 的 FP32 吞吐高 8-16x（以精度换速度）
//
//   3. Register 级分块：超大 tile（如 128×128 的 C 块）
//      每个 Warp 负责更大的输出区域，提升寄存器利用率
//
//   4. 软件流水线（Software Pipelining）：
//      计算与内存访问的 overlap，消除 __syncthreads() 的延迟
//
//   5. 指令级优化：精心排列指令顺序，隐藏延迟，提升指令发射率
//
// 【cuBLAS 的列主序约定】
//   cuBLAS API 使用列主序（Column-major，Fortran 传统）
//   而我们的矩阵是行主序（Row-major，C/C++ 传统）
//
//   转换技巧（利用 C = A×B 的转置性质）：
//     C = A × B
//     C^T = B^T × A^T
//
//   cuBLAS 以列主序看待矩阵时，row-major 的 A 等价于列主序的 A^T
//   所以调用 cublasSgemm(B^T, A^T) 在 cuBLAS 视角下等价于计算：
//   cuBLAS sees: result = B^T × A^T = (A×B)^T = C^T
//   cuBLAS 以列主序写回 C^T = 行主序的 C ✓
//
//   代码实现：
//   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
//               N, M, K, &alpha,
//               d_B, N,   // B，lda=N（cuBLAS 视角：N×K 列主序矩阵）
//               d_A, K,   // A，ldb=K（cuBLAS 视角：K×M 列主序矩阵）
//               &beta, d_C, N)  // C，ldc=N
//   这实际上计算了 C = A × B（行主序）✓
//
// 【与我们实现的性能差距】
//   典型差距（RTX 系列 GPU）：
//     我们的 V9   ≈ cuBLAS 的 30-60%
//     我们的 V10  ≈ cuBLAS 的 40-70%
//     还有差距的原因：cuBLAS 使用了 Tensor Core + 极致软件流水线
// ============================================================

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include "../include/benchmark_utils.h"

// cuBLAS 错误检查宏
#define CUBLAS_CHECK(call) do { \
    cublasStatus_t _status = (call); \
    if (_status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "[CUBLAS ERROR] %s:%d — status=%d\n", \
                __FILE__, __LINE__, (int)_status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================
// 全局 cuBLAS Handle（在程序生命周期内保持）
// ============================================================
// cublasHandle_t 包含 cuBLAS 的内部状态和工作空间
// 创建和销毁都有开销，应该复用而非每次计算都创建新的

static cublasHandle_t g_cublas_handle = nullptr;

static void ensure_cublas_handle() {
    if (g_cublas_handle == nullptr) {
        CUBLAS_CHECK(cublasCreate(&g_cublas_handle));
    }
}

// 调用 gemm_v11_cleanup() 在程序结束时释放资源
void gemm_v11_cleanup() {
    if (g_cublas_handle != nullptr) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
    }
}

// ============================================================
// Host Wrapper
// ============================================================

void gemm_v11(const float* A, const float* B, float* C, int M, int N, int K) {
    ensure_cublas_handle();

    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_B = (size_t)K * N * sizeof(float);
    size_t bytes_C = (size_t)M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, bytes_B, cudaMemcpyHostToDevice));

    const float alpha = 1.0f, beta = 0.0f;

    // 调用 cublasSgemm
    // 参数说明（以行主序的 C = A × B 为目标）：
    //   op(B) = CUBLAS_OP_N（B 不转置，cuBLAS 按列主序读，等价于行主序的 B^T）
    //   op(A) = CUBLAS_OP_N（A 不转置）
    //   cuBLAS 计算 C = op(B) × op(A) = B^T × A^T（cuBLAS 列主序视角）
    //   结果写回后，以行主序读 C，得到正确的 A×B
    CUBLAS_CHECK(cublasSgemm(
        g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // 不转置（cuBLAS 内部处理行/列主序转换）
        N, M, K,                    // cuBLAS 的 m,n,k（注意 M 和 N 对换！）
        &alpha,
        d_B, N,                     // B 矩阵，leading dimension = N
        d_A, K,                     // A 矩阵，leading dimension = K
        &beta,
        d_C, N                      // C 矩阵，leading dimension = N
    ));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// Device 版本：直接操作 GPU 内存（供 benchmark 使用，避免重复内存分配）
void gemm_v11_device(const float* d_A, const float* d_B, float* d_C,
                     int M, int N, int K) {
    ensure_cublas_handle();
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(
        g_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha,
        d_B, N, d_A, K,
        &beta, d_C, N
    ));
}
