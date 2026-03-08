// ============================================================
// CUDA Benchmark：对比Naive vs Shared Memory vs cuBLAS
// ============================================================
// 编译：nvcc -O3 -arch=sm_86 cuda/gemm_cuda_bench.cu \
//            -lcublas -I include -o cuda_bench
// 注意：-arch=sm_86 对应RTX 30系列，根据你的GPU调整：
//       sm_75: RTX 20系列 / Tesla T4
//       sm_80: A100
//       sm_86: RTX 30系列
//       sm_89: RTX 40系列
// ============================================================

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1024  // CUDA测试用更大矩阵
#define TILE_SIZE 16

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

// 从前面的.cu文件引入的kernel（实际项目中分开编译）
extern __global__ void gemm_naive_kernel(const float*, const float*, float*, int);
extern __global__ void gemm_shared_kernel(const float*, const float*, float*, int);

// CUDA事件计时（比CPU计时更精确）
float time_kernel(std::function<void()> fn, int repeat = 5) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    fn(); // 预热

    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / repeat;
}

bool verify_gpu(const float* ref, const float* result, int n) {
    for (int i = 0; i < n * n; i++) {
        float diff = fabs(ref[i] - result[i]);
        if (diff > 1e-2f * fabs(ref[i]) + 1e-2f) {
            printf("Mismatch at %d: ref=%.4f, got=%.4f\n", i, ref[i], result[i]);
            return false;
        }
    }
    return true;
}

int main() {
    int n = N;
    printf("CUDA GEMM Benchmark: %dx%d matrices\n", n, n);
    printf("GPU: "); 
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("%s (SM %d.%d, %.1f GB, %.1f TFLOPS peak)\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1e9,
           prop.warpSize * prop.multiProcessorCount * prop.clockRate * 2e-9f);
    printf("%-30s | %10s | %13s | %10s\n",
           "Version", "Time(ms)", "GFLOPS", "vs cuBLAS");
    printf("-------------------------------------------------------------------\n");

    size_t bytes = (size_t)n * n * sizeof(float);

    // 初始化主机内存
    float* h_A = new float[n * n];
    float* h_B = new float[n * n];
    float* h_C = new float[n * n];
    float* h_ref = new float[n * n];

    srand(42);
    for (int i = 0; i < n * n; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((n + TILE_SIZE-1)/TILE_SIZE, (n + TILE_SIZE-1)/TILE_SIZE);

    // ---- cuBLAS作为参考基准 ----
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f, beta = 0.0f;

    float cublas_ms = time_kernel([&]() {
        // cuBLAS使用列主序，这里通过转置技巧实现行主序：
        // C = A*B  →  C^T = B^T * A^T
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    n, n, n, &alpha,
                    d_B, n, d_A, n, &beta, d_C, n);
    });
    float cublas_gflops = 2.0 * n * n * n / (cublas_ms / 1000.0) / 1e9;
    CUDA_CHECK(cudaMemcpy(h_ref, d_C, bytes, cudaMemcpyDeviceToHost));
    printf("%-30s | %10.2f | %13.2f | %10s\n",
           "cuBLAS (reference)", cublas_ms, cublas_gflops, "100%");

    // ---- CUDA Naive ----
    float naive_ms = time_kernel([&]() {
        gemm_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
    });
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    float naive_gflops = 2.0 * n * n * n / (naive_ms / 1000.0) / 1e9;
    printf("%-30s | %10.2f | %13.2f | %9.1f%%\n",
           "CUDA Naive", naive_ms, naive_gflops,
           naive_gflops / cublas_gflops * 100);
    printf("   Correctness: %s\n", verify_gpu(h_ref, h_C, n) ? "PASS ✓" : "FAIL ✗");

    // ---- CUDA Shared Memory ----
    float shared_ms = time_kernel([&]() {
        gemm_shared_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
    });
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    float shared_gflops = 2.0 * n * n * n / (shared_ms / 1000.0) / 1e9;
    printf("%-30s | %10.2f | %13.2f | %9.1f%%\n",
           "CUDA Shared Memory", shared_ms, shared_gflops,
           shared_gflops / cublas_gflops * 100);
    printf("   Correctness: %s\n", verify_gpu(h_ref, h_C, n) ? "PASS ✓" : "FAIL ✗");

    printf("\n==== 性能分析命令 ====\n");
    printf("# 分析显存带宽利用率：\n");
    printf("ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum ./cuda_bench\n");
    printf("# 分析SM利用率：\n");
    printf("ncu --metrics sm__throughput.avg.pct_of_peak_sustained_active ./cuda_bench\n");

    // 清理
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_ref;

    return 0;
}
