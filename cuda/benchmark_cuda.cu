// ============================================================
// benchmark_cuda.cu — CUDA GEMM 版本基准测试主程序
// ============================================================
//
// 编译（由 CMakeLists.txt 处理）：
//   nvcc -O3 -arch=sm_75 \
//        -I include cuda/benchmark_cuda.cu \
//        cuda/gemm_v7_naive.cu cuda/gemm_v8_shared.cu \
//        cuda/gemm_v9_register.cu cuda/gemm_v10_vectorized.cu \
//        cuda/cublas_reference.cu \
//        -lcublas -o benchmark_cuda
//
// 用法：
//   ./benchmark_cuda                              # 默认大小 512 1024 2048 4096
//   ./benchmark_cuda --size 1024 2048 4096        # 指定矩阵大小
//   ./benchmark_cuda --size 2048 --version 9      # 只测试特定版本
//
// 输出：
//   1. 控制台：格式化表格
//   2. results/cuda_results.csv：供 Python 分析
// ============================================================

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <cstdio>
#include <algorithm>
#include <functional>

#include "../include/benchmark_utils.h"  // CUDA_CHECK, CudaTimer, verify_result 等

// ============================================================
// 前向声明（来自各 CUDA 文件的 device 版本）
// ============================================================
void gemm_v7_device(const float* d_A, const float* d_B, float* d_C, int M, int N, int K);
void gemm_v8_device(const float* d_A, const float* d_B, float* d_C, int M, int N, int K);
void gemm_v9_device(const float* d_A, const float* d_B, float* d_C, int M, int N, int K);
void gemm_v10_device(const float* d_A, const float* d_B, float* d_C, int M, int N, int K);
void gemm_v11_device(const float* d_A, const float* d_B, float* d_C, int M, int N, int K);
void gemm_v11_cleanup();

// ============================================================
// 命令行参数解析
// ============================================================

struct Config {
    std::vector<int> sizes;
    int version_filter = -1;  // -1 表示测试所有版本
    int warm_up = 3;
    int repeat  = 5;
};

static Config parse_args(int argc, char* argv[]) {
    Config cfg;
    cfg.sizes = {512, 1024, 2048, 4096};

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--size") {
            cfg.sizes.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-')
                cfg.sizes.push_back(std::atoi(argv[++i]));
        } else if (arg == "--version") {
            cfg.version_filter = std::atoi(argv[++i]);
        } else if (arg == "--warmup") {
            cfg.warm_up = std::atoi(argv[++i]);
        } else if (arg == "--repeat") {
            cfg.repeat  = std::atoi(argv[++i]);
        }
    }
    return cfg;
}

// ============================================================
// CUDA Event 计时函数
// ============================================================

/**
 * 使用 CUDA Event 对 Kernel 计时（比 CPU 计时更精确）
 *
 * CUDA Event 在 GPU 时间轴上打时间戳，不受 CPU 调度影响
 * 精度：约 0.5 微秒
 *
 * @param fn      Kernel 函数（通常是 lambda，内部调用 device 版本）
 * @param warm_up 预热次数
 * @param repeat  正式测量次数（取均值，GPU 执行稳定，均值比中位数更合适）
 * @return 平均耗时（毫秒）
 */
static float benchmark_cuda_ms(std::function<void()> fn, int warm_up, int repeat) {
    // 预热：让 GPU 驱动完成初始化，Cache 预热
    for (int i = 0; i < warm_up; i++) {
        fn();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 使用 CUDA Event 计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));  // 等待 GPU 完成

    float total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / repeat;  // 返回平均值（GPU 执行比较稳定，均值合适）
}

// ============================================================
// 主函数
// ============================================================

int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    // 打印 GPU 信息
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("============================================================\n");
    printf("CUDA GEMM Benchmark（V7-V11）\n");
    printf("GPU: %s (SM %d.%d, 显存 %.1f GB)\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1e9);
    // 理论 FP32 峰值 = SM数 × 每SM的CUDA Core数 × 主频 × 2（FMA=2FLOPs）
    // 注意：prop.warpSize=32，multiProcessorCount=SM数，clockRate单位是kHz
    printf("理论 FP32 峰值：%.1f TFLOPS\n",
           (double)prop.multiProcessorCount * prop.clockRate * 1e3 * 2 / 1e12);
    printf("预热 %d 次，测量 %d 次（取均值）\n", cfg.warm_up, cfg.repeat);
    printf("============================================================\n");

    // 打开 CSV
    CsvWriter csv("results/cuda_results.csv");
    csv.write_header({"version","M","N","K","time_ms","gflops",
                      "bandwidth_gb_s","pct_of_cublas"});

    // ====== 遍历矩阵大小 ======
    for (int sz : cfg.sizes) {
        int M = sz, N = sz, K = sz;
        size_t bytes = (size_t)M * N * sizeof(float);

        printf("\n【矩阵大小：%d × %d（M=N=K）】\n", sz, sz);

        // 分配主机内存（pinned memory 加速 H2D/D2H 传输）
        float *h_A, *h_B, *h_C, *h_ref;
        CUDA_CHECK(cudaMallocHost(&h_A, (size_t)M * K * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_B, (size_t)K * N * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_C, bytes));
        CUDA_CHECK(cudaMallocHost(&h_ref, bytes));

        // 随机初始化
        srand(42);
        for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX - 0.5f;
        for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX - 0.5f;

        // 分配 GPU 显存（一次分配，多版本共用）
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, (size_t)M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, (size_t)K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));

        // 拷贝输入到 GPU
        CUDA_CHECK(cudaMemcpy(d_A, h_A, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, (size_t)K * N * sizeof(float), cudaMemcpyHostToDevice));

        // ---- cuBLAS 计算参考结果 ----
        printf("计算参考结果（cuBLAS V11）...\n");
        gemm_v11_device(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_ref, d_C, bytes, cudaMemcpyDeviceToHost));

        // cuBLAS 计时（用于计算各版本占比）
        float cublas_ms = benchmark_cuda_ms([&](){
            gemm_v11_device(d_A, d_B, d_C, M, N, K);
        }, cfg.warm_up, cfg.repeat);

        double cublas_gflops = calc_gflops(M, N, K, cublas_ms);
        printf("cuBLAS: %.2f ms, %.2f GFLOPS（参考上界）\n", cublas_ms, cublas_gflops);

        // 打印表头
        printf("\n%-22s | %8s | %8s | %10s | %10s\n",
               "版本", "时间(ms)", "GFLOPS", "BW(GB/s)", "占cuBLAS%");
        printf("%-22s-+-%8s-+-%8s-+-%10s-+-%10s\n",
               "----------------------","--------","--------","----------","----------");

        // ---- 定义要测试的版本 ----
        struct CudaVersion {
            const char* name;
            int ver_num;
            std::function<void(const float*, const float*, float*, int, int, int)> fn;
        };

        std::vector<CudaVersion> versions = {
            {"V7_CUDA_Naive",   7, gemm_v7_device},
            {"V8_Shared_Mem",   8, gemm_v8_device},
            {"V9_Register",     9, gemm_v9_device},
            {"V10_Vectorized", 10, gemm_v10_device},
            {"V11_cuBLAS",     11, gemm_v11_device},
        };

        for (auto& ver : versions) {
            if (cfg.version_filter != -1 && ver.ver_num != cfg.version_filter) continue;

            float ms = benchmark_cuda_ms([&](){
                ver.fn(d_A, d_B, d_C, M, N, K);
            }, cfg.warm_up, cfg.repeat);

            // 验证正确性（将结果拷回 CPU 对比）
            CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
            bool passed = verify_result(h_ref, h_C, M, N);

            double gflops   = calc_gflops(M, N, K, ms);
            double bw_gbs   = calc_bandwidth_gbs(M, N, K, ms);
            double pct      = gflops / cublas_gflops * 100.0;

            printf("%-22s | %8.2f | %8.2f | %10.1f | %9.1f%%  %s\n",
                   ver.name, ms, gflops, bw_gbs, pct,
                   passed ? "✓" : "✗ FAIL");

            csv.write_row({
                ver.name, to_str(M), to_str(N), to_str(K),
                to_str(ms), to_str(gflops), to_str(bw_gbs), to_str(pct, 1)
            });
        }

        // 释放资源
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaFreeHost(h_A); cudaFreeHost(h_B);
        cudaFreeHost(h_C); cudaFreeHost(h_ref);
    }

    printf("\n============================================================\n");
    printf("结果已写入 results/cuda_results.csv\n");
    printf("运行 python3 analysis/plot_results.py 生成图表\n");

    gemm_v11_cleanup();
    return 0;
}
