// ============================================================
// benchmark_cpu.cpp — CPU GEMM 版本基准测试主程序
// ============================================================
//
// 编译（由 CMakeLists.txt 处理）：
//   g++ -O3 -march=native -mavx2 -mfma -fopenmp \
//       -I include src/benchmark_cpu.cpp \
//       src/gemm_v1_naive.cpp src/gemm_v2_ikj.cpp src/gemm_v3_tiling.cpp \
//       src/gemm_v4_avx2.cpp src/gemm_v5_avx2_tiling.cpp src/gemm_v6_openmp.cpp \
//       -o benchmark_cpu
//
// 用法：
//   ./benchmark_cpu                        # 默认大小 512 1024 2048，线程数 1 4
//   ./benchmark_cpu --size 256 512 1024    # 指定矩阵大小
//   ./benchmark_cpu --threads 1 4 8        # 指定线程数
//   ./benchmark_cpu --size 1024 --threads 8
//
// 输出：
//   1. 控制台：格式化表格（版本、大小、时间、GFLOPS、加速比、验证）
//   2. results/cpu_results.csv：CSV 格式，供 Python 分析使用
// ============================================================

#include "../include/gemm.h"
#include "../include/benchmark_utils.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <omp.h>

// ============================================================
// 随机矩阵初始化
// ============================================================

/**
 * 用 [-1, 1] 均匀分布的随机数填充矩阵
 * 使用固定种子（42）保证可重现性
 */
static void rand_init(float* mat, int rows, int cols, unsigned int seed = 42) {
    srand(seed);
    for (int i = 0; i < rows * cols; i++) {
        // 生成 [-1, 1] 范围的随机数
        mat[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

// ============================================================
// 命令行参数解析
// ============================================================

struct Config {
    std::vector<int> sizes;     // 矩阵大小列表（方阵 M=N=K）
    std::vector<int> threads;   // 线程数列表
    int warm_up = 3;            // 预热次数
    int repeat  = 5;            // 正式测量次数
};

static Config parse_args(int argc, char* argv[]) {
    Config cfg;
    cfg.sizes   = {512, 1024, 2048};  // 默认值
    cfg.threads = {1};

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--size") {
            cfg.sizes.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                cfg.sizes.push_back(std::atoi(argv[++i]));
            }
        } else if (arg == "--threads") {
            cfg.threads.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                cfg.threads.push_back(std::atoi(argv[++i]));
            }
        } else if (arg == "--warmup") {
            cfg.warm_up = std::atoi(argv[++i]);
        } else if (arg == "--repeat") {
            cfg.repeat  = std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            printf("用法：benchmark_cpu [--size N...] [--threads T...] [--warmup W] [--repeat R]\n");
            printf("  --size    : 矩阵大小列表（方阵 M=N=K），例：--size 512 1024 2048\n");
            printf("  --threads : 线程数列表，例：--threads 1 4 8\n");
            printf("  --warmup  : 预热次数（默认 3）\n");
            printf("  --repeat  : 测量次数（默认 5，取中位数）\n");
            exit(0);
        }
    }

    return cfg;
}

// ============================================================
// 单版本 Benchmark
// ============================================================

struct BenchResult {
    const char* name;
    int M, N, K;
    int threads;
    double ms;
    double gflops;
    double speedup;
    bool passed;
};

/**
 * 对单个 GEMM 版本进行基准测试
 *
 * @param name     版本名称（用于打印）
 * @param fn       要测试的函数
 * @param A,B,C    输入/输出矩阵（C 会被清零后测试）
 * @param C_ref    V1 计算的参考结果（用于验证正确性）
 * @param M,N,K    矩阵维度
 * @param warm_up  预热次数
 * @param repeat   正式测量次数（取中位数）
 */
static BenchResult run_single(
    const char* name,
    void (*fn)(const float*, const float*, float*, int, int, int),
    const float* A, const float* B, float* C, const float* C_ref,
    int M, int N, int K,
    int threads, int warm_up, int repeat)
{
    // 设置线程数（对 V6/OpenMP 生效，其他版本无影响）
    omp_set_num_threads(threads);

    // 预热（warm-up）阶段：
    //   目的：1. 让 JIT/branch predictor 稳定
    //          2. 让 Cache 预热（第一次运行数据是冷的）
    //          3. 让 CPU 频率提升到全速（turbo boost 延迟）
    for (int i = 0; i < warm_up; i++) {
        fn(A, B, C, M, N, K);
    }

    // 正式测量：多次运行，取中位数
    std::vector<double> times(repeat);
    for (int i = 0; i < repeat; i++) {
        double t0 = get_time_ms();
        fn(A, B, C, M, N, K);
        times[i] = get_time_ms() - t0;
    }

    // 排序取中位数（比均值更鲁棒，不受偶发的 OS 抖动影响）
    std::sort(times.begin(), times.end());
    double median_ms = times[repeat / 2];

    // 验证正确性（使用最后一次运行的结果 C）
    bool passed = verify_result(C_ref, C, M, N);

    double gflops = calc_gflops(M, N, K, median_ms);

    return {name, M, N, K, threads, median_ms, gflops, 0.0, passed};
}

// ============================================================
// 主函数
// ============================================================

int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    printf("============================================================\n");
    printf("CPU GEMM Benchmark（V1-V6）\n");
    printf("预热 %d 次，测量 %d 次（取中位数）\n", cfg.warm_up, cfg.repeat);
    printf("============================================================\n");

    // 打开 CSV 文件
    // 确保 results/ 目录存在（由 run_all.sh 创建，或手动 mkdir）
    CsvWriter csv("results/cpu_results.csv");
    csv.write_header({"version", "M", "N", "K", "threads",
                      "time_ms", "gflops", "speedup_vs_naive"});

    // 定义所有 CPU 版本
    struct Version {
        const char* name;
        void (*fn)(const float*, const float*, float*, int, int, int);
    };

    std::vector<Version> versions = {
        {"V1_Naive_ijk",    gemm_v1},
        {"V2_ikj",          gemm_v2},
        {"V3_Tiling",       gemm_v3},
        {"V4_AVX2",         gemm_v4},
        {"V5_AVX2+Tiling",  gemm_v5},
        {"V6_OMP+AVX2+Tile",gemm_v6},
    };

    // ====== 遍历矩阵大小 ======
    for (int sz : cfg.sizes) {
        int M = sz, N = sz, K = sz;
        printf("\n【矩阵大小：%d × %d（M=N=K）】\n", sz, sz);
        printf("内存占用：A+B+C = %.1f MB\n",
               3.0 * M * N * sizeof(float) / (1024.0 * 1024.0));

        // 分配并初始化矩阵
        std::vector<float> A(M * K), B(K * N), C(M * N), C_ref(M * N);
        rand_init(A.data(), M, K);
        rand_init(B.data(), K, N, 43);  // 不同种子

        // 用 V1 计算参考结果（只算一次，用于后续所有版本的验证）
        printf("计算参考结果（V1 Naive）...\n");
        gemm_v1(A.data(), B.data(), C_ref.data(), M, N, K);

        // 打印表头
        print_table_header();

        double v1_ms = 0.0;  // 记录 V1 时间，用于计算加速比

        // ====== 遍历版本（使用默认线程数 1，V6 用 --threads 参数） ======
        for (auto& ver : versions) {
            // 对非 OpenMP 版本用线程数 1；V6 用配置中的第一个线程数
            int t = (ver.fn == gemm_v6) ? cfg.threads[0] : 1;

            BenchResult r = run_single(
                ver.name, ver.fn,
                A.data(), B.data(), C.data(), C_ref.data(),
                M, N, K, t, cfg.warm_up, cfg.repeat);

            if (ver.fn == gemm_v1) v1_ms = r.ms;
            r.speedup = (v1_ms > 0) ? v1_ms / r.ms : 1.0;

            print_table_row(r.name, sz, r.ms, r.gflops, r.speedup, r.passed);

            // 写 CSV（单线程版本）
            csv.write_row({
                r.name, to_str(M), to_str(N), to_str(K), to_str(t),
                to_str(r.ms), to_str(r.gflops), to_str(r.speedup)
            });
        }

        // ====== V6 多线程扩展测试 ======
        if (cfg.threads.size() > 1) {
            printf("\n  【V6 多线程扩展性测试（矩阵大小 %d）】\n", sz);
            printf("  %-10s | %8s | %8s | %8s\n", "线程数", "时间(ms)", "GFLOPS", "效率");
            printf("  ----------+---------+---------+----------\n");

            for (int t : cfg.threads) {
                BenchResult r = run_single(
                    "V6_OpenMP", gemm_v6,
                    A.data(), B.data(), C.data(), C_ref.data(),
                    M, N, K, t, cfg.warm_up, cfg.repeat);

                double speedup = v1_ms / r.ms;
                double efficiency = speedup / t * 100.0;  // 并行效率
                printf("  %-10d | %8.2f | %8.2f | %8.1f%%\n",
                       t, r.ms, r.gflops, efficiency);

                csv.write_row({
                    "V6_OpenMP", to_str(M), to_str(N), to_str(K), to_str(t),
                    to_str(r.ms), to_str(r.gflops), to_str(speedup)
                });
            }
        }
    }

    printf("\n============================================================\n");
    printf("结果已写入 results/cpu_results.csv\n");
    printf("运行 python3 analysis/plot_results.py 生成图表\n");

    return 0;
}
