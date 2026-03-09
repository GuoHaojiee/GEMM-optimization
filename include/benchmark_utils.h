#pragma once
// ============================================================
// benchmark_utils.h — 计时、验证、CSV 输出工具（全内联实现）
// ============================================================
// 设计为 header-only，直接 #include 即可使用
// 无外部依赖（CUDA_CHECK 宏仅在 CUDA 代码中需要）
// ============================================================

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

// ============================================================
// 计时工具
// ============================================================

/**
 * 返回当前时间（毫秒），使用高精度时钟
 * 用法：
 *   double t0 = get_time_ms();
 *   do_work();
 *   double elapsed = get_time_ms() - t0;  // 耗时（毫秒）
 */
inline double get_time_ms() {
    using namespace std::chrono;
    auto now = high_resolution_clock::now();
    // duration_cast 转换为纳秒，再除以 1e6 得毫秒（保留小数精度）
    return duration_cast<nanoseconds>(now.time_since_epoch()).count() / 1e6;
}

/**
 * 对函数计时：warm_up 次预热 + repeat 次正式运行，返回中位数（毫秒）
 *
 * 使用中位数而非均值：避免偶发的 OS 调度抖动影响结果
 * 预热原因：首次运行可能触发 JIT 编译、Cache 冷启动等一次性开销
 *
 * @param fn      要计时的函数（零参数，返回 void）
 * @param warm_up 预热次数（默认 3）
 * @param repeat  正式测量次数（默认 5）
 * @return 中位数耗时（毫秒）
 */
template<typename Func>
inline double benchmark_ms(Func fn, int warm_up = 3, int repeat = 5) {
    // 预热阶段：执行但不计时
    for (int i = 0; i < warm_up; i++) fn();

    // 正式测量
    std::vector<double> times(repeat);
    for (int i = 0; i < repeat; i++) {
        double t0 = get_time_ms();
        fn();
        times[i] = get_time_ms() - t0;
    }

    // 排序后取中位数
    std::sort(times.begin(), times.end());
    return times[repeat / 2];
}

// ============================================================
// 正确性验证
// ============================================================

/**
 * 验证 test 结果与 ref 参考结果是否一致
 *
 * 使用相对误差（而非绝对误差）原因：
 *   - 矩阵元素数值范围可能很大，绝对误差阈值难以设置
 *   - 浮点累积误差与数值大小成正比
 *
 * 公式：|test[i] - ref[i]| / max(|ref[i]|, 1e-7) < 1e-2
 *
 * 注意：容差设为 1e-2（1%），原因是 GEMM 的浮点累加顺序不同会产生
 * 约 K×eps（K=矩阵维度，eps=machine epsilon≈1e-7）的累积误差。
 * 当 K=512 时，理论误差约 5e-5，但不同优化版本的累加顺序不同，
 * 实际误差可达 1e-3 量级，因此 1e-2 是合理的验证阈值。
 *
 * @param ref    参考结果（V1 naive 计算）
 * @param test   待验证结果
 * @param M, N   矩阵行列数
 * @param tol    相对误差容忍度（默认 1e-2）
 * @return       true 表示通过验证
 */
inline bool verify_result(const float* ref, const float* test, int M, int N,
                           float rel_tol = 1e-2f, float abs_tol = 1e-4f) {
    for (int i = 0; i < M * N; i++) {
        float diff     = std::fabs(test[i] - ref[i]);
        float abs_ref  = std::fabs(ref[i]);
        // 混合误差验证：绝对误差 OR 相对误差满足容差即通过
        // 当 |ref| 很小时，用绝对误差判断（避免近零值的假阳性失败）
        // 当 |ref| 较大时，用相对误差判断
        bool pass = (diff <= abs_tol) || (diff / std::max(abs_ref, 1e-7f) <= rel_tol);
        if (!pass) {
            int row = i / N, col = i % N;
            printf("  [FAIL] C[%d][%d]: ref=%.6f, got=%.6f, abs_err=%.2e, rel_err=%.2e\n",
                   row, col, ref[i], test[i], diff,
                   diff / std::max(abs_ref, 1e-7f));
            return false;
        }
    }
    return true;
}

// ============================================================
// 性能指标计算
// ============================================================

/**
 * 计算 GEMM 的 GFLOPS（每秒十亿次浮点运算）
 *
 * GEMM 浮点运算量：每个 C[i][j] 需要 K 次乘法 + K 次加法 = 2K 次 FP 运算
 * 总 FP 运算量 = 2 × M × N × K FLOP
 *
 * @param M, N, K  矩阵维度
 * @param ms       执行时间（毫秒）
 * @return         GFLOPS（越高越好）
 */
inline double calc_gflops(int M, int N, int K, double ms) {
    // 2.0 * M * N * K 可能超过 int 范围（4096³ ≈ 6.8e10），必须用 double
    double flops = 2.0 * M * N * K;
    double seconds = ms / 1000.0;
    return (flops / seconds) / 1e9;
}

/**
 * 计算内存带宽利用率（GB/s）
 *
 * 理论最低内存访问量：
 *   读 A（M×K）+ 读 B（K×N）+ 读写 C（M×N）= (M*K + K*N + 2*M*N) × sizeof(float)
 *
 * @param M, N, K  矩阵维度
 * @param ms       执行时间（毫秒）
 * @return         带宽（GB/s）
 */
inline double calc_bandwidth_gbs(int M, int N, int K, double ms) {
    // 最优情况下每个元素只读写一次
    double bytes = (double)(M * K + K * N + 2LL * M * N) * sizeof(float);
    double seconds = ms / 1000.0;
    return (bytes / seconds) / 1e9;
}

// ============================================================
// CSV 输出工具
// ============================================================

/**
 * CSV 写入器：管理一个 CSV 文件的打开、写标题行、追加数据行
 *
 * 用法：
 *   CsvWriter csv("results/cpu_results.csv");
 *   csv.write_header({"version","M","N","K","time_ms","gflops"});
 *   csv.write_row({"V1",512,512,512,123.4,5.6});
 */
class CsvWriter {
public:
    explicit CsvWriter(const std::string& filename) {
        file_.open(filename);
        if (!file_.is_open()) {
            printf("[WARNING] 无法打开 CSV 文件：%s\n", filename.c_str());
        }
    }

    ~CsvWriter() { if (file_.is_open()) file_.close(); }

    // 写标题行
    void write_header(const std::vector<std::string>& cols) {
        if (!file_.is_open()) return;
        for (size_t i = 0; i < cols.size(); i++) {
            if (i > 0) file_ << ",";
            file_ << cols[i];
        }
        file_ << "\n";
        file_.flush();
    }

    // 写一行数据（所有字段用逗号分隔）
    void write_row(const std::vector<std::string>& values) {
        if (!file_.is_open()) return;
        for (size_t i = 0; i < values.size(); i++) {
            if (i > 0) file_ << ",";
            file_ << values[i];
        }
        file_ << "\n";
        file_.flush();
    }

    bool is_open() const { return file_.is_open(); }

private:
    std::ofstream file_;
};

/**
 * 辅助函数：将数值转为固定精度字符串
 */
inline std::string to_str(double v, int precision = 3) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.*f", precision, v);
    return std::string(buf);
}

inline std::string to_str(int v) {
    return std::to_string(v);
}

// ============================================================
// 格式化打印工具
// ============================================================

/**
 * 打印 benchmark 结果表格的标题行
 */
inline void print_table_header() {
    printf("\n%-22s | %8s | %8s | %8s | %10s | %10s\n",
           "版本", "M=N=K", "时间(ms)", "GFLOPS", "相对V1", "验证");
    printf("%-22s-+-%8s-+-%8s-+-%8s-+-%10s-+-%10s\n",
           "----------------------", "--------", "--------",
           "--------", "----------", "----------");
}

/**
 * 打印一行 benchmark 结果
 */
inline void print_table_row(const char* name, int size,
                             double ms, double gflops,
                             double speedup, bool passed) {
    printf("%-22s | %8d | %8.2f | %8.2f | %9.2fx | %10s\n",
           name, size, ms, gflops, speedup, passed ? "PASS ✓" : "FAIL ✗");
}

// ============================================================
// CUDA 辅助宏（仅在 CUDA 代码中有效）
// ============================================================

#ifdef __CUDACC__
#include <cuda_runtime.h>

/**
 * CUDA 错误检查宏
 * 在任何 CUDA API 调用后使用，出错时打印文件/行号并退出
 *
 * 用法：CUDA_CHECK(cudaMalloc(&ptr, size));
 */
#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/**
 * CUDA Event 计时器（比 CPU 计时更精确，直接在 GPU 时间轴上测量）
 *
 * 用法：
 *   CudaTimer timer;
 *   timer.start();
 *   kernel<<<...>>>(...);
 *   float ms = timer.stop();  // 自动同步
 */
class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { cudaEventRecord(start_); }
    float stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);  // 等待 GPU 完成
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
private:
    cudaEvent_t start_, stop_;
};

#endif  // __CUDACC__
