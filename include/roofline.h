#pragma once
// ============================================================
// roofline.h — Roofline 模型硬件参数 + 算术强度计算
// ============================================================
//
// 【什么是 Roofline 模型】
//   性能上界 = min(峰值算力, 算术强度 × 峰值带宽)
//   其中算术强度（Arithmetic Intensity, AI）= FLOP / Byte（字节访问量）
//
//   - 若 AI < 峰值算力/峰值带宽  → Memory Bound（内存瓶颈）
//   - 若 AI > 峰值算力/峰值带宽  → Compute Bound（算力瓶颈）
//
// 【如何获取硬件参数】
//   CPU 峰值算力：核心数 × 频率 × FMA 吞吐(每周期FMA数) × SIMD宽度(float数)
//     例：8核 @ 3.5GHz，每核每周期 2 个 FMA，AVX2=8 float
//     = 8 × 3.5e9 × 2 × 8 × 2 (FMA=2 FLOPs) = ~896 GFLOPS
//     命令：lscpu | grep MHz; nproc
//
//   CPU 内存带宽：用 stream benchmark 实测
//     命令：./stream（或 sudo lshw -C memory 查看理论值）
//     https://www.cs.virginia.edu/stream/
//
//   GPU 峰值算力：查 GPU 规格表（spec sheet）
//     命令：nvidia-smi --query-gpu=name --format=csv
//     参考：RTX 3090=35.6 TFLOPS FP32, A100=312 TFLOPS TF32
//
//   GPU 内存带宽：查规格表或用 bandwidthTest 实测
//     命令：./bandwidthTest（CUDA 示例程序）
//     参考：RTX 3090=936 GB/s, A100 HBM2e=2000 GB/s
// ============================================================

// ============================================================
// CPU 硬件参数（根据你的机器修改！）
// ============================================================
// 如何修改：运行 `lscpu` 获取核心数、主频，运行 stream 得带宽

// 估算公式：核心数 × 主频(GHz) × 每周期FMA数 × SIMD宽度 × 2(FMA=2FLOPs)
// 示例（8核 i7, 3.5GHz, AVX2）：8 × 3.5 × 2 × 8 × 2 = 896 GFLOPS
constexpr double CPU_PEAK_GFLOPS = 100.0;   // 请根据实际机器修改

// 用 stream benchmark 实测：./stream | grep Triad
// 典型台式机 DDR4-3200 双通道 ≈ 40-50 GB/s
constexpr double CPU_PEAK_BW_GBS = 50.0;    // 请根据实际机器修改

// 计算 bound/memory bound 分界点（"屋顶线的拐点"）
// 低于此 AI → memory bound；高于此 AI → compute bound
constexpr double CPU_RIDGE_POINT = CPU_PEAK_GFLOPS / CPU_PEAK_BW_GBS;  // FLOP/Byte

// ============================================================
// GPU 硬件参数（根据你的 GPU 修改！）
// ============================================================
// 如何修改：nvidia-smi 查型号，然后查 NVIDIA 规格表

// FP32 峰值（单精度）：查 GPU datasheet 中 "FP32 Performance"
constexpr double GPU_PEAK_GFLOPS = 10000.0; // 请根据实际 GPU 修改（GFLOPS）

// DRAM 带宽：查 datasheet 中 "Memory Bandwidth"
constexpr double GPU_PEAK_BW_GBS = 500.0;   // 请根据实际 GPU 修改（GB/s）

// GPU Roofline 拐点
constexpr double GPU_RIDGE_POINT = GPU_PEAK_GFLOPS / GPU_PEAK_BW_GBS;  // FLOP/Byte

// ============================================================
// 各版本算术强度计算（理论分析）
// ============================================================
//
// 【算术强度推导方法】
//   AI = FLOPs / Memory Traffic (字节)
//   GEMM FLOPs = 2 × M × N × K（每个输出元素做 K 次 FMA）
//
// ──────────────────────────────────────────────────────────
// V1/V2 Naive（无 Cache 重用）：
//   每次 C[i][j] += A[i][k] * B[k][j] 都从内存读取 A、B
//   Memory Traffic ≈ 4 × M×N×K × 4 字节（A 被读 N 次，B 被读 M 次，C 读写各一次，无重用）
//   简化：AI ≈ 2MNK / (4 × (MK×N + KN×M)) = 2MNK / (8MNK) = 0.25 FLOP/Byte
//   → 极低！严重 memory bound
//
// V3 Tiling（块大小 T）：
//   A 块被复用 T 次（计算 T 列 B 时 A 块不变）
//   B 块被复用 T 次（计算 T 行 A 时 B 块不变）
//   Memory Traffic ≈ 4 × (MK + KN + MN) × 4 字节（每个元素只从 DRAM 读一次）
//   但实际分析：AI = T/4 FLOP/Byte（块越大，复用越好）
//   T=64: AI ≈ 16 FLOP/Byte
//
// V8 CUDA Shared Memory（TILE_SIZE=T）：
//   每个 Tile 从 Global Memory 加载一次后被 T 个 Thread 复用 T 次
//   AI = 2MNK / (4 × 2 × MNK/T) = T/4 FLOP/Byte
//   T=16: AI ≈ 4 FLOP/Byte（仍然 memory bound，有提升空间）
//
// V9 Register Tiling（BM=64, BN=64, BK=8, TM=8, TN=8）：
//   从 Shared Memory 加载 TM+TN 个 float，计算 TM×TN=64 次 FMA
//   AI ≈ TM×TN×2 / ((TM+TN)×4) = 128/64 = 2 FLOP/Byte（相对于 Shared Memory）
//   总体从 Global Memory 看：AI ≈ BK/4 × (BM+BN)/(BM+BN) ≈ 相同结构下更好的复用
// ──────────────────────────────────────────────────────────

/**
 * 计算 GEMM 理论算术强度（FLOP/Byte）
 *
 * 三种分析模式：
 *   - mode=0：无 Cache 重用（Naive，理论最差）
 *   - mode=1：完美 Cache 复用（每个数据元素只从 DRAM 读一次，理论最优）
 *   - mode=2：分块复用，复用因子 tile_size
 *
 * @return 算术强度（FLOP/Byte）
 */
inline double arithmetic_intensity(long long M, long long N, long long K,
                                    int mode = 1, int tile_size = 16) {
    double flops = 2.0 * M * N * K;

    if (mode == 0) {
        // 无复用：A 被读 N 次，B 被读 M 次，C 读写各一次
        double bytes = (double)(M * K * N + K * N * M + 2LL * M * N) * 4.0;
        return flops / bytes;
    } else if (mode == 1) {
        // 完美复用：每个元素只从 DRAM 读一次
        double bytes = (double)(M * K + K * N + M * N) * 4.0;
        return flops / bytes;
    } else {
        // 分块复用：AI ≈ tile_size / 4
        return (double)tile_size / 4.0;
    }
}

/**
 * 给定实测 GFLOPS 和算术强度，判断是否达到 Roofline
 * 返回实测性能占 Roofline 上界的百分比
 */
inline double roofline_efficiency(double actual_gflops, double ai_flop_per_byte,
                                   bool is_gpu = false) {
    double peak_gflops = is_gpu ? GPU_PEAK_GFLOPS : CPU_PEAK_GFLOPS;
    double peak_bw     = is_gpu ? GPU_PEAK_BW_GBS  : CPU_PEAK_BW_GBS;

    // Roofline 上界 = min(算力上界, 带宽上界)
    double compute_bound = peak_gflops;
    double memory_bound  = ai_flop_per_byte * peak_bw;  // FLOP/Byte × Byte/s = FLOP/s
    double ceiling_gflops = std::min(compute_bound, memory_bound);

    return actual_gflops / ceiling_gflops * 100.0;
}
