// ============================================================
// Benchmark 入口：统一测试所有CPU版本性能
// ============================================================
// 编译：g++ -O3 -mavx2 -mfma -march=native -I include \
//           src/benchmark.cpp src/gemm_naive.cpp src/gemm_simd.cpp \
//           -o benchmark
// 运行：./benchmark
// ============================================================

#include "gemm.h"
#include <cstdlib>
#include <ctime>
#include <stdio.h>

// 随机初始化矩阵（值在0~1之间）
void rand_init(float* M, int n) {
    for (int i = 0; i < n * n; i++)
        M[i] = (float)rand() / RAND_MAX;
}

int main() {
    srand(42);  // 固定随机种子，保证可复现

    const int n = N;  // 来自gemm.h的宏，默认512
    printf("Matrix size: %d x %d\n", n, n);
    printf("%-30s | %10s | %13s\n", "Version", "Time(ms)", "GFLOPS");
    printf("%s\n", "--------------------------------------------------------------");

    // 分配内存（对齐到32字节，AVX2要求）
    float* A = (float*)aligned_alloc(32, n * n * sizeof(float));
    float* B = (float*)aligned_alloc(32, n * n * sizeof(float));
    float* C = (float*)aligned_alloc(32, n * n * sizeof(float));
    float* C_ref = (float*)aligned_alloc(32, n * n * sizeof(float));

    rand_init(A, n);
    rand_init(B, n);

    // ---- 版本1: Naive ijk（基准） ----
    double ms = measure_ms([&]() { gemm_naive_ijk(A, B, C_ref, n); });
    print_result("1. Naive (ijk)", ms, calc_gflops(n, ms));
    // 保存为参考结果
    float* ref_copy = new float[n * n];
    memcpy(ref_copy, C_ref, n * n * sizeof(float));

    // ---- 版本2: 循环重排 ikj ----
    ms = measure_ms([&]() { gemm_ikj(A, B, C, n); });
    print_result("2. Loop reorder (ikj)", ms, calc_gflops(n, ms));
    printf("   Correctness: %s\n", verify(ref_copy, C, n) ? "PASS ✓" : "FAIL ✗");

    // ---- 版本3: 分块 Tiling ----
    ms = measure_ms([&]() { gemm_tiling(A, B, C, n); });
    print_result("3. Tiling (32x32 blocks)", ms, calc_gflops(n, ms));
    printf("   Correctness: %s\n", verify(ref_copy, C, n) ? "PASS ✓" : "FAIL ✗");

    // ---- 版本4: AVX2 SIMD ----
    ms = measure_ms([&]() { gemm_simd_avx2(A, B, C, n); });
    print_result("4. AVX2 SIMD", ms, calc_gflops(n, ms));
    printf("   Correctness: %s\n", verify(ref_copy, C, n) ? "PASS ✓" : "FAIL ✗");

    // ---- 清理 ----
    free(A); free(B); free(C); free(C_ref);
    delete[] ref_copy;

    printf("\n");
    printf("提示：用以下命令查看编译器生成的汇编，观察向量化情况：\n");
    printf("  g++ -O3 -march=native -S src/gemm_naive.cpp -o /tmp/gemm.s\n");
    printf("  grep -E 'vmovups|vfmadd|vmovss' /tmp/gemm.s | head -20\n");

    return 0;
}
