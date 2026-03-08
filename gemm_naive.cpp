// ============================================================
// 版本1：Naive GEMM - 三重循环朴素实现
// ============================================================
// 知识点：为什么慢？
//
// 内存布局：C/C++中矩阵按行存储（Row-major）
//   A[i][j] 在内存中 = A[i*n + j]
//   一行的元素在内存中是连续的
//
// 问题分析（以ijk顺序为例）：
//   最内层循环 k，访问 B[k][j]
//   B矩阵是按列访问的！相邻k值对应的B[k][j]和B[k+1][j]
//   在内存中相差n个float（即n*4字节）
//   当n=512时，步长=2048字节，远超Cache line（64字节）
//   → 每次访问B都是Cache miss → 性能极差
//
// Cache基础知识：
//   L1 Cache: ~32KB,  延迟4个时钟周期
//   L2 Cache: ~256KB, 延迟12个时钟周期
//   L3 Cache: ~8MB,   延迟40个时钟周期
//   主内存:   ~GB,    延迟200+个时钟周期
//   → Cache miss代价是命中的50倍！
// ============================================================

#include "gemm.h"

// 最朴素的实现，ijk顺序
// 问题：B矩阵按列访问，大量Cache miss
void gemm_naive_ijk(const float* A, const float* B, float* C, int n) {
    // 清零输出矩阵
    memset(C, 0, n * n * sizeof(float));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                // A[i][k]: i固定，k递增 → 访问A的一行 → 连续 ✓
                // B[k][j]: j固定，k递增 → 访问B的一列 → 跨行 ✗ Cache miss!
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// ============================================================
// 版本2：循环重排 - ikj顺序
// ============================================================
// 关键改动：把 j 和 k 的循环顺序交换
//
// 分析ikj顺序：
//   最内层循环 j，访问 B[k][j]
//   k固定，j递增 → 访问B的一行 → 连续 ✓
//   C[i][j] 也是连续访问 ✓
//   A[i][k] 是常数（k和i都固定），被编译器提到寄存器中 ✓
//
// 结论：三个操作数都Cache友好，性能显著提升
//
// 用godbolt.org验证：
//   把这个函数粘贴到 https://godbolt.org
//   编译器选 g++ -O3 -march=native
//   观察是否生成了 vmovups/vfmadd 等向量指令（自动向量化）
// ============================================================
void gemm_ikj(const float* A, const float* B, float* C, int n) {
    memset(C, 0, n * n * sizeof(float));

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            // 把A[i][k]提出来作为临时变量
            // 编译器会将其放入寄存器，避免重复内存访问
            float a_ik = A[i * n + k];
            for (int j = 0; j < n; j++) {
                // 这一行的访问模式：
                // C[i][j]: i固定，j递增 → 连续 ✓
                // B[k][j]: k固定，j递增 → 连续 ✓
                // a_ik: 寄存器 ✓
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}

// ============================================================
// 版本3：分块优化（Blocking / Tiling）
// ============================================================
// 问题：即使ikj顺序，当n很大时（如2048），
//       A和B的工作集超过L2 Cache，仍然有Cache miss
//
// 解决思路：把大矩阵切成小块，每次处理一个小块
//           让小块数据在Cache中保持活跃，实现数据复用
//
// 分块大小选择原则：
//   3个块（A块+B块+C块）同时在L1 Cache中
//   每块大小 = sqrt(L1_size / 3 / sizeof(float))
//   L1=32KB: sqrt(32*1024 / 3 / 4) ≈ 52
//   → 通常选32或64作为tile size
//
// 性能提升原因（Roofline模型角度）：
//   Naive: 每次乘加都需要从内存取数，算术强度低
//   Tiling: 每个数据从内存取一次后，在Cache中被复用TILE次
//           算术强度提升TILE倍，逼近计算bound而非内存bound
// ============================================================

#define TILE 32  // 分块大小，可以尝试16/32/64

void gemm_tiling(const float* A, const float* B, float* C, int n) {
    memset(C, 0, n * n * sizeof(float));

    // 外层三重循环：遍历块
    for (int i = 0; i < n; i += TILE) {
        for (int k = 0; k < n; k += TILE) {
            for (int j = 0; j < n; j += TILE) {

                // 内层三重循环：在块内计算
                // 这里的数据（A块和B块）尺寸是TILE*TILE*4字节
                // 三个块合计 3*32*32*4 = 12KB，可以放进L1 Cache
                int i_end = std::min(i + TILE, n);
                int k_end = std::min(k + TILE, n);
                int j_end = std::min(j + TILE, n);

                for (int ii = i; ii < i_end; ii++) {
                    for (int kk = k; kk < k_end; kk++) {
                        float a_ik = A[ii * n + kk];
                        for (int jj = j; jj < j_end; jj++) {
                            C[ii * n + jj] += a_ik * B[kk * n + jj];
                        }
                    }
                }
            }
        }
    }
}
