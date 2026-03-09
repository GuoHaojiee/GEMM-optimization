cmake_minimum_required(VERSION 3.18)
project(GEMM_Optimization LANGUAGES CXX)

# ============================================================
# 基本设置
# ============================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # 生成 compile_commands.json（IDE 用）

# 默认 Release 构建
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================
# CUDA 检测（可选）
# ============================================================
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA 编译器：${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA 版本：${CMAKE_CUDA_COMPILER_VERSION}")

    # CUDA 架构（可通过 cmake -DCUDA_ARCH=sm_80 指定）
    # 查询当前 GPU 架构：nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    # sm_75: RTX 2080, T4（Turing）
    # sm_80: A100（Ampere）
    # sm_86: RTX 3090（Ampere）
    # sm_89: RTX 4090（Ada Lovelace）
    set(CUDA_ARCH "sm_75" CACHE STRING "CUDA 目标架构（例：sm_75, sm_80, sm_86）")
    message(STATUS "CUDA 架构：${CUDA_ARCH}")
else()
    message(STATUS "未找到 CUDA 编译器，跳过 GPU 目标")
endif()

# ============================================================
# 头文件目录
# ============================================================
include_directories(${CMAKE_SOURCE_DIR}/include)

# ============================================================
# CPU Benchmark 目标
# ============================================================

# CPU 源文件
set(CPU_SOURCES
    src/gemm_v1_naive.cpp
    src/gemm_v2_ikj.cpp
    src/gemm_v3_tiling.cpp
    src/gemm_v4_avx2.cpp
    src/gemm_v5_avx2_tiling.cpp
    src/gemm_v6_openmp.cpp
    src/benchmark_cpu.cpp
)

add_executable(benchmark_cpu ${CPU_SOURCES})

target_compile_options(benchmark_cpu PRIVATE
    -O3                  # 最高优化级别（开启自动向量化、循环展开等）
    -march=native        # 使用当前 CPU 支持的所有指令集（不可移植，但性能最佳）
    -mavx2               # 显式开启 AVX2（通常 march=native 已包含）
    -mfma                # 显式开启 FMA 指令
    -fopenmp             # OpenMP 多线程支持（V6 需要）
    -funroll-loops       # 循环展开（可选，通常对 GEMM 有帮助）
    # -ffast-math        # 宽松浮点（允许重排，可能影响数值精度，谨慎使用）
)

target_link_libraries(benchmark_cpu PRIVATE
    -fopenmp             # 链接 OpenMP 运行时
)

# ============================================================
# CUDA Benchmark 目标（如果 CUDA 可用）
# ============================================================

if(CMAKE_CUDA_COMPILER)
    # CUDA 源文件
    set(CUDA_SOURCES
        cuda/gemm_v7_naive.cu
        cuda/gemm_v8_shared.cu
        cuda/gemm_v9_register.cu
        cuda/gemm_v10_vectorized.cu
        cuda/cublas_reference.cu
        cuda/benchmark_cuda.cu
    )

    add_executable(benchmark_cuda ${CUDA_SOURCES})

    # CUDA 架构设置（两种写法等价）
    set_target_properties(benchmark_cuda PROPERTIES
        CUDA_ARCHITECTURES "75"  # 对应 sm_75（去掉 "sm_" 前缀）
    )

    target_compile_options(benchmark_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math         # GPU 端的 fast math（相当于 -ffast-math）
            -lineinfo               # 保留行信息（用于 Nsight 调试）
            --generate-line-info    # 同上
        >
        $<$<COMPILE_LANGUAGE:CXX>:
            -O3
        >
    )

    target_link_libraries(benchmark_cuda PRIVATE
        cublas      # cuBLAS 库（V11 需要）
        cudart      # CUDA 运行时
    )

    message(STATUS "CUDA benchmark_cuda 目标已配置")
endif()

# ============================================================
# 输出编译信息
# ============================================================

message(STATUS "")
message(STATUS "======================================")
message(STATUS "GEMM 优化项目配置摘要")
message(STATUS "======================================")
message(STATUS "构建类型：${CMAKE_BUILD_TYPE}")
message(STATUS "C++ 编译器：${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA 编译器：${CMAKE_CUDA_COMPILER_VERSION}")
    message(STATUS "CUDA 目标架构：${CUDA_ARCH}")
endif()
message(STATUS "======================================")
message(STATUS "")
message(STATUS "编译命令：")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=sm_75")
message(STATUS "  make -j\$(nproc)")
message(STATUS "")
message(STATUS "运行命令：")
message(STATUS "  ./build/benchmark_cpu --size 512 1024 2048")
if(CMAKE_CUDA_COMPILER)
    message(STATUS "  ./build/benchmark_cuda --size 512 1024 2048")
endif()
message(STATUS "======================================")
