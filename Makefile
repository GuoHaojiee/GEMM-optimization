CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -march=native -mavx2 -mfma -std=c++17 -I include
NVCCFLAGS = -O3 -arch=sm_86 -std=c++17 -I include
# 根据你的GPU修改sm版本：
#   sm_75: GTX 16xx / RTX 20xx / Tesla T4
#   sm_80: A100
#   sm_86: RTX 30xx
#   sm_89: RTX 40xx

# ---- CPU 目标 ----
cpu: benchmark

benchmark: benchmark.cpp gemm_naive.cpp gemm_simd.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@
	@echo "✓ CPU benchmark built. Run: ./benchmark"

# 查看汇编（用于分析向量化情况）
asm:
	$(CXX) $(CXXFLAGS) -S src/gemm_naive.cpp -o /tmp/gemm_naive.s
	@echo "Assembly written to /tmp/gemm_naive.s"
	@echo "Look for vmovups/vfmadd to confirm AVX2 vectorization:"
	@grep -c "ymm" /tmp/gemm_naive.s && echo "✓ YMM registers found (AVX2)" || echo "✗ No YMM registers"

# ---- CUDA 目标 ----
cuda: cuda_bench

cuda_bench: cuda/gemm_cuda_bench.cu cuda/gemm_cuda_naive.cu cuda/gemm_cuda_shared.cu
	$(NVCC) $(NVCCFLAGS) cuda/gemm_cuda_naive.cu cuda/gemm_cuda_shared.cu \
	        cuda/gemm_cuda_bench.cu -lcublas -o $@
	@echo "✓ CUDA benchmark built. Run: ./cuda_bench"

# ---- 性能分析 ----
profile_cpu:
	perf stat -e cache-misses,cache-references,instructions,cycles ./benchmark

profile_cuda:
	ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,\
	    sm__throughput.avg.pct_of_peak_sustained_active \
	    ./cuda_bench

# ---- 清理 ----
clean:
	rm -f benchmark cuda_bench /tmp/gemm_*.s

.PHONY: cpu cuda asm profile_cpu profile_cuda clean
