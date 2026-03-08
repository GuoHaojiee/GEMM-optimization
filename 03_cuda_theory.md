# 知识点3：CUDA性能优化核心概念

## GPU vs CPU架构差异

```
CPU：少核心，每个核心很强
  - 4~32个大核
  - 深度流水线，分支预测，乱序执行
  - 适合：延迟敏感的串行任务

GPU：多核心，每个核心简单
  - 数千个小核（CUDA Core）
  - 通过海量线程隐藏内存延迟
  - 适合：高度并行的计算密集任务
```

---

## 关键概念：Warp

- GPU中32个Thread为一组，称为Warp，**同时执行相同指令**
- 这是GPU调度的基本单位（SIMT：Single Instruction Multiple Thread）
- 分支（if/else）如果Warp内不同Thread走不同分支 → **Warp Divergence**，性能下降

```
Warp内32个Thread：
  Thread 0~31 同时执行 fmadd ...
                        ↑ 同一条指令
```

---

## 内存合并访问（Memory Coalescing）

**最重要的CUDA优化原则之一**

```
// 好的访问模式（合并）：
// Warp中Thread i访问 arr[blockIdx*32 + i]
// 32个Thread访问连续的32个float = 128字节 = 2个Cache Line
// 硬件合并为1~2次内存事务

// 差的访问模式（非合并）：
// Thread i访问 arr[i * stride]（stride>1）
// 每个Thread访问不同的Cache Line
// 需要32次内存事务
```

在GEMM中：
- 访问B矩阵的同一列 → 非合并（stride=N）→ 性能差
- Shared Memory优化后 → 合并加载A和B的行 → 性能好

---

## Shared Memory优化的本质

```
Global Memory带宽（A100）: ~2 TB/s
Shared Memory带宽:         ~19 TB/s（快10倍）

Naive GEMM：
  计算 N×N 的C，每个元素需要2N次Global Memory读
  总读取量 = N³ × 8字节（大量重复读）

Shared Memory GEMM（Tile=T）：
  每个Tile把A[T×T]和B[T×T]从Global搬到Shared
  在Shared中复用T次
  Global Memory读取量减少T倍
```

---

## Occupancy（占用率）

- Occupancy = 实际活跃Warp数 / SM最大支持Warp数
- 高Occupancy → 可以用其他Warp的计算隐藏内存延迟
- 影响Occupancy的因素：
  - 每个Thread用的寄存器数（寄存器是有限资源）
  - 每个Block用的Shared Memory量
  - Block大小

```bash
# 用nvcc内置工具查看理论Occupancy
nvcc --ptxas-options=-v gemm_cuda_shared.cu
# 输出类似：
# ptxas info: Used 32 registers, 2048+0 bytes smem, ...
# 然后用 CUDA Occupancy Calculator 计算
```

---

## 常用性能分析命令

```bash
# 1. 基础性能分析
ncu ./cuda_bench

# 2. 查看显存带宽利用率（最重要的指标之一）
ncu --metrics dram__bytes_read.sum,\
              dram__bytes_write.sum,\
              dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./cuda_bench

# 3. 查看Shared Memory效果
ncu --metrics \
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,\
    smsp__sass_average_data_bytes_per_sector_mem_shared.pct \
    ./cuda_bench

# 4. 查看Warp Divergence
ncu --metrics smsp__thread_inst_executed_pred_on.sum ./cuda_bench

# 5. nvprof（旧版，但更容易用）
nvprof ./cuda_bench
```

---

## 进阶方向（如果时间充裕）

1. **Double Buffering**：在计算当前Tile时，预加载下一个Tile
   → 隐藏Global Memory延迟，效果显著

2. **向量化内存加载**：用 `float4` 一次加载4个float
   ```cuda
   float4 val = reinterpret_cast<const float4*>(&A[idx])[0];
   ```

3. **寄存器级分块**：每个Thread计算更多输出元素
   → 减少同步次数，提升寄存器复用

---

## 面试话术

> "我用Nsight Compute分析了Naive版本，
>  发现dram__bytes.sum是理论最优值的约16倍，
>  原因是B矩阵的列访问导致无法合并，
>  每32个Thread的内存事务数是合并情况的32倍。
>  引入Shared Memory Tiling后，
>  通过协同加载将合并访问率提升到接近100%，
>  dram__bytes降至理论值的1.1倍，
>  性能从约X GFLOPS提升到Y GFLOPS，
>  达到cuBLAS的约65%。"
