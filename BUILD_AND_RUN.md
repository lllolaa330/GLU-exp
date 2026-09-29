# GLU v3.0 编译与运行指南

## 环境要求

- **操作系统**: Linux
- **编译器**: GCC/G++ (支持 C++11)
- **CUDA**: NVIDIA CUDA Toolkit 12.x，GPU 架构 ≥ sm_60 (Pascal)
- **依赖**: pthreads, libm, librt

## 编译

```bash
cd src/
make_hpcc clean    # 清理旧编译产物
make_hpcc MAIN     # 编译 lu_cmd
```

编译后在 `src/` 下生成可执行文件 `lu_cmd`。

## 运行

```bash
./lu_cmd -i matrix/add32_csr.mtx        # 标准 GESP 求解

```

- `-i <file>`: 指定输入矩阵文件（CSR 排序的 Matrix Market 格式）

- 解向量输出到 `x.dat`（每行一个值）
- 右端项隐式为 `b = [1, 1, ..., 1]`
- 运行结束后输出残差 `||Ax - b||` 的 1-范数、2-范数、无穷范数

## 矩阵文件

`src/matrix/` 目录下包含 13 个测试矩阵，均为 CSR 排序格式：

| 矩阵 | 阶数 n | 非零元 | 规模 |
|------|--------|--------|------|
| add32 | 4,960 | 23,884 | 微型 |
| rajat13 | 7,598 | 48,922 | 小型 |
| dianwangmatrix1 | 16,327 | 75,827 | 小型 |
| rajat27 | 20,640 | 99,777 | 中小型 |
| rajat26 | 51,032 | 249,302 | 中型 |
| bcircuit | 68,902 | 375,558 | 中型 |
| rajat25 | 87,190 | 607,235 | 中大型 |
| ASIC_100ks | 99,190 | 578,890 | 大型 |
| ASIC_100k | 99,340 | 954,163 | 大型 |
| twotone | 120,750 | 1,224,224 | 大型 |
| dianwangmatrix2 | 126,905 | 542,895 | 大型 |
| G2_circuit | 150,102 | 438,388 | 大型 |
| ASIC_680ks | 682,712 | 2,329,176 | 超大型 |

## 示例

```bash
$ cd src/
$ make clean && make MAIN
$ ./lu_cmd -i matrix/add32_csr.mtx
Reading matrix...
Preprocessing matrix...
Preprocessing time: 2.522 ms
Matrix Row: 4960
Original nonzero: 23884
Symbolic nonzero: 23942
Symbolic time: 1.282 ms
CSR time: 1.214 ms
PredictLU time: 0.276 ms
Number of levels: 54
Leveling time: 0.085 ms
Device 0: NVIDIA A800 80GB PCIe has been selected.
Total GPU time: 20.539 ms
Ax-b (1-norm): 6.527e-12
Ax-b (2-norm): 1.915e-13
Ax-b (infinite-norm): 3.553e-14
```

## 精度说明

- 使用双精度浮点（`double`）

-

