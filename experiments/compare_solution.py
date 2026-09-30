import math
import sys

def read_vector(path):
    with open(path) as f:
        return [float(line) for line in f if line.strip()]

x_glu = read_vector(sys.argv[1])
x_klu = read_vector(sys.argv[2])

# 先检查文件，避免把缺失数据或 NaN 当作正常结果。
if not x_glu or len(x_glu) != len(x_klu):
    raise ValueError("解向量为空或长度不一致")
if not all(math.isfinite(v) for v in x_glu + x_klu):
    raise ValueError("解向量中存在 NaN 或 Inf")

diff_norm = math.sqrt(math.fsum(
    (g - k) ** 2 for g, k in zip(x_glu, x_klu)
))
ref_norm = math.sqrt(math.fsum(k ** 2 for k in x_klu))
if ref_norm == 0:
    raise ValueError("参考解为零，不能使用此相对误差公式")

error = diff_norm / ref_norm
passed = math.isfinite(error) and error <= 1e-6

print(f"n = {len(x_glu)}")
print(f"relative_l2_error = {error:.17e}")
print(f"correctness = {'PASS' if passed else 'FAIL'}")
sys.exit(0 if passed else 1)