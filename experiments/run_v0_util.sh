#!/usr/bin/env bash
set -euo pipefail

repo=/workspace/jjy-exp/GLU-exp
results=$(mktemp -d "$repo/experiments/v0/add32-util-load-XXXXXX")
echo "结果目录：$results"

# 开始监控；退出脚本时只停止这次启动的监控进程。
/opt/htdriver/bin/ht-smi -l 100 \
  -o "$results/util.csv" --show-usage \
  > "$results/monitor.log" 2>&1 &
monitor_pid=$!

cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

# 留一小段空闲记录，便于观察负载开始的位置。
sleep 1
kill -0 "$monitor_pid"

python3 -c 'from datetime import datetime; print(datetime.now().isoformat())' \
  > "$results/load-start.txt"

for round in $(seq 1 100); do
  mkdir "$results/round-$round"
  (
    cd "$results/round-$round"
    timeout 120s "$repo/src/lu_cmd" \
      -i "$repo/src/matrix/add32_csr.mtx" > glu.txt 2>&1
  )
done

python3 -c 'from datetime import datetime; print(datetime.now().isoformat())' \
  > "$results/load-end.txt"

sleep 1
cleanup
trap - EXIT

# 在监控结束后检查每次的解，避免比较脚本打断求解负载。
for round in $(seq 1 100); do
  python3 "$repo/experiments/compare_solution.py" \
    "$results/round-$round/x.dat" \
    "$repo/experiments/v0/klu-add32-stages/x_klu.txt" \
    > "$results/round-$round/error.txt" 2>&1
done

echo "100 次求解及正确性检查全部完成"