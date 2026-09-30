#!/usr/bin/env bash
set -euo pipefail

repo=/workspace/jjy-exp/GLU-exp
klu=/workspace/jjy-exp/KLU_standalone/klu_demo
matrix="$repo/src/matrix/add32_csr.mtx"

# 创建新目录，避免覆盖以前的结果。
results=$(mktemp -d "$repo/experiments/v2/add32-repeat-XXXXXX")
echo "结果目录：$results"

/opt/htdriver/bin/ht-smi > "$results/device-before.txt"

for round in 0 1 2 3 4 5; do
    mkdir "$results/round-$round"
    (
        cd "$results/round-$round"

        timeout 120s "$klu" "$matrix" x_klu.txt > klu.txt 2>&1
        timeout 120s "$repo/src/lu_cmd" -i "$matrix" > glu.txt 2>&1

        python3 "$repo/experiments/compare_solution.py" \
            x.dat x_klu.txt > error.txt 2>&1

        echo "===== round $round ====="
        awk '/Total solve wall time/ {print FILENAME ": " $0}' klu.txt glu.txt
        cat error.txt
    )
done

/opt/htdriver/bin/ht-smi > "$results/device-after.txt"