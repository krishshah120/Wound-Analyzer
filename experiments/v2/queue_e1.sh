#!/bin/zsh
EXP=${0:A:h:h}
PY=$EXP/../.venv/bin/python
seed=${1:-1}
name=E1_effv2s
[[ -f $EXP/runs/${name}_s$seed.json ]] && { echo "skip ${name}_s$seed"; exit 0; }
$PY -u $EXP/experiment9.py --name $name --seed $seed --backbone efficientnetv2s --label-smoothing 0.1 \
  --train-dir $EXP/v2/b84_train --val-dir $EXP/v2/b84_val > $EXP/logs/${name}_s$seed.log 2>&1 || echo "FAILED ${name}_s$seed"
grep -h RESULT $EXP/logs/${name}_s$seed.log
echo E1_DONE_s$seed
