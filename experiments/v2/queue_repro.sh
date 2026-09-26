#!/bin/zsh
EXP=${0:A:h:h}
PY=$EXP/../.venv/bin/python
run() {
  local name=$1 seed=$2; shift 2
  [[ -f $EXP/runs/${name}_s$seed.json ]] && { echo "skip ${name}_s$seed"; return; }
  $PY -u $EXP/experiment9.py --name $name --seed $seed "$@" > $EXP/logs/${name}_s$seed.log 2>&1 || echo "FAILED ${name}_s$seed"
  grep -h RESULT $EXP/logs/${name}_s$seed.log
}
A=(--alpha 1.4 --label-smoothing 0.1 --train-dir $EXP/v2/b84_train --val-dir $EXP/v2/b84_val)
run R_exact_a 42 $A
run R_exact_b 42 $A
for s in 1 2 3; do run R_exact $s $A; done
echo QUEUE_REPRO_DONE
