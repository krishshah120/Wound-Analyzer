#!/bin/zsh
# X6 (PLAN.md addendum): RE3 members = deployed recipe, seeds 1-3, weights saved; then X6C pilot
# (b84_train minus the 24 identical-image conflicts). Resumable.
EXP=${0:A:h:h}; PY=$EXP/../.venv/bin/python
R=(--alpha 1.4 --label-smoothing 0.1 --val-dir $EXP/v2/b84_val)
run() { local name=$1 s=$2 td=$3
  [[ -f $EXP/runs/${name}_s$s.json ]] && { echo "skip ${name}_s$s"; return; }
  $PY -u $EXP/experiment9.py --name $name --seed $s $R --train-dir $td --save-model $EXP/v3/models/${name}_s$s.keras > $EXP/v3/logs/${name}_s$s.log 2>&1 || echo "FAILED ${name}_s$s"
  grep -h RESULT $EXP/v3/logs/${name}_s$s.log; }
for s in 1 2 3; do run RE3 $s $EXP/v2/b84_train; done
for s in ${@:-1}; do run X6C $s $EXP/v3/b84_train_t1; done
echo QUEUE_X6_DONE
