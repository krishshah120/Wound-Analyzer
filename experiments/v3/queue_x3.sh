#!/bin/zsh
# X3 (PLAN.md): deployed recipe on b84_train minus the 86 objective-conflict photos. Usage: queue_x3.sh SEED...
EXP=${0:A:h:h}; PY=$EXP/../.venv/bin/python
A=(--alpha 1.4 --label-smoothing 0.1 --train-dir $EXP/v3/b84_train_x3 --val-dir $EXP/v2/b84_val)
for s in "$@"; do
  [[ -f $EXP/runs/X3B_s$s.json ]] && { echo "skip X3B_s$s"; continue; }
  $PY -u $EXP/experiment9.py --name X3B --seed $s $A --save-model $EXP/v3/models/X3B_s$s.keras > $EXP/v3/logs/X3B_s$s.log 2>&1 || echo "FAILED X3B_s$s"
  grep -h RESULT $EXP/v3/logs/X3B_s$s.log
done
echo QUEUE_X3_DONE
