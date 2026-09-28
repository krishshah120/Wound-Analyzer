#!/bin/zsh
# X4b (PLAN.md): deployed recipe on the merged taxonomy (abrasion, bruise, burn, cut, out_of_scope). Usage: queue_x4b.sh SEED...
EXP=${0:A:h:h}; PY=$EXP/../.venv/bin/python
for s in "$@"; do
  [[ -f $EXP/runs/X4B_s$s.json ]] && { echo "skip X4B_s$s"; continue; }
  $PY -u $EXP/experiment9.py --name X4B --seed $s --alpha 1.4 --label-smoothing 0.1 --train-dir $EXP/v3/b84_train_burn --val-dir $EXP/v3/b84_val_burn --save-model $EXP/v3/models/X4B_s$s.keras > $EXP/v3/logs/X4B_s$s.log 2>&1 || echo "FAILED X4B_s$s"
  grep -h RESULT $EXP/v3/logs/X4B_s$s.log
done
echo QUEUE_X4B_DONE
