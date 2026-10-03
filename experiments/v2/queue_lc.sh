#!/bin/zsh
EXP=${0:A:h:h}
PY=$EXP/../.venv/bin/python
for frac in 25 50; do for s in 1 2 3; do
  name=LC_${frac}pct
  [[ -f $EXP/runs/${name}_s$s.json ]] && continue
  $PY -u $EXP/experiment9.py --name $name --seed $s --alpha 1.4 --label-smoothing 0.1 --train-dir $EXP/v2/b84_train_$frac --val-dir $EXP/v2/b84_val > $EXP/logs/${name}_s$s.log 2>&1 || echo "FAILED ${name}_s$s"
  grep -h RESULT $EXP/logs/${name}_s$s.log
done; done
echo LC_DONE
