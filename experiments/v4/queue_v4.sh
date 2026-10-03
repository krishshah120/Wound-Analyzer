#!/bin/zsh
# v4 (PLAN.md): gate recipe, control on data/train vs data/train + new healthy-skin photos. Usage: queue_v4.sh "G4ctrl:1 G4new:1"
EXP=${0:A:h:h}; PY=$EXP/../.venv/bin/python
for job in ${=1}; do
  name=${job%%:*}; s=${job##*:}
  [[ $name == G4new ]] && TD=(--train-dir $EXP/v4/train_v4g) || TD=()
  [[ -f $EXP/runs/${name}_s$s.json ]] && { echo "skip ${name}_s$s"; continue; }
  $PY -u $EXP/experiment9.py --name $name --seed $s --alpha 1.4 --label-smoothing 0.1 $TD --save-model $EXP/v4/models/${name}_s$s.keras > $EXP/v4/logs/${name}_s$s.log 2>&1 || echo "FAILED ${name}_s$s"
  grep -h RESULT $EXP/v4/logs/${name}_s$s.log
  (cd $EXP/v3 && $PY -u views.py ${name}_s$s --gate $EXP/v4/models/${name}_s$s.keras > $EXP/v4/logs/views_${name}_s$s.log 2>&1) || echo "VIEWS FAILED ${name}_s$s"
done
echo QUEUE_V4_DONE
