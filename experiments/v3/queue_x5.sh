#!/bin/zsh
# X5 (PLAN.md): out-of-scope gate recipe (MobileNetV2 1.4, label smoothing 0.1, early stop data/val),
# control on data/train vs hard negatives (train_hn = data/train + healthy-foot pool). Usage: queue_x5.sh "Gctrl:1 Ghn:1"
EXP=${0:A:h:h}; PY=$EXP/../.venv/bin/python
for job in ${=1}; do
  name=X5_${job%%:*}; s=${job##*:}
  [[ $name == X5_Ghn ]] && TD=(--train-dir $EXP/v3/train_hn) || TD=()
  [[ -f $EXP/runs/${name}_s$s.json ]] && { echo "skip ${name}_s$s"; continue; }
  $PY -u $EXP/experiment9.py --name $name --seed $s --alpha 1.4 --label-smoothing 0.1 $TD --save-model $EXP/v3/models/${name}_s$s.keras > $EXP/v3/logs/${name}_s$s.log 2>&1 || echo "FAILED ${name}_s$s"
  grep -h RESULT $EXP/v3/logs/${name}_s$s.log
done
echo QUEUE_X5_DONE
