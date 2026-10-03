#!/bin/zsh
# v6: wait for archive-8 manifest, build training folders, train N6 then N6k, then views for all models.
V6=${0:A:h}; EXP=${V6:h}; PY=$EXP/../.venv/bin/python; SRV=~/Downloads/wound-analyzer-vercel/models
until [[ -f $V6/a8_manifest.csv ]]; do sleep 30; done
grep "split:" $V6/logs/prep_archive8.log
$PY $V6/build_train.py || exit 1
for N in N6 N6k; do
  $PY -u $EXP/experiment9.py --name $N --seed 1 --alpha 1.4 --label-smoothing 0.1 --train-dir $V6/train_$N --val-dir $EXP/v2/b84_val --save-model $V6/models/${N}_s1.keras > $V6/logs/${N}_s1.log 2>&1 || echo "FAILED $N"
  grep -h RESULT $V6/logs/${N}_s1.log
done
for pair in deployed:$SRV/wound_model.tflite shipped:$SRV/out_of_scope_gate.tflite healthy:$SRV/out_of_scope_gate_healthy.tflite N6:$V6/models/N6_s1.keras N6k:$V6/models/N6k_s1.keras; do
  $PY $V6/views6.py ${pair%%:*} ${pair#*:} > $V6/logs/views6_${pair%%:*}.log 2>&1 || echo "VIEWS FAILED ${pair%%:*}"
done
echo QUEUE6_DONE
