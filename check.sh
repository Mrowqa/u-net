#!/bin/bash

EVAL_DIR=/d/students/dnn/titan-dump/eval_output
CHECK_SH="python eval_score.py"

for d in $EVAL_DIR/*valid*; do
    echo $d
    $CHECK_SH $d
done

