#!/bin/bash

ALGO_ANCILLARY="SGD"
ALGO_MAIN="Ave"
BATCH="8"
CDFSIZE="250"
DATA="$1"
ENTROPY="256117190779556056928268872043329970341"
EPOCHS="50"
LOSS="logistic"
MODEL="linreg_multi"
STEP="1.0"
TASK="off"
TRIALS="10"

python "learn_driver.py" --algo-ancillary="$ALGO_ANCILLARY" --algo-main="$ALGO_MAIN" --batch-size="$BATCH" --cdf-size="$CDFSIZE" --data="$DATA" --entropy="$ENTROPY" --loss="$LOSS" --model="$MODEL" --no-srisk --num-epochs="$EPOCHS" --num-trials="$TRIALS" --step-size="$STEP" --task-name="$TASK"




