#!/bin/bash

ALGO_ANCILLARY="SGD"
ALGO_MAIN="Ave"
BATCH="8"
CDFSIZE="250"
DATA="adult"
ENTROPY="256117190779556056928268872043329970341"
EPOCHS="60"
LOSS="logistic"
MODEL="linreg_multi"
STEP="1.0"
TASK="fast"
TRIALS="10"

python "learn_driver.py" --algo-ancillary="$ALGO_ANCILLARY" --algo-main="$ALGO_MAIN" --batch-size="$BATCH" --cdf-size="$CDFSIZE" --data="$DATA" --entropy="$ENTROPY" --fast --loss="$LOSS" --model="$MODEL" --num-epochs="$EPOCHS" --num-trials="$TRIALS" --step-size="$STEP" --task-name="$TASK"
