#!/bin/bash

SAVEDIR=$1
EXP_NAME=$2

# Try to determine base path automatically
CUDA_LAUNCH_BLOCKING=1 python gui.py --data "$SAVEDIR" --expname "$EXP_NAME" 2>&1 | tee crash.log