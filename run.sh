#!/bin/bash

# Generate timestamp for log file
TIMESTAMP=$(date +"%m%d_%H%M%S")
LOG_FILE="Logs/${TIMESTAMP}.log"

# Run the Python script with argparse inputs
C:/Users/aaaa/miniconda3/envs/lane/python.exe d:/scripts/lane-and-road-marking-detection/pretrain_vpp.py \
   --timestamp $TIMESTAMP \
   --phase 1 \
   -lr 1e-3 \
   --max_lr 1 \
   --batch_size 30 
#  > $LOG_FILE 2>&1

# Check the exit status of the Python script
if [ $? -eq 0 ]; then
    echo -e "\033[1;32m Script ran successfully. Log saved in $LOG_FILE \033[0m";
else
    echo -e "\033[1;31m Error: Script encountered a problem. See $LOG_FILE for details. \033[0m";
fi