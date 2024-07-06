#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00

WORK_DIR="/tmp/2024-05-10_LEAP/logs"
LOG_DIR="2024-07-06_18-05-38"
SOURCE_DIR="/mnt/pool/6/ovkomleva/LEAP"

cp -r "$WORK_DIR/$LOG_DIR" "$SOURCE_DIR/logs"
