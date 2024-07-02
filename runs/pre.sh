#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00

WORK_DIR="/tmp/2024-05-10_LEAP"
INIT_DIR="/mnt/pool/5/ovkomleva/LEAP"

mkdir -p "$WORK_DIR/data"
chmod 700 "$WORK_DIR"
cp -r "$INIT_DIR/data/train" "$WORK_DIR/data"
cp "$INIT_DIR/data/test.csv" "$WORK_DIR/data"
cp "$INIT_DIR/data/sample_submission.csv" "$WORK_DIR/data"
cp -r "$INIT_DIR/.venv" "$WORK_DIR"
cp -r /mnt/pool/5/ovkomleva/metric_learning/miniforge3 "$WORK_DIR"
