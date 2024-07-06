#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00

WORK_DIR="/tmp/2024-05-10_LEAP"
SOURCE_DIR="/mnt/pool/6/ovkomleva/LEAP"

mkdir -p "$WORK_DIR/data"
mkdir -p "$WORK_DIR/logs"
chmod 700 "$WORK_DIR"
cp -r "$SOURCE_DIR/data/train" "$WORK_DIR/data"
cp "$SOURCE_DIR/data/test.csv" "$WORK_DIR/data"
cp "$SOURCE_DIR/data/sample_submission.csv" "$WORK_DIR/data"
cp -r "$SOURCE_DIR/.venv" "$WORK_DIR"
cp -r /mnt/pool/5/ovkomleva/metric_learning/miniforge3 "$WORK_DIR"
