#!/bin/bash
# shellcheck disable=SC1090
# shellcheck disable=SC2012
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00

WORK_DIR="/tmp/2024-05-10_LEAP"
SOURCE_DIR="/mnt/pool/6/ovkomleva/LEAP/"
CONDA_DIR="$WORK_DIR/miniforge3/bin/activate"
RUN_SCRIPT="$SOURCE_DIR/src/run.py"
VENV_NAME="$WORK_DIR/.venv"

cd $WORK_DIR || exit 1
source "$CONDA_DIR" "$VENV_NAME"

python $RUN_SCRIPT
