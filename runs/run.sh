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
SOURCE_DIR="/mnt/pool/5/ovkomleva/LEAP/"
CONDA_DIR="$WORK_DIR/miniforge3/bin/activate"
RUN_SCRIPT="$WORK_DIR/src/run.py"
VENV_NAME="$WORK_DIR/.venv"
EXCLUDE_DIRS=("data" "logs" "notebooks" "runs" ".venv" ".git")

rsync -av --exclude="${EXCLUDE_DIRS[*]}" "$SOURCE_DIR" "$WORK_DIR"

cd $WORK_DIR || exit 1
source "$CONDA_DIR" "$VENV_NAME"

if python $RUN_SCRIPT; then

  LATEST_LOG_DIR=$(ls -td "$WORK_DIR/logs"/*/ | head -1)
  if [ -d "$LATEST_LOG_DIR" ]; then
    cp -r "$LATEST_LOG_DIR" "$SOURCE_DIR/logs"
  else
    echo "No directories found in logs."
  fi
else
  echo "Python script did not finish successfully."
fi
