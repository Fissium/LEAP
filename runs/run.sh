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

WORKDIR="/tmp/2024-05-10_LEAP"
CONDA_DIR="$WORKDIR/miniforge3/bin/activate"
RUN_SCRIPT="$WORKDIR/src/run.py"
VENV_NAME="$WORKDIR/.venv"
LOGS_DIR="/mnt/pool/5/ovkomleva/LEAP/logs"
EXCLUDE_DIRS=("data" "logs" "notebooks" "runs")

mkdir -p "$WORKDIR/logs"

rsync -av --exclude="${EXCLUDE_DIRS[*]}" . "$WORKDIR"

cd $WORKDIR || exit 1
source "$CONDA_DIR" "$VENV_NAME"

if python $RUN_SCRIPT; then

  LATEST_LOG_DIR=$(ls -td "$WORKDIR/logs"/*/ | head -1)
  if [ -d "$LATEST_LOG_DIR" ]; then
    cp -r "$LATEST_LOG_DIR" "$LOGS_DIR"
  else
    echo "No directories found in logs."
  fi
else
  echo "Python script did not finish successfully."
fi
