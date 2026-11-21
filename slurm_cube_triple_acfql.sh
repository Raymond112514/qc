#!/bin/bash

#SBATCH --job-name=qc_fql_h
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=/global/scratch/users/r112358/qc-exp/logs/%x_%A_%a.out
#SBATCH --error=/global/scratch/users/r112358/qc-exp/logs/%x_%A_%a.err
#SBATCH --exclude=n0141.savio4,n0142.savio4

# QC-FQL on cube-triple-play-singletask-task2-v0.
# Same flags as the README QC-FQL command, with horizon_length 10, 25, and 50.
# Action dim is 5, so the policy outputs 50, 125, and 250.

source ~/.bashrc
set -eo pipefail

export CONDA_PREFIX=/global/scratch/users/r112358/miniconda3/envs/qc
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
PY="${CONDA_PREFIX}/bin/python"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONFAULTHANDLER=1

# Home is on a tight user quota. Keep wandb, JAX, and XDG caches on scratch.
export SCRATCH_BASE=/global/scratch/users/r112358
export WANDB_DIR="${SCRATCH_BASE}/wandb"
export WANDB_CACHE_DIR="${SCRATCH_BASE}/cache/wandb"
export WANDB_CONFIG_DIR="${SCRATCH_BASE}/cache/wandb-config"
export XDG_CACHE_HOME="${SCRATCH_BASE}/cache"
export JAX_COMPILATION_CACHE_DIR="${SCRATCH_BASE}/cache/jax"
export TMPDIR=/tmp

mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" \
  "${JAX_COMPILATION_CACHE_DIR}" \
  /global/scratch/users/r112358/qc-exp/logs \
  /global/scratch/users/r112358/qc-exp
mkdir -p /global/scratch/users/r112358/ogbench-data
mkdir -p "${HOME}/.ogbench"
ln -sfn /global/scratch/users/r112358/ogbench-data "${HOME}/.ogbench/data"

cd /global/home/users/r112358/qc

HORIZONS=(10 25 50)
H=${HORIZONS[${SLURM_ARRAY_TASK_ID}]}

echo "host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset} horizon=${H}"

"${PY}" -u main.py \
  --run_group="cube-triple-acfql-h${H}" \
  --agent.alpha=100 \
  --env_name=cube-triple-play-singletask-task2-v0 \
  --sparse=False \
  --horizon_length="${H}" \
  --save_dir=/global/scratch/users/r112358/qc-exp \
  --seed=0
