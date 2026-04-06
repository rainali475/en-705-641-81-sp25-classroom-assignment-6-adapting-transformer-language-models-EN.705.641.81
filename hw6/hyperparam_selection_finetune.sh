#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-17
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1

module load conda
conda activate ssm_hw6

types=(
  full
  head
  prefix
)

lrs=(
  1e-4
  5e-4
  1e-3
)

epochs_list=(
  7
  9
)

num_types=${#types[@]}
num_lrs=${#lrs[@]}
num_epochs=${#epochs_list[@]}

task_id=${SLURM_ARRAY_TASK_ID}

type_idx=$(( task_id / (num_lrs * num_epochs) ))
remainder=$(( task_id % (num_lrs * num_epochs) ))
lr_idx=$(( remainder / num_epochs ))
epoch_idx=$(( remainder % num_epochs ))

type=${types[$type_idx]}
lr=${lrs[$lr_idx]}
epochs=${epochs_list[$epoch_idx]}

echo "Task ID: $task_id"
echo "Type: $type"
echo "LR: $lr"
echo "Epochs: $epochs"

python classification.py \
  --device cuda \
  --type "$type" \
  --model "roberta-base" \
  --lr "$lr" \
  --num_epochs "$epochs"