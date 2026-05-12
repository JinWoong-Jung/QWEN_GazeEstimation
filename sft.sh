#!/bin/bash
#SBATCH --job-name=jinwoong_training
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --output=logs/result_%j.log
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /home/jinwoongjung/envs/jinwoong

export PYTHONNOUSERSITE=1


python main.py --config sft.yaml