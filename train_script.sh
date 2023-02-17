#!/bin/bash
# Job name:
#SBATCH --job-name=MandarinToneGAN
#
# Account:
#SBATCH --account=fc_bsclab
#
# Partition:
#SBATCH --partition=savio2_1080ti
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task:
# Always at least twice the number of GPUs (savio2_gpu and GTX2080TI in savio3_gpu)
# Four times the number for TITAN and V100 in savio3_gpu
# Eight times the number for A40 in savio3_gpu
#SBATCH --cpus-per-task=2
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1
#
# Wall clock limit:
#SBATCH --time=16:00:00
#
## Command(s) to run (example):
module load ml/tensorflow/1.12.0-py36
nohup python3 backup.py '/global/scratch/users/nzxyin/mandarin_tone/results' 60 &
python3 ./train_ciwgan.py train '/global/scratch/users/nzxyin/mandarin_tone/results/' --data_dir '/global/scratch/users/nzxyin/mandarin_tone/data/tone_perfect_wav_16' --data_slice_len 32000 --num_categ 4 --data_first_slice --data_pad_end --data_fast_wav
