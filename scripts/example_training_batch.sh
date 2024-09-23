#!/bin/bash

################ Batch Params ################

#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=8
#SBATCH --output=output.out
#SBATCH --error=output.out

################ Environment Params ################

MODULE_DIR="/home/disk/brume/nacc/dlesm/zephyr"
DEVICE_NUMBERS="1,2"

################ Training Params ################

NUM_WORKERS=8
PORT=29450
LEARNING_RATE=2e-4
BATCH_SIZE=8
EXPERIMENT_NAME="hpx64_unet_136-68-34_cnxt_skip_dil_gru_6h_300_example"
MODEL="hpx_rec_unet"
ENCODER_CONV_BLOCK="conv_next_block"
DECODER_CONV_BLOCK="conv_next_block"
ENCODER_N_CHANNELS="[136,68,34]"
DECODER_N_CHANNELS="[34,68,136]"
MAX_EPOCHS=300
DATA="era5_hpx64_7var_6h_24h"
DST_DIRECTORY="/home/quicksilver2/dlwp/data/HPX64"
PREBUILT_DATASET=True
DROP_LAST=True
LR_SCHEDULER="cosine"
OPTIMIZER="adam"
ENABLE_HEALPIXPAD=True

#############################################################
############ Boiler plate to execute training ###############
#############################################################

cd ${MODULE_DIR}
export PYTHONPATH=${MODULE_DIR}

RUN_CMD="python -u scripts/train.py \
num_workers=${NUM_WORKERS} \
port=${PORT} \
learning_rate=${LEARNING_RATE} \
batch_size=${BATCH_SIZE} \
experiment_name=${EXPERIMENT_NAME} \
model=${MODEL} \
model/modules/blocks@model.encoder.conv_block=${ENCODER_CONV_BLOCK} \
model/modules/blocks@model.decoder.conv_block=${DECODER_CONV_BLOCK} \
model.encoder.n_channels=${ENCODER_N_CHANNELS} \
model.decoder.n_channels=${DECODER_N_CHANNELS} \
trainer.max_epochs=${MAX_EPOCHS} \
data=${DATA} \
data.dst_directory=${DST_DIRECTORY} \
data.prebuilt_dataset=${PREBUILT_DATASET} \
data.module.drop_last=${DROP_LAST} \
trainer/lr_scheduler=${LR_SCHEDULER} \
trainer/optimizer=${OPTIMIZER} \
model.enable_healpixpad=${ENABLE_HEALPIXPAD}"

export WORLD_RANK=${SLURM_PROCID}
export HDF5_USE_FILE_LOCKING=False
export CUDA_VISIBLE_DEVICES=${DEVICE_NUMBERS}
export HYDRA_FULL_ERROR=1 

${RUN_CMD}