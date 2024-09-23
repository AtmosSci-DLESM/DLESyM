#!/bin/bash

EXP_NAME="test_hpx32_dlom_sst-only_48H-dt_basic-conv_no-gru"

# Create output file name 
OUTPUT_DIR="/home/disk/quicksilver/nacc/dlesm/zephyr/outputs/${EXP_NAME}"
mkdir -p $OUTPUT_DIR
OUTPUT_FILE="${OUTPUT_DIR}/output.out"

RUN_CMD="python -u scripts/train.py num_workers=4 port=4550 learning_rate=2e-4 batch_size=16 experiment_name=${EXP_NAME} model=hpx32_dlom_sst-only_no-gru_48H-dt model.encoder.n_channels=[64,128,256] model.decoder.n_channels=[256,128,64] trainer.max_epochs=100 trainer/criterion=ocean_mse data=hpx32_dlom_sst-only_48H-dt trainer/lr_scheduler=cosine trainer/optimizer=adam model.enable_healpixpad=True"

# Run configuration
NUM_GPU=2
NUM_CPU=32
GPU_NAME=A100
DEVICE_NUMBERS="0,1"

# Command to run model on srun -u --ntasks=${NUM_GPU} \
srun -u --ntasks=${NUM_GPU} \
     --ntasks-per-node=${NUM_GPU} \
     --gres=gpu:${GPU_NAME}:${NUM_GPU} \
     --output=$OUTPUT_FILE \
     --error=$OUTPUT_FILE \
     --cpu_bind=sockets \
     -c $(( ${NUM_CPU} / ${NUM_GPU} )) \
     bash -c "
     export WORLD_RANK=\${SLURM_PROCID}
     export HDF5_USE_FILE_LOCKING=False
     export CUDA_VISIBLE_DEVICES=${DEVICE_NUMBERS}
     export HYDRA_FULL_ERROR=1 
     ${RUN_CMD}"
