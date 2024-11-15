#!/bin/bash

RUN_CMD="python -u scripts/train.py num_workers=1 port=4555 learning_rate=2e-4 batch_size=16 experiment_name=hpx32_dlom_sst-only_48H-dt_pp1 model=hpx32_dlom_sst-only_48H-dt model/modules/blocks@model.encoder.conv_block=conv_next_block model/modules/blocks@model.decoder.conv_block=conv_next_block model.encoder.n_channels=[128,64,32] model.decoder.n_channels=[32,64,128] trainer/criterion=ocean_mse trainer.max_epochs=100 data=hpx32_dlom_sst-only_48H-dt trainer/lr_scheduler=cosine trainer/optimizer=adam model.enable_healpixpad=False"

# Command to run model on CPU (useful for prototyping and verifying code)
#NUM_CPU=4
#srun -u --ntasks=1 \
#     --ntasks-per-node=1 \
#     --cpu_bind=sockets \
#     -c $(( ${NUM_CPU} )) \
#     bash -c "
#     export WORLD_RANK=\${SLURM_PROCID}
#     export HDF5_USE_FILE_LOCKING=True
#     export CUDA_VISIBLE_DEVICES=
#     export HYDRA_FULL_ERROR=1 
#     ${RUN_CMD}"
#exit

# Run configuration
NUM_GPU=2
NUM_CPU=8
GPU_NAME=V100
DEVICE_NUMBERS="4,5"

# Command to run model on srun -u --ntasks=${NUM_GPU} \
srun -u --ntasks=${NUM_GPU} \
     --ntasks-per-node=${NUM_GPU} \
     --gres=gpu:${GPU_NAME}:${NUM_GPU} \
     --cpu_bind=sockets \
     -c $(( ${NUM_CPU} / ${NUM_GPU} )) \
     bash -c "
     export WORLD_RANK=\${SLURM_PROCID}
     export HDF5_USE_FILE_LOCKING=False
     export CUDA_VISIBLE_DEVICES=${DEVICE_NUMBERS}
     export HYDRA_FULL_ERROR=1 
     ${RUN_CMD}"
