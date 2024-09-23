#!/bin/bash

EXP_NAME="HPX32_coupled_atmos"
mkdir -p /home/disk/quicksilver/nacc/dlesm/zephyr/outputs/${EXP_NAME} 
OUTPUT_FILE="/home/disk/quicksilver/nacc/dlesm/zephyr/outputs/${EXP_NAME}/output.out"

RUN_CMD="python -u scripts/train.py seed=5 num_workers=8 port=29460 learning_rate=2e-4 batch_size=32 experiment_name=${EXP_NAME} model=hpx_rec_unet model/modules/blocks@model.encoder.conv_block=conv_next_block model/modules/blocks@model.decoder.conv_block=conv_next_block model.encoder.n_channels=[136,68,34] model.decoder.n_channels=[34,68,136] trainer.max_epochs=300 data=era5_hpx32_8var-coupled_6h_24h data.dst_directory=/home/quicksilver2/nacc/Data/HPX32 trainer/lr_scheduler=cosine +trainer.max_norm=0.25 trainer/optimizer=adam model.enable_healpixpad=True"

# Run configuration
NUM_GPU=1
NUM_CPU=32
GPU_NAME=A100
DEVICE_NUMBERS="1"

cd /home/disk/quicksilver/nacc/dlesm/zephyr
source /home/disk/brume/nacc/.bashrc
source activate zephyr-1.0
export HYDRA_FULL_ERROR=1
export HDF5_USE_FILE_LOCKING=False
PYTHONPATH=/home/disk/quicksilver/nacc/dlesm/zephyr
NUMEXPR_MAX_THREADS=128

# Command to run model on 
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
#srun -u --ntasks=${NUM_GPU} \
#     "--output=${OUTPUT_FILE}" \
#     "--error=${OUTPUT_FILE}" \
#     --ntasks-per-node=${NUM_GPU} \
#     --gres=gpu:${GPU_NAME}:${NUM_GPU} \
#     --cpu_bind=sockets \
#     -c $(( ${NUM_CPU} / ${NUM_GPU} )) \
#     bash -c "
#     export WORLD_RANK=\${SLURM_PROCID}
#     export HDF5_USE_FILE_LOCKING=False
#     export CUDA_VISIBLE_DEVICES=${DEVICE_NUMBERS}
#     export HYDRA_FULL_ERROR=1 
#     ${RUN_CMD}" & 

