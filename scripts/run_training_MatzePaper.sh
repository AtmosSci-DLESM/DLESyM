#!/bin/bash

seed="0"
output_file="/home/disk/quicksilver/nacc/dlesm/zephyr/scripts/run_training_MatzePaper_seed${seed}.out"

RUN_CMD="python -u scripts/train.py seed=${seed} num_workers=8 port=29450 learning_rate=2e-4 batch_size=6 experiment_name=hpx64_unet_136-68-34_cnxt_skip_dil_gru_6h_300_seed${seed} model=hpx_rec_unet model/modules/blocks@model.encoder.conv_block=conv_next_block model/modules/blocks@model.decoder.conv_block=conv_next_block model.encoder.n_channels=[136,68,34] model.decoder.n_channels=[34,68,136] trainer.max_epochs=300 data=era5_hpx64_7var_6h_24h data.dst_directory=/home/quicksilver2/karlbam/Data/DLWP/HPX64 data.prefix=era5_0.25deg_3h_HPX64_1979-2021_ data.prebuilt_dataset=True data.module.drop_last=True trainer/lr_scheduler=cosine trainer/optimizer=adam model.enable_healpixpad=True"

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
NUM_CPU=32
GPU_NAME=A100
DEVICE_NUMBERS="0"

# Command to run model on 
srun -u --ntasks=${NUM_GPU} \
     --ntasks-per-node=${NUM_GPU} \
     --gres=gpu:${GPU_NAME}:${NUM_GPU} \
     --cpu_bind=sockets \
     -c $(( ${NUM_CPU} / ${NUM_GPU} )) \
     "--output=${output_file}" \
     "--error=${output_file}" \
     bash -c "
     export WORLD_RANK=\${SLURM_PROCID}
     export HDF5_USE_FILE_LOCKING=False
     export CUDA_VISIBLE_DEVICES=${DEVICE_NUMBERS}
     export HYDRA_FULL_ERROR=1 
     ${RUN_CMD}"
