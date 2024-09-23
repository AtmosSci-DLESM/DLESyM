#!/bin/bash

################ Batch Params ################

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --output=output.out
#SBATCH --error=output.out
#SBATCH --job-name=ExampleBatchForecast

################ Environment Params ################
MODULE_DIR="path/to/zephyr"
DEVICE_NUMBERS="1"

################ Forecast Params ################

# Training parameters
OUTPUT_DIR="path/to/output/directory"
MODEL_PATH="/home/disk/rhodium/dlwp/models/hpx64_unet_136-68-34_cnxt_skip_dil_gru_6h_300"
OUTPUT_FILENAME="forecast_336h_biweekly_hpx64_unet_136-68-34_cnxt_skip_dil_gru_6h_300"
DATA_DIR="/home/disk/gold3/dlwp/data/HPX64/"
LEAD_TIME="336"
INIT_START="2017-01-02"
INIT_END="2018-12-30"
FREQ="biweekly"

#############################################################
############ Boiler plate to execute forecast ###############
#############################################################

cd ${MODULE_DIR}
export PYTHONPATH=${MODULE_DIR}

RUN_CMD="python scripts/forecast.py \
    --model-path ${MODEL_PATH} \
    --non-strict \
    --lead-time ${LEAD_TIME} \
    --forecast-init-start ${INIT_START} \
    --forecast-init-end ${INIT_END} \
    --freq ${FREQ} \
    --output-directory ${OUTPUT_DIR} \
    --output-filename ${OUTPUT_FILENAME} \
    --data-directory ${DATA_DIR} \
    --gpu 0"

# Set environment variables and run the command
export WORLD_RANK=${SLURM_PROCID}
export HDF5_USE_FILE_LOCKING=False
export CUDA_VISIBLE_DEVICES=${DEVICE_NUMBERS}
export HYDRA_FULL_ERROR=1 
${RUN_CMD}