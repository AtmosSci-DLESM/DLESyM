#!/bin/bash


# This example presents a batch script for forecasting using the SLURM
# scheduler configuration on our machines at the University of Washington.

################ Batch Params ################

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --output=output.out
#SBATCH --error=error.out

################ Environment Params ################
MODULE_DIR="/path/to/zephyr"

################ Forecast Params ################

# Training parameters
OUTPUT_DIR="/path/to/desired/output/directory"
ATMOS_MODEL="path/to/atmos/model_directory"
OCEAN_MODEL="path/to/ocean/model_directory"
ATMOS_OUTPUT_FILENAME="atmos_model_output_filename"
OCEAN_OUTPUT_FILENAME="ocean_model_output_filename"
DATA_DIR="/path/to/initializations/data/directory"
LEAD_TIME="336"
INIT_START="2017-01-01"
INIT_END="2017-01-02"
FREQ="biweekly"

#############################################################
############ Boiler plate to execute forecast ###############
#############################################################

cd ${MODULE_DIR}
export PYTHONPATH=${MODULE_DIR}

RUN_CMD="python scripts/coupled_forecast.py \
    --atmos-model-path ${ATMOS_MODEL} \
    --ocean-model-path ${OCEAN_MODEL} \
    --non-strict \
    --lead-time ${LEAD_TIME} \
    --forecast-init-start ${INIT_START} \
    --forecast-init-end ${INIT_END} \
    --freq ${FREQ} \
    --output-directory ${OUTPUT_DIR} \
    --atmos-output-filename ${ATMOS_OUTPUT_FILENAME} \
    --ocean-output-filename ${OCEAN_OUTPUT_FILENAME} \
    --data-directory ${DATA_DIR} \
    --gpu 0"

# Set environment variables and run the command
export WORLD_RANK=${SLURM_PROCID}
export HDF5_USE_FILE_LOCKING=False
export HYDRA_FULL_ERROR=1 
${RUN_CMD}