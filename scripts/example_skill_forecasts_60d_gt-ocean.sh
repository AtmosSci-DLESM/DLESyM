#!/bin/bash

# This example presents a shell script for creating 104 forecasts initialized 
# biweekly over 1 year. This script was configured to use a CUDA GPU as configured 
# on our machines at the University of Washington.

# NOTE: initialization for all 104 forecast are not included in this repository. 
# The data will have to be retrieved and processed seperately. 

################ Environment Params ################
# Path to DLESyM module
MODULE_DIR="./"
# set to -1 to use CPU
DEVICE_NUMBERS="4"

################ Forecast Params ################

# Destination directory for forecast files
OUTPUT_DIR="path/to/output/dir"
ATMOS_OUTPUT_FILE="atmos_forecast_skill_60d"
OCEAN_OUTPUT_FILE="ocean_forecast_skill_60d"
# Output logs sent here
# OUTPUT_FILE="${OUTPUT_DIR}/forecast_logs.out"
# Path to models 
ATMOS_MODEL="${MODULE_DIR}/models/dlwp"
OCEAN_MODEL="${MODULE_DIR}/models/ocean_gt_model"
# Define the range of initializations
INIT_START="2016-05-01"
INIT_END="2017-05-01"
FREQ="biweekly"
# Path to directory with initialization data. 
DATA_DIR="path/to/init/data"
# Name of dataset
ATMOS_DATASET_NAME="hpx64_atmos_ds_name"
OCEAN_DATASET_NAME="hpx64_ocean_ds_name"
# 60 days 
LEAD_TIME="1440" 

#############################################################
############ Boiler plate to execute forecast ###############
#############################################################

cd ${MODULE_DIR}
export PYTHONPATH=${MODULE_DIR}

# delete the output file if it exists
if [ -f ${OUTPUT_FILE} ]; then
    rm ${OUTPUT_FILE}
fi


RUN_CMD="python scripts/coupled_forecast.py \
    --atmos-model-path ${ATMOS_MODEL} \
    --ocean-model-path ${OCEAN_MODEL} \
    --non-strict \
    --lead-time ${LEAD_TIME} \
    --forecast-init-start ${INIT_START} \
    --forecast-init-end ${INIT_END} \
    --freq ${FREQ} \
    --output-directory ${OUTPUT_DIR} \
    --atmos-output-filename ${ATMOS_OUTPUT_FILE} \
    --ocean-output-filename ${OCEAN_OUTPUT_FILE} \
    --data-directory ${DATA_DIR} \
    --atmos-dataset-name ${ATMOS_DATASET_NAME} \
    --ocean-dataset-name ${OCEAN_DATASET_NAME} \
    --gpu ${DEVICE_NUMBERS}"

# Set environment variables and run the command
export HDF5_USE_FILE_LOCKING=False
export HYDRA_FULL_ERROR=1 

# if output file is given, redirect output
if [[ -n "${OUTPUT_FILE}" ]]; then
    ${RUN_CMD} &>> ${OUTPUT_FILE}
else
    ${RUN_CMD}
fi
