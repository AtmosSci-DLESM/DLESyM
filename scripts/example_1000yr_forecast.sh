#!/bin/bash

# This example presents a shell script for creating a 1000 year forecast. 
# This script was configured to use a CUDA GPU as configured on our machines
# at the University of Washington.

################ Environment Params ################
# Path to DLESyM module
MODULE_DIR="/home/disk/brume/nacc/DLESyM"
# set to -1 to use CPU
DEVICE_NUMBERS="-1"

################ Forecast Params ################
# Destination directory for forecast files
OUTPUT_DIR="./"
# Output logs sent here
OUTPUT_FILE="${OUTPUT_DIR}/1000yr_forecast.out"
# Path to models 
ATMOS_MODEL="${MODULE_DIR}/models/dlwp"
OCEAN_MODEL="${MODULE_DIR}/models/dlom"
# Path to directory with initialization data. Initializaion for example forecasts are included in repo
DATA_DIR="${MODULE_DIR}/example_data"
# Name of dataset
ATMOS_DATASET_NAME="hpx64_9varCoupledAtmos-sst"
OCEAN_DATASET_NAME="hpx64_1varCoupledOcean-z1000-ws10-olr"
# These are parameters used to create the first 100 year forecast initialized in jan. To create other inits we have to change these values for availability of olr 
ATMOS_OUTPUT_FILENAME="atmos_dlesym_1000year"
OCEAN_OUTPUT_FILENAME="ocean_dlesym_1000year"

# Parameters for resolving intialization dates
INIT_STARTS="2017-01-01"
INIT_ENDS="2017-01-02"
FREQ="MS"
TIME_CHUNK=1
STEP_CHUNK=1000
DATETIME="True"
END_DATE="3017-01-03"

# This is where intermediate data is stored, make sure it's big enough. 
# 1000 year forecast takes ~2.4TB
CACHE_DIR="${OUTPUT_DIR}/"

#############################################################
############ Boiler plate to execute forecast ###############
#############################################################

cd ${MODULE_DIR}
export PYTHONPATH=${MODULE_DIR}

LEN=${#INIT_STARTS[@]}

# delete the output file if it exists
if [ -f ${OUTPUT_FILE} ]; then
    rm ${OUTPUT_FILE}
fi


RUN_CMD="python scripts/coupled_forecast_hdf5.py \
    --atmos-model-path ${ATMOS_MODEL} \
    --ocean-model-path ${OCEAN_MODEL} \
    --non-strict \
    --forecast-init-start ${INIT_STARTS[$i]} \
    --forecast-init-end ${INIT_ENDS[$i]} \
    --freq ${FREQ} \
    --time-chunk ${TIME_CHUNK} \
    --step-chunk ${STEP_CHUNK} \
    --cache-dir ${CACHE_DIR} \
    --output-directory ${OUTPUT_DIR} \
    --atmos-output-filename ${ATMOS_OUTPUT_FILENAME} \
    --ocean-output-filename ${OCEAN_OUTPUT_FILENAME} \
    --data-directory ${DATA_DIR} \
    --atmos-dataset-name ${ATMOS_DATASET_NAME} \
    --ocean-dataset-name ${OCEAN_DATASET_NAME} \
    --datetime ${DATETIME} \
    --end-date ${END_DATE} \
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