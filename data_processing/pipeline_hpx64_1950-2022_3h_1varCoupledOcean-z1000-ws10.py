# This script offers examples of how to structure a data processing pipeline for
# DLESyM-style atmosphere training and intialization data. It provides examples
# of how to included data utilities. You'll notice it includes directory paths 
# and file names that are specfic to machines at University of Washington. 
# As such it cannot be run as is on non-UW machines. 

# NOTE: This script will note recreate the training data used for training 
# the DLESyM model presented in Cresswell-Clay et al. 2024. In particular, 
# the OLR retirval and preparation routines are not included here, 
# however, full initialization data for simulations in the paper are included in
# the repository.

from utils import (
    era5_retrieval,
    data_imputation,
    map2hpx,
    windspeed,
    trailing_average,
    scale_topography,
    tau_calculation,
    update_scaling,
)
from training.dlwp.data import data_loading as dl
import yaml
import os
import numpy as np
from omegaconf import OmegaConf

era5_requests = [
    # lsm (Land Sea Mask)
    {
        "constant": True,  # Indicates that the land sea mask is a constant field (doesn't change over time)
        "single_level_variable": True,  # Indicates that the land sea mask is a single level variable (not a vertical profile)
        "variable_name": "land_sea_mask",  # The name of the variable in the dataset
        "grid": [0.25, 0.25],  # The grid resolution of the data
        "target_file": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_lsm.nc",  # The file where the processed data will be saved
    },
    # u 10m wind component
    {
        "constant": False,  # Indicates that the 10m wind component is not a constant field (it changes over time)
        "single_level_variable": True,  # Indicates that the 10m wind component is a single level variable (not a vertical profile)
        "variable_name": "10u",  # The name of the variable in the dataset
        "grid": [0.25, 0.25],  # The grid resolution of the data
        "year": [
            y for y in range(1950, 2023)
        ],  # The years for which the data is required
        "month": [
            month + 1 for month in range(0, 12)
        ],  # The months for which the data is required
        "day": [d + 1 for d in range(0, 31)],  # The days for which the data is required
        "time": np.arange(
            0, 24, 3
        ).tolist(),  # The times for which the data is required (every 3 hours)
        "target_file": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_u10m.nc",  # The file where the processed data will be saved
    },
    # v 10m wind component
    {
        "constant": False,
        "single_level_variable": True,
        "variable_name": "10v",
        "grid": [0.25, 0.25],
        "year": [y for y in range(1950, 2023)],
        "month": [month + 1 for month in range(0, 12)],
        "day": [d + 1 for d in range(0, 31)],
        "time": np.arange(0, 24, 3).tolist(),
        "target_file": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_v10m.nc",
    },
    # z1000
    {
        "constant": False,
        "single_level_variable": False,
        "variable_name": "z",
        "pressure_level": "1000",
        "grid": [0.25, 0.25],
        "year": [y for y in range(1950, 2023)],
        "month": [month + 1 for month in range(0, 12)],
        "day": [d + 1 for d in range(0, 31)],
        "time": np.arange(0, 24, 3).tolist(),
        "target_file": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_z1000.nc",
    },
    # sst
    {
        "constant": False,
        "single_level_variable": True,
        "variable_name": "sst",
        "pressure_level": "1000",
        "grid": [0.25, 0.25],
        "year": [y for y in range(1950, 2023)],
        "month": [month + 1 for month in range(0, 12)],
        "day": [d + 1 for d in range(0, 31)],
        "time": np.arange(0, 24, 3).tolist(),
        "target_file": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_sst.nc",
    },
]
# Parameters for imputing sst data over land
impute_params = {
    "filename": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_sst.nc",  # File with data to be imputed
    "variable": "sst",  # Variable in the file that needs imputation
    "chunks": {"time": 1024},  # Chunk size for processing the data
    "imputed_file": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_sst-ti.nc",  # File to save the imputed data
}
# Parameters for calculating wind speed
windspeed_params = {
    "u_file": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_u10m.nc",  # File with U component of wind
    "v_file": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_v10m.nc",  # File with V component of wind
    "target_file": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_windspeed.nc",  # File to save the calculated wind speed
    "chunks": {"time": 8},  # Chunk size for processing the data
}
# parameters for healpix remapping
hpx_params = [
    {
        "file_name": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_lsm.nc",  # The path to the input file
        "target_variable_name": "lsm",  # The name of the variable in the input dataset
        "file_variable_name": "lsm",  # The name of the variable in in the newly generated file
        "prefix": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1950-2022_",  # The prefix for the output file names
        "nside": 64,  # The number of divisions on the side of the grid
        "order": "bilinear",  # The interpolation method to use when regridding
        "resolution_factor": 1.0,  # The factor by which to change the resolution of the data
        "visualize": False,  # Whether to generate a visualization of the regridded data
    },
    {
        "file_name": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_windspeed.nc",
        "target_variable_name": "ws10",
        "file_variable_name": "ws10",
        "prefix": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1950-2022_",
        "nside": 64,
        "order": "bilinear",
        "resolution_factor": 1.0,
        "visualize": False,
    },
    {
        "file_name": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_z1000.nc",
        "target_variable_name": "z1000",
        "file_variable_name": "z",
        "prefix": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1950-2022_",
        "nside": 64,
        "order": "bilinear",
        "resolution_factor": 1.0,
        "visualize": False,
    },
    {
        "file_name": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_sst-ti.nc",
        "target_variable_name": "sst",
        "file_variable_name": "sst",
        "prefix": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1950-2022_",
        "nside": 64,
        "order": "bilinear",
        "resolution_factor": 1.0,
        "visualize": False,
    },
]
trailing_average_params = [
    {
        "filename": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1950-2022_ws10.nc",  # input file name
        "variable_name": "ws10",  # input variable name in netcdf file
        "output_variable_name": "ws10-48H",  # name of variables in output file
        "coupled_dt": "6H",  # temporal resolution of coupled model. This is used find the samples over which to calculate the average
        "output_filename": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1950-2022_ws10-48H.nc",  # output file name
        "influence_window": np.timedelta64(
            2, "D"
        ),  # time window over which to calculate the average
        "chunks": None,  # chunk size for loading large datasets
        "load_first": True,
    },  # if True, load the data into memory before calculating the average
    {
        "filename": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1950-2022_z1000.nc",
        "variable_name": "z1000",
        "output_variable_name": "z1000-48H",
        "coupled_dt": "6H",
        "output_filename": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1950-2022_z1000-48H.nc",
        "influence_window": np.timedelta64(2, "D"),
        "chunks": None,
        "load_first": True,
    },
]
# Define the parameters for updating the scaling parameters of various variables
update_scaling_params = {
    "scale_file": "/home/disk/quicksilver/nacc/dlesm/zephyr/training/configs/data/scaling/hpx64.yaml",  # Path to the YAML file containing the scaling parameters
    "variable_file_prefix": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1950-2022_",  # Prefix for the file names containing the variables
    "variable_names": [  # List of variable names to update
        "ws10_48H",
        "z1000_48H",
        "sst",
    ],
    "selection_dict": {
        "sample": slice(np.datetime64("1950-01-01"), np.datetime64("2022-12-31"))
    },  # Dictionary defining the data subset to use for the calculation
    "overwrite": False,  # Whether to overwrite existing scaling parameters
    "chunks": None,  # Dictionary defining the chunk sizes for the data loading
}
zarr_params = {
    "src_directory": "/home/disk/rhodium/dlwp/data/HPX64/",
    "dst_directory": "/home/disk/rhodium/dlwp/data/HPX64/",
    "dataset_name": "hpx64_1950-2022_3h_1varCoupledOcean-z1000-ws10",
    "input_variables": [
        "ws10-48H",
        "z1000-48H",
        "sst",
    ],
    "output_variables": [
        "sst",
    ],
    "constants": {"lsm": "lsm"},
    "prefix": "era5_0.25deg_3h_HPX64_1950-2022_",
    "batch_size": 16,
    "scaling": OmegaConf.load(
        update_scaling.create_yaml_if_not_exists(update_scaling_params["scale_file"])
    ),
    "overwrite": False,
}

### Use the parameters defined above to run the pipeline ###

# Retrive raw data
for request in era5_requests:
    era5_retrieval.main(request)

# Impute ocean data
data_imputation.triple_interp(impute_params)
# windspeed calculation
windspeed.main(windspeed_params)
# scale topography
scale_topography.main(scale_topography_params)
# Remap data to HPX mesh
for hpx_param in hpx_params:
    map2hpx.main(hpx_param)
# 48 hour trailing average of atmospheric fields
for trailing_average_param in trailing_average_params:
    trailing_average.main(trailing_average_param)
# update scaling dictionary
update_scaling.main(update_scaling_params)
# create zarr file for optimized training
dl.create_time_series_dataset_classic(**zarr_params)
