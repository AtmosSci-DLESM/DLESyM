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
    # topography -- follows same format as land sea mask
    {
        "constant": True,
        "single_level_variable": True,
        "variable_name": "geopotential",
        "grid": [0.25, 0.25],
        "target_file": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_unscaled_topography.nc",
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
    # t2m
    {
        "constant": False,
        "single_level_variable": True,
        "variable_name": "2m_temperature",
        "pressure_level": "1000",
        "grid": [0.25, 0.25],
        "year": [y for y in range(1950, 2023)],
        "month": [month + 1 for month in range(0, 12)],
        "day": [d + 1 for d in range(0, 31)],
        "time": np.arange(0, 24, 3).tolist(),
        "target_file": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_t2m.nc",
    },
    # t850
    {
        "constant": False,
        "single_level_variable": False,
        "variable_name": "temperature",
        "pressure_level": "850",
        "grid": [0.25, 0.25],
        "year": [y for y in range(1950, 2023)],
        "month": [month + 1 for month in range(0, 12)],
        "day": [d + 1 for d in range(0, 31)],
        "time": np.arange(0, 24, 3).tolist(),
        "target_file": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_t850.nc",
    },
    # z500
    {
        "constant": False,
        "single_level_variable": False,
        "variable_name": "z",
        "pressure_level": "500",
        "grid": [0.25, 0.25],
        "year": [y for y in range(1950, 2023)],
        "month": [month + 1 for month in range(0, 12)],
        "day": [d + 1 for d in range(0, 31)],
        "time": np.arange(0, 24, 3).tolist(),
        "target_file": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_z500.nc",
    },
    # z700 for tau
    {
        "constant": False,
        "single_level_variable": False,
        "variable_name": "z",
        "pressure_level": "700",
        "grid": [0.25, 0.25],
        "year": [y for y in range(1950, 2023)],
        "month": [month + 1 for month in range(0, 12)],
        "day": [d + 1 for d in range(0, 31)],
        "time": np.arange(0, 24, 3).tolist(),
        "target_file": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_z700.nc",
    },
    # z300 for tau
    {
        "constant": False,
        "single_level_variable": False,
        "variable_name": "z",
        "pressure_level": "300",
        "grid": [0.25, 0.25],
        "year": [y for y in range(1950, 2023)],
        "month": [month + 1 for month in range(0, 12)],
        "day": [d + 1 for d in range(0, 31)],
        "time": np.arange(0, 24, 3).tolist(),
        "target_file": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_z300.nc",
    },
    # z250
    {
        "constant": False,
        "single_level_variable": False,
        "variable_name": "z",
        "pressure_level": "250",
        "grid": [0.25, 0.25],
        "year": [y for y in range(1950, 2023)],
        "month": [month + 1 for month in range(0, 12)],
        "day": [d + 1 for d in range(0, 31)],
        "time": np.arange(0, 24, 3).tolist(),
        "target_file": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_z250.nc",
    },
    # tcwv
    {
        "constant": False,
        "single_level_variable": True,
        "variable_name": "total_column_water_vapour",
        "pressure_level": "1000",
        "grid": [0.25, 0.25],
        "year": [y for y in range(1950, 2023)],
        "month": [month + 1 for month in range(0, 12)],
        "day": [d + 1 for d in range(0, 31)],
        "time": np.arange(0, 24, 3).tolist(),
        "target_file": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_tcwv.nc",
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
# Parameters for scaling topography
scale_topography_params = {
    "src_file": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_unscaled_topography.nc",  # Source file with unscaled topography data
    "target_file": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_topography.nc",  # Target file to save the scaled topography data
}
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
# parameters for calculating tau
tau_params = {
    "upper_file": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_z300.nc",  # Path to the file containing the 300 hPa geopotential height data
    "lower_file": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_z700.nc",  # Path to the file containing the 700 hPa geopotential height data
    "chunks": {"time": 128},  # Dictionary defining the chunk sizes for the data loading
    "target_file": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_tau300-700.nc",  # Path to the file where the calculated tau data will be saved
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
        "poolsize": 30 # The number of parallel processes to use
    },
    {
        "file_name": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_topography.nc",
        "target_variable_name": "z",
        "target_file_variable_name": "topography",
        "file_variable_name": "z",
        "prefix": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1950-2022_",
        "nside": 64,
        "order": "bilinear",
        "resolution_factor": 1.0,
        "visualize": False,
        "poolsize": 30 # The number of parallel processes to use
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
        "poolsize": 30 # The number of parallel processes to use
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
        "poolsize": 30 # The number of parallel processes to use
    },
    {
        "file_name": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_t2m.nc",
        "target_variable_name": "t2m",
        "file_variable_name": "t2m",
        "prefix": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1950-2022_",
        "nside": 64,
        "order": "bilinear",
        "resolution_factor": 1.0,
        "visualize": False,
        "poolsize": 30 # The number of parallel processes to use
    },
    {
        "file_name": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_t850.nc",
        "target_variable_name": "t850",
        "file_variable_name": "t",
        "prefix": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1950-2022_",
        "nside": 64,
        "order": "bilinear",
        "resolution_factor": 1.0,
        "visualize": False,
        "poolsize": 30 # The number of parallel processes to use
    },
    {
        "file_name": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_z500.nc",
        "target_variable_name": "z500",
        "file_variable_name": "z",
        "prefix": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1950-2022_",
        "nside": 64,
        "order": "bilinear",
        "resolution_factor": 1.0,
        "visualize": False,
        "poolsize": 30 # The number of parallel processes to use
    },
    {
        "file_name": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_tau300-700.nc",
        "target_variable_name": "tau300-700",
        "file_variable_name": "tau300-700",
        "prefix": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1950-2022_",
        "nside": 64,
        "order": "bilinear",
        "resolution_factor": 1.0,
        "visualize": False,
        "poolsize": 30 # The number of parallel processes to use
    },
    {
        "file_name": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_tcwv.nc",
        "target_variable_name": "tcwv",
        "file_variable_name": "tcwv0",
        "prefix": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1950-2022_",
        "nside": 64,
        "order": "bilinear",
        "resolution_factor": 1.0,
        "visualize": False,
        "poolsize": 30 # The number of parallel processes to use
    },
    {
        "file_name": "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_z250.nc",
        "target_variable_name": "z250",
        "file_variable_name": "z",
        "prefix": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1950-2022_",
        "nside": 64,
        "order": "bilinear",
        "resolution_factor": 1.0,
        "visualize": False,
        "poolsize": 30 # The number of parallel processes to use
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
        "poolsize": 30 # The number of parallel processes to use
    },
]
# Define the parameters for updating the scaling parameters of various variables
update_scaling_params = {
    "scale_file": "/home/disk/quicksilver/nacc/dlesm/zephyr/training/configs/data/scaling/hpx64.yaml",  # Path to the YAML file containing the scaling parameters
    "variable_file_prefix": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1950-2022_",  # Prefix for the file names containing the variables
    "variable_names": [  # List of variable names to update
        "ws10",
        "z1000",
        "t2m0",
        "t850",
        "z500",
        "tau300-700",
        "tcwv0",
        "z250",
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
    "dataset_name": "hpx64_1950-2022_3h_8varCoupledAtmos-sst",
    "input_variables": [
        "ws10",
        "z1000",
        "t2m0",
        "t850",
        "z500",
        "tau300-700",
        "tcwv0",
        "z250",
        "sst",
    ],
    "output_variables": [
        "ws10",
        "z1000",
        "t2m0",
        "t850",
        "z500",
        "tau300-700",
        "tcwv0",
        "z250",
    ],
    "constants": {
        "lsm": "lsm",
        "topography": "z",
    },
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
# tau calculation
tau_calculation.main(tau_params)
# scale topography
scale_topography.main(scale_topography_params)
# Remap data to HPX mesh
for hpx_param in hpx_params:
    map2hpx.main(hpx_param)
# update scaling dictionary
update_scaling.main(update_scaling_params)
# create zarr file for optimized training
dl.create_time_series_dataset_classic(**zarr_params)
