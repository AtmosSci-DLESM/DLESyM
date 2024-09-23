from utils import (
    era5_retrieval,
    data_imputation,
    map2hpx,
    windspeed,
    trailing_average,
    update_scaling,
)
from training.dlwp.data import data_loading as dl
import numpy as np
from omegaconf import OmegaConf


"""
This data pipline creates training, validation, and test data for a coupled deep learning ocean model (DLOM). The coupled DLOM will forecast on a HEALPix 32 mesh, have 48 hour resolution, and predict a single prognostic field: sea surface temperature (SST). The model is trained to receive 10m windspeed and geopotential height at 1000 hPa for the 4 day period forecast. 
"""

era5_requests = [
    {
        "constant": True,
        "single_level_variable": True,
        "variable_name": "land_sea_mask",
        "grid": [1.0, 1.0],
        "target_file": "/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_lsm.nc",
    },
    # u and v 10m wind components
    {
        "constant": False,
        "single_level_variable": True,
        "variable_name": "10u",
        "pressure_level": "1000",
        "grid": [1.0, 1.0],
        "year": [y for y in range(1950, 2023)],
        "month": [month + 1 for month in range(0, 12)],
        "day": [d + 1 for d in range(0, 31)],
        "time": np.arange(0, 24, 3).tolist(),
        "target_file": "/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_u10m.nc",
    },
    {
        "constant": False,
        "single_level_variable": True,
        "variable_name": "10v",
        "pressure_level": "1000",
        "grid": [1.0, 1.0],
        "year": [y for y in range(1950, 2023)],
        "month": [month + 1 for month in range(0, 12)],
        "day": [d + 1 for d in range(0, 31)],
        "time": np.arange(0, 24, 3).tolist(),
        "target_file": "/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_v10m.nc",
    },
    # z1000
    {
        "constant": False,
        "single_level_variable": False,
        "variable_name": "z",
        "pressure_level": "1000",
        "grid": [1.0, 1.0],
        "year": [y for y in range(1950, 2023)],
        "month": [month + 1 for month in range(0, 12)],
        "day": [d + 1 for d in range(0, 31)],
        "time": np.arange(0, 24, 3).tolist(),
        "target_file": "/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_z1000.nc",
    },
    # sst
    {
        "constant": False,
        "single_level_variable": True,
        "variable_name": "sst",
        "pressure_level": "1000",
        "grid": [1.0, 1.0],
        "year": [y for y in range(1950, 2023)],
        "month": [month + 1 for month in range(0, 12)],
        "day": [d + 1 for d in range(0, 31)],
        "time": np.arange(0, 24, 3).tolist(),
        "target_file": "/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_sst.nc",
    },
]
# Parameters for imputing sst data over land
impute_params = {
    "filename": "/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_sst.nc",
    "variable": "sst",
    "chunks": {"time": 10000},
    "imputed_file": "/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_sst-ti.nc",
}
# Parameters for calculating wind speed
windspeed_params = {
    "u_file": "/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_u10m.nc",
    "v_file": "/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_v10m.nc",
    "target_file": "/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_windspeed.nc",
    "chunks": {"time": 10},
}
# parameters for healpix remapping
hpx_params = [
    {
        "file_name": "/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_sst-ti.nc",
        "target_variable_name": "sst",
        "file_variable_name": "sst",
        "prefix": "/home/disk/rhodium/dlwp/data/HPX32/era5_1deg_3h_HPX32_1950-2022_",
        "nside": 32,
        "order": "bilinear",
        "resolution_factor": 1.0,
        "visualize": False,
    },
    {
        "file_name": "/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_windspeed.nc",
        "target_variable_name": "ws10",
        "file_variable_name": "ws10",
        "prefix": "/home/disk/rhodium/dlwp/data/HPX32/era5_1deg_3h_HPX32_1950-2022_",
        "nside": 32,
        "order": "bilinear",
        "resolution_factor": 1.0,
        "visualize": False,
    },
    {
        "file_name": "/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_z1000.nc",
        "target_variable_name": "z1000",
        "file_variable_name": "z",
        "prefix": "/home/disk/rhodium/dlwp/data/HPX32/era5_1deg_3h_HPX32_1950-2022_",
        "nside": 32,
        "order": "bilinear",
        "resolution_factor": 1.0,
        "visualize": False,
    },
    {
        "file_name": "/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_lsm.nc",
        "target_variable_name": "lsm",
        "file_variable_name": "lsm",
        "prefix": "/home/disk/rhodium/dlwp/data/HPX32/era5_1deg_3h_HPX32_1950-2022_",
        "nside": 32,
        "order": "bilinear",
        "resolution_factor": 1.0,
        "visualize": False,
    },
]
# parameters for preparing coupled atmosphere training data
trailing_average_params = [
    {
        "filename": "/home/disk/rhodium/dlwp/data/HPX32/era5_1deg_3h_HPX32_1950-2022_ws10.nc",
        "variable_name": "ws10",
        "output_variable_name": "ws10-48H",
        "coupled_dt": "6H",
        "output_filename": "/home/disk/rhodium/dlwp/data/HPX32/era5_1deg_3h_HPX32_1950-2022_ws10-48H.nc",
        "influence_window": np.timedelta64(2, "D"),
        "chunks": None,
        "load_first": True,
    },
    {
        "filename": "/home/disk/rhodium/dlwp/data/HPX32/era5_1deg_3h_HPX32_1950-2022_z1000.nc",
        "variable_name": "z1000",
        "output_variable_name": "z1000-48H",
        "coupled_dt": "6H",
        "output_filename": "/home/disk/rhodium/dlwp/data/HPX32/era5_1deg_3h_HPX32_1950-2022_z1000-48H.nc",
        "influence_window": np.timedelta64(2, "D"),
        "chunks": None,
        "load_first": True,
    },
]
# Define the parameters for updating the scaling parameters of various variables
update_scaling_params = {
    "scale_file": "/home/disk/quicksilver/nacc/dlesm/zephyr/training/configs/data/scaling/hpx32.yaml",
    "variable_file_prefix": "/home/disk/rhodium/dlwp/data/HPX32/era5_1deg_3h_HPX32_1950-2022_",
    "variable_names": [
        "sst",
    ],
    "selection_dict": {
        "sample": slice(np.datetime64("1950-01-01"), np.datetime64("2022-12-31"))
    },
    "overwrite": False,
    "chunks": None,
}
# parameters used to write optimized zarr file
zarr_params = {
    "src_directory": "/home/disk/rhodium/dlwp/data/HPX32/",
    "dst_directory": "/home/disk/rhodium/dlwp/data/HPX32/",
    "dataset_name": "hpx32_1950-2022_3h_sst_coupled",
    "input_variables": [
        "sst",
        "ws10-48H",
        "z1000-48H",
    ],
    "output_variables": [
        "sst",
    ],
    "constants": {"lsm": "lsm"},
    "prefix": "era5_1deg_3h_HPX32_1950-2022_",
    "batch_size": 16,
    "scaling": OmegaConf.load(
        update_scaling.create_yaml_if_not_exists(update_scaling_params["scale_file"])
    ),
    "overwrite": False,
}
# Retrive raw data
for request in era5_requests:
    era5_retrieval.main(request)
# Impute ocean data
data_imputation.triple_interp(impute_params)
# windspeed calculation
windspeed.main(windspeed_params)
# Remap data to HPX mesh
for hpx_param in hpx_params:
    map2hpx.main(hpx_param)
# 48 hour trailing average of atmospheric fields
for trailing_average_param in trailing_average_params:
    trailing_average.main(trailing_average_param)
# create zarr file for optimized training
dl.create_time_series_dataset_classic(**zarr_params)
