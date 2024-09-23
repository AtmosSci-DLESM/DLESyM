from utils import (
    era5_retrieval,
    data_imputation,
    map2hpx,
    windspeed,
    trailing_average,
    update_scaling,
    duacs_processing,
    write_zarr,
)
from training.dlwp.data import data_loading as dl
import numpy as np
from omegaconf import OmegaConf


"""
This data pipline creates training, validation, and test data for a coupled deep learning ocean model (DLOM). The coupled DLOM will forecast on a HEALPix 32 mesh, have 48 hour resolution, and predict a two prognostic fields: sea surface temperature (SST), and absolute dynamic topography (adt; similar to SSH). The model is trained to receive 10m windspeed and geopotential height at 1000 hPa for the 4 day period forecast. 
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
duacs_requests = {
    "years": np.arange(1993, 2023),
    "output_directory": "/home/disk/rhodium/dlwp/data/DUACS/raw_data",
    "overwrite": False,
}
# Parameters for imputing sst, adt data over land
impute_params = [
    # sst
    {
        "filename": "/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_sst.nc",
        "variable": "sst",
        "chunks": {"time": 10000},
        "imputed_file": "/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_sst-ti.nc",
    },
    # adt
    {
        "filename": "/home/disk/rhodium/dlwp/data/DUACS/raw_data/dt_global_twosat_phy_l4_*_vDT2021.nc",
        "variable": "adt",
        "chunks": {"time": 1},
        "imputed_file": "/home/disk/rhodium/dlwp/data/DUACS/dt_global_twosat_phy_l4_imputed_1993-2022_vDT2021_adt.nc",
    },
]
# Parameters for calculating wind speed
windspeed_params = {
    "u_file": "/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_u10m.nc",
    "v_file": "/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_v10m.nc",
    "target_file": "/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_windspeed.nc",
    "chunks": {"time": 10},
}
# Parameters for fixing coordinates of DUACS data. This insures ERA5 data and DUACS data are on the same grid.
duacs_fix_coords_params = {
    "variable_name": "adt",
    "input_file": "/home/disk/rhodium/dlwp/data/DUACS/dt_global_twosat_phy_l4_imputed_1993-2022_vDT2021_adt.nc",
    "output_file": "/home/disk/rhodium/dlwp/data/DUACS/dt_global_twosat_phy_l4_imputed_1993-2022_vDT2021_adt.pp.nc",
}
# parameters for healpix remapping
hpx_params = [
    # sst
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
    # adt
    {
        "file_name": "/home/disk/rhodium/dlwp/data/DUACS/dt_global_twosat_phy_l4_imputed_1993-2022_vDT2021_adt.pp.nc",
        "target_variable_name": "adt",
        "file_variable_name": "adt",
        "prefix": "/home/disk/rhodium/dlwp/data/HPX32/duacs_1deg_3h_HPX32_1993-2022_",
        "nside": 32,
        "order": "bilinear",
        "resolution_factor": 1.0,
        "visualize": False,
    },
    # windspeed
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
    # z1000
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
    # lsm
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
# Define the parameters for updating the scaling parameters of various variables from era5
update_scaling_params_era5 = {
    "scale_file": "/home/disk/quicksilver/nacc/dlesm/zephyr/training/configs/data/scaling/hpx32_1993-2022.yaml",
    "variable_file_prefix": "/home/disk/rhodium/dlwp/data/HPX32/era5_1deg_3h_HPX32_1950-2022_",
    "variable_names": [
        "sst",
        "ws10-48H",
        "z1000-48H",
    ],
    "selection_dict": {
        "sample": slice(np.datetime64("1993-01-01"), np.datetime64("2022-01-01"))
    },
    "overwrite": False,
    "chunks": None,
}
# Define the parameters for updating the scaling parameters of various variables from duacs
update_scaling_params_duacs = {
    "scale_file": "/home/disk/quicksilver/nacc/dlesm/zephyr/training/configs/data/scaling/hpx32_1993-2022.yaml",
    "variable_file_prefix": "/home/disk/rhodium/dlwp/data/HPX32/duacs_1deg_3h_HPX32_1993-2022_",
    "variable_names": [
        "adt",
    ],
    "selection_dict": {
        "sample": slice(np.datetime64("1993-01-01"), np.datetime64("2022-01-01"))
    },
    "overwrite": False,
    "chunks": None,
}
# make sure the scale file is the same for both era5 and duacs
if (
    update_scaling_params_era5["scale_file"]
    is not update_scaling_params_duacs["scale_file"]
):
    raise ValueError("Scale file used for era5 and duacs must be the same.")
# parameters used to write optimized zarr file
# TODO allow for different prefix for each variable and subsetting
zarr_params = {
    "dst_directory": "/home/disk/rhodium/dlwp/data/HPX32/",
    "dataset_name": "hpx32_1993-2022_3h_2varCoupledOcean-z1000-ws10",
    "inputs": {
        "sst": "/home/disk/rhodium/dlwp/data/HPX32/era5_1deg_3h_HPX32_1950-2022_sst.nc",
        "adt": "/home/disk/rhodium/dlwp/data/HPX32/duacs_1deg_3h_HPX32_1993-2022_adt.nc",
        "ws10-48H": "/home/disk/rhodium/dlwp/data/HPX32/era5_1deg_3h_HPX32_1950-2022_ws10-48H.nc",
        "z1000-48H": "/home/disk/rhodium/dlwp/data/HPX32/era5_1deg_3h_HPX32_1950-2022_z1000-48H.nc",
    },
    "outputs": {
        "sst": "/home/disk/rhodium/dlwp/data/HPX32/era5_1deg_3h_HPX32_1950-2022_sst.nc",
        "adt": "/home/disk/rhodium/dlwp/data/HPX32/duacs_1deg_3h_HPX32_1993-2022_adt.nc",
    },
    "constants": {
        "lsm": "/home/disk/rhodium/dlwp/data/HPX32/era5_1deg_3h_HPX32_1950-2022_lsm.nc"
    },
    # "prefix": "era5_1deg_3h_HPX32_1950-2022_",
    "batch_size": 16,
    "time_subset": slice(np.datetime64("1993-01-01"), np.datetime64("2022-01-01")),
    "scaling": OmegaConf.load(
        update_scaling.create_yaml_if_not_exists(
            update_scaling_params_era5["scale_file"]
        )
    ),
    "time_dim" : np.arange(np.datetime64("1993-01-01"), np.datetime64("2022-01-01T01"), np.timedelta64(3, 'h')),
    "overwrite": True,
}
# Retrive raw data
for request in era5_requests:
    era5_retrieval.main(request)
# retrive raw duacs data
duacs_processing.retrieve(duacs_requests)
# Impute ocean data
for impute_param in impute_params:
    data_imputation.triple_interp(impute_param)
# fix coordinates of duacs data
duacs_processing.fix_coords(duacs_fix_coords_params)
# windspeed calculation
windspeed.main(windspeed_params)
# Remap data to HPX mesh
for hpx_param in hpx_params:
    map2hpx.main(hpx_param)
# update scaling dictionary
update_scaling.main(update_scaling_params_era5)
update_scaling.main(update_scaling_params_duacs)
# 48 hour trailing average of atmospheric fields
for trailing_average_param in trailing_average_params:
    trailing_average.main(trailing_average_param)
# create zarr file for optimized training
write_zarr.create_prebuilt_zarr(**zarr_params)
