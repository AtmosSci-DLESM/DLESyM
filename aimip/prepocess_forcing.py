# basic imports
import os 
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# data processing libraries
import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar
from omegaconf import OmegaConf

# data pipeline utilities
from data_processing.utils import (
    data_imputation,
    map2hpx,
    update_scaling,
    write_zarr
)

# basic reformatting for ingestion into DLESyM data pipeline:
# 1. changes sea_surface_temperature to sst and land_sea_mask to lsm
# 2. drops sea_ice_cover (not used in DLESyM)
# 2. save LSM as seperate file
def reformat_aimip_forcing(params):
    logger.info(f"Reformatting AIMIP forcing data from {params['filename']} to {params['imputed_file']}")
    ds = xr.open_dataset(params['filename'])
    ds = ds.rename({
        "sea_surface_temperature": "sst",
        "land_sea_mask": "lsm",
    })
    ds = ds.drop("sea_ice_cover")
    lsm = ds.lsm.squeeze()
    ds = ds.drop("lsm")
    logger.info(f"Saving reformatted data to {params['imputed_file']}...")
    ds.to_netcdf(params['imputed_file'])
    lsm.to_netcdf(params['lsm_file'])
    logger.info(f"...done!")

def resample_monthly_zarr_to_daily(
    src_zarr: str,
    dst_zarr: str,
    time_chunk: int = 128,
) -> None:
    """
    Resample a monthly HPX64 forcing Zarr dataset to daily resolution using
    linear interpolation in time, writing the result with chunked Dask IO.

    Parameters
    ----------
    src_zarr : str
        Path to the source monthly Zarr store (e.g., the output of
        write_zarr.create_prebuilt_zarr).
    dst_zarr : str
        Path where the daily Zarr store will be written.
    time_chunk : int, optional
        Chunk size along the time dimension for both computation and writing.
    """
    logger.info(f"Opening monthly forcing Zarr dataset from {src_zarr}")
    ds = xr.open_zarr(src_zarr, chunks={"time": time_chunk})

    # Build a daily time axis spanning the monthly range
    time_values = ds["time"].values
    start = time_values[0].astype("datetime64[D]")
    end = time_values[-1].astype("datetime64[D]")
    daily_time = np.arange(start, end + np.timedelta64(1, "D"), np.timedelta64(1, "D"))

    logger.info(
        f"Resampling from monthly ({len(time_values)} steps) to daily "
        f"({len(daily_time)} steps) using linear interpolation in time."
    )
    ds_daily = ds.interp(time=daily_time)

    # Ensure channel coordinate labels are proper strings for Zarr's VLenUTF8 codec
    for coord_name in ("channel_c", "channel_in", "channel_out"):
        if coord_name in ds_daily.coords:
            ds_daily = ds_daily.assign_coords(
                {coord_name: ds_daily[coord_name].astype(str)}
            )

    # Ensure reasonable chunking before writing
    ds_daily = ds_daily.chunk({"time": time_chunk})

    logger.info(f"Writing daily-resampled Zarr dataset to {dst_zarr}")
    write_job = ds_daily.to_zarr(
        dst_zarr,
        mode="w",
        encoding={"time": {"dtype": "float64"}},
        compute=False,
    )
    with ProgressBar():
        write_job.compute()
    logger.info("Finished writing daily-resampled forcing dataset.")

def increase_sst_uniform(
    input_zarr: str,
    output_zarr: str,
    delta_K: float,
    overwrite: bool = False,
) -> None:
    """
    Increase the SST field uniformly by ``delta_K``.
    """
    if os.path.exists(output_zarr) and not overwrite:
        logger.info(f"Output {output_zarr} already exists and overwrite=False. Skipping.")
        return
    logger.info(f"Increasing SST field uniformly by {delta_K} K")
    ds = xr.open_zarr(input_zarr)
    
    # Increment sst channel in both input and targets (xr.where preserves dask laziness for chunked writes)
    inputs = ds["inputs"]
    targets = ds["targets"]
    inputs = inputs + xr.where(inputs.channel_in == "sst", delta_K, 0)
    targets = targets + xr.where(targets.channel_out == "sst", delta_K, 0)
    ds = ds.assign(inputs=inputs)
    ds = ds.assign(targets=targets)

    logger.info(f"Writing increased SST field to {output_zarr}")
    with ProgressBar():
        ds.to_zarr(output_zarr)
        logger.info(f"Finished writing increased SST field to {output_zarr}")
    return
# basic reformatting
reformat_params = {
    "filename": '/home/disk/mercury2/nacc/AIMIP2026/DLESyM/aimip/forcing_data/ERA5-0.25deg-monthly-mean-forcing-1978-2024.nc',
    "imputed_file": "/home/disk/mercury2/nacc/AIMIP2026/DLESyM/aimip/forcing_data/ERA5-0.25deg-monthly-mean-forcing-1978-2024_reformatted.nc",  # File to save the imputed data
    "lsm_file": "/home/disk/mercury2/nacc/AIMIP2026/DLESyM/aimip/forcing_data/ERA5-0.25deg-monthly-mean-forcing-1978-2024_lsm.nc",  # File to save the LSM data
}
# Parameters for imputing sst data over land
impute_params = {
    "filename": '/home/disk/mercury2/nacc/AIMIP2026/DLESyM/aimip/forcing_data/ERA5-0.25deg-monthly-mean-forcing-1978-2024_reformatted.nc',
    "variable": "sst",  # Variable in the file that needs imputation
    "chunks": {"time": 1024},  # Chunk size for processing the data
    "imputed_file": "/home/disk/mercury2/nacc/AIMIP2026/DLESyM/aimip/forcing_data/ERA5-0.25deg-monthly-mean-forcing-1978-2024_sst-imputed.nc",  # File to save the imputed data
}
# parameters for healpix remapping
hpx_params = [
    {
        "file_name": "/home/disk/mercury2/nacc/AIMIP2026/DLESyM/aimip/forcing_data/ERA5-0.25deg-monthly-mean-forcing-1978-2024_sst-imputed.nc",  # The path to the input file
        "target_variable_name": "sst",  # The name of the variable in the input dataset
        "file_variable_name": "sst",  # The name of the variable in in the newly generated file
        "prefix": "/home/disk/mercury2/nacc/AIMIP2026/DLESyM/aimip/forcing_data/ERA5-0.25deg-monthly-mean-forcing-1978-2024_HPX64_",  # The prefix for the output file names
        "nside": 64,  # The number of divisions on the side of the grid
        "order": "bilinear",  # The interpolation method to use when regridding
        "resolution_factor": 1.0,  # The factor by which to change the resolution of the data
        "visualize": False,  # Whether to generate a visualization of the regridded data
    },
    {
        "file_name": "/home/disk/mercury2/nacc/AIMIP2026/DLESyM/aimip/forcing_data/ERA5-0.25deg-monthly-mean-forcing-1978-2024_lsm.nc",  # The path to the input file
        "target_variable_name": "lsm",  # The name of the variable in the input dataset
        "file_variable_name": "lsm",  # The name of the variable in in the newly generated file
        "prefix": "/home/disk/mercury2/nacc/AIMIP2026/DLESyM/aimip/forcing_data/ERA5-0.25deg-monthly-mean-forcing-1978-2024_HPX64_",  # The prefix for the output file names
        "nside": 64,  # The number of divisions on the side of the grid
        "order": "bilinear",  # The interpolation method to use when regridding
        "resolution_factor": 1.0,  # The factor by which to change the resolution of the data
        "visualize": False,  # Whether to generate a visualization of the regridded data
    },
]
# params for compilation of zarr dataset 
zarr_params = {
    "dst_directory": "/home/disk/mercury2/nacc/AIMIP2026/DLESyM/aimip/forcing_data/",
    "dataset_name": "AIMIP_1978-2024_monthly_HPX64",
    "inputs": {
        "sst": "/home/disk/mercury2/nacc/AIMIP2026/DLESyM/aimip/forcing_data/ERA5-0.25deg-monthly-mean-forcing-1978-2024_HPX64_sst.nc",
    },
    "outputs": {
        "sst": "/home/disk/mercury2/nacc/AIMIP2026/DLESyM/aimip/forcing_data/ERA5-0.25deg-monthly-mean-forcing-1978-2024_HPX64_sst.nc",
    },
    "constants": {
        "lsm": "/home/disk/mercury2/nacc/AIMIP2026/DLESyM/aimip/forcing_data/ERA5-0.25deg-monthly-mean-forcing-1978-2024_HPX64_lsm.nc",
    },
    "batch_size": 16,
    "scaling": OmegaConf.load(
        update_scaling.create_yaml_if_not_exists("/home/disk/mercury2/nacc/AIMIP2026/DLESyM/aimip/forcing_data/scaling_aimip.yaml")
    ),
    "overwrite": False,
}

# params for resampling of zarr dataset to daily resolution
resample_params = {
    "src_zarr": "/home/disk/mercury2/nacc/AIMIP2026/DLESyM/aimip/forcing_data/AIMIP_1978-2024_monthly_HPX64.zarr",
    "dst_zarr": "/home/disk/mercury2/nacc/AIMIP2026/DLESyM/aimip/forcing_data/AIMIP_1978-2024_daily_HPX64.zarr",
    "time_chunk": 128,
}

# params for increasing SST field uniformly
increase_sst_uniform_params_2k = {
    "input_zarr": "/home/disk/mercury2/nacc/AIMIP2026/DLESyM/aimip/forcing_data/AIMIP_1978-2024_daily_HPX64.zarr",
    "output_zarr": "/home/disk/mercury2/nacc/AIMIP2026/DLESyM/aimip/forcing_data/AIMIP_1978-2024_daily_HPX64_2k.zarr",
    "delta_K": 2.0,
}
# params for increasing SST by 4k
increase_sst_uniform_params_4k = {
    "input_zarr": "/home/disk/mercury2/nacc/AIMIP2026/DLESyM/aimip/forcing_data/AIMIP_1978-2024_daily_HPX64.zarr",
    "output_zarr": "/home/disk/mercury2/nacc/AIMIP2026/DLESyM/aimip/forcing_data/AIMIP_1978-2024_daily_HPX64_4k.zarr",
    "delta_K": 4.0,
}

if __name__ == "__main__":
    logger.info("Starting AIMIP forcing data preprocessing...")
    logger.info("Reformatting AIMIP forcing data...")
    reformat_aimip_forcing(reformat_params)
    logger.info("Imputing SST data over land...")
    data_imputation.triple_interp(impute_params)
    logger.info("Remapping data to HPX mesh...")
    for hpx_param in hpx_params:
        map2hpx.main(hpx_param)
    logger.info("Compiling zarr dataset...")
    write_zarr.create_prebuilt_zarr(**zarr_params)
    logger.info("Resampling zarr dataset to daily resolution...")
    resample_monthly_zarr_to_daily(**resample_params)

    # create incremented experiment forcing files
    increase_sst_uniform(**increase_sst_uniform_params_2k)
    increase_sst_uniform(**increase_sst_uniform_params_4k)