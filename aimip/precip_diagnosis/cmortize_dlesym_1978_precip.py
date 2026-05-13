import xarray as xr
import numpy as np
import pandas as pd
import cftime
import os
import logging
from dask.diagnostics import ProgressBar
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from aimip.cmortize_dlesym_1978 import _valid_times_to_cf_numeric, save_daily_average, save_monthly_average, _set_metadata_coordinates

def convert_to_kg_m2_s(da, dt=6):
    """
    Convert precipitation from meters to kg/(m^2 s).

    Parameters:
        da: xarray.DataArray
            Precipitation data array in meters.
        dt: int, optional
            Time step in hours. Default is 6 hours.

    Returns:
        xarray.DataArray: Precipitation data array in kg/(m^2 s).
    """
    rho = 1000.0  # kg/m^3
    dt = dt * 3600  # s
    rate = (rho * da) / dt 
    return rate

def fix_multi_init(ds):
    """
    Take forecast with multiple inits and single step and return a forecast with single init and multiple steps.
    """
    logger.info(f"Fixing multi-init forecast with single step to single init and multiple steps")
    ds = ds.rename({'time': 'step_'})
    ds = ds.rename({'step': 'time', 'step_':'step'})
    return ds

def cmortize_precip(forecast_file, output_dir, r, experiment):
    """
    Cmortize DLESyM precipitation forecasts.
    """
    logger.info(f"Cmortizing DLESyM precipitation forecasts from {forecast_file} to {output_dir}")
    ds = xr.open_dataset(forecast_file, chunks={'time': 768})
    variables = ds.keys()

    # Valid times: init (time) + lead (step) for each (time, step) pair
    valid_times = (ds.step.values[0] + ds.time.values)
    logger.info(f"Valid time range: {valid_times.flat[0]} to {valid_times.flat[-1]}")

    # CF-compliant time coordinate: numeric values with units and calendar
    time_cf = _valid_times_to_cf_numeric(valid_times)   

    # an idiosyncrosy of precip is that the full simulation is registered as a series of inits
    # here we fix this so that precip follows normal organization
    ds = fix_multi_init(ds)
    ## precip
    pr_da = ds.tp6.rename('pr').isel(time=0)
    # fix time
    pr_da = pr_da.drop('time').rename({'step': 'time'}).assign_coords(time=time_cf) 
    # [time, face, height, width]
    pr_da = pr_da.transpose('time', 'face', 'height', 'width')
    # Enforce 32 bit precision for data vars and coordinates
    pr_da = pr_da.astype(np.float32)
    pr_da.coords['time'] = pr_da.coords['time'].astype(np.float32)
    pr_da.coords['face'] = pr_da.coords['face'].astype(np.int32)
    pr_da.coords['height'] = pr_da.coords['height'].astype(np.int32)
    pr_da.coords['width'] = pr_da.coords['width'].astype(np.int32)

    # convert to kg/(m^2 s) from meters
    pr_da = convert_to_kg_m2_s(pr_da)

    # make dataset
    pr_ds = pr_da.to_dataset(name='pr')
    # save jobs
    save_monthly_average(pr_ds, output_dir, experiment, r, 'pr', units='kg/(m^2 s)', long_name='surface_precipitation_rate', surface=True)
    save_daily_average(pr_ds, output_dir, experiment, r, 'pr', units='kg/(m^2 s)', long_name='surface_precipitation_rate', surface=True)

if __name__ == "__main__":

    logger.info('cmortizing aimip-p2k r5')
    cmortize_precip(
        forecast_file='/home/disk/brass/nacc/forecasts/aimip/p2k_r5/precip_aimip_forced_forecast_1978-2025_p2k_r5.nc',
        output_dir='/home/disk/mercury3/nacc/aimip_subission_1978',
        r = 5,
        experiment = 'aimip-p2k',
    )