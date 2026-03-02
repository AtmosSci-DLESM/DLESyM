import xarray as xr
import numpy as np
import pandas as pd
import cftime
import os
import logging
from dask.diagnostics import ProgressBar
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _valid_times_to_cf_numeric(valid_times):
    """Convert valid-time array to CF-compliant numeric values (Optimized)."""
    
    # 1850 is the standard CMIP6/7 reference epoch
    TIME_UNITS = "days since 1850-01-01 00:00:00"
    TIME_CALENDAR = "gregorian"

    # Vectorized conversion to pandas datetime
    dt_index = pd.to_datetime(np.asarray(valid_times).ravel())

    # Use cftime.date2num directly on the pydatetime objects.
    # We cast to float64 to prevent coordinate 'drift' in long time series.
    numeric_times = cftime.date2num(
        dt_index.to_pydatetime(), 
        units=TIME_UNITS, 
        calendar=TIME_CALENDAR
    ).astype(np.float64)

    return numeric_times

def save_daily_average(ds, output_dir, experiment, r, variable, surface=False):
    # recursively make directory structure for output
    path = os.path.join(output_dir, 'university_of_washington', 'DLESyM', experiment, f'r{r}i1p1f1', 'day', variable, 'gn', 'v20250825')
    os.makedirs(path, exist_ok=True)

    # encoding instructions for netcdf
    if surface:
        encoding = {
            variable: {'dtype': 'float32'},
            'time': {'dtype': 'float32'},
            'face': {'dtype': 'int32'},
            'height': {'dtype': 'int32'},
            'width': {'dtype': 'int32'},
        }
    else:
        encoding = {
            variable: {'dtype': 'float32'},
            'time': {'dtype': 'float32'},
            'plev': {'dtype': 'float32'},
            'face': {'dtype': 'int32'},
            'height': {'dtype': 'int32'},
            'width': {'dtype': 'int32'},
        }

    TIME_UNITS = ds.attrs['time_units']
    TIME_CALENDAR = ds.attrs['time_calendar']
    # Resample to daily mean with labels at middle of day (12:00)
    times_cf = cftime.num2date(ds.time.values, units=TIME_UNITS, calendar=TIME_CALENDAR)
    ds = ds.assign_coords(time=times_cf)
    ds = ds.resample(time='1D').mean()
    # select first 2.25 years data : October 1983 to December 1984
    ds = ds.sel(time=slice('1983-10-01', '1984-12-31'))
    # Shift labels from day-start to mid-day (12:00 at 00:00)
    mid_day = np.array([type(t)(t.year, t.month, t.day, 12, 0, 0) for t in ds.time.values])
    time_num = cftime.date2num(mid_day, units=TIME_UNITS, calendar=TIME_CALENDAR)
    ds = ds.assign_coords(time=time_num.astype(np.float32))    
    # resolve start and end time from time coordinate for filename
    start_dt = cftime.num2date(float(ds.time.values[0]), units=TIME_UNITS, calendar=TIME_CALENDAR)
    end_dt = cftime.num2date(float(ds.time.values[-1]), units=TIME_UNITS, calendar=TIME_CALENDAR)
    start_time = f"{start_dt.year:04d}{start_dt.month:02d}{start_dt.day:02d}"
    end_time = f"{end_dt.year:04d}{end_dt.month:02d}{end_dt.day:02d}"
    # filename: {variable}_{frequency}_{model_name}_{experiment}_{ensemble_member}_{grid}_{start_time}-{end_time}.nc
    file_name = f'{variable}_day_DLESyM_{experiment}_r{r}i1p1f1_gn_{start_time}-{end_time}.nc'
    logger.info(f"Saving daily average to {os.path.join(path, file_name)}")
    with ProgressBar():
        ds.to_netcdf(os.path.join(path, file_name), encoding=encoding)
    logger.info(f"Saved daily average to {os.path.join(path, file_name)}")

def save_monthly_average(ds, output_dir, experiment, r, variable, surface=False):

    # recursively make directory structure for output
    path = os.path.join(output_dir, 'university_of_washington', 'DLESyM', experiment, f'r{r}i1p1f1', 'Amon', variable, 'gn', 'v20250825')
    os.makedirs(path, exist_ok=True)

    # Resample to monthly mean with labels at middle of month (16th)
    TIME_UNITS = ds.attrs['time_units']
    TIME_CALENDAR = ds.attrs['time_calendar']
    times_cf = cftime.num2date(ds.time.values, units=TIME_UNITS, calendar=TIME_CALENDAR)
    ds = ds.assign_coords(time=times_cf)
    ds = ds.resample(time='1MS').mean()

    # Shift labels from month-start to mid-month (16th at 00:00)
    mid_month = np.array([type(t)(t.year, t.month, 16, 0, 0, 0) for t in ds.time.values])
    time_num = cftime.date2num(mid_month, units=TIME_UNITS, calendar=TIME_CALENDAR)
    ds = ds.assign_coords(time=time_num.astype(np.float32))

    # resolve start and end time from time coordinate for filename
    start_dt = cftime.num2date(float(ds.time.values[0]), units=TIME_UNITS, calendar=TIME_CALENDAR)
    end_dt = cftime.num2date(float(ds.time.values[-1]), units=TIME_UNITS, calendar=TIME_CALENDAR)
    start_time = f"{start_dt.year:04d}{start_dt.month:02d}{start_dt.day:02d}"
    end_time = f"{end_dt.year:04d}{end_dt.month:02d}{end_dt.day:02d}"
    # encoding instructions for netcdf
    if surface:
        encoding = {
            variable: {'dtype': 'float32'},
            'time': {'dtype': 'float32'},
            'face': {'dtype': 'int32'},
            'height': {'dtype': 'int32'},
            'width': {'dtype': 'int32'},
        }
    else:
        encoding = {
            variable: {'dtype': 'float32'},
            'time': {'dtype': 'float32'},
            'plev': {'dtype': 'float32'},
            'face': {'dtype': 'int32'},
            'height': {'dtype': 'int32'},
            'width': {'dtype': 'int32'},
        }

    # filename: {variable}_{frequency}_{model_name}_{experiment}_{ensemble_member}_{grid}_{start_time}-{end_time}.nc
    file_name = f'{variable}_Amon_DLESyM_{experiment}_r{r}i1p1f1_gn_{start_time}-{end_time}.nc'
    logger.info(f"Saving monthly average to {os.path.join(path, file_name)}")
    with ProgressBar():
        ds.to_netcdf(os.path.join(path, file_name), encoding=encoding)
    logger.info(f"Saved monthly average to {os.path.join(path, file_name)}")

def cmortize_geopotential_height(forecast_file, output_dir, r, experiment):
    """
    Cmortize DLESyM geopotential height forecasts.
    """
    logger.info(f"Cmortizing DLESyM forecasts from {forecast_file} to {output_dir}")
    ds = xr.open_dataset(forecast_file, chunks={'step': 768})
    variables = ds.keys()

    # assert that there is only one initialization time
    assert len(ds.time.values) == 1, "Rollouts should only have one initialization time. Instead got {len(ds.time.values)}"

    # Valid times: init (time) + lead (step) for each (time, step) pair
    valid_times = (ds.time.values[0] + ds.step.values)
    logger.info(f"Valid time range: {valid_times.flat[0]} to {valid_times.flat[-1]}")

    # CF-compliant time coordinate: numeric values with units and calendar
    time_cf = _valid_times_to_cf_numeric(valid_times)

    ## Geopotential height (plev in Pa: 250, 500, 1000 hPa -> 25000, 50000, 100000 Pa)
    zg_da = xr.concat(
        [ds.z250.rename('zg').isel(time=0), 
        ds.z500.rename('zg').isel(time=0), 
        ds.z1000.rename('zg').isel(time=0)], 
        dim='plev').assign_coords(plev=[25000., 50000., 100000.])
    # fix time
    zg_da = zg_da.drop('time').rename({'step': 'time'}).assign_coords(time=time_cf) 
    # [time, pressure, face, height, width]
    zg_da = zg_da.transpose('time', 'plev', 'face', 'height', 'width')
    # convert to m
    zg_da = zg_da / 9.81 

    # Enforce 32 bit precision for data vars and coordinates
    zg_da = zg_da.astype(np.float32)
    zg_da.coords['time'] = zg_da.coords['time'].astype(np.float32)
    zg_da.coords['plev'] = zg_da.coords['plev'].astype(np.float32)
    zg_da.coords['face'] = zg_da.coords['face'].astype(np.int32)
    zg_da.coords['height'] = zg_da.coords['height'].astype(np.int32)
    zg_da.coords['width'] = zg_da.coords['width'].astype(np.int32)
    zg_da.attrs = {
        'long_name': 'geopotential_height',
        'units': 'm',
        'time_units': 'days since 1850-01-01 00:00:00',
        'time_calendar': 'gregorian',
    }   
    # units for pressure levels
    zg_da.coords['plev'].attrs['units'] = 'Pa'

    save_monthly_average(zg_da, output_dir, experiment, r, 'zg')
    save_daily_average(zg_da, output_dir, experiment, r, 'zg')

def cmortize_temperature(forecast_file, output_dir, r, experiment):
    """
    Cmortize DLESyM temperature forecasts (only available at 850hPa level)
    """
    logger.info(f"Cmortizing DLESyM temperature forecasts from {forecast_file} to {output_dir}")
    ds = xr.open_dataset(forecast_file, chunks={'step': 768})
    variables = ds.keys()

    # assert that there is only one initialization time 
    assert len(ds.time.values) == 1, "Rollouts should only have one initialization time. Instead got {len(ds.time.values)}"

    # Valid times: init (time) + lead (step) for each (time, step) pair
    valid_times = (ds.time.values[0] + ds.step.values)
    logger.info(f"Valid time range: {valid_times.flat[0]} to {valid_times.flat[-1]}")

    # CF-compliant time coordinate: numeric values with units and calendar
    time_cf = _valid_times_to_cf_numeric(valid_times)   

    ## Temperature (plev in Pa: 850 hPa -> 85000 Pa)
    ta_da = ds.t850.rename('ta').isel(time=0).expand_dims('plev').assign_coords(plev=[85000.])
    # fix time
    ta_da = ta_da.drop('time').rename({'step': 'time'}).assign_coords(time=time_cf) 
    # [time, pressure, face, height, width]
    ta_da = ta_da.transpose('time', 'plev', 'face', 'height', 'width')

    # Enforce 32 bit precision for data vars and coordinates
    ta_da = ta_da.astype(np.float32)
    ta_da.coords['time'] = ta_da.coords['time'].astype(np.float32)
    ta_da.coords['plev'] = ta_da.coords['plev'].astype(np.float32)
    ta_da.coords['face'] = ta_da.coords['face'].astype(np.int32)
    ta_da.coords['height'] = ta_da.coords['height'].astype(np.int32)
    ta_da.coords['width'] = ta_da.coords['width'].astype(np.int32)
    ta_da.attrs = {
        'long_name': 'air_temperature',
        'units': 'K',
        'time_units': 'days since 1850-01-01 00:00:00',
        'time_calendar': 'gregorian',
    }
    # units for pressure levels
    ta_da.coords['plev'].attrs['units'] = 'Pa'

    save_monthly_average(ta_da, output_dir, experiment, r, 'ta')
    save_daily_average(ta_da, output_dir, experiment, r, 'ta')

def cmortize_surface_temperature(forecast_file, output_dir, r, experiment):
    """
    Cmortize DLESyM surface temperature forecasts.
    """
    logger.info(f"Cmortizing DLESyM surface temperature forecasts from {forecast_file} to {output_dir}")
    ds = xr.open_dataset(forecast_file, chunks={'step': 768})
    variables = ds.keys()

    # assert that there is only one initialization time 
    assert len(ds.time.values) == 1, "Rollouts should only have one initialization time. Instead got {len(ds.time.values)}"

    # Valid times: init (time) + lead (step) for each (time, step) pair
    valid_times = (ds.time.values[0] + ds.step.values)
    logger.info(f"Valid time range: {valid_times.flat[0]} to {valid_times.flat[-1]}")

    # CF-compliant time coordinate: numeric values with units and calendar
    time_cf = _valid_times_to_cf_numeric(valid_times)   
    ## Surface temperature
    tas_da = ds.t2m0.rename('tas').isel(time=0)
    # fix time
    tas_da = tas_da.drop('time').rename({'step': 'time'}).assign_coords(time=time_cf) 
    # [time, face, height, width]
    tas_da = tas_da.transpose('time', 'face', 'height', 'width')
    # Enforce 32 bit precision for data vars and coordinates
    tas_da = tas_da.astype(np.float32)
    tas_da.coords['time'] = tas_da.coords['time'].astype(np.float32)
    tas_da.coords['face'] = tas_da.coords['face'].astype(np.int32)
    tas_da.coords['height'] = tas_da.coords['height'].astype(np.int32)
    tas_da.coords['width'] = tas_da.coords['width'].astype(np.int32)
    tas_da.attrs = {
        'long_name': 'surface_temperature',
        'units': 'K',
        'time_units': 'days since 1850-01-01 00:00:00',
        'time_calendar': 'gregorian',
    }
    save_monthly_average(tas_da, output_dir, experiment, r, 'tas', surface=True)
    save_daily_average(tas_da, output_dir, experiment, r, 'tas', surface=True)

def cmortize_dlesym(forecast_file, output_dir, r, experiment):
    """
    Cmortize DLESyM forecasts.
    """
    logger.info(f"Cmortizing DLESyM forecasts from {forecast_file} to {output_dir}")
    cmortize_geopotential_height(forecast_file, output_dir, r, experiment)
    cmortize_temperature(forecast_file, output_dir, r, experiment)
    cmortize_surface_temperature(forecast_file, output_dir, r, experiment)
    
if __name__ == "__main__":

    # aimip
    logger.info('cmortizing aimip r1')
    cmortize_dlesym(
        forecast_file='/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1983-2025_n01.nc',
        output_dir='/home/disk/mercury3/nacc/aimip_subission',
        r = 1,
        experiment = 'aimip',
    )
    logger.info('cmortizing aimip r2')
    cmortize_dlesym(
        forecast_file='/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1983-2025_n02.nc',
        output_dir='/home/disk/mercury3/nacc/aimip_subission',
        r = 2,
        experiment = 'aimip',
    )
    logger.info('cmortizing aimip r3')
    cmortize_dlesym(
        forecast_file='/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1983-2025_n03.nc',
        output_dir='/home/disk/mercury3/nacc/aimip_subission',
        r = 3,
        experiment = 'aimip',
    )
    logger.info('cmortizing aimip r4')
    cmortize_dlesym(
        forecast_file='/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1983-2025_n04.nc',
        output_dir='/home/disk/mercury3/nacc/aimip_subission',
        r = 4,
        experiment = 'aimip',
    )
    logger.info('cmortizing aimip r5')
    cmortize_dlesym(
        forecast_file='/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1983-2025_n05.nc',
        output_dir='/home/disk/mercury3/nacc/aimip_subission',
        r = 5,
        experiment = 'aimip',
    )

    # aimip-p2k
    logger.info('cmortizing aimip-p2k r1')
    cmortize_dlesym(
        forecast_file='/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1983-2025_2k_n01.nc',
        output_dir='/home/disk/mercury3/nacc/aimip_subission',
        r = 1,
        experiment = 'aimip-p2k',
    )
    logger.info('cmortizing aimip-p2k r2')
    cmortize_dlesym(
        forecast_file='/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1983-2025_2k_n02.nc',
        output_dir='/home/disk/mercury3/nacc/aimip_subission',
        r = 2,
        experiment = 'aimip-p2k',
    )
    logger.info('cmortizing aimip-p2k r3')
    cmortize_dlesym(
        forecast_file='/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1983-2025_2k_n03.nc',
        output_dir='/home/disk/mercury3/nacc/aimip_subission',
        r = 3,
        experiment = 'aimip-p2k',
    )
    logger.info('cmortizing aimip-p2k r4')
    cmortize_dlesym(
        forecast_file='/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1983-2025_2k_n04.nc',
        output_dir='/home/disk/mercury3/nacc/aimip_subission',
        r = 4,
        experiment = 'aimip-p2k',
    )
    logger.info('cmortizing aimip-p2k r5')
    cmortize_dlesym(
        forecast_file='/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1983-2025_2k_n05.nc',
        output_dir='/home/disk/mercury3/nacc/aimip_subission',
        r = 5,
        experiment = 'aimip-p2k',
    )

    # aimip-p4k
    logger.info('cmortizing aimip-p4k r1')
    cmortize_dlesym(
        forecast_file='/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1983-2025_4k_n01.nc',
        output_dir='/home/disk/mercury3/nacc/aimip_subission',
        r = 1,
        experiment = 'aimip-p4k',
    )
    logger.info('cmortizing aimip-p4k r2')
    cmortize_dlesym(
        forecast_file='/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1983-2025_4k_n02.nc',
        output_dir='/home/disk/mercury3/nacc/aimip_subission',
        r = 2,
        experiment = 'aimip-p4k',
    )
    logger.info('cmortizing aimip-p4k r3')
    cmortize_dlesym(
        forecast_file='/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1983-2025_4k_n03.nc',
        output_dir='/home/disk/mercury3/nacc/aimip_subission',
        r = 3,
        experiment = 'aimip-p4k',
    )
    logger.info('cmortizing aimip-p4k r4')
    cmortize_dlesym(
        forecast_file='/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1983-2025_4k_n04.nc',
        output_dir='/home/disk/mercury3/nacc/aimip_subission',
        r = 4,
        experiment = 'aimip-p4k',
    )
    logger.info('cmortizing aimip-p4k r5')
    cmortize_dlesym(
        forecast_file='/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1983-2025_4k_n05.nc',
        output_dir='/home/disk/mercury3/nacc/aimip_subission',
        r = 5,
        experiment = 'aimip-p4k',
    )