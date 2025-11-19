import os
import xarray as xr 
import numpy as np
from xarray.conventions import SerializationWarning
import pandas as pd
import cftime
import warnings
# Suppress SerializationWarning from xarray
warnings.filterwarnings("ignore", category=SerializationWarning)
# Suppress RuntimeWarning for CFTimeIndex conversion
warnings.filterwarnings("ignore", category=RuntimeWarning)

# custom slicing function that allows for non unique dates 
# this is necessary for handling non-gregorian calendars that appear in 
# some CMIP6 models 
def custom_time_slice(ds, time_slice):

    start_date = pd.Timestamp(time_slice.start)
    end_date = pd.Timestamp(time_slice.stop)

    # Create a mask for selecting data within the time range
    time_mask = (ds['time'].values >= start_date) & (ds['time'].values <= end_date)

    # Apply the mask to select data within the time range
    return ds.isel(time=time_mask)

def cesm2(model_dir):

    ds = xr.open_mfdataset(f'{model_dir}/{f}' for f in os.listdir(model_dir)).zg
    # select 500hPa level
    ds = ds.sel(plev=50000).squeeze()
    # Check if the time values are cftime objects and convert them to datetime
    if isinstance(ds['time'].values[0], cftime._cftime.DatetimeNoLeap):
        # Convert cftime to numpy datetime64
        ds['time'] = xr.coding.cftimeindex.CFTimeIndex(ds['time'].values).to_datetimeindex()

    return ds*9.81 # convert to gp

def mpi_esm1(model_dir):

    ds = xr.open_mfdataset(f'{model_dir}/{f}' for f in os.listdir(model_dir)).zg
    # select 500hPa level
    ds = ds.sel(plev=50000).squeeze()
    return ds*9.81 # convert to gp

def gfdl_cm4(model_dir):
    # Open the dataset
    ds = xr.open_mfdataset(f'{model_dir}/{f}' for f in os.listdir(model_dir)).zg
    
    # Select 500hPa level and squeeze out the single level dimensions
    ds = ds.sel(plev=50000).squeeze()
    
    # Define the new 1 degree mesh, this is necessary as GFDL native mesh is not 
    # compatible with blocking calculation
    new_lat = xr.DataArray(data=np.arange(-90, 90.1, 1), dims="lat", name="lat")
    new_lon = xr.DataArray(data=np.arange(0, 360.1, 1), dims="lon", name="lon")
    
    # Interpolate the dataset onto the new mesh
    ds_interp = ds.interp(lat=new_lat, lon=new_lon, kwargs={'fill_value': 'extrapolate'})
    
    # Convert to geopotential
    return ds_interp * 9.81

def had_gem(model_dir):

    ds = xr.open_mfdataset([f'{model_dir}/{f}' for f in os.listdir(model_dir)], chunks={'time':-1}).zg
    # select 500hPa level
    ds = ds.sel(plev=50000).squeeze()
    
    # Change the time dimension from CFTime 360-day year to Gregorian calendar
    ds = ds.convert_calendar(calendar='gregorian',align_on='date')
    ds = ds.chunk({'time': -1})
    ds = ds.interpolate_na(dim='time',fill_value="extrapolate")
    return ds*9.81 # convert to gp
    
def ipsl(model_dir):

    ds = xr.open_mfdataset(f'{model_dir}/{f}' for f in os.listdir(model_dir)).zg
    # select 500hPa level
    ds = ds.sel(plev=50000).squeeze()

    return ds*9.81 # convert to gp
    