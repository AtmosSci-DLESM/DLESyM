import xarray as xr 
import os
from dask.diagnostics import ProgressBar
import numpy as np 

# params for triple interpolation imputation of sst 
PARAMS_TI = {
    'filename':'/home/disk/brass/nacc/data/ERA5/1-deg/1979-2021_era5_1deg_3h_sea_surface_temperature.nc',
    'variable':'sst',
    'chunks':{'time':10},
    'imputed_file':'/home/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_sea_surface_temperature_ti-imputed.nc'
}

# params for zonal climotological imputation of sst 
PARAMS_CLIMO = {
    'filename':'/home/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_sea_surface_temperature.nc',
    'variable':'sst',
    'chunks':{'time':10},
    'imputed_file':'/home/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_sea_surface_temperature_climo-imputed.nc'
}

def zonal_climo_impute(params):

    if os.path.isfile(params["imputed_file"]):
        print(f'Target file {params["imputed_file"]} already exists. Aborting.')
        return

    da = xr.open_dataset(params['filename'],chunks = params['chunks'])[params['variable']]
    # calculate climotology 
    climo = xr.open_dataset(params['filename'],chunks = params['chunks'])[params['variable']].mean(dim='time')
    # get zonal climatology and expand dimensions so that it is defined in lat-lon dimensions  
    zonal_climo = climo.mean(dim='longitude').expand_dims(dim={'longitude':climo.longitude.values},axis=1)
    # use polar interpolation to fill in polar NaNs
    zonal_climo_polar_interp = zonal_climo.roll(latitude=int(len(zonal_climo.latitude)/2)).interpolate_na(dim='latitude',method='linear',use_coordinate=False).roll(latitude=len(zonal_climo.latitude)-int(len(zonal_climo.latitude)/2))
    # impute NaN pixels with new polar interpolated  zonally climatology 
    da_climo_imputed = xr.where(np.isnan(da),zonal_climo_polar_interp,da)
     
    print(f'Beginning computation of climotological imputation and writing...')
    print(f'INPUT filename: {params["filename"]}')
    print(f'IMPUTED filename: {params["imputed_file"]}') 
    with ProgressBar():
        result = da_climo_imputed.to_netcdf(params['imputed_file']) 
    if result is None: 
        print(f'Processes exited successfully. Hooray!')

def triple_interp(params):

    if os.path.isfile(params["imputed_file"]):
        if 'overwrite' in params.keys() and params['overwrite']:
            print(f'Overwriting target file {params["imputed_file"]}.')
            os.remove(params["imputed_file"])
        else:
            print(f'Target file {params["imputed_file"]} already exists. Aborting.')
            return

    # Account for multiple file datasets 
    if "*" in params['filename']:
        da = xr.open_mfdataset(params['filename'],chunks = params['chunks'])[params['variable']]
    else:
        da = xr.open_dataset(params['filename'],chunks = params['chunks'])[params['variable']]
    # first go at interpolation along lines of latitude 
    da_interp = da.interpolate_na(dim='longitude',method='linear',use_coordinate=False)
    # offset and interpolate again, this allows interpolation where nans persist to boundary 
    da_double_interp = da_interp.roll(longitude=int(len(da_interp.longitude)/2)).interpolate_na(dim='longitude',method='linear',use_coordinate=False).roll(longitude=len(da_interp.longitude)-int(len(da_interp.longitude)/2))
    # last round of interpolation fills in N-S boundary nans by offsetting and interpolating pole to pole. 
    da_triple_interp = da_double_interp.roll(latitude=int(len(da_interp.latitude)/2)).interpolate_na(dim='latitude',method='linear',use_coordinate=False).roll(latitude=len(da_interp.latitude)-int(len(da_interp.latitude)/2))

    print(f'Beginning computation of interpolation and writing...')
    print(f'INPUT filename: {params["filename"]}')
    print(f'IMPUTED filename: {params["imputed_file"]}') 
    with ProgressBar():
        result = da_triple_interp.to_netcdf(params['imputed_file']) 
    if result is None: 
        print(f'Processes exited successfully. Hooray!')
    

if __name__ == "__main__" :

    triple_interp(params)
    zonal_climo_impute(params)
