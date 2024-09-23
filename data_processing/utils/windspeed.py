import xarray as xr
import os
from dask.diagnostics import ProgressBar
import numpy as np

PARAMS = {
    'u_file':'/home/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_u10m.nc',
    'v_file':'/home/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_v10m.nc',
    'target_file':'/home/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_windspeed.nc',
    'chunks':{'time':10},
}

def main(params):

    if os.path.isfile(params["target_file"]):
        print(f'Target file {params["target_file"]} already exists. Aborting.')
        return
 
    print(f'Calculating windspeed from {params["u_file"]} and {params["v_file"]}...')
    u = xr.open_dataset(params['u_file'],chunks=params['chunks'])
    v = xr.open_dataset(params['v_file'],chunks=params['chunks']) 
    # change variable names and combine 
    ws = np.sqrt(u.rename({'u10':'ws10'})**2+v.rename({'v10':'ws10'})**2)
    
    print(f'Beginning computation of windspeed and writing...')
    print(f'TARGET filename: {params["target_file"]}') 
    with ProgressBar():
        result = ws.to_netcdf(params['target_file']) 
    if result is None: 
        print(f'Processes exited successfully. Hooray!')
    print(ws)

if __name__=="__main__":

    main(PARAMS)
