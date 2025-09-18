import xarray as xr
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from data_processing.utils import write_zarr

split_fc_prefix = "path/hpx64_forecasts_"

input_vars = [
    "z500",
    "tau300-700",
    "z1000",
    "t2m0",
    "tcwv0",
    "t850",
    "z250",
    "ws10",
    "olr",
    "sst"
]

inputs = {var: f"{split_fc_prefix}{var}.nc" for var in input_vars}
outputs = {
    "tp6": f"{split_fc_prefix}tcwv0.nc"  # dummy tp6 data
}

params = {
    "atmos_input": "path/to/atmos/forecast.nc",
    "ocean_input": "path/to/ocean/forecast.nc",
    "split_fc_prefix": split_fc_prefix,
    "dst_directory": "new/path/",
    "dataset_name": "forecast_for_precip_diagnosis",
    "inputs": inputs,
    "outputs": outputs,
    "constants": {
        "lsm": "path/to/lsm.nc",
        "z": "path/to/topography.nc"
    },
    "scaling": OmegaConf.load("path/to/scaling.yaml"),
}

def split_nc_file(input_file, output_prefix):
    ds = xr.open_dataset(input_file)
    
    # Define different varlev coordinates
    varlev_values = ['500.0', '500.0', '1000.0', 't2m0/0', 'tcwv0/0', '850.0', '250.0', 'ws10/0', 'olr/0']
    
    for idx, var_name in enumerate(ds.variables):
        if var_name not in ds.dims:
            var_ds = ds[var_name]
            dates = var_ds.time.values + var_ds.step.values
            
            ds_flat = var_ds.stack(times=('time', 'step'))
            ds_flat = ds_flat.assign_coords(times=dates)
            
            var_final = ds_flat.rename({'times':'sample'})
            var_final = var_final.assign_coords(varlev=varlev_values[idx])
            var_final['varlev'] = var_final['varlev'].astype('object')
            var_final = var_final.expand_dims(varlev=[varlev_values[idx]])
            var_final = var_final.rename("predictors")
            var_final = var_final.transpose('sample','varlev','face','height','width')
            print(f"Formatted complete for {var_name}. Saving to netcdf...")
            
            output_file = f'{output_prefix}{var_name}.nc'
            var_final.to_netcdf(output_file)
            print(f"Variable '{var_name}' saved to '{output_file}'.")
            
    ds.close()
    
def format_sst_file(input_file, output_prefix):

    print("Starting sst file prep.")
    sst = xr.open_dataset(input_file)['sst']

    selected_forecast = sst.isel(time=0)
    merged_time = selected_forecast['time'] + selected_forecast['step']

    selected_forecast = selected_forecast.assign_coords(time=merged_time)
    selected_forecast = selected_forecast.swap_dims({'step': 'time'})
    selected_forecast = selected_forecast.drop('step')

    var_final = selected_forecast.rename({'time':'sample'})
    var_final = var_final.assign_coords(varlev='sst/0')
    var_final['varlev'] = var_final['varlev'].astype('object')
    var_final = var_final.expand_dims(varlev=['sst/0'])
    var_final = var_final.rename('predictors')

    print("Interpolating...")
    time_start = var_final.sample.min().values
    time_end = var_final.sample.max().values
    new_time = pd.date_range(start=time_start, end=time_end, freq='6H')
    new_time_da = xr.DataArray(new_time, dims='sample')
    ds_interp = var_final.interp(sample=new_time_da)
    print("Formatted complete for sst. Saving to netcdf...")

    output_file = f'{output_prefix}sst.nc'
    ds_interp.to_netcdf(output_file)
    print(f"Variable sst saved to '{output_file}'.")
    sst.close()
    
def main(params):
    split_nc_file(params['atmos_input'], params['split_fc_prefix'])
    format_stt_file(params['ocean_input'], params['split_fc_prefix'])
    
    write_zarr.create_prebuilt_zarr(
    params["dst_directory"],
    params["dataset_name"],
    params["inputs"],
    params["outputs"],
    constants=params["constants"],
    scaling=params["scaling"]
)
    
if __name__ == "__main__": 
    main(params)