import os

import xarray as xr
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from data_processing.utils import write_zarr

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

params_2k_r5 = {
    "atmos_input": "/home/disk/mercury3/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1978-2025_2k_n05.nc",
    "ocean_input": "/home/disk/mercury3/nacc/forecasts/aimip/ocean_aimip_forcing_1978-2025_2k_n05.nc",
    "dst_directory": "/home/disk/brass/nacc/forecasts/aimip/p2k_r5",
    "dataset_name": "precip-init_aimip_forced_forecast_1978-2025_2k_n05",
    "constants": {
        "lsm": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_lsm.nc",
        "z": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_topography.nc"
    },
    "scaling": OmegaConf.load("/home/disk/mercury2/nacc/AIMIP2026/DLESyM/training/configs/data/scaling/hpx64_1983-2017.yaml"),
    "overwrite_nc_cache": False,
}

def split_nc_file(input_file, output_prefix, overwrite_nc_cache=False):
    ds = xr.open_dataset(input_file)

    # Define different varlev coordinates (one entry per non-dimension data variable, in order)
    varlev_values = ['500.0', '500.0', '1000.0', 't2m0/0', 'tcwv0/0', '850.0', '250.0', 'ws10/0', 'olr/0']

    data_var_names = [v for v in ds.variables if v not in ds.dims]
    if (
        not overwrite_nc_cache
        and data_var_names
        and all(os.path.isfile(f"{output_prefix}/{vn}.nc") for vn in data_var_names)
    ):
        print("All split NetCDF caches present; skipping split_nc_file.")
        ds.close()
        return

    data_idx = 0
    for var_name in ds.variables:
        if var_name in ds.dims:
            continue

        output_file = f"{output_prefix}/{var_name}.nc"
        if not overwrite_nc_cache and os.path.isfile(output_file):
            print(f"Using cached NetCDF for {var_name}: {output_file}")
            data_idx += 1
            continue

        var_ds = ds[var_name]
        dates = var_ds.time.values + var_ds.step.values

        ds_flat = var_ds.stack(times=('time', 'step'))
        ds_flat = ds_flat.assign_coords(times=dates)

        var_final = ds_flat.rename({'times': 'sample'})
        vlev = varlev_values[data_idx]
        var_final = var_final.assign_coords(varlev=vlev)
        var_final['varlev'] = var_final['varlev'].astype('object')
        var_final = var_final.expand_dims(varlev=[vlev])
        var_final = var_final.rename("predictors")
        var_final = var_final.transpose('sample', 'varlev', 'face', 'height', 'width')
        print(f"Formatted complete for {var_name}. Saving to netcdf...")

        var_final.to_netcdf(output_file)
        print(f"Variable '{var_name}' saved to '{output_file}'.")
        data_idx += 1

    ds.close()

def format_sst_file(input_file, output_prefix, overwrite_nc_cache=False):
    output_file = f"{output_prefix}/sst.nc"
    if not overwrite_nc_cache and os.path.isfile(output_file):
        print(f"Using cached SST NetCDF: {output_file}")
        return

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

    output_file = f'{output_prefix}/sst.nc'
    ds_interp.to_netcdf(output_file)
    print(f"Variable sst saved to '{output_file}'.")
    sst.close()

def main(params):

    # create dst_directory recursively if it doesn't exist
    os.makedirs(params["dst_directory"], exist_ok=True)

    overwrite_nc_cache = params.get("overwrite_nc_cache", False)
    split_nc_file(
        params["atmos_input"],
        params["dst_directory"],
        overwrite_nc_cache=overwrite_nc_cache,
    )
    format_sst_file(
        params["ocean_input"],
        params["dst_directory"],
        overwrite_nc_cache=overwrite_nc_cache,
    )

    inputs = {var: f"{params['dst_directory']}/{var}.nc" for var in input_vars}
    outputs = {
        "tp6": f"{params['dst_directory']}/tcwv0.nc"  # dummy tp6 data
    }
    
    write_zarr.create_prebuilt_zarr(
    params["dst_directory"],
    params["dataset_name"],
    inputs,
    outputs,
    constants=params["constants"],
    scaling=params["scaling"]
)
    
if __name__ == "__main__": 
    main(params_2k_r5)