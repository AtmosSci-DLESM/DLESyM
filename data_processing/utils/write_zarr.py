from typing import DefaultDict, Optional, Sequence, Union
from omegaconf import DictConfig, OmegaConf
import logging
import os
import shutil
import time
from dask.diagnostics import ProgressBar
from pprint import pprint
import numpy as np
import pandas as pd
import xarray as xr


def create_prebuilt_zarr(
        dst_directory: str,
        dataset_name: str,
        inputs: dict,
        outputs: dict,
        constants: Optional[DefaultDict] = None,
        batch_size: int = 32,
        time_subset: slice = None,
        scaling: Optional[DictConfig] = None,
        time_dim: Sequence = None,
        overwrite: bool = False,
        ) -> xr.Dataset:
    
    file_exists = os.path.exists(os.path.join(dst_directory, dataset_name + ".zarr"))

    if file_exists and not overwrite:
        print(f"Dataset {dataset_name} already exists in {dst_directory}. To overwrite, set params['overwrite'] to True. Aborting zarr creation.")
        return
    elif file_exists and overwrite:
        print(f"Dataset {dataset_name} already exists in {dst_directory}. Overwriting.")
        shutil.rmtree(os.path.join(dst_directory, dataset_name + ".zarr"))

    input_variables = list(inputs.keys())
    output_variables = list(outputs.keys()) or input_variables
    all_variables = np.union1d(input_variables , output_variables)
    merged_dict = {**inputs, **outputs}

    print('Creating a zarr dataset by merging the following netcdf files:')
    pprint(merged_dict)
    if time_subset is not None:
        print(f"Time subset: {time_subset}")

    def get_file_name(path, var):
        return os.path.join(path, f"{prefix}{var}{suffix}.nc")

    merge_time = time.time()

    datasets = []
    remove_attrs = ['varlev', 'mean', 'std']
    for variable in all_variables:
        file_name = merged_dict[variable]
        if "sample" in list(xr.open_dataset(file_name).dims.keys()):
            ds = xr.open_dataset(file_name, chunks={'sample': batch_size}).rename({"sample": "time"})
        else:
            ds = xr.open_dataset(file_name, chunks={"time": batch_size})
        if "varlev" in ds.dims:
            ds = ds.isel(varlev=0)
        # Subset time
        if time_subset is not None:
            ds = ds.sel(time=time_subset)
        for attr in remove_attrs:
            try:
                ds = ds.drop(attr)
            except ValueError:
                pass
        # Rename variable
        if "predictors" in list(ds.keys()):
            ds = ds.rename({"predictors": variable})

        # Change lat/lon to coordinates
        try:
            ds = ds.set_coords(['lat', 'lon'])
        except (ValueError, KeyError):
            pass

        # Apply log scaling lazily
        if variable in scaling and scaling[variable].get('log_epsilon', None) is not None:
            ds[variable] = np.log(ds[variable] + scaling[variable]['log_epsilon']) \
                           - np.log(scaling[variable]['log_epsilon'])
            
        # check time dimension is as desired and if not, resample 
        if time_dim is not None:
            if not np.array_equal(ds.time.values.astype('datetime64[h]'),time_dim.astype('datetime64[h]')):
                print(f'Time dimension of {merged_dict[variable]} is {ds.time.values.astype("datetime64[h]")}')
                print(f' This is different from the specified time_dim. Resampling using forward fill to {time_dim.astype("datetime64[h]")}')
                ds = ds.reindex(time=time_dim, method='ffill')
        datasets.append(ds)

    # Merge datasets
    data = xr.merge(datasets, compat="override")

    # Convert to input/target array by merging along the variables
    input_da = data[list(input_variables)].to_array('channel_in', name='inputs').transpose(
        'time', 'channel_in', 'face', 'height', 'width')
    target_da = data[list(output_variables)].to_array('channel_out', name='targets').transpose(
        'time', 'channel_out', 'face', 'height', 'width')

    result = xr.Dataset()
    result['inputs'] = input_da
    result['targets'] = target_da

    # Get constants
    if constants is not None:
        constants_ds = []
        for name, filename in constants.items():
            constants_ds.append(xr.open_dataset(
                filename
                ).set_coords(['lat', 'lon'])[name].astype(np.float32))
        constants_ds = xr.merge(constants_ds, compat='override')
        constants_da = constants_ds.to_array('channel_c', name='constants').transpose(
            'channel_c', 'face', 'height', 'width')
        result['constants'] = constants_da

    # writing out
    def write_zarr(data, path):
        #write_job = data.to_netcdf(path, compute=False)
        write_job = data.to_zarr(path, encoding={'time':{'dtype':'float64'}}, compute=False) # we have to enforce float64 for time to avoid percision issues with zarr writing
        with ProgressBar():
            print(f"writing zarr dataset to {path}")
            write_job.compute()
        print('Successfully wrote zarr:')
        print(data)

    # enforce chunking before writing
    result = result.chunk({'time': batch_size})

    write_zarr(data=result, path=os.path.join(dst_directory, dataset_name + ".zarr"))
    
    return True