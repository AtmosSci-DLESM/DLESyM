import xarray as xr
import os
from dask.diagnostics import ProgressBar
import numpy as np 

EXAMPLE_PARAM = {
    'filename' : 'data_processing/utils/test_input_trailing_average.nc', # path to intantaneous file 
    'variable_name' : 'ws10', # name of variable in dataset
    'output_variable_name' : 'ws10-48H', # name of variable in output dataset
    'ds_dt' : np.timedelta64(3, 'h'), # the time resoluton of the atmos dataset
    'output_filename' : 'test_trailingAverage.nc', # output file name including path
    'influence_window' : np.timedelta64(2, 'D'), # range for averaging 
    'chunks' : {'sample':10}, # for parallel processing
    'load_first' : False, # if data can fit in memory, this will speed up calculations 
}

def main(params):
    """
    This function calculates the trailing average of a data set in parallel, assigning at each time the average of the previous
    n hours, where n is the influence window. The function uses the rolling mean method from xarray to calculate the trailing average.
    Parameters:
    params (dict): A dictionary containing the following parameters
        filename (str): The path to the input file
        variable_name (str): The name of the variable in the input dataset
        output_variable_name (str): The name of the variable in the output dataset
        ds_dt (np.timedelta64): The time resolution of the input dataset
        output_filename (str): The path to the output file
        influence_window (np.timedelta64): The range for averaging
        chunks (dict): The chunk size for parallel processing
        load_first (bool): If True, the input data will be loaded into memory
    """
    if os.path.isfile(params["output_filename"]):
        print(f'Trailing Average: Target file {params["output_filename"]} already exists. Aborting.')
        return

    # Open dataset lazily with Dask
    ds = xr.open_dataset(params['filename'], chunks=params['chunks'])
    da = ds[params['variable_name']]

    # Apply rolling mean instead of manual looping
    window_size = int(params["influence_window"] / params['ds_dt'])  # Convert timedelta to integer
    result = da.rolling(sample=window_size, center=False).mean()

    # enforce chunks 
    result = result.chunk(params['chunks'])

    # Rename variable
    result = result.rename(params['output_variable_name'])

    print(f'Writing ATMOS influence array to {params["output_filename"]}...')
    with ProgressBar():
        result.to_netcdf(params['output_filename'], compute=True)  # Compute only once at write stage

    print('DONE!')
    ds.close()

if __name__=="__main__":
    
    main(EXAMPLE_PARAM)
