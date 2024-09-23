import xarray as xr
import os
from dask.diagnostics import ProgressBar
import numpy as np 
import pandas as pd 
from tqdm import tqdm 

EXAMPLE_PARAM = {
    'filename' : 'data_processing/utils/test_input_trailing_average.nc', # path to intantaneous file 
    'variable_name' : 'ws10', # name of variable in dataset
    'output_variable_name' : 'ws10-48H', # name of variable in output dataset
    'coupled_dt' : '6H', # the time resoluton of the atmos model to be coupled 
    'output_filename' : 'test_trailingAverage.nc', # output file name including path
    'influence_window' : np.timedelta64(2, 'D'), # range for averaging 
    'chunks' : {'sample':10}, # for parallel processing
    'load_first' : False, # if data can fit in memory, this will speed up calculations 
}

def main(params):

    if os.path.isfile(params["output_filename"]):
        print(f'Trailing Average: Target file {params["output_filename"]} already exists. Aborting.')
        return

    # open data array 
    if params['load_first']:
        print('attempting to load ds into memory...')
        da = xr.open_dataset(params['filename'],chunks=params['chunks'])[params['variable_name']].load()
        print('...done!')
    else:
        da = xr.open_dataset(params['filename'],chunks=params['chunks'])[params['variable_name']]
    # initialize resulting array 
    print(f'initializing result array...')
    result = xr.zeros_like(da)
    print('done..')
    # snip array where the last forward influence is possible to calculate 
    first_valid_sample = da.sample[0].values + params['influence_window']
    result = result.sel(sample=slice(first_valid_sample, result.sample.values[-1]))

    print(f'Generating ATMOS influence array from {params["filename"]}') 
    for s in tqdm(result.sample.values):
        
        coupled_influence = da.sel(sample=pd.date_range(s-params["influence_window"], s, freq=params["coupled_dt"])).mean(dim='sample')
        result.loc[{'sample':s}] = coupled_influence 
    # rename variable after trailing average is taken 
    result = result.rename(params['output_variable_name'])
    
    print(f'Writing ATMOS influence array to {params["output_filename"]}...') 
    with ProgressBar():
        write_return = result.to_netcdf(params['output_filename']) 
    if write_return is None: 
        print(f'Processes exited successfully. Hooray!')
    print(result)
    print('DONE!')
    # clean up 
    result.close()
    da.close()

if __name__=="__main__":
    
    main(EXAMPLE_PARAM)
