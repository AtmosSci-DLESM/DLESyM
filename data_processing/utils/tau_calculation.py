import xarray as xr
import os
from dask.diagnostics import ProgressBar
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

PARAMS = {
    "upper_file" : "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_z300.nc",
    "lower_file" : "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_z700.nc",
    "target_file" : "/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_tau300-700.nc",
    "chunks": {"time" : 128},
}

def main(params):
    """
    Calculate the difference between two datasets and save the result to a file.
    
    Parameters:
    params (dict): A dictionary with the following keys:
        - 'upper_file': Path to the upper dataset file.
        - 'lower_file': Path to the lower dataset file.
        - 'target_file': Path to the file where the result will be saved.
        - 'chunks': Chunk size for processing the data.
    """
    logging.info("Calculating difference between %s and %s...", params['upper_file'], params['lower_file'])
    
    if os.path.isfile(params["target_file"]):
        logging.error('Target file %s already exists. Aborting.', params["target_file"])
        return
    
    try:
        upper = xr.open_dataset(params['upper_file'], chunks=params['chunks']).assign_coords(level=500)
        lower = xr.open_dataset(params['lower_file'], chunks=params['chunks']).assign_coords(level=500)
        difference = upper - lower
    except Exception as e:
        logging.error('Failed to calculate difference: %s', e)
        return

    logging.info('Beginning computation and writing to %s...', params["target_file"])
    
    with ProgressBar():
        result = difference.to_netcdf(params['target_file'])
    
    if result is None:
        logging.info('Processes exited successfully. Hooray!')
    
    logging.info(difference)
    return

if __name__=="__main__":
    main(PARAMS)
