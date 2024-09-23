import xarray as xr
import os
import numpy as np

# Define the source and target file paths
EXAMPLE_PARAMS = {
    'src_file':'/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_unscaled_topography.nc',  # The path to the source file
    'target_file':'/home/disk/rhodium/dlwp/data/era5/0.25deg/era5_1950-2022_3h_0.25deg_topography.nc',  # The path to the target file
}

def main(params):
    """
    This function scales the topography data from the source file and writes it to the target file.
    The scaling is done by subtracting the mean and dividing by the standard deviation.

    Parameters:
    params (dict): A dictionary containing the source and target file paths.
    """

    # Check if the target file already exists
    if os.path.isfile(params["target_file"]):
        print(f'Target file {params["target_file"]} already exists. Aborting.')
        return
    
    # Load the unscaled topography data from the source file
    unscaled_topo = xr.open_dataset(params['src_file'])['z'].load()

    # Calculate the mean and standard deviation of the unscaled topography data
    mean = unscaled_topo.values.mean()
    std = unscaled_topo.values.std()

    print('Scaling topography...')
    # Scale the topography data
    scaled_topo = (unscaled_topo - mean) / std

    print(f'Writing to {params["target_file"]}...')
    # Write the scaled topography data to the target file
    scaled_topo.to_netcdf(params['target_file'])

    return

# module is runnable as a script
if __name__=="__main__":

    main(EXAMPLE_PARAMS)
