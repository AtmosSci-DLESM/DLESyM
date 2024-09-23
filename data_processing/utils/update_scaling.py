import numpy as np 
import xarray as xr
from omegaconf import OmegaConf, open_dict
import logging
from pprint import pprint
import json
import os
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)

def create_yaml_if_not_exists(file_name):
    """
    Creates a new YAML file if it doesn't already exist.

    Parameters:
    file_name (str): The path to the file to be created. This should include the full path along with the file name.

    Returns:
    None
    """
    # Check if the file already exists
    if not os.path.exists(file_name):
        # If the file doesn't exist, create it
        with open(file_name, "w") as file:
            # Write an empty dictionary to the file. This will be written as an empty YAML document.
            yaml.dump({}, file)
    return file_name

EXAMPLE_PARAMS = {
    'scale_file' : '/home/disk/quicksilver/nacc/dlesm/zephyr/training/configs/data/scaling/hpx32.yaml',
    'variable_file_prefix' : '/home/disk/rhodium/dlwp/data/HPX32/era5_1deg_3h_HPX32_1950-2022_',
    'variable_names' : ['z1000-24H'],
    'selection_dict' : {'sample':slice(np.datetime64('2022-12-21'),np.datetime64('2022-12-31'))},
    'overwrite' : False,
    'chunks' : None,
}

def main(params):
    """
    Update the scaling parameters for a list of variables in a dataset.

    Parameters:
    params (dict): A dictionary with the following keys:
        - 'scale_file': Path to the YAML file containing the scaling parameters.
        - 'variable_file_prefix': Prefix for the file names containing the variables. Can also pass a list of file names. instead
        - 'variables_files' : list of file names containing the variables,
        - 'variable_names': List of variable names to update.
        - 'selection_dict': Dictionary defining the data subset to use for the calculation.
        - 'overwrite': Whether to overwrite existing scaling parameters.
        - 'chunks': Dictionary defining the chunk sizes for the data loading.
    """

    # Load the current scaling parameters
    scale_dict = OmegaConf.to_container(OmegaConf.load(params['scale_file']))

    # Loop over the variable names
    for i, variable_name in enumerate(params['variable_names']):
        # Check if the variable is already in the scale file
        if variable_name in scale_dict.keys() and not params['overwrite']:
            logging.info(f"{variable_name} is already in {params['scale_file']}. If you wish to overwrite existing value, pass 'overwrite'=True")
            continue
        else: 
            logging.info(f"Adding {variable_name} to {params['scale_file']}")

        # Load the data for the variable
        if 'variable_file_prefix' in params:
            da = xr.open_dataset(params['variable_file_prefix'] + variable_name + '.nc',chunks=params['chunks'])[variable_name].sel(params['selection_dict'])
        else: 
            da = xr.open_dataset(params['variable_files'][i],chunks=params['chunks'])[variable_name].sel(params['selection_dict'])

        # Calculate the mean and standard deviation
        if params['chunks'] is None:
            da.load()
            mean = da.values.mean()
            std = da.values.std()
        else:
            raise NotImplementedError("chunked calculation of mean and std is not yet implemented. Xarray's built-in std function is bad!")

        # Update the scaling parameters for the variable
        scale_dict[variable_name] = { 'mean':mean.item(),'std':std.item() }

    # Save the updated scaling parameters
    print('updating scale dict to:')
    pprint(scale_dict)
    scale_conf = OmegaConf.create(scale_dict)  
    OmegaConf.save(config = scale_conf, f=params['scale_file'])

if __name__ == "__main__":
    # Run the main function with the example parameters
    main(EXAMPLE_PARAMS)
    
    
    


