import sys 
import argparse
import sys
import os
import numpy as np
import xarray as xr
import healpy as hp
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt

from data_processing.remap.healpix import HEALPixRemap
from data_processing.remap.cubesphere import to_chunked_dataset

EXAMPLE_PARAMS = {
    'file_name' : '/home/quicksilver2/nacc/Data/pipeline_dev/era5_1950-2022_3h_1deg_sst-ti.nc',
    'target_variable_name' : 'sst', # this is used to extract the DataArray from the Dataset
    'file_variable_name' : 'sst', # this is how the variable will be saved in the new file 
    # this is the "prefix" for the output file. Should include the desired path to the output file
    'prefix' : '/home/quicksilver2/nacc/Data/pipeline_dev/era5_1deg_3h_HPX32_1950-2022_',
    'nside' : 32,
    'order' : 'bilinear', # order of the interpolation
    'resolution_factor' : 1.0,
    'visualize':False, # This determines whether to visualize remap. Warning: buggy
    'pool_size': 1,
}
EXAMPLE_PARAMS_CONSTANT = {
    'file_name' : '/home/quicksilver2/nacc/Data/pipeline_dev/era5_1950-2022_3h_1deg_lsm.nc',
    'target_variable_name' : 'lsm', 
    'file_variable_name' : 'lsm', 
    'prefix' : '/home/quicksilver2/nacc/Data/pipeline_dev/era5_1deg_3h_HPX32_1950-2022_',
    'nside' : 32,
    'order' : 'bilinear', 
    'resolution_factor' : 1.0,
    'visualize':False,
}

def main(params, output_file=None):  

    """
    This function remaps a latlon dataset to a HEALPix dataset.
    
    Parameters:
    params (dict): A dictionary with the following keys:
        - 'file_name': including absolute path of lat lon file to be remapped.
        - 'target_variable_name': variable name to be used in the output dataset.
        - 'file_variable_name': name of variable inside input file.
        - 'target_file_variable_name': name of variable used to create output file name, only if different from 'target_variable_name'.
        - 'prefix': path and file prefix for output file name (to be combined with file_variable_name).
        - 'nside': HEALPix nside parameter e.g. 32.
        - 'order': interpolation order e.g. 'bilinear'.
        - 'resolution_factor': resolution factor for remap (adjust with caution) e.g. 1.0.
        - 'visualize': boolean to visualize remap (warning: buggy).
        - 'pool_size': number of parallel processes to use (defaults to 1).
    output_file (str): Intended for use only in testing. Otherwise established conventions are used to determine the output file name.
    """
        
    # create namespace object for quick attribute referencing
    args = argparse.Namespace(**params)

    # get varaible identifier within otuput file name, this should only be different from target_variable_name for topography
    target_file_variable_name = getattr(args,'target_file_variable_name', args.target_variable_name)

    # save for later 
    nside = args.nside
    prefix = args.prefix

    # make sure target file doesn't already exits 
    print('checking parameters...')
    if not os.path.isfile(args.file_name):
        print(f'source file ~{args.file_name}~ not found. Aborting.')
        return
    if os.path.isfile(prefix+target_file_variable_name+'.nc'):
        print(f'target file ~{prefix+target_file_variable_name+".nc"}~ already exists. Aborting.')
        return

    assert f"HPX{nside}" in prefix, (f"'HPX{nside}' could not be found in the prefix '{prefix}'. Please make sure "
                                    f"that the nside={nside} parameter and the prefix provided to this function "
                                    f"match.")

    # Load .nc file in latlon format to extract latlon information and to initialize the remapper module
    ds_ll = xr.open_dataset(args.file_name)
    if 'time' in xr.open_dataset(args.file_name).dims:
        ds_ll.rename({"time": "sample"}).squeeze()
    try:
        latitudes, longitudes = ds_ll.dims["latitude"], ds_ll.dims["longitude"]
    except KeyError:
        latitudes, longitudes = ds_ll.dims["lat"], ds_ll.dims["lon"]
    mapper = HEALPixRemap(
        latitudes=latitudes,
        longitudes=longitudes,
        nside=nside,
        resolution_factor=args.resolution_factor,
        order=args.order,
        )
    
    # determine weather remap is to be done in parallel 
    mapper.remap(
        file_path=args.file_name,
        prefix=args.prefix,
        file_variable_name=args.file_variable_name,
        target_variable_name=args.target_variable_name,
        target_file_variable_name=target_file_variable_name,
        poolsize=getattr(args,'pool_size',1),
        chunk_ds=True,
        output_file=output_file,
    )

if __name__ == "__main__":
     
    main(EXAMPLE_PARAMS)
    main(EXAMPLE_PARAMS_CONSTANT)
