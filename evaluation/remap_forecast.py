import os 
import sys 
import logging

import numpy as np
import xarray as xr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(
    infile: str,
    outfile: str,
    var: str,
    ):
    """
    simple routine to remap forecast data from healpix to lat lon grid. Assumes HPX64 input, grid and 1-degree lat lon output
    param infile: str
        path to input healpix forecast file
    param outfile: str
        path to output lat lon remapped forecast file
    param var: str
        variable name to remap
    """

    # first let's make sure we can import the remap module
    try: 
        import data_processing.remap.healpix as hpx 
    except ImportError as e:
        logger.error("Failed to import reamp data_processing.remap.healpix, make sure you are running this utility from the project root directory (DLESyM/) and it's added to PYTHONPATH")
        raise ImportError(str(e))

    # initialize the remopper 
    mapper = hpx.HEALPixRemap(
            latitudes=181,
            longitudes=360,
            nside=64,
        )
    
    # open the forecast file
    fcst = xr.open_dataset(infile)

    # select variable to remap
    if var is not None:
        if var in fcst.data_vars:
            fcst = fcst[var]
        else:
            raise ValueError(f"Variable {var} not found in forecast file. Available variables are: {list(fcst.data_vars.keys())}")
    else:
        raise ValueError(f"No variable specified for remapping. Available variables are: {list(fcst.data_vars.keys())}")

    #perserve time and step dimensions
    time = fcst['time']
    step = fcst['step']

    # buffer for remaped data 
    try:
        remap_buffer = np.zeros((len(time), len(step), 181, 360), dtype=np.float32)
    except MemoryError as e:
        logger.error("Not enough memory to remap the forecast data, try running on a machine with more memory or splitting forecast into smaller chunks.")
        raise MemoryError(str(e))
    
    logger.info(f'Remapping forecast data from healpix to lat lon grid')
    # remap the forecast data to lat lon grid
    for i, t in tqdm(
        enumerate(time),
        desc="Time",
        unit="step",
        total=len(time), 
    ):
        for j, s in tqdm(
            enumerate(step),
            desc=f"Step (t={i})",
            unit="step",
            total=len(step),
            leave=False,
        ):
            remap_buffer[i, j] = mapper.hpx2ll(fcst.sel(time=t, step=s).values)
        
    # create new xarray DataArray with remapped data
    fcst_ll = xr.DataArray(
        remap_buffer,
        dims=['time', 'step', 'lat', 'lon'],
        coords={
            'time': time,
            'step': step,
            'lat': np.arange(90, -90.1, -1),
            'lon': np.arange(0, 360, 1)
        }
    )

    # save the remapped forecast data to a new file
    logger.info(f'Saving remapped forecast data to {outfile}')
    fcst_ll.to_netcdf(outfile)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Remap forecast data from healpix to lat-lon grid."
    )
    parser.add_argument(
        "--infile",
        type=str,
        required=True,
        help="Path to input healpix forecast file.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Path to output lat-lon remapped forecast file.",
    )
    parser.add_argument(
        "--var",
        type=str,
        required=True,
        help="Variable name to remap.",
    )

    args = parser.parse_args()

    main(
        infile=args.infile,
        outfile=args.outfile,
        var=args.var,
    )


