import argparse
import glob
import logging
import os
from pathlib import Path
import time

from dask.distributed import LocalCluster, Client
import xarray as xr
import zarr

from dlwp.utils import configure_logging, encode_variables_as_int, remove_chars, to_chunked_dataset

logger = logging.getLogger(__name__)


class ParameterConfigError(Exception):
    pass


def convert_file(args, input_file, chunking):
    total_time = time.time()

    # Get output file name and check if it exists
    output_file = Path(input_file).name.replace('.nc', '.zarr')
    if args.output_prefix is not None:
        if args.prefix != '':
            output_file = output_file.replace(args.prefix, args.output_prefix)
        else:
            output_file = f"{args.output_prefix}{output_file}"
    output_file = os.path.join(args.output_directory, output_file)
    if os.path.exists(output_file) and not args.overwrite:
        logger.warning("output zarr group %s already exists", output_file)
        return

    # Open original netCDF file
    open_time = time.time()
    logger.info("open netCDF file %s", input_file)
    # Open the data file once first to determine whether this file has the correct time dimension. Constants files do
    # not have the time dimension.
    ds = xr.open_dataset(input_file)
    if 'sample' not in ds.dims:
        try:
            chunking.pop('sample')
        except KeyError:
            pass
    ds.close()
    ds = None
    ds = xr.open_dataset(input_file, chunks=chunking, cache=False)
    for key in chunking.keys():
        if key not in ds.dims:
            raise ParameterConfigError(f"chunking key '{key}' not found in data dimensions {tuple(ds.dims.keys())}")
    ds = to_chunked_dataset(ds, chunking)
    logger.debug("opened file in %0.1f s", time.time() - open_time)

    # Conversions applied to the file
    #   1. re-format to new schema
    #      - set lat/lon as coords not vars
    #      - remove varlev dimension
    #      - rename predictors: varlev, sample: time
    #      - remove 'mean' and 'std'
    #   2. compute int16 encoding from min/max
    #   3. set the compressor in the encoding

    # Change lat/lon to coordinates
    try:
        ds = ds.set_coords(['lat', 'lon'])
    except (ValueError, KeyError):
        pass
    if 'predictors' in ds.data_vars:
        variable = remove_chars(ds.varlev.values[0])
        ds = ds.isel(varlev=0).drop('varlev')
        ds = ds.rename({'predictors': variable, 'sample': 'time'})
    for attr in ['mean', 'std']:
        try:
            ds = ds.drop(attr)
        except ValueError:
            pass

    # Compute int16 and compression
    if args.int16:
        encode_time = time.time()
        logger.info("computing variable int encoding")
        ds = encode_variables_as_int(ds, 'int16')
        logger.debug("computed variable int encoding in %0.1f s", time.time() - encode_time)
    compressor = zarr.Blosc(cname=args.comp_method, clevel=args.comp_level)
    for var in ds.data_vars.keys():
        ds[var].encoding['compressor'] = compressor

    # Output new zarr group
    write_time = time.time()
    logger.info("writing zarr file %s", output_file)
    ds.to_zarr(output_file, mode='w', compute=True)
    logger.debug("wrote file in %0.1f s", time.time() - write_time)

    logger.info("total time for converting data file %s: %0.1f s", output_file, time.time() - total_time)


def main(args):
    os.makedirs(args.output_directory, exist_ok=True)

    # Create chunking dict
    chunking = {}
    for dim in args.chunks:
        try:
            key, value = dim.split('=')
            value = int(value)
            chunking[key] = value
        except (IndexError, ValueError, TypeError):
            logger.error("invalid chunking dimension specified (expects dim=[int], got '%s'", dim)
            raise
    logger.debug("chunking: %s", chunking)

    # Find input files
    input_files = glob.glob(os.path.join(args.input_directory, f"{args.prefix}*.nc"))

    # Create dask cluster and client
    cluster = LocalCluster(dashboard_address=f':{args.client_port}' if args.client_port is not None else None)
    client = Client(cluster)  # pylint: disable=unused-variable

    # Iterate over files
    for file in input_files:
        convert_file(args, file, dict(chunking))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-directory', required=True, type=str,
                        help="Input directory for netCDF files")
    parser.add_argument('-p', '--prefix', required=True, type=str,
                        help="Prefix for input files. Searches for files of name "
                             "'{input-directory}/{prefix}{variable}.nc'.")
    parser.add_argument('-o', '--output-directory', required=True, type=str,
                        help="Output directory for zarr files")
    parser.add_argument('-c', '--chunks', nargs='+', type=str, required=True,
                        help="Chunking for the data dimensions. Specify a list of dim=[int] arguments. Any data "
                             "dimensions not specified here will result in a single chunk.")
    parser.add_argument('--comp-method', choices=['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd'], default='zstd',
                        help="Blosc compression method")
    parser.add_argument('--comp-level', type=int, default=1,
                        help="Compression level (1-9)")
    parser.add_argument('--int16', action='store_true',
                        help="Updates data variable encodings to store data in int16 format. Requires substantial "
                             "compute at write time but saves > 2x disk I/O later.")
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite existing output zarr stores")
    parser.add_argument('--output-prefix', type=str, default=None,
                        help="Optional new prefix for output files")
    parser.add_argument('--client-port', type=int, default=8880,
                        help="Port to connect to the dask distributed client for visualization of progress")

    run_args = parser.parse_args()
    configure_logging(2)
    main(run_args)
