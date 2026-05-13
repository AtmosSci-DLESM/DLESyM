from scripts.diagnose import inference
import xarray as xr
import pandas as pd
import os 
import argparse
from training.dlwp.utils import configure_logging
import logging

logger = logging.getLogger(__name__)

logging.getLogger('cfgrib').setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.INFO) # IN FOREGROUND
logging.getLogger('xarray').setLevel(logging.INFO)

PARAMS = {
    'model_path': '/home/disk/mercury2/nacc/AIMIP2026/DLESyM/models/precip',
    # 'destination_directory': '/home/disk/brass/nacc/forecasts/aimip/p2k_r5/',
    'destination_directory': '/home/disk/mercury2/nacc/forecasts/aimip/p2k_r5/',
    'data_name': 'precip-init_aimip_forced_forecast_1978-2025_2k_n05',
    'gpu': 0,
    'output_directory': '/home/disk/brass/nacc/forecasts/aimip/p2k_r5/',
    'tag': 'p2k_r5',
    'overwrite_forecast': True,
}

def main(params):


    # configure logging
    configure_logging(2)

    # create namespace object 
    params = argparse.Namespace(**params)
    # create args namespace object
    args = argparse.Namespace()

    # get init time from dataset 
    ds = xr.open_zarr(os.path.join(params.destination_directory, params.data_name + '.zarr'))
    init_time = ds.time.values[1]
    args.forecast_init_start = init_time
    args.forecast_init_end = ds.time.values[-1]
    # args.forecast_init_end = ds.time.values[2]
    # construct output_filename
    args.output_filename = f"precip_aimip_forced_forecast_1978-2025_{params.tag}"
    args.output_directory = params.output_directory
    # check output file already exists
    if os.path.exists(os.path.join(args.output_directory, args.output_filename + '.nc')):
        if not params.overwrite_forecast:
            logger.info(f"Output file {args.output_filename + '.nc'} already exists. To overwrite, set params['overwrite_forecast'] to True. Aborting diagnosis.")
            return
        else:
            logger.info(f"Output file {args.output_filename + '.nc'} already exists. Overwriting.")
            os.remove(os.path.join(args.output_directory, args.output_filename + '.nc'))

    # params necessary for diagnosis but not independent for this ablation
    args.batch_size = None  
    args.destination_directory = params.destination_directory
    args.data_name = params.data_name
    args.data_directory = None
    args.data_prefix = None
    args.data_suffix = None
    args.model_checkpoint = None
    args.prebuilt = True
    args.encode_int = False
    args.to_zarr = False
    args.lead_time = "0h"
    args.freq = '6H'
    args.gpu = params.gpu

    # here we need to set the hydra path to the model path
    # by getting relative path from current working directory
    args.model_path = params.model_path
    args.hydra_path = os.path.relpath(params.model_path, os.path.join(os.getcwd(), 'scripts'))


    inference(args=args)
    

if __name__ == "__main__":
    main(PARAMS)