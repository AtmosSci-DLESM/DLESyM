import xarray as xr
import numpy as np
import pandas as pd
import argparse
from omegaconf import OmegaConf
import time

from prep_precip_inputs import main as prep_precip_inputs
from run_diagnosis import main as run_diagnosis
from cmortize_dlesym_1978_precip import cmortize_precip

import logging

logger = logging.getLogger(__name__)

def main(params):

    # make namespace object
    params = argparse.Namespace(**params)

    start_time = time.time()
    logger.info(f'Starting precipitation pipeline for {params.tag} at {time.strftime("%Y-%m-%d %H:%M:%S")}')

    prep_precip_inputs(params.prep_precip_inputs_params)
    run_diagnosis(params.run_diagnosis_params)
    cmortize_precip(**params.cmortize_precip_params)
    logger.info(f'Precipitation pipeline for {params.tag} completed at {time.strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'Total time taken: {time.time() - start_time} seconds')

if __name__ == "__main__":

    PARAMS_p2k_r5 = {
        "tag": "p2k_r5",
        "prep_precip_inputs_params": {
            "atmos_input": "/home/disk/mercury3/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1978-2025_2k_n05.nc",
            "ocean_input": "/home/disk/mercury3/nacc/forecasts/aimip/ocean_aimip_forcing_1978-2025_2k_n05.nc",
            "dst_directory": "/home/disk/brass/nacc/forecasts/aimip/precip/p2k_r5",
            "dataset_name": "precip-init_aimip_forced_forecast_1978-2025_2k_n05",
            "constants": {
                "lsm": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_lsm.nc",
                "z": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_topography.nc"
            },
            "scaling": OmegaConf.load("/home/disk/mercury2/nacc/AIMIP2026/DLESyM/training/configs/data/scaling/hpx64_1983-2017.yaml"),
            "overwrite_nc_cache": False,
        },
        "run_diagnosis_params": {
            'model_path': '/home/disk/mercury2/nacc/AIMIP2026/DLESyM/models/precip',
            'destination_directory': '/home/disk/brass/nacc/forecasts/aimip/precip/p2k_r5/',
            # 'destination_directory': '/home/disk/mercury2/nacc/forecasts/aimip/p2k_r5/',
            'data_name': 'precip-init_aimip_forced_forecast_1978-2025_2k_n05',
            'gpu': 0,
            'output_directory': '/home/disk/brass/nacc/forecasts/aimip/precip/p2k_r5/',
            'tag': 'p2k_r5',
            'overwrite_forecast': False,
        }, 
        "cmortize_precip_params": {
            "forecast_file": "/home/disk/brass/nacc/forecasts/aimip/precip/p2k_r5/precip_aimip_forced_forecast_1978-2025_p2k_r5.nc",
            "output_dir": "/home/disk/mercury3/nacc/aimip_subission_1978",
            "r": 5,
            "experiment": "aimip-p2k",
        },
    }
    main(PARAMS_p2k_r5)