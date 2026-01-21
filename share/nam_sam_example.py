import numpy as np
import pandas as pd
from evaluation.NAM_SAM_hpx import main as nam_sam_main

params_nam = dict(
        # directory of input forecast
        input_dir = '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/',
        # Northern or Southern Annular mode
        mode = 'NAM',
        # We save the computed nam or sam pattern to an itermediate cache before plotting.
        output_cache = '/home/disk/brume/nacc/gradient_dissent/DLESyM/scratch/nam_sam/2in2out_cache_nam.nc',
        # this is where output plots will be saved. 
        output_plot_file = '/home/disk/brume/nacc/gradient_dissent/DLESyM/scratch/nam_sam/2in2out_map_nam.png'
    )
params_sam = dict(
        input_dir = '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/',
        mode = 'SAM',
        output_cache = '/home/disk/brume/nacc/gradient_dissent/DLESyM/scratch/nam_sam/2in2out_cache_sam.nc',
        output_plot_file = '/home/disk/brume/nacc/gradient_dissent/DLESyM/scratch/nam_sam/2in2out_map_sam.png'
    )

if __name__ == "__main__":
    nam_sam_main(**params_nam)
    nam_sam_main(**params_sam)