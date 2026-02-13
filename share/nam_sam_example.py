import numpy as np
import pandas as pd
from evaluation.NAM_SAM_hpx import main as nam_sam_main

params_nam_24hr = dict(
        # directory of input forecast
        input_file = '/home/disk/mercury5/nacc/forecasts/hpx64_coupled-dlwp-olr_1-to-1_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_100-year_JanInit_z1000_ll.nc',
        # Northern or Southern Annular mode
        mode = 'NAM',
        # We save the computed nam or sam pattern to an itermediate cache before plotting.
        output_cache = '/home/disk/brume/shaanj/gradient_dissent/DLESyM/scratch/nam_sam/1in1out24hr_cache_nam.nc',
        # this is where output plots will be saved. 
        output_plot_file = '/home/disk/brume/shaanj/gradient_dissent/DLESyM/scratch/nam_sam/1in1out24hr_map_nam.png'

    )
params_sam_24hr = dict(
        input_file = '/home/disk/mercury5/nacc/forecasts/hpx64_coupled-dlwp-olr_1-to-1_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_100-year_JanInit_z500_ll.nc',
        mode = 'SAM',
        output_cache = '/home/disk/brume/shaanj/gradient_dissent/DLESyM/scratch/nam_sam/1in1out24hr_cache_sam.nc',
        output_plot_file = '/home/disk/brume/shaanj/gradient_dissent/DLESyM/scratch/nam_sam/1in1out24hr_map_sam.png'
    )


params_nam_12hr = dict(
        # directory of input forecast
        input_file = '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_1-to-1_seed0_12h-AR+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_100-year_JanInit_z1000_ll.nc ',
        # Northern or Southern Annular mode
        mode = 'NAM',
        # We save the computed nam or sam pattern to an itermediate cache before plotting.
        output_cache = '/home/disk/brume/shaanj/gradient_dissent/DLESyM/scratch/nam_sam/1in1out12hr_cache_nam.nc',
        # this is where output plots will be saved. 
        output_plot_file = '/home/disk/brume/shaanj/gradient_dissent/DLESyM/scratch/nam_sam/1in1out12hr_map_nam.png'

    )
params_sam_12hr = dict(
        input_file = '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_1-to-1_seed0_12h-AR+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_100-year_JanInit_z500_ll.nc ',
        mode = 'SAM',
        output_cache = '/home/disk/brume/shaanj/gradient_dissent/DLESyM/scratch/nam_sam/1in1out12hr_cache_sam.nc',
        output_plot_file = '/home/disk/brume/shaanj/gradient_dissent/DLESyM/scratch/nam_sam/1in1out12hr_map_sam.png'
    )




if __name__ == "__main__":
    nam_sam_main(**params_nam_24hr)
    nam_sam_main(**params_sam_24hr)
    nam_sam_main(**params_nam_12hr)
    nam_sam_main(**params_sam_12hr)


