import numpy as np
import pandas as pd
from DLESyM.evaluation.NAM_SAM_hpx import main as nam_sam_main

params_nam = dict(
        # directory of input forecast
        input_dir = '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/',
        # Northern or Southern Annular mode
        mode = 'NAM',
        # where to save output plots
        output_dir = './scratch'
    )
params_sam = dict(
        input_dir = '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/',
        mode = 'SAM',
        output_dir = './scratch'
    )

if __name__ == "__main__":
    nam_sam_main(params_nam)
    nam_sam_main(params_sam)