import numpy as np
import pandas as pd
from evaluation.seasonal_cycle import main as seasonal_cycle_main

PARAMS_Z500_1in1out_24AR = {
    # forecast file 
    'forecast_file': '/home/disk/mercury5/nacc/forecasts/hpx64_coupled-dlwp-olr_1-to-1_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_100-year_JanInit.nc',
    # params for evaluator initialization  
    'eval_variable' : 'z500',
    # params for seasonal cycle calculation 
    'levels' : np.arange(490,591,10),
    'scale_factor':98.1, # transform geopotential to deka meters
    'time': slice(pd.Timedelta(365*90,'D'),pd.Timedelta(365*100,'D')),
    'init_index' : 1, # corresponds to july initialization 
    # 'add_verif_ref':True,
    'rolling_params': {'dim':{'step':int(12)},
                       'center':True},
    'ref_line':560,
    'cmap':'Spectral_r',
    'colorbar_label':'Z$_{500}$ (dkm)',
    'title':'Simulation Seasonal Cycle',
    'savefig_params': {
        'fname' : './scratch/forecast_seasonal_cycle_z500_1in1out_24AR.png',
        'dpi' : 300,
    },
    'verif_title':'6-year ERA5 Seasonal Cycle',
}

if __name__ == "__main__":
    seasonal_cycle_main(PARAMS_Z500_1in1out_24AR)