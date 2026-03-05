import numpy as np
import pandas as pd
from evaluation.blocking import get_custom_cmap


PARAMS_Z500_1in1out_24h_seasonal_cycle = {
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
        'fname' : '/home/disk/brume/nacc/gradient_dissent/DLESyM/scratch/24h_1in1out_full_eval/forecast_seasonal_cycle_z500_1in1out_24h.png',
        'dpi' : 300,
    },
    'verif_title':'6-year ERA5 Seasonal Cycle',
}
PARAMS_Z500_1in1out_24h_NH_40yr_blocking = {
    'forecast_file' : '/home/disk/mercury5/nacc/forecasts/hpx64_coupled-dlwp-olr_1-to-1_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_100-year_JanInit',
    'forecast_time_range': slice('2070-01-01', '2110-12-31'),
    'verification_file' : '/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_z500.nc',
    'verification_time_range': slice('1970-01-01', '2010-12-31'),
    'months':[1,2,3,4,5,6,7,8,9,10,11,12],
    'hemisphere': 'north',
    # 'freq_levels': np.arange(1, 15.1, 1),
    'freq_levels': np.arange(0.0, 0.101, 0.01),
    # 'freq_ticks': np.arange(1, 15.1, 2),
    'freq_ticks': np.arange(0.0, 0.101, 0.01),
    'freq_cmap': get_custom_cmap('plasma_r'),
    'std_cmap': get_custom_cmap('Greens'),
    'std_levels': np.arange(0, 0.101, .02),
    'std_ticks': np.arange(0, 0.101, .02),
    'output_dir' : '/home/disk/brume/nacc/gradient_dissent/DLESyM/scratch/24h_1in1out_full_eval/',
    'plot_file_suffix': 'nh_40yr',
    'map_suffix': 'nh_map_40yr',
}
params_nam_24hr = dict(
        input_file = '/home/disk/mercury5/nacc/forecasts/hpx64_coupled-dlwp-olr_1-to-1_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_100-year_JanInit_z1000_ll.nc',
        mode = 'NAM',
        output_cache = '/home/disk/brume/nacc/gradient_dissent/DLESyM/scratch/24h_1in1out_full_eval/1in1out24hr_cache_nam.nc',
        output_plot_file =  '/home/disk/brume/nacc/gradient_dissent/DLESyM/scratch/24h_1in1out_full_eval/1in1out24hr_map_nam.png'

    )
params_sam_24hr = dict(
        input_file = '/home/disk/mercury5/nacc/forecasts/hpx64_coupled-dlwp-olr_1-to-1_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_100-year_JanInit_z500_ll.nc',
        mode = 'SAM',
        output_cache = '/home/disk/brume/nacc/gradient_dissent/DLESyM/scratch/24h_1in1out_full_eval/1in1out24hr_cache_sam.nc',
        output_plot_file = '/home/disk/brume/nacc/gradient_dissent/DLESyM/scratch/24h_1in1out_full_eval/1in1out24hr_map_sam.png'
    )
tc_params_24h = dict(
        # directory of input file
        input_dir = '/home/disk/mercury5/nacc/forecasts/hpx64_coupled-dlwp-olr_1-to-1_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/',
        # name of z1000 file
        z1000_file = 'atmos_100-year_JanInit_z1000_ll.nc',
        # name of tau input file
        tau_file = 'atmos_100-year_JanInit_tau300-700_ll.nc',
        # where to put output plots
        output_prefix = '/home/disk/brume/nacc/gradient_dissent/DLESyM/scratch/24h_1in1out_full_eval/1in1out24hr_TC-eval',
    )

# ToDo: add taylor diagrams
# todo add enso evals

if __name__ == "__main__":

    run_seasonal_cycle = True
    run_blocking = True
    run_tc = True  
    # activate environment with xeofs installed  
    run_nam = False
    run_sam = False

    if run_seasonal_cycle:
        from evaluation.seasonal_cycle import main as seasonal_cycle_main
        seasonal_cycle_main(PARAMS_Z500_1in1out_24h_seasonal_cycle)
    if run_blocking:
        from evaluation.blocking import blocking_frequency
        blocking_frequency(**PARAMS_Z500_1in1out_24h_NH_40yr_blocking)
    if run_tc:
        from evaluation.TC_freq_hpx import main as tc_main
        tc_main(**tc_params_24h)
    if run_nam:
        from evaluation.NAM_SAM_hpx import main as nam_sam_main
        nam_sam_main(**params_nam_24hr)
    if run_sam:
        from evaluation.NAM_SAM_hpx import main as nam_sam_main
        nam_sam_main(**params_sam_24hr)
