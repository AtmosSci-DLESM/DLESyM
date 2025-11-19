import numpy as np
from evaluation.blocking import get_custom_cmap, blocking_frequency

PARAMS_NH_40yr = {
    'forecast_file' : '/home/disk/rhodium/WEB/DLESyM_AGU-Advances/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit',
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
    'output_dir' : './scratch/blocking_last40/',
    'plot_file_suffix': 'nh_40yr',
    'map_suffix': 'nh_map_40yr',
}

if __name__ == "__main__":

    blocking_frequency(**PARAMS_NH_40yr)