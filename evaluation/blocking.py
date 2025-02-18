# region ### IMPORTS ###

import xarray as xr 
import os
import numpy as np
import sys
import copy
from tqdm import tqdm
from typing import List, Tuple, Union
from PIL import Image
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import cartopy.feature as cft 
sys.path.append('/home/disk/brume/nacc/dlesm/zephyr')
import scripts.coupled_forecast as cf 
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from evaluation.evaluators import EvaluatorHPX
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors 
import matplotlib.patches as mpatches
import glob
import pandas as pd 
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from cmip6_utils import cesm2, mpi_esm1, gfdl_cm4, had_gem, ipsl

# endregion 


#region ### PARAMS ###

# adds grey as the lowest color. good for sparse fields.
def get_custom_cmap(cmap_base='YlOrRd', middle=False, white_levels=20, start_color=0):

    if middle:
        # Create a colormap that goes from the base colormap to 'whitesmoke'
        cmap_base = plt.get_cmap(cmap_base)
        
        # Split the base colormap into two parts
        colors_lower = cmap_base(np.linspace(0, 0.45, 128))
        colors_upper = cmap_base(np.linspace(0.55, 1, 128))

        colors_middle = mcolors.LinearSegmentedColormap.from_list("mycmap", ['whitesmoke', 'whitesmoke'])
        
        # Create a new colormap with 'whitesmoke' in the middle
        colors = np.vstack((colors_lower, colors_middle(np.linspace(.4,.6,32)), colors_upper))
        cmap_combined = mcolors.LinearSegmentedColormap.from_list('custom_divergent_colormap', colors)
        
        return cmap_combined
    else:
        # Create a colormap that goes from grey to white
        cmap1 = mcolors.LinearSegmentedColormap.from_list("mycmap", ['whitesmoke', 'whitesmoke'])

        # Get the 'YlOrRd' colormap
        cmap2 = plt.get_cmap(cmap_base)

        # Combine the two colormaps
        colors = np.vstack((cmap1(np.linspace(0, 1, white_levels)), cmap2(np.linspace(start_color, 1, 128))))
        cmap_combined = mcolors.LinearSegmentedColormap.from_list('colormap', colors)
        return cmap_combined

PARAMS_NH = {
    'forecast_file' : '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit',
    'forecast_time_range': slice('2087-01-01', '2116-12-31'),
    'verification_file' : '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_geopotential_500.nc',
    'verification_time_range': slice('1987-01-01', '2016-12-31'),
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
    'output_dir' : '/home/disk/brume/nacc/WeeklyNotebooks/2024.05.13/FigureScripts/blocking_cache',
    'plot_file_suffix': 'nh',
    'map_suffix': 'nh_map',
}

PARAMS_NH_40yr = {
    'forecast_file' : '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit',
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
    'output_dir' : '/home/disk/brume/nacc/WeeklyNotebooks/2024.05.13/FigureScripts/blocking_cache',
    'plot_file_suffix': 'nh_40yr',
    'map_suffix': 'nh_map_40yr',
}

PARAMS_NH_CESM = {
    'simulation_dir' : '/home/disk/mercury5/nacc/cmip6/CESM2',
    'selection_function' : cesm2,
    'forecast_time_range': slice('2087-01-01', '2116-12-31'),
    'verification_file' : '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_geopotential_500.nc',
    'verification_time_range': slice('1987-01-01', '2016-12-31'),
    'months':[1,2,3,4,5,6,7,8,9,10,11,12],
    'hemisphere': 'north',
    'model_label':'CESM2',
    # 'freq_levels': np.arange(1, 15.1, 1),
    'freq_levels': np.arange(0.0, 0.101, 0.01),
    # 'freq_ticks': np.arange(1, 15.1, 2),
    'freq_ticks': np.arange(0.0, 0.101, 0.01),
    'freq_cmap': get_custom_cmap('plasma_r'),
    'std_cmap': get_custom_cmap('Greens'),
    'std_levels': np.arange(0, 0.101, .02),
    'std_ticks': np.arange(0, 0.101, .02),
    'output_dir' : '/home/disk/brume/nacc/WeeklyNotebooks/2024.05.13/FigureScripts/blocking_cache',
    'plot_file_suffix': 'cesm_nh',
    'map_suffix': 'cesm_nh_map',
}
PARAMS_NH_CESM_40yr = {
    'simulation_dir' : '/home/disk/mercury5/nacc/cmip6/CESM2',
    'selection_function' : cesm2,
    'forecast_time_range': slice('2070-01-01', '2110-12-31'),
    'verification_file' : '/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_z500.nc',
    'verification_time_range': slice('1970-01-01', '2010-12-31'),
    'months':[1,2,3,4,5,6,7,8,9,10,11,12],
    'hemisphere': 'north',
    'model_label':'CESM2',
    # 'freq_levels': np.arange(1, 15.1, 1),
    'freq_levels': np.arange(0.0, 0.101, 0.01),
    # 'freq_ticks': np.arange(1, 15.1, 2),
    'freq_ticks': np.arange(0.0, 0.101, 0.01),
    'freq_cmap': get_custom_cmap('plasma_r'),
    'std_cmap': get_custom_cmap('Greens'),
    'std_levels': np.arange(0, 0.101, .02),
    'std_ticks': np.arange(0, 0.101, .02),
    'output_dir' : '/home/disk/brume/nacc/WeeklyNotebooks/2024.05.13/FigureScripts/blocking_cache',
    'plot_file_suffix': 'cesm_nh_40yr',
    'map_suffix': 'cesm_nh_map_40yr',
}

PARAMS_NH_MPI = {
    'simulation_dir' : '/home/disk/mercury5/nacc/cmip6/MPI-ESM1-2-LR/day',
    'selection_function' : mpi_esm1,
    'forecast_time_range': slice('2087-01-01', '2116-12-31'),
    'verification_file' : '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_geopotential_500.nc',
    'verification_time_range': slice('1987-01-01', '2016-12-31'),
    'months':[1,2,3,4,5,6,7,8,9,10,11,12],
    'hemisphere': 'north',
    'max_lat': 73.659, # closest lat to 75 on mesh
    'model_label':'MPI-ESM1-LR',
    'model_label_size': 12,
    # 'freq_levels': np.arange(1, 15.1, 1),
    'freq_levels': np.arange(0.0, 0.101, 0.01),
    # 'freq_ticks': np.arange(1, 15.1, 2),
    'freq_ticks': np.arange(0.0, 0.101, 0.01),
    'freq_cmap': get_custom_cmap('plasma_r'),
    'std_cmap': get_custom_cmap('Greens'),
    'std_levels': np.arange(0, 0.101, .02),
    'std_ticks': np.arange(0, 0.101, .02),
    'output_dir' : '/home/disk/brume/nacc/WeeklyNotebooks/2024.05.13/FigureScripts/blocking_cache',
    'plot_file_suffix': 'mpi-esp1-2-lr_nh',
    'map_suffix': 'mpi-esp1-2-lr_nh_map',
}

PARAMS_NH_MPI_HR = {
    'simulation_dir' : '/home/disk/mercury5/nacc/cmip6/MPI-ESM1-2-HR',
    'selection_function' : mpi_esm1,
    'forecast_time_range': slice('2087-01-01', '2116-12-31'),
    'verification_file' : '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_geopotential_500.nc',
    'verification_time_range': slice('1987-01-01', '2016-12-31'),
    'months':[1,2,3,4,5,6,7,8,9,10,11,12],
    'hemisphere': 'north',
    'max_lat': 73.659, # closest lat to 75 on mesh
    'model_label':'MPI',
    # 'freq_levels': np.arange(1, 15.1, 1),
    'freq_levels': np.arange(0.0, 0.101, 0.01),
    # 'freq_ticks': np.arange(1, 15.1, 2),
    'freq_ticks': np.arange(0.0, 0.101, 0.01),
    'freq_cmap': get_custom_cmap('plasma_r'),
    'std_cmap': get_custom_cmap('Greens'),
    'std_levels': np.arange(0, 0.101, .02),
    'std_ticks': np.arange(0, 0.101, .02),
    'output_dir' : '/home/disk/brume/nacc/WeeklyNotebooks/2024.05.13/FigureScripts/blocking_cache',
    'plot_file_suffix': 'mpi-esp1-2-hr_nh',
    'map_suffix': 'mpi-esp1-2-hr_nh_map',
}

PARAMS_NH_MPI_HR_40yr = {
    'simulation_dir' : '/home/disk/mercury5/nacc/cmip6/MPI-ESM1-2-HR',
    'selection_function' : mpi_esm1,
    'forecast_time_range': slice('2070-01-01', '2110-12-31'),
    'verification_file' : '/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_z500.nc',
    'verification_time_range': slice('1970-01-01', '2010-12-31'),
    'months':[1,2,3,4,5,6,7,8,9,10,11,12],
    'hemisphere': 'north',
    'max_lat': 73.659, # closest lat to 75 on mesh
    'model_label':'MPI',
    # 'freq_levels': np.arange(1, 15.1, 1),
    'freq_levels': np.arange(0.0, 0.101, 0.01),
    # 'freq_ticks': np.arange(1, 15.1, 2),
    'freq_ticks': np.arange(0.0, 0.101, 0.01),
    'freq_cmap': get_custom_cmap('plasma_r'),
    'std_cmap': get_custom_cmap('Greens'),
    'std_levels': np.arange(0, 0.101, .02),
    'std_ticks': np.arange(0, 0.101, .02),
    'output_dir' : '/home/disk/brume/nacc/WeeklyNotebooks/2024.05.13/FigureScripts/blocking_cache',
    'plot_file_suffix': 'mpi-esp1-2-hr_nh_40yr',
    'map_suffix': 'mpi-esp1-2-hr_nh_map_40yr',
}

PARAMS_NH_gfdl = {
    'simulation_dir' : '/home/disk/mercury5/nacc/cmip6/GFDL-CM4',
    'selection_function' : gfdl_cm4,
    'forecast_time_range': slice('2087-01-01', '2116-12-31'),
    'verification_file' : '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_geopotential_500.nc',
    'verification_time_range': slice('1987-01-01', '2016-12-31'),
    'months':[1,2,3,4,5,6,7,8,9,10,11,12],
    'hemisphere': 'north',
    'model_label':'GFDL-CM4',
    'model_label_size': 12,
    # 'freq_levels': np.arange(1, 15.1, 1),
    'freq_levels': np.arange(0.0, 0.101, 0.01),
    # 'freq_ticks': np.arange(1, 15.1, 2),
    'freq_ticks': np.arange(0.0, 0.101, 0.01),
    'freq_cmap': get_custom_cmap('plasma_r'),
    'std_cmap': get_custom_cmap('Greens'),
    'std_levels': np.arange(0, 0.101, .02),
    'std_ticks': np.arange(0, 0.101, .02),
    'output_dir' : '/home/disk/brume/nacc/WeeklyNotebooks/2024.05.13/FigureScripts/blocking_cache',
    'plot_file_suffix': 'gfdl_nh',
    'map_suffix': 'gfdl_nh_map',
}

PARAMS_NH_gfdl_40yr = {
    'simulation_dir' : '/home/disk/mercury5/nacc/cmip6/GFDL-CM4',
    'selection_function' : gfdl_cm4,
    'forecast_time_range': slice('2070-01-01', '2110-12-31'),
    'verification_file' : '/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_z500.nc',
    'verification_time_range': slice('1970-01-01', '2010-12-31'),
    'months':[1,2,3,4,5,6,7,8,9,10,11,12],
    'hemisphere': 'north',
    'model_label':'GFDL',
    # 'freq_levels': np.arange(1, 15.1, 1),
    'freq_levels': np.arange(0.0, 0.101, 0.01),
    # 'freq_ticks': np.arange(1, 15.1, 2),
    'freq_ticks': np.arange(0.0, 0.101, 0.01),
    'freq_cmap': get_custom_cmap('plasma_r'),
    'std_cmap': get_custom_cmap('Greens'),
    'std_levels': np.arange(0, 0.101, .02),
    'std_ticks': np.arange(0, 0.101, .02),
    'output_dir' : '/home/disk/brume/nacc/WeeklyNotebooks/2024.05.13/FigureScripts/blocking_cache',
    'plot_file_suffix': 'gfdl_nh_40yr',
    'map_suffix': 'gfdl_nh_map_40yr',
}

PARAMS_NH_HadGEM = {
    'simulation_dir' : '/home/disk/mercury5/nacc/cmip6/HadGEM3-GC31-LL',
    'selection_function' : had_gem,
    'forecast_time_range': slice('2087-01-01', '2116-12-31'),
    'verification_file' : '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_geopotential_500.nc',
    'verification_time_range': slice('1987-01-01', '2016-12-31'),
    'months':[1,2,3,4,5,6,7,8,9,10,11,12],
    'hemisphere': 'north',
    'max_lat': 74, # closest lat to 75 on mesh
    'model_label':'HadGEM3',
    'model_label_size': 13,
    # 'freq_levels': np.arange(1, 15.1, 1),
    'freq_levels': np.arange(0.0, 0.101, 0.01),
    # 'freq_ticks': np.arange(1, 15.1, 2),
    'freq_ticks': np.arange(0.0, 0.101, 0.01),
    'freq_cmap': get_custom_cmap('plasma_r'),
    'std_cmap': get_custom_cmap('Greens'),
    'std_levels': np.arange(0, 0.101, .02),
    'std_ticks': np.arange(0, 0.101, .02),
    'output_dir' : '/home/disk/brume/nacc/WeeklyNotebooks/2024.05.13/FigureScripts/blocking_cache',
    'plot_file_suffix': 'hadgem_nh',
    'map_suffix': 'hadgem_nh_map',
}

PARAMS_NH_HadGEM_40yr = {
    'simulation_dir' : '/home/disk/mercury5/nacc/cmip6/HadGEM3-GC31-LL',
    'selection_function' : had_gem,
    'forecast_time_range': slice('2070-01-01', '2110-12-31'),
    'verification_file' : '/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_z500.nc',
    'verification_time_range': slice('1970-01-01', '2010-12-31'),
    'months':[1,2,3,4,5,6,7,8,9,10,11,12],
    'hemisphere': 'north',
    'max_lat': 74, # closest lat to 75 on mesh
    'model_label':'HadGEM3',
    'model_label_size': 13,
    # 'freq_levels': np.arange(1, 15.1, 1),
    'freq_levels': np.arange(0.0, 0.101, 0.01),
    # 'freq_ticks': np.arange(1, 15.1, 2),
    'freq_ticks': np.arange(0.0, 0.101, 0.01),
    'freq_cmap': get_custom_cmap('plasma_r'),
    'std_cmap': get_custom_cmap('Greens'),
    'std_levels': np.arange(0, 0.101, .02),
    'std_ticks': np.arange(0, 0.101, .02),
    'output_dir' : '/home/disk/brume/nacc/WeeklyNotebooks/2024.05.13/FigureScripts/blocking_cache',
    'plot_file_suffix': 'hadgem_nh_40yr',
    'map_suffix': 'hadgem_nh_map_40yr',
}

PARAMS_NH_IPSL = {
    'simulation_dir' : '/home/disk/mercury5/nacc/cmip6/IPSL-CM6A-LR',
    'selection_function' : ipsl,
    'forecast_time_range': slice('2087-01-01', '2116-12-31'),
    'verification_file' : '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_geopotential_500.nc',
    'verification_time_range': slice('1987-01-01', '2016-12-31'),
    'months':[1,2,3,4,5,6,7,8,9,10,11,12],
    'hemisphere': 'north',
    'model_label':'IPSL-CM6A-LR',
    'model_label_size': 10,
    # 'freq_levels': np.arange(1, 15.1, 1),
    'freq_levels': np.arange(0.0, 0.101, 0.01),
    # 'freq_ticks': np.arange(1, 15.1, 2),
    'freq_ticks': np.arange(0.0, 0.101, 0.01),
    'freq_cmap': get_custom_cmap('plasma_r'),
    'std_cmap': get_custom_cmap('Greens'),
    'std_levels': np.arange(0, 0.101, .02),
    'std_ticks': np.arange(0, 0.101, .02),
    'output_dir' : '/home/disk/brume/nacc/WeeklyNotebooks/2024.05.13/FigureScripts/blocking_cache',
    'plot_file_suffix': 'ipsl_nh',
    'map_suffix': 'ipsl_nh_map',
}


PARAMS_NH_DJF = {
    'forecast_file' : '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit',
    'forecast_time_range': slice('2087-01-01', '2116-12-31'),
    'verification_file' : '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_geopotential_500.nc',
    'verification_time_range': slice('1987-01-01', '2016-12-31'),
    'months':[12,1,2],
    'hemisphere': 'north',
    # 'freq_levels': np.arange(1, 15.1, 1),
    'freq_levels': np.arange(0.0, 0.101, 0.01),
    # 'freq_ticks': np.arange(1, 15.1, 2),
    'freq_ticks': np.arange(0.0, 0.101, 0.01),
    'freq_cmap': get_custom_cmap('plasma_r'),
    'std_cmap': get_custom_cmap('Greens'),
    'std_levels': np.arange(0, 0.101, .02),
    'std_ticks': np.arange(0, 0.101, .02),
    'output_dir' : '/home/disk/brume/nacc/WeeklyNotebooks/2024.05.13/FigureScripts/blocking_cache',
    'plot_file_suffix': 'nh_djf',
    # ATLANTIC SECTOR
    # 'spatial_correlation_params': {
    #     'region': {'lat':slice(50,75),'lon':slice(-90,90)},
    #     'extra_suffix': '_atl'
    # },
    # 'average_params' : {
    #     'region': {'lat':slice(50,75),'lon':slice(-90,90)},
    #     'extra_suffix': '_atl'
    # },
    # 'rmse_params' : {
    #     'region': {'lat':slice(50,75),'lon':slice(-90,90)},
    #     'extra_suffix': '_atl'
    # },
    # PACIFIC SECTOR
    # 'spatial_correlation_params': {
    #     'region': {'lat':slice(50,75),'lon':slice(90,270)},
    #     'extra_suffix': '_pac',
    #     'yticks': np.arange(.6,1.01,0.05)
    # },
    # 'average_params' : {
    #     'region': {'lat':slice(50,75),'lon':slice(90,270)},
    #     'extra_suffix': '_pac',
    #     'yticks': np.arange(0,0.036,0.005)
    # },
    # 'rmse_params' : {
    #     'region': {'lat':slice(50,75),'lon':slice(90,270)},
    #     'extra_suffix': '_pac',
    #     'yticks': np.arange(0.0,0.026,0.005)
    # },
}

PARAMS_NH_MAM = {
    'forecast_file' : '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit',
    'forecast_time_range': slice('2087-01-01', '2116-12-31'),
    'verification_file' : '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_geopotential_500.nc',
    'verification_time_range': slice('1987-01-01', '2016-12-31'),
    'months':[3,4,5],
    'hemisphere': 'north',
    'freq_levels': np.arange(0.0, 0.101, 0.01),
    'freq_ticks': np.arange(0.0, 0.101, 0.01),
    'freq_cmap': get_custom_cmap('plasma_r'),
    'std_cmap': get_custom_cmap('Greens'),
    'std_levels': np.arange(0, 0.101, .02),
    'std_ticks': np.arange(0, 0.101, .02),
    'output_dir' : '/home/disk/brume/nacc/WeeklyNotebooks/2024.05.13/FigureScripts/blocking_cache',
    'plot_file_suffix': 'nh_mam',
    # ATLANTIC SECTOR
    # 'spatial_correlation_params': {
    #     'region': {'lat':slice(50,75),'lon':slice(-90,90)},
    #     'extra_suffix': '_atl'
    # },
    # 'average_params' : {
    #     'region': {'lat':slice(50,75),'lon':slice(-90,90)},
    #     'extra_suffix': '_atl'
    # },
    # 'rmse_params' : {
    #     'region': {'lat':slice(50,75),'lon':slice(-90,90)},
    #     'extra_suffix': '_atl'
    # },
    # PACIFIC SECTOR
    # 'spatial_correlation_params': {
    #     'region': {'lat':slice(50,75),'lon':slice(90,270)},
    #     'extra_suffix': '_pac',
    #     'yticks': np.arange(.6,1.01,0.05)
    # },
    # 'average_params' : {
    #     'region': {'lat':slice(50,75),'lon':slice(90,270)},
    #     'extra_suffix': '_pac',
    #     'yticks': np.arange(0,0.036,0.005)
    # },
    # 'rmse_params' : {
    #     'region': {'lat':slice(50,75),'lon':slice(90,270)},
    #     'extra_suffix': '_pac',
    #     'yticks': np.arange(0.0,0.026,0.005)
    # },
}

PARAMS_NH_JJA = {
    'forecast_file' : '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit',
    'forecast_time_range': slice('2087-01-01', '2116-12-31'),
    'verification_file' : '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_geopotential_500.nc',
    'verification_time_range': slice('1987-01-01', '2016-12-31'),
    'months':[6,7,8],
    'hemisphere': 'north',
    # 'freq_levels': np.arange(1, 15.1, 1),
    'freq_levels': np.arange(0.0, 0.101, 0.01),
    # 'freq_ticks': np.arange(1, 15.1, 2),
    'freq_ticks': np.arange(0.0, 0.101, 0.01),
    'freq_cmap': get_custom_cmap('plasma_r'),
    'std_cmap': get_custom_cmap('Greens'),
    'std_levels': np.arange(0, 0.101, .02),
    'std_ticks': np.arange(0, 0.101, .02),
    'plot_file_suffix': 'nh_jja', 
    'output_dir' : '/home/disk/brume/nacc/WeeklyNotebooks/2024.05.13/FigureScripts/blocking_cache',
    # ATLANTIC SECTOR
    # 'spatial_correlation_params': {
    #     'region': {'lat':slice(50,75),'lon':slice(-90,90)},
    #     'extra_suffix': '_atl'
    # },
    # 'average_params' : {
    #     'region': {'lat':slice(50,75),'lon':slice(-90,90)},
    #     'extra_suffix': '_atl'
    # },
    # 'rmse_params' : {
    #     'region': {'lat':slice(50,75),'lon':slice(-90,90)},
    #     'extra_suffix': '_atl'
    # },
    # PACIFIC SECTOR
    # 'spatial_correlation_params': {
    #     'region': {'lat':slice(50,75),'lon':slice(90,270)},
    #     'extra_suffix': '_pac',
    #     'yticks': np.arange(.6,1.01,0.05)
    # },
    # 'average_params' : {
    #     'region': {'lat':slice(50,75),'lon':slice(90,270)},
    #     'extra_suffix': '_pac',
    #     'yticks': np.arange(0,0.036,0.005)
    # },
    # 'rmse_params' : {
    #     'region': {'lat':slice(50,75),'lon':slice(90,270)},
    #     'extra_suffix': '_pac',
    #     'yticks': np.arange(0.0,0.026,0.005)
    # },
}

PARAMS_NH_SON = {
    'forecast_file' : '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit',
    'forecast_time_range': slice('2087-01-01', '2116-12-31'),
    'verification_file' : '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_geopotential_500.nc',
    'verification_time_range': slice('1987-01-01', '2016-12-31'),
    'months':[9,10,11],
    'hemisphere': 'north',
    # 'freq_levels': np.arange(1, 15.1, 1),
    'freq_levels': np.arange(0.0, 0.101, 0.01),
    # 'freq_ticks': np.arange(1, 15.1, 2),
    'freq_ticks': np.arange(0.0, 0.101, 0.01),
    'freq_cmap': get_custom_cmap('plasma_r'),
    'std_cmap': get_custom_cmap('Greens'),
    'std_levels': np.arange(0, 0.101, .02),
    'std_ticks': np.arange(0, 0.101, .02),
    'output_dir' : '/home/disk/brume/nacc/WeeklyNotebooks/2024.05.13/FigureScripts/blocking_cache',
    'plot_file_suffix': 'nh_son',
    # ATLANTIC SECTOR
    # 'spatial_correlation_params': {
    #     'region': {'lat':slice(50,75),'lon':slice(-90,90)},
    #     'extra_suffix': '_atl'
    # },
    # 'average_params' : {
    #     'region': {'lat':slice(50,75),'lon':slice(-90,90)},
    #     'extra_suffix': '_atl'
    # },
    # 'rmse_params' : {
    #     'region': {'lat':slice(50,75),'lon':slice(-90,90)},
    #     'extra_suffix': '_atl'
    # },
    # PACIFIC SECTOR
    # 'spatial_correlation_params': {
    #     'region': {'lat':slice(50,75),'lon':slice(90,270)},
    #     'extra_suffix': '_pac',
    #     'yticks': np.arange(.6,1.01,0.05)
    # },
    # 'average_params' : {
    #     'region': {'lat':slice(50,75),'lon':slice(90,270)},
    #     'extra_suffix': '_pac',
    #     'yticks': np.arange(0,0.036,0.005)
    # },
    # 'rmse_params' : {
    #     'region': {'lat':slice(50,75),'lon':slice(90,270)},
    #     'extra_suffix': '_pac',
    #     'yticks': np.arange(0.0,0.026,0.005)
    # },
}

#endregion

#region ### SOURCE ###

def cyclic_select(data, lon_range):
    if type(lon_range) == slice:
        lon_min, lon_max = lon_range.start, lon_range.stop
    else:
        lon_min, lon_max = lon_range
    
    if lon_min < 0:
        lon_min += 360

    if lon_min < lon_max:
        return data.sel(lon=slice(lon_min, lon_max))
    else:
        return xr.concat([data.sel(lon=slice(lon_min, 360)), data.sel(lon=slice(0, lon_max))], dim='lon')

# Count number of true values in each year
def count_blocking(blocking):
    blocking = blocking.astype(int)
    # blocking = blocking.resample(time="Y").sum()
    blocking = blocking.resample(time="Y").mean()
    return blocking

def AGP_Blocking_Index(
      z500: Union[xr.Dataset, xr.DataArray], 
      hemisphere: str = "north",
      target_file: str = None
   ) -> xr.Dataset:
   """
   Blocking Index at described in Schiemann et al. 2020
   1. Equator-to-pole gradient of the 500 hPa geopotential height Z to the south 
      reverses: (Z500(lat) - Z500(lat - 15))/15 > 0
   2. Winds to the north via pressure gradient force: 
      (Z500(lat + 15) - Z500(lat))/15 < -10 m / latitude
   3. Must last for at least 5 days

   Region is defines 35N - 75N and 35S - 75S
   """

   # If southern hemisphere, reverse latitudes and change sign (i.e. pretend like it's the northern hemisphere)
   if hemisphere == "south":
      z500 = z500.sel(lat=slice(-90, 0))
      z500 = z500.reindex(lat=z500["lat"][::-1])
      z500["lat"] = np.abs(z500["lat"])

   upper_lat = 75
   lower_lat = 35
   del_lat = 15 
   # 1. Equator-to-pole gradient of the 500 hPa geopotential height Z to the south reverses: (Z500(lat) - Z500(lat - 15))/15 > 0
   z500_o = z500.sel(lat=slice(lower_lat, upper_lat))
   z500_south = z500.sel(lat=slice(lower_lat - del_lat, upper_lat - del_lat))

   z500_south["lat"] = z500_o["lat"]
   grad_reversal = z500_o - z500_south
   grad_reversal = (grad_reversal / del_lat) > 0

   # 2. Winds to the north via pressure gradient force: (Z500(lat + 15) - Z500(lat))/15 < -10 m / latitude
   z500_o = z500.sel(lat=slice(lower_lat, upper_lat))
   z500_north = z500.sel(lat=slice(lower_lat + del_lat, upper_lat + del_lat))
   z500_north["lat"] = z500_o["lat"]
   winds_pgf = z500_north - z500_o
   winds_pgf = (winds_pgf / del_lat) < -10

   # Find where both conditions are met
   blocking = grad_reversal & winds_pgf

   # 3. Must be True for at least 5 days
   blocking = (blocking.rolling(time=5).sum() == 5)

   # If southern hemisphere, revert back to original latitudes
   if hemisphere == "south":
      blocking = blocking.reindex(lat=blocking["lat"][::-1])
      blocking["lat"] = -1 * blocking["lat"]

   # Save to file
   if target_file is not None:
       blocking.to_netcdf(target_file)

   return blocking

def blocking_frequency(
        forecast_file: str,
        forecast_time_range: slice,
        verification_file: str,
        verification_time_range: slice,
        months: List[int],
        output_dir: str = '.',
        plot_file_suffix: str = None,
        map_suffix: str = '',
        hemisphere: str = "north",
        freq_levels: List[int] = np.arange(1, 45.1, 2),
        freq_ticks: List[int] = np.arange(5, 45.1, 5),
        freq_cmap: str = 'ocean_r',
        std_levels: List[int] = np.arange(2, 11.1, 1),
        std_ticks: List[int] = np.arange(2, 11.1, 1),
        std_cmap: str = 'Greens',
        average_params: dict = None,
        spatial_correlation_params: dict = None,
        rmse_params: dict = None
):
    
    # create output directory
    os.makedirs(output_dir, exist_ok=True)

    #####    DATA PREPARATION    #####
    # initialize evaluator for z500 forecast
    fcst_z500 = EvaluatorHPX(
        forecast_path = f'{forecast_file}.nc',
        verification_path = verification_file,
        eval_variable = 'z500',
        on_latlon = True,
        poolsize = 20,
        ll_file=f'{forecast_file}_z500_ll.nc'
    )

    ####    CALCULATION    #####
    # cache filename 
    fcst_blocking_freq_cache = f'{output_dir}/AGP_Blocking_Index_fcst_{plot_file_suffix}.nc'
    if os.path.exists(fcst_blocking_freq_cache):
        print(f'Loading {fcst_blocking_freq_cache}...')
        fcst_blocking = xr.open_dataarray(fcst_blocking_freq_cache)
    else:
        # extract forecast dataarrays, fix time dimensions, grab storm forecasted time, eliminate singleton dimensions, scale heights to dekameters
        fcst_ll_z500 = fcst_z500.forecast_da.assign_coords(step = fcst_z500.forecast_da.time.values + fcst_z500.forecast_da.step).sel(step=forecast_time_range).squeeze()
        fcst_ll_z500 = fcst_ll_z500.drop('time').squeeze().rename({'step':'time'})[:,::-1,:]
        fcst_ll_z500 = fcst_ll_z500.resample(time='1D').mean()
        fcst_ll_z500 = fcst_ll_z500.sel(time=fcst_ll_z500['time.month'].isin(months))
        fcst_blocking = AGP_Blocking_Index(fcst_ll_z500, hemisphere=hemisphere, target_file=fcst_blocking_freq_cache)
    fcst_blocking_count = count_blocking(fcst_blocking).mean('time')
    fcst_blocking_std = count_blocking(fcst_blocking).std('time')

    if map_suffix != '':
        map_cache_file = f'{output_dir}/blocking_freq_{map_suffix}.nc'
        print(f'Saving blocking frequency to {map_cache_file}.')
        fcst_blocking_count.to_netcdf(map_cache_file)

    # do the same for the observed comparison
    # cache filename
    verif_blocking_freq_cache = f'{output_dir}/AGP_Blocking_Index_verif_{plot_file_suffix}.nc'
    if os.path.exists(verif_blocking_freq_cache):
        print(f'Loading {verif_blocking_freq_cache}...')
        verif_blocking = xr.open_dataarray(verif_blocking_freq_cache)
    else:
        verif_ll_z500 = xr.open_dataset(verification_file)['z'].sel(time=verification_time_range).rename({'longitude':'lon','latitude':'lat'}).squeeze()
        verif_ll_z500 = verif_ll_z500.resample(time='1D').mean()[:,::-1,:]
        verif_ll_z500 = verif_ll_z500.sel(time=verif_ll_z500['time.month'].isin(months))
        verif_blocking = AGP_Blocking_Index(verif_ll_z500, hemisphere=hemisphere, target_file=verif_blocking_freq_cache)
    verif_blocking_count = count_blocking(verif_blocking).mean('time')
    verif_blocking_std = count_blocking(verif_blocking).std('time')

    if map_suffix != '':
        map_cache_file_verif = f'{output_dir}/blocking_freq_{map_suffix}-verif.nc'
        print(f'Saving blocking frequency to {map_cache_file_verif}.')
        verif_blocking_count.to_netcdf(map_cache_file_verif)

    #####    CALCULATING METRICS    #####
    def spatial_correlation(data1, data2, region, extra_suffix='', yticks=np.arange(0,1.1,0.2)):
        # TODO calculate and plot correlation
        data1 = cyclic_select(data1.sel(lat=region['lat']), region['lon'])
        data2 = cyclic_select(data2.sel(lat=region['lat']), region['lon'])

        # Calculate correlation
        corr = np.corrcoef(data1.values.flatten(), data2.values.flatten())[0,1]
        print(f'Correlation: {corr}')

        # plot correlations 
        fig, ax = plt.subplots(1, 1, figsize=(1,5))
        ax.set_facecolor('lightgray')
        ax.yaxis.tick_right()  # Move the y-ticks to the right side
        ax.yaxis.set_label_position("right")
        ax.plot(1, corr, '^', color='black', markersize=10)
        ax.set_xlim(.9,1.1)
        ax.set_ylim(yticks[0],yticks[-1])
        ax.set_xticks([1])
        ax.set_yticks(yticks)
        ax.set_xticklabels(['DLESM'], fontsize=12,rotation='vertical')
        print(f'Saving blocking correlation plot to blocking_correlation_{plot_file_suffix}{extra_suffix}.png')
        fig.tight_layout()
        fig.savefig(f'{output_dir}/blocking_correlation_{plot_file_suffix}{extra_suffix}.png', dpi=300)

        return 

    def calculate_average(data1, data2, region, extra_suffix='', yticks=np.arange(0,0.031,0.005)):
        # TODO calculate and plot average
        data1 = cyclic_select(data1.sel(lat=region['lat']), region['lon'])
        data2 = cyclic_select(data2.sel(lat=region['lat']), region['lon'])

        # Calculate average
        avg1 = data1.mean().values
        avg2 = data2.mean().values

        print(plot_file_suffix)
        # plot averages
        fig, ax = plt.subplots(1, 1, figsize=(1.25,5))
        ax.set_facecolor('lightgray')
        ax.yaxis.tick_right()  # Move the y-ticks to the right side
        ax.yaxis.set_label_position("right")
        ax.plot(1, avg1, '^', color='black', markersize=10)
        # ax.plot(1, avg2, '*', color='blue', markersize=10)
        ax.set_xlim(.9,1.1)
        ax.set_ylim(yticks[0],yticks[-1])
        ax.set_xticks([1])
        ax.set_yticks(yticks)
        ax.set_xticklabels(['DLESM'], fontsize=12,rotation='vertical')
        print(f'Saving blocking average plot to blocking_average_{plot_file_suffix}{extra_suffix}.png')
        fig.tight_layout()
        fig.savefig(f'{output_dir}/blocking_freq_average_{plot_file_suffix}{extra_suffix}.png', dpi=300)

        return
    
    def calculate_rmse(data1, data2, region, extra_suffix='',yticks=np.arange(0,0.026,0.005)):
        # TODO calculate and plot rmse
        data1 = cyclic_select(data1.sel(lat=region['lat']), region['lon'])
        data2 = cyclic_select(data2.sel(lat=region['lat']), region['lon'])

        # Calculate RMSE
        rmse = np.sqrt(np.mean((data1 - data2)**2)).values
        print(f'RMSE: {rmse}')

        # plot rmse
        fig, ax = plt.subplots(1, 1, figsize=(1.25,5))
        ax.set_facecolor('lightgray')
        ax.yaxis.tick_right()  # Move the y-ticks to the right side
        ax.yaxis.set_label_position("right")
        ax.plot(1, rmse, '^', color='black', markersize=10)
        ax.set_xlim(.9,1.1)
        ax.set_ylim(yticks[0],yticks[-1])
        ax.set_xticks([1])
        ax.set_yticks(yticks)
        ax.set_xticklabels(['DLESM'], fontsize=12,rotation='vertical')
        print(f'Saving blocking RMSE plot to blocking_rmse_{plot_file_suffix}{extra_suffix}.png')
        fig.tight_layout()
        fig.savefig(f'{output_dir}/blocking_rmse_{plot_file_suffix}{extra_suffix}.png', dpi=300)
        return

    if average_params is not None:
        calculate_average(fcst_blocking_count,
                          verif_blocking_count,
                          **average_params)
    if spatial_correlation_params is not None:
        spatial_correlation(fcst_blocking_count,
                            verif_blocking_count,
                            **spatial_correlation_params)
    if rmse_params is not None:
        calculate_rmse(fcst_blocking_count,
                       verif_blocking_count,
                       **rmse_params)

    #####    PLOTTING MAPS   #####
    def init_polar_axs(hemisphere):

        # initialize figure and gridspec
        fig = plt.figure(figsize=(10,5))
        gs = GridSpec(1, 2)
        proj=ccrs.NorthPolarStereo(central_longitude=0) if hemisphere == "north" else ccrs.SouthPolarStereo(central_longitude=0)

        def draw_circle(ax, theta, center_x=0.5, center_y=0.5):
            center, radius = [center_x, center_y], 0.5
            # Adjust theta to span only half the circle
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            ax.set_boundary(circle, transform=ax.transAxes)
            return ax
        
        # function to cleanly write a label in the center of the north pole where 
        # blocking metric is undefined
        def write_center_label(ax, label):
            # proj = ccrs.NorthPolarStereo(central_longitude=0)
            projx1, projy1 = proj.transform_point(0, 90 if hemisphere == 'north' else -90, ccrs.Geodetic()) #get proj coord of (lon,lat)
            def compute_radius(ortho, radius_degrees, lon=0):
                lat = 90 if hemisphere == 'north' else -90
                phi1 = lat + radius_degrees if lat <= 0 else lat - radius_degrees
                _, y1 = ortho.transform_point(lon, phi1, ccrs.PlateCarree())
                return abs(y1)
            ax.add_patch(mpatches.Circle(xy=[projx1, projy1], radius=compute_radius(proj,15), color='white', \
                                alpha=1, transform=proj, zorder=25))
            ax.add_patch(mpatches.Circle(xy=[projx1, projy1], radius=compute_radius(proj,15), edgecolor='black', \
                    facecolor='none', transform=proj, zorder=25))
            # add label 
            ax.text(projx1, projy1, label, transform=proj, ha='center', va='center', fontsize=17, zorder=30, color='black')
            return ax
        
        # format first axis
        ax1 = fig.add_subplot(gs[0], projection=proj)
        ax1.set_extent([0, 360, 45, 90] if hemisphere == 'north' else [0, 360, -45, -90], ccrs.PlateCarree())
        # ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
        #               linewidth=2, color='gray', alpha=0.5, linestyle='--',
        #               xlocs=[0,60,120,180,-120,-60], ylocs=[])
        ax1.coastlines(resolution='50m', linewidth=0.6, zorder=20)
        ax1 = write_center_label(ax1, 'DL$\it{ESy}$M')
        ax1 = draw_circle(ax1, np.linspace(0, 2*np.pi, 100),.5, .5)

        # format second axis
        ax2 = fig.add_subplot(gs[1], projection=proj)#ccrs.NorthPolarStereo(central_longitude=0) if hemisphere == "north" else ccrs.SouthPolarStereo(central_longitude=0))
        ax2.set_extent([0, 360, 45, 90] if hemisphere == 'north' else [0, 360, -45, -90], ccrs.PlateCarree())
        # ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
        #               linewidth=2, color='gray', alpha=0.5, linestyle='--',
        #               xlocs=[0,60,120,180,-120,-60], ylocs=[])
        ax2.coastlines(resolution='50m', linewidth=0.6)
        ax2 = write_center_label(ax2, 'ERA5')
        ax2 = draw_circle(ax2, np.linspace(0, 2*np.pi, 100),.5, .5)

        return fig, [ax1, ax2]
    def _global_plot(ax, data, lon, lat, levels, cmap):
        data, lon = add_cyclic_point(data, coord=lon)
        # plot map of global data with central longitude 180
        img = ax.contourf(
            lon, lat, data,
            transform=ccrs.PlateCarree(), 
            cmap=cmap,
            extend="both",
            levels=levels,
        )
        _ = ax.contour(
            lon, lat, data,
            transform=ccrs.PlateCarree(),
            colors='black',
            linewidths=0.5,
            levels=levels,
        )
        return img
    
    # frequency plot 
    fig, axs = init_polar_axs(hemisphere)
    img = _global_plot(axs[0], fcst_blocking_count, fcst_blocking_count.lon, fcst_blocking_count.lat, levels=freq_levels, cmap=freq_cmap)
    _ = _global_plot(axs[1], verif_blocking_count, verif_blocking_count.lon, verif_blocking_count.lat, levels=freq_levels, cmap=freq_cmap)
    
    # add horizontal colorbar
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.075, 0.7, 0.035])
    cbar = fig.colorbar(img, cax=cbar_ax, orientation='horizontal', ticks=freq_ticks)
    cbar_ax.set_title('Blocks / Day', fontsize=15)

    #save
    plot_file = f'{output_dir}/blocking_freq_{plot_file_suffix}.svg'
    print(f'Saving blocking frequency plot to {plot_file}')
    fig.savefig(plot_file,dpi=300)

    # std plot 
    fig, axs = init_polar_axs(hemisphere)
    img = _global_plot(axs[0], fcst_blocking_std, fcst_blocking_std.lon, fcst_blocking_std.lat, 
                       cmap=std_cmap, levels=std_levels)
    _ = _global_plot(axs[1], verif_blocking_std, verif_blocking_std.lon, verif_blocking_std.lat,
                        cmap=std_cmap, levels=std_levels)
    
    # add horizontal colorbar
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.075, 0.7, 0.035])
    cbar = fig.colorbar(img, cax=cbar_ax, orientation='horizontal', ticks=std_ticks)
    cbar_ax.set_title('Event STD', fontsize=15)

    #save
    plot_file_std = f'{output_dir}/blocking_std_{plot_file_suffix}.png'
    print(f'Saving blocking frequency plot to {plot_file_std}')
    fig.savefig(plot_file_std,dpi=300)

def blocking_frequency_cmip(
        simulation_dir: str,
        selection_function: callable,
        forecast_time_range: slice,
        verification_file: str,
        verification_time_range: slice,
        months: List[int],
        output_dir: str = '.',
        plot_file_suffix: str = None,
        map_suffix: str = '',
        hemisphere: str = "north",
        max_lat: float = 75,
        model_label: str = None,
        model_label_size: int = 17,
        freq_levels: List[int] = np.arange(1, 45.1, 2),
        freq_ticks: List[int] = np.arange(5, 45.1, 5),
        freq_cmap: str = 'ocean_r',
        std_levels: List[int] = np.arange(2, 11.1, 1),
        std_ticks: List[int] = np.arange(2, 11.1, 1),
        std_cmap: str = 'Greens',
        average_params: dict = None,
        spatial_correlation_params: dict = None,
        rmse_params: dict = None
):
    
   
    # create output directory
    os.makedirs(output_dir, exist_ok=True)

    ####    CALCULATION    #####
    # cache filename 
    fcst_blocking_freq_cache = f'{output_dir}/AGP_Blocking_Index_fcst_{plot_file_suffix}.nc'
    if os.path.exists(fcst_blocking_freq_cache):
        print(f'Loading {fcst_blocking_freq_cache}...')
        fcst_blocking = xr.open_dataarray(fcst_blocking_freq_cache)
    else:
        print(f'calculating blocking and storing in {fcst_blocking_freq_cache}')
        # extract forecast dataarrays, fix time dimensions, grab storm forecasted time, eliminate singleton dimensions, scale heights to dekameters
        fcst_ll_z500 = selection_function(simulation_dir)
        fcst_ll_z500 = fcst_ll_z500.resample(time='1D').mean()
        fcst_ll_z500 = fcst_ll_z500.sel(time=fcst_ll_z500['time.month'].isin(months))
        fcst_blocking = AGP_Blocking_Index(fcst_ll_z500, hemisphere=hemisphere, target_file=fcst_blocking_freq_cache)
    fcst_blocking_count = count_blocking(fcst_blocking).mean('time')
    fcst_blocking_std = count_blocking(fcst_blocking).std('time')

    if map_suffix != '':
        map_cache_file = f'{output_dir}/blocking_freq_{map_suffix}.nc'
        print(f'saving blocking frequency to {map_cache_file}')
        fcst_blocking_count.to_netcdf(map_cache_file)

    # do the same for the observed comparison
    # cache filename
    verif_blocking_freq_cache = f'{output_dir}/AGP_Blocking_Index_verif_{plot_file_suffix}.nc'
    if os.path.exists(verif_blocking_freq_cache):
        print(f'Loading {verif_blocking_freq_cache}...')
        verif_blocking = xr.open_dataarray(verif_blocking_freq_cache)
    else:
        verif_ll_z500 = xr.open_dataset(verification_file)['z'].sel(time=verification_time_range).rename({'longitude':'lon','latitude':'lat'}).squeeze()
        verif_ll_z500 = verif_ll_z500.resample(time='1D').mean()[:,::-1,:]
        verif_ll_z500 = verif_ll_z500.sel(time=verif_ll_z500['time.month'].isin(months))
        verif_blocking = AGP_Blocking_Index(verif_ll_z500, hemisphere=hemisphere, target_file=verif_blocking_freq_cache)
    verif_blocking_count = count_blocking(verif_blocking).mean('time')
    verif_blocking_std = count_blocking(verif_blocking).std('time')

    #####    CALCULATING METRICS    #####
    def spatial_correlation(data1, data2, region, extra_suffix='', yticks=np.arange(0,1.1,0.2)):
        # TODO calculate and plot correlation
        data1 = cyclic_select(data1.sel(lat=region['lat']), region['lon'])
        data2 = cyclic_select(data2.sel(lat=region['lat']), region['lon'])

        # Calculate correlation
        corr = np.corrcoef(data1.values.flatten(), data2.values.flatten())[0,1]
        print(f'Correlation: {corr}')

        # plot correlations 
        fig, ax = plt.subplots(1, 1, figsize=(1,5))
        ax.set_facecolor('lightgray')
        ax.yaxis.tick_right()  # Move the y-ticks to the right side
        ax.yaxis.set_label_position("right")
        ax.plot(1, corr, '^', color='black', markersize=10)
        ax.set_xlim(.9,1.1)
        ax.set_ylim(yticks[0],yticks[-1])
        ax.set_xticks([1])
        ax.set_yticks(yticks)
        ax.set_xticklabels(['DLESM'], fontsize=12,rotation='vertical')
        print(f'Saving blocking correlation plot to blocking_correlation_{plot_file_suffix}{extra_suffix}.png')
        fig.tight_layout()
        fig.savefig(f'{output_dir}/blocking_correlation_{plot_file_suffix}{extra_suffix}.png', dpi=300)

        return 

    def calculate_average(data1, data2, region, extra_suffix='', yticks=np.arange(0,0.031,0.005)):
        # TODO calculate and plot average
        data1 = cyclic_select(data1.sel(lat=region['lat']), region['lon'])
        data2 = cyclic_select(data2.sel(lat=region['lat']), region['lon'])

        # Calculate average
        avg1 = data1.mean().values
        avg2 = data2.mean().values

        print(plot_file_suffix)
        # plot averages
        fig, ax = plt.subplots(1, 1, figsize=(1.25,5))
        ax.set_facecolor('lightgray')
        ax.yaxis.tick_right()  # Move the y-ticks to the right side
        ax.yaxis.set_label_position("right")
        ax.plot(1, avg1, '^', color='black', markersize=10)
        # ax.plot(1, avg2, '*', color='blue', markersize=10)
        ax.set_xlim(.9,1.1)
        ax.set_ylim(yticks[0],yticks[-1])
        ax.set_xticks([1])
        ax.set_yticks(yticks)
        ax.set_xticklabels(['DLESM'], fontsize=12,rotation='vertical')
        print(f'Saving blocking average plot to blocking_average_{plot_file_suffix}{extra_suffix}.png')
        fig.tight_layout()
        fig.savefig(f'{output_dir}/blocking_freq_average_{plot_file_suffix}{extra_suffix}.png', dpi=300)

        return
    
    def calculate_rmse(data1, data2, region, extra_suffix='',yticks=np.arange(0,0.026,0.005)):
        # TODO calculate and plot rmse
        data1 = cyclic_select(data1.sel(lat=region['lat']), region['lon'])
        data2 = cyclic_select(data2.sel(lat=region['lat']), region['lon'])

        # Calculate RMSE
        rmse = np.sqrt(np.mean((data1 - data2)**2)).values
        print(f'RMSE: {rmse}')

        # plot rmse
        fig, ax = plt.subplots(1, 1, figsize=(1.25,5))
        ax.set_facecolor('lightgray')
        ax.yaxis.tick_right()  # Move the y-ticks to the right side
        ax.yaxis.set_label_position("right")
        ax.plot(1, rmse, '^', color='black', markersize=10)
        ax.set_xlim(.9,1.1)
        ax.set_ylim(yticks[0],yticks[-1])
        ax.set_xticks([1])
        ax.set_yticks(yticks)
        ax.set_xticklabels(['DLESM'], fontsize=12,rotation='vertical')
        print(f'Saving blocking RMSE plot to blocking_rmse_{plot_file_suffix}{extra_suffix}.png')
        fig.tight_layout()
        fig.savefig(f'{output_dir}/blocking_rmse_{plot_file_suffix}{extra_suffix}.png', dpi=300)
        return

    if average_params is not None:
        calculate_average(fcst_blocking_count,
                          verif_blocking_count,
                          **average_params)
    if spatial_correlation_params is not None:
        spatial_correlation(fcst_blocking_count,
                            verif_blocking_count,
                            **spatial_correlation_params)
    if rmse_params is not None:
        calculate_rmse(fcst_blocking_count,
                       verif_blocking_count,
                       **rmse_params)

    #####    PLOTTING MAPS   #####
    def init_polar_axs(hemisphere):

        # initialize figure and gridspec
        fig = plt.figure(figsize=(10,5))
        gs = GridSpec(1, 2)
        proj=ccrs.NorthPolarStereo(central_longitude=0) if hemisphere == "north" else ccrs.SouthPolarStereo(central_longitude=0)

        def draw_circle(ax, theta, center_x=0.5, center_y=0.5):
            center, radius = [center_x, center_y], 0.5
            # Adjust theta to span only half the circle
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            ax.set_boundary(circle, transform=ax.transAxes)
            return ax
        
        # function to cleanly write a label in the center of the north pole where 
        # blocking metric is undefined
        def write_center_label(ax, label):
            # proj = ccrs.NorthPolarStereo(central_longitude=0)
            projx1, projy1 = proj.transform_point(0, 90 if hemisphere == 'north' else -90, ccrs.Geodetic()) #get proj coord of (lon,lat)
            def compute_radius(ortho, radius_degrees, lon=0):
                lat = 90 if hemisphere == 'north' else -90
                phi1 = lat + radius_degrees if lat <= 0 else lat - radius_degrees
                _, y1 = ortho.transform_point(lon, phi1, ccrs.PlateCarree())
                return abs(y1)
            ax.add_patch(mpatches.Circle(xy=[projx1, projy1], radius=compute_radius(proj,90-max_lat), color='white', \
                                alpha=1, transform=proj, zorder=25))
            ax.add_patch(mpatches.Circle(xy=[projx1, projy1], radius=compute_radius(proj,90-max_lat), edgecolor='black', \
                    facecolor='none', transform=proj, zorder=25))
            # add label 
            ax.text(projx1, projy1, label, transform=proj, ha='center', va='center', fontsize=model_label_size, zorder=30, color='black')
            return ax
        
        # format first axis
        ax1 = fig.add_subplot(gs[0], projection=proj)
        ax1.set_extent([0, 360, 45, 90] if hemisphere == 'north' else [0, 360, -45, -90], ccrs.PlateCarree())
        # ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
        #               linewidth=2, color='gray', alpha=0.5, linestyle='--',
        #               xlocs=[0,60,120,180,-120,-60], ylocs=[])
        ax1.coastlines(resolution='50m', linewidth=0.6, zorder=20)
        ax1 = write_center_label(ax1, model_label)
        ax1 = draw_circle(ax1, np.linspace(0, 2*np.pi, 100),.5, .5)

        # format second axis
        ax2 = fig.add_subplot(gs[1], projection=proj)#ccrs.NorthPolarStereo(central_longitude=0) if hemisphere == "north" else ccrs.SouthPolarStereo(central_longitude=0))
        ax2.set_extent([0, 360, 45, 90] if hemisphere == 'north' else [0, 360, -45, -90], ccrs.PlateCarree())
        # ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
        #               linewidth=2, color='gray', alpha=0.5, linestyle='--',
        #               xlocs=[0,60,120,180,-120,-60], ylocs=[])
        ax2.coastlines(resolution='50m', linewidth=0.6)
        ax2 = write_center_label(ax2, 'ERA5')
        ax2 = draw_circle(ax2, np.linspace(0, 2*np.pi, 100),.5, .5)

        return fig, [ax1, ax2]
    def _global_plot(ax, data, lon, lat, levels, cmap):
        data, lon = add_cyclic_point(data, coord=lon)
        # plot map of global data with central longitude 180
        img = ax.contourf(
            lon, lat, data,
            transform=ccrs.PlateCarree(), 
            cmap=cmap,
            extend="both",
            levels=levels,
        )
        _ = ax.contour(
            lon, lat, data,
            transform=ccrs.PlateCarree(),
            colors='black',
            linewidths=0.5,
            levels=levels,
        )
        return img
    
    # frequency plot 
    fig, axs = init_polar_axs(hemisphere)
    img = _global_plot(axs[0], fcst_blocking_count, fcst_blocking_count.lon, fcst_blocking_count.lat, levels=freq_levels, cmap=freq_cmap)
    _ = _global_plot(axs[1], verif_blocking_count, verif_blocking_count.lon, verif_blocking_count.lat, levels=freq_levels, cmap=freq_cmap)
    
    # add horizontal colorbar
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.075, 0.7, 0.035])
    cbar = fig.colorbar(img, cax=cbar_ax, orientation='horizontal', ticks=freq_ticks)
    cbar_ax.set_title('Blocks / Day', fontsize=15)

    #save
    plot_file = f'{output_dir}/blocking_freq_{plot_file_suffix}.svg'
    print(f'Saving verif blocking frequency plot to {plot_file}')
    fig.savefig(plot_file,dpi=300)

    # std plot 
    fig, axs = init_polar_axs(hemisphere)
    img = _global_plot(axs[0], fcst_blocking_std, fcst_blocking_std.lon, fcst_blocking_std.lat, 
                       cmap=std_cmap, levels=std_levels)
    _ = _global_plot(axs[1], verif_blocking_std, verif_blocking_std.lon, verif_blocking_std.lat,
                        cmap=std_cmap, levels=std_levels)
    
    # add horizontal colorbar
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.075, 0.7, 0.035])
    cbar = fig.colorbar(img, cax=cbar_ax, orientation='horizontal', ticks=std_ticks)
    cbar_ax.set_title('Event STD', fontsize=15)

    #save
    plot_file_std = f'{output_dir}/blocking_std_{plot_file_suffix}.png'
    print(f'Saving blocking frequency plot to {plot_file_std}')
    fig.savefig(plot_file_std,dpi=300)

def blocking_nam_sam(
    verif_blocking: str,
    dlesm_blocking: str,
    blocking_levels: np.ndarray,
    blocking_colormap: str,
    plot_file: str,
):
    """
    Midlatitude figure compiles blocking, NAM and SAM metrics for ERA5, DLESM, and CMIP6 models.
    """
    def init_polar_axs(axs):

        # initialize figure and gridspec
        fig = plt.figure(figsize=(12,12))
        gs = GridSpec(3,3)

        def draw_circle(ax, theta, center_x=0.5, center_y=0.5):
            center, radius = [center_x, center_y], 0.5
            # Adjust theta to span only half the circle
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            ax.set_boundary(circle, transform=ax.transAxes)
            return ax
        
        # function to cleanly write a label in the center of the north pole where 
        # blocking metric is undefined
        def write_center_label(ax, label):
            projx1, projy1 = proj.transform_point(0, 90 if hemisphere == 'north' else -90, ccrs.Geodetic()) #get proj coord of (lon,lat)
            def compute_radius(ortho, radius_degrees, lon=0):
                lat = 90 if hemisphere == 'north' else -90
                phi1 = lat + radius_degrees if lat <= 0 else lat - radius_degrees
                _, y1 = ortho.transform_point(lon, phi1, ccrs.PlateCarree())
                return abs(y1)
            ax.add_patch(mpatches.Circle(xy=[projx1, projy1], radius=compute_radius(proj,15), color='white', \
                                alpha=1, transform=proj, zorder=25))
            ax.add_patch(mpatches.Circle(xy=[projx1, projy1], radius=compute_radius(proj,15), edgecolor='black', \
                    facecolor='none', transform=proj, zorder=25))
            # add label 
            ax.text(projx1, projy1, label, transform=proj, ha='center', va='center', fontsize=17, zorder=30, color='black')
            return ax
        
        geo_axis = [0,1,6,7]
        axs = []
        for i in geo_axis:

            # blocking and nam in norhtern hemisphere, sam in southern hemisphere
            if i== 6 or i == 7:
                hemisphere = 'south'
            else:
                hemisphere = 'north'
            proj = ccrs.NorthPolarStereo(central_longitude=0) if hemisphere == "north" else ccrs.SouthPolarStereo(central_longitude=0)
            ax = fig.add_subplot(gs[i], projection=proj)
            ax.set_extent([0, 360, 45, 90] if hemisphere == 'north' else [0, 360, -45, -90], ccrs.PlateCarree())
            ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                        linewidth=2, color='gray', alpha=0.5, linestyle='--',
                        xlocs=[0,60,120,180,-120,-60], ylocs=[])
            ax.coastlines(resolution='50m', linewidth=0.6, zorder=20)
            ax = write_center_label(ax, 'DLESM' if i%3 == 0 else 'ERA5')
            ax = draw_circle(ax, np.linspace(0, 2*np.pi, 100),.5, .5)
            axs.append(ax)

        return fig, axs
    
    # initiaize axes 
    fig, axs = init_polar_axs('north')
    # push subplots to right to make room for colorbars
    fig.tight_layout()
    fig.subplots_adjust(left=0.15)

    # plot blocking maps 
    def plot_blocking(data, ax, ):
        data_cyc, lon = add_cyclic_point(data, coord=data.lon)
        ax.contour(
            lon, data.lat, data_cyc,
            transform=ccrs.PlateCarree(),
            levels=blocking_levels,
            colors='black',
            linewidths=.5,
        )
        img = ax.contourf(
            lon, data.lat, data_cyc,
            transform=ccrs.PlateCarree(), 
            cmap=blocking_colormap,
            extend="both",
            levels=blocking_levels,
        )
        return img
    _ = plot_blocking(xr.open_dataarray(dlesm_blocking), axs[0])
    img_blocking = plot_blocking(xr.open_dataarray(verif_blocking), axs[1])
    # place colorbar
    cbar_ax_blocking = fig.add_axes([0.21875, 0.64, 0.4, 0.01])
    cbar_blocking = fig.colorbar(img_blocking, cax=cbar_ax_blocking, orientation='horizontal', ticks=blocking_levels)
    cbar_ax_blocking.set_title('Blocks per Day', fontsize=15)

    
    # plot nam maps

    # plot sam maps 

    # plot taylor diagrams

    # tighten and save figure
    
    logger.info(f'Saving blocking, NAM, and SAM figure to {plot_file}.')
    fig.savefig(plot_file,dpi=300)

#endregion

if __name__=="__main__":

    # blocking_nam_sam(
    #     verif_blocking = '/home/disk/brume/nacc/WeeklyNotebooks/2024.06.10/FigureScripts/blocking_cache/blocking_freq_nh_map-verif.nc',
    #     dlesm_blocking = '/home/disk/brume/nacc/WeeklyNotebooks/2024.05.06/FigureScripts/blocking_cache/blocking_freq_nh_map_40yr.nc',
    #     blocking_levels = np.arange(0.0, 0.101, 0.01),
    #     blocking_colormap = get_custom_cmap('plasma_r'),
    #     plot_file = '/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/FigureScripts/blocking_nam_sam.pdf',
    # )

    # blocking_frequency_cmip(**PARAMS_NH_CESM)
    blocking_frequency_cmip(**PARAMS_NH_CESM_40yr)
    # blocking_frequency_cmip(**PARAMS_NH_MPI)
    # blocking_frequency_cmip(**PARAMS_NH_MPI_HR)
    blocking_frequency_cmip(**PARAMS_NH_MPI_HR_40yr)
    # blocking_frequency_cmip(**PARAMS_NH_gfdl)
    blocking_frequency_cmip(**PARAMS_NH_gfdl_40yr)
    # blocking_frequency_cmip(**PARAMS_NH_HadGEM)
    blocking_frequency_cmip(**PARAMS_NH_HadGEM_40yr)
    # blocking_frequency_cmip(**PARAMS_NH_IPSL)

    # blocking_frequency(**PARAMS_NH)
    blocking_frequency(**PARAMS_NH_40yr)
    # blocking_frequency(**PARAMS_NH_DJF)
    # blocking_frequency(**PARAMS_NH_MAM)
    # blocking_frequency(**PARAMS_NH_JJA)
    # blocking_frequency(**PARAMS_NH_SON)
