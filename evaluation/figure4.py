import numpy as np
import pandas as pd
import xarray as xr
import pprint
from typing import List, Tuple, Union
import sys 
import copy
import os
import argparse
from dask.diagnostics import ProgressBar
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
sys.path.append('./')
import evaluation.evaluators as ev
from matplotlib.gridspec import GridSpec
from figure1 import get_year
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import cartopy.feature as cft 
import logging
from drift import linear_fit
logging.basicConfig(level=logging.INFO)
from blocking import get_custom_cmap
from taylor_diagrams import TaylorDiagram

# Creates maps of blocking frequency, and spatial distribution of annular modes. Also plots taylor diagrams for 
# blocking and annular modes patterns from CMIP files

# NOTE - this script requires prepared CMIP6 output files which are not included in this repository. 

FIGURE4_PARAMS = {
    'params_blocking' : {
        'forecast_file': 'path/to/100yr/forecast/100yr_forecast_filename', # path to forecast file
        'forecast_time_range': slice('2070-01-01', '2110-12-31'),
        'verification_file' : '/path/to/verif/era5_z500.nc',
        'verification_time_range': slice('1970-01-01', '2010-12-31'),
        'months':[1,2,3,4,5,6,7,8,9,10,11,12],
        'hemisphere': 'north',
        # 'freq_levels': np.arange(1, 15.1, 1),
        'freq_levels': np.arange(0.0, 0.101, 0.01),
        # 'freq_ticks': np.arange(1, 15.1, 2),
        'freq_ticks': np.arange(0.0, 0.101, 0.02),
        'freq_cmap': get_custom_cmap('plasma_r'),
        'output_dir' : '/path/to/cache/dir/dir_name',
        'plot_file_suffix': 'nh_40yr',
        'map_suffix': 'nh_map_40yr',
    },
    # 
    'params_taylor_blocking': {
        'simulation_dicts' : [
            {'model_name': 'DL$\it{ESy}$M', 'file_path': 'path/to/blocking_cache/blocking_freq_nh_map_40yr.nc','cmip':False, 'var_name': 'z500',
                'marker_specs': {"labelColor": "k","symbol": "o", "size": 11,"faceColor": "r","edgeColor": "r",}},
            {'model_name': 'CESM2', 'file_path': 'path/to/blocking_cache/blocking_freq_cesm_nh_map_40yr.nc','cmip':True, 'var_name': 'zg',
                'marker_specs': {"labelColor": "k","symbol": "D", "size": 11,"faceColor": "w","edgeColor": "b",}},
            {'model_name': 'GFDL', 'file_path': 'path/to/blocking_freq_gfdl_nh_map_40yr.nc','cmip':True, 'var_name': 'zg',
                'marker_specs': {"labelColor": "k","symbol": "o", "size": 11,"faceColor": "w","edgeColor": "g",}},
            {'model_name': 'HadGEM3', 'file_path': 'path/to/blocking_cache/blocking_freq_hadgem_nh_map_40yr.nc','cmip':True, 'var_name': 'zg',
                'marker_specs': {"labelColor": "k","symbol": ">", "size": 11,"faceColor": "w","edgeColor": "m",}},
            {'model_name': 'MPI', 'file_path': 'path/to/blocking_cache/blocking_freq_mpi-esp1-2-hr_nh_map_40yr.nc','cmip':True, 'var_name': 'zg',
                'marker_specs': {"labelColor": "k","symbol": "<", "size": 11,"faceColor": "w","edgeColor": "c",}},
        ],
        'ref_file' : 'path/to/blocking_cache/blocking_freq_nh_map-verif.nc',
        'ref_var_name' : 'z',
        'ylabel' : 'STD Ratio',
    },
    'params_NAM' : {
        'dlesm_pattern_file':'path/to/nam_cache/eof1_monthly_hpx64.nc',   
        'obs_pattern_file':'path/to/nam_cache/eof1_monthly_era5.nc',
        # 'levels': np.arange(-54.5, 54.5, 8.5),
        'levels': np.arange(-59.5, 60, 8.5),
        'colorbar_ticks': np.arange(-51, 52, 17),
        'cmap': get_custom_cmap('bwr', middle=True),
    },
    'params_taylor_NAM' : {
        'simulation_dicts' : [
            {'model_name': 'DLE$\it{Sy}$M', 'file_path': 'path/to/nam_cache/eof1_monthly_hpx64.nc','cmip':False, 'var_name': 'eof',
                'marker_specs': {"labelColor": "k","symbol": "o", "size": 11,"faceColor": "r","edgeColor": "r",}},
            {'model_name': 'CESM2', 'file_path': 'path/to/nam_cache/eof1_slp_CESM2.nc','cmip':True, 'var_name': 'eof',
                'marker_specs': {"labelColor": "k","symbol": "D", "size": 11,"faceColor": "w","edgeColor": "b",}},
            {'model_name': 'GFDL-CM4', 'file_path': 'path/to/nam_cache/eof1_slp_GFDL-CM4.nc','cmip':True, 'var_name': 'eof',
                'marker_specs': {"labelColor": "k","symbol": "o", "size": 11,"faceColor": "w","edgeColor": "g",}},
            {'model_name': 'HadGEM3-GC31-LL', 'file_path': 'path/to/nam_cache/eof1_slp_HadGEM3-GC31-LL.nc','cmip':True, 'var_name': 'eof',
                'marker_specs': {"labelColor": "k","symbol": ">", "size": 11,"faceColor": "w","edgeColor": "m",}},
            {'model_name': 'MPI-ESM1-2-HR', 'file_path': 'path/to/nam_cache/eof1_slp_MPI-ESM1-2-HR.nc','cmip':True, 'var_name': 'eof',
                'marker_specs': {"labelColor": "k","symbol": "<", "size": 11,"faceColor": "w","edgeColor": "c",}},
        ],
        'ref_file_z1000' : 'path/to/nam_cache/eof1_monthly_era5.nc',
        'ref_file_slp' : 'path/to/nam_cache/eof1_monthly_slp_era5.nc',
        'ref_var_name': 'eof',
        'ylabel' : 'STD Ratio',
    },
    'params_SAM' : {
        'dlesm_pattern_file':'path/to/sam_cache/eof1_monthly_SAM500_hpx64.nc',   
        'obs_pattern_file':'path/to/sam_cache/eof1_monthly_SAM500_era5.nc',
        # 'levels': np.arange(-64, 64, 8.5),
        'levels': np.arange(-64, 66, 8),
        'colorbar_ticks': np.arange(-64, 66, 16),
        'cmap': get_custom_cmap('bwr', middle=True),
        'hemisphere': 'south',
    },
    'params_taylor_SAM' : {
        'simulation_dicts' : [
            {'model_name': 'DLE$\it{Sy}$M', 'file_path': 'path/to/sam_cache/eof1_monthly_SAM500_hpx64.nc','cmip':False, 'var_name': 'eof',
                'marker_specs': {"labelColor": "k","symbol": "o", "size": 11,"faceColor": "r","edgeColor": "r",}},
            {'model_name': 'CESM2', 'file_path': 'path/to/sam_cache/eof1_SAM500_CESM2.nc','cmip':True, 'var_name': 'eof',
                'marker_specs': {"labelColor": "k","symbol": "D", "size": 11,"faceColor": "w","edgeColor": "b",}},
            {'model_name': 'GFDL-CM4', 'file_path': 'path/to/sam_cache/eof1_SAM500_GFDL-CM4.nc','cmip':True, 'var_name': 'eof',
                'marker_specs': {"labelColor": "k","symbol": "o", "size": 11,"faceColor": "w","edgeColor": "g",}},
            {'model_name': 'HadGEM3-GC31-LL', 'file_path': 'path/to/sam_cache/eof1_SAM500_HadGEM3-GC31-LL.nc','cmip':True, 'var_name': 'eof',
                'marker_specs': {"labelColor": "k","symbol": ">", "size": 11,"faceColor": "w","edgeColor": "m",}},
            {'model_name': 'MPI-ESM1-2-HR', 'file_path': 'path/to/sam_cache/eof1_SAM500_MPI-ESM1-2-HR.nc','cmip':True, 'var_name': 'eof',
                'marker_specs': {"labelColor": "k","symbol": "<", "size": 11,"faceColor": "w","edgeColor": "c",}},
        ],
        'ref_file' : 'path/to/sam_cache/eof1_monthly_SAM500_era5.nc',
        'ref_var_name': 'eof',
        'ylabel' : 'STD Ratio',
    },
    'save_params': {
        'fname': './figure4.png',
        'dpi': 400,
        'bbox_inches': 'tight',
    },
}

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
    levels_c = []
    for l in levels: 
        if l != 0: levels_c.append(l)
    _ = ax.contour(
        lon, lat, data,
        transform=ccrs.PlateCarree(),
        colors='black',
        linewidths=0.5,
        levels=levels_c,
    )
    return img

def blocking_frequency(
        fig, dlesm_ax_loc, obs_ax_loc,
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
):
    
    # create output directory
    os.makedirs(output_dir, exist_ok=True)

    #####    DATA PREPARATION    #####
    # initialize evaluator for z500 forecast
    fcst_z500 = ev.EvaluatorHPX(
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

    #####    PLOTTING MAPS   #####
    def init_polar_axs(loc1, loc2 , hemisphere):

        # initialize figure and gridspec
        # fig = plt.figure(figsize=(10,5))
        # gs = GridSpec(1, 2)
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
            # ax.text(projx1, projy1, label, transform=proj, ha='center', va='center', fontsize=17, zorder=30, color='black')
            return ax
        
        # format first axis
        ax1 = fig.add_subplot(loc1, projection=proj)
        ax1.set_extent([0, 360, 45, 90] if hemisphere == 'north' else [0, 360, -45, -90], ccrs.PlateCarree())
        # ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
        #               linewidth=2, color='gray', alpha=0.5, linestyle='--',
        #               xlocs=[0,60,120,180,-120,-60], ylocs=[])
        ax1.coastlines(resolution='50m', linewidth=0.6, zorder=20)
        ax1 = write_center_label(ax1, 'DLESM')
        ax1 = draw_circle(ax1, np.linspace(0, 2*np.pi, 100),.5, .5)

        # format second axis
        ax2 = fig.add_subplot(loc2, projection=proj)#ccrs.NorthPolarStereo(central_longitude=0) if hemisphere == "north" else ccrs.SouthPolarStereo(central_longitude=0))
        ax2.set_extent([0, 360, 45, 90] if hemisphere == 'north' else [0, 360, -45, -90], ccrs.PlateCarree())
        # ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
        #               linewidth=2, color='gray', alpha=0.5, linestyle='--',
        #               xlocs=[0,60,120,180,-120,-60], ylocs=[])
        ax2.coastlines(resolution='50m', linewidth=0.6)
        ax2 = write_center_label(ax2, 'ERA5')
        ax2 = draw_circle(ax2, np.linspace(0, 2*np.pi, 100),.5, .5)

        return [ax1, ax2]
        
    # frequency plot 
    axs = init_polar_axs(dlesm_ax_loc, obs_ax_loc, hemisphere)
    img = _global_plot(axs[0], fcst_blocking_count, fcst_blocking_count.lon, fcst_blocking_count.lat, levels=freq_levels, cmap=freq_cmap)
    _ = _global_plot(axs[1], verif_blocking_count, verif_blocking_count.lon, verif_blocking_count.lat, levels=freq_levels, cmap=freq_cmap)
    
    # add horizontal colorbar beneath axs
    pos_ax = axs[1].get_position()
    cbar_ax = fig.add_axes([pos_ax.x0+.041, pos_ax.y0-.03, 0.35, 0.007])
    cbar = fig.colorbar(img, cax=cbar_ax, orientation='horizontal', ticks=freq_ticks)
    cbar_ax.set_title('Blocks / Day', fontsize=15)

    return axs[0], axs[1]

def am_pattern(fig, dlesm_loc, obs_loc, dlesm_pattern_file, obs_pattern_file, levels, cmap, hemisphere='north', colorbar_ticks=None, add_variance=True):

    # load nam patterns
    dlesm_pattern = xr.open_dataset(dlesm_pattern_file)
    obs_pattern = xr.open_dataset(obs_pattern_file)

    #####    PLOTTING MAPS   #####
    def init_polar_axs(loc1, loc2 , hemisphere):

        # initialize figure and gridspec
        # fig = plt.figure(figsize=(10,5))
        # gs = GridSpec(1, 2)
        proj=ccrs.NorthPolarStereo(central_longitude=0) if hemisphere == "north" else ccrs.SouthPolarStereo(central_longitude=180)

        def draw_circle(ax, theta, center_x=0.5, center_y=0.5):
            center, radius = [center_x, center_y], 0.5
            # Adjust theta to span only half the circle
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            ax.set_boundary(circle, transform=ax.transAxes)
            return ax
        
        # format first axis
        ax1 = fig.add_subplot(loc1, projection=proj)
        ax1.set_extent([0, 360, 20, 90] if hemisphere == 'north' else [0, 360, -20, -90], ccrs.PlateCarree())
        ax1.coastlines(resolution='50m', linewidth=0.6, zorder=20)
        ax1 = draw_circle(ax1, np.linspace(0, 2*np.pi, 100),.5, .5)

        # format second axis
        ax2 = fig.add_subplot(loc2, projection=proj)#ccrs.NorthPolarStereo(central_longitude=0) if hemisphere == "north" else ccrs.SouthPolarStereo(central_longitude=0))
        ax2.set_extent([0, 360, 20, 90] if hemisphere == 'north' else [0, 360, -20, -90], ccrs.PlateCarree())
        ax2.coastlines(resolution='50m', linewidth=0.6)
        ax2 = draw_circle(ax2, np.linspace(0, 2*np.pi, 100),.5, .5)

        return [ax1, ax2]

    ax_dlesm, ax_obs = init_polar_axs(dlesm_loc, obs_loc, hemisphere)

    # plot dlesm pattern
    img = _global_plot(ax_dlesm, dlesm_pattern['eof'], dlesm_pattern['longitude'], dlesm_pattern['latitude'], levels=levels, cmap=cmap)
    # plot obs pattern
    _ = _global_plot(ax_obs, obs_pattern['eof'], obs_pattern['longitude'], obs_pattern['latitude'], levels=levels, cmap=cmap)

    # add explained variance to plot
    if add_variance:
        ax_dlesm.text(.93, .91, f'{dlesm_pattern["expvar_ratio"].values[0]*100:.1f}%', transform=ax_dlesm.transAxes, ha='center', fontsize=15,)
        ax_obs.text(.93, .91, f'{obs_pattern["expvar_ratio"].values[0]*100:.1f}%', transform=ax_obs.transAxes, ha='center', fontsize=15,)

    # add horizontal colorbar beneath axs
    pos_ax = ax_obs.get_position()
    cbar_ax = fig.add_axes([pos_ax.x0+.041, pos_ax.y0-.03, 0.35, 0.007])
    cbar = fig.colorbar(img, cax=cbar_ax, orientation='horizontal', ticks=levels if colorbar_ticks is None else colorbar_ticks)
    cbar_ax.set_title('Meters', fontsize=15)

    return ax_dlesm, ax_obs

def nam_taylor_smip_comp(
    simulation_dicts: List[dict],
    ref_file_z1000: str,
    ref_file_slp: str,
    ref_var_name: str,
    ax: plt.Axes,
    ylabel: str,
):
    """
    NAM is calculated using SLP in CMIP output and z1000 in DLESM output. This 
    function calculates taylor sats both and plots them on the same plot
    """

    # empty param dicts
    z1000_taylor_params = {}
    slp_taylor_params = {}

    # empty sim lists
    z1000_taylor_params['simulation_dicts'] = []
    slp_taylor_params['simulation_dicts'] = []
    for sim in simulation_dicts:
        if sim['cmip']:
            slp_taylor_params['simulation_dicts'].append(sim)
        else:
            z1000_taylor_params['simulation_dicts'].append(sim)
    
    # asign ref files 
    z1000_taylor_params['ref_file'] = ref_file_z1000
    slp_taylor_params['ref_file'] = ref_file_slp

    # var name
    z1000_taylor_params['ref_var_name'] = ref_var_name
    slp_taylor_params['ref_var_name'] = ref_var_name

    # axis assignment
    z1000_taylor_params['ax'] = ax
    slp_taylor_params['ax'] = ax

    # ylabel
    z1000_taylor_params['ylabel'] = ylabel
    slp_taylor_params['ylabel'] = ylabel

    # plot taylor diagrams
    ax = TaylorDiagram(**z1000_taylor_params)
    ax = TaylorDiagram(**slp_taylor_params)

    return ax

def plot_figure4(
    params_blocking: dict,
    params_taylor_blocking: dict,    
    params_NAM: dict,
    params_SAM: dict,
    params_taylor_NAM: dict,
    params_taylor_SAM: dict,
    save_params: dict   
):
    
    def scootch_right(ax, scootch=.075):
        pos = ax.get_position()
        ax.set_position([pos.x0 + scootch, pos.y0, pos.width, pos.height])
        return ax
    
    def shrink_to_left(ax, shrink=.9):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width*shrink, pos.height*shrink])
        return ax
    
    def stretch_up(ax, stretch=1):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width*stretch, pos.height*stretch])
        return ax
    
    def show_legend(ax):
        legend = ax.get_legend()
        legend.set_visible(True)
        legend.set_alpha(1)
        legend.set_bbox_to_anchor((0.38, .96))
        for text in legend.get_texts():
            text.set_fontsize(14)
        return ax
    def place_line(ax, loc=1.05):
        ax.axhline(loc, xmax=.713,color='black', linewidth=1.5)
        return ax
    def add_subplot_label(label, ax, x=0.1, y=.98):
        ax.text(x, y, label, transform=ax.transAxes, fontsize=18, fontweight='bold')
    
    fig = plt.figure(figsize=(13, 15))
    gs = GridSpec(3, 7, figure=fig)

    # Adjust the spacing between rows
    fig.subplots_adjust(wspace=0.2, hspace=0.3)

    # plot blocking frequency maps, pass grid specs to locate axis
    ax_dlesm_blocking, ax_obs_blocking = blocking_frequency(fig, gs[0,2:4], gs[0,0:2], **params_blocking)
    ax_dlesm_blocking.set_title('DL$\it{ESy}$M', fontsize=20, pad=20)
    ax_obs_blocking.set_title('ERA5', fontsize=20, pad=20)
    # set sublot labels
    ax_dlesm_blocking = add_subplot_label('B', ax_dlesm_blocking)
    ax_obs_blocking = add_subplot_label('A', ax_obs_blocking)

    # initialize taylor diagram axis
    ax_taylor_blocking = fig.add_subplot(gs[0, 4:7])
    ax_taylor_blocking = TaylorDiagram(**params_taylor_blocking, ax=ax_taylor_blocking)
    # postion taylor diagram to the right and show legend, add subplot label
    ax_taylor_blocking = add_subplot_label('C',scootch_right(place_line(show_legend(ax_taylor_blocking))), x=-.15, y=.94)

    # plot nam for dlesm and obs
    ax_dlesm_NAM, ax_obs_NAM = am_pattern(fig, gs[1,2:4], gs[1, 0:2], **params_NAM)
    # set subplot labels
    ax_dlesm_NAM = add_subplot_label('E', ax_dlesm_NAM)
    ax_obs_NAM = add_subplot_label('D', ax_obs_NAM)

    # # initialize taylor diagram axis
    ax_taylor_NAM = fig.add_subplot(gs[1, 4:7])
    ax_taylor_NAM = nam_taylor_smip_comp(**params_taylor_NAM, ax=ax_taylor_NAM)
    # postion taylor diagram to the right and add subplot label
    ax_taylor_NAM = add_subplot_label('F',scootch_right(place_line(ax_taylor_NAM)), x=-.15, y=.94)
    # initilaize obs SAM axis 
    ax_dlesm_SAM, ax_obs_SAM = am_pattern(fig, gs[2,2:4], gs[2, 0:2], **params_SAM)

    # set subplot labels
    ax_dlesm_SAM = add_subplot_label('H', ax_dlesm_SAM)
    ax_obs_SAM = add_subplot_label('G', ax_obs_SAM)

    # initialize taylor diagram axis
    ax_taylor_SAM = fig.add_subplot(gs[2, 4:7])
    ax_taylor_SAM = TaylorDiagram(**params_taylor_SAM, ax=ax_taylor_SAM)
    # postion taylor diagram to the right, add subplot label
    ax_taylor_SAM = add_subplot_label('I',scootch_right(place_line(ax_taylor_SAM)), x=-.15, y=.94)

    # save plot 
    fig.savefig(**save_params)


    # blocking_frequency(**params_blocking)

if __name__ == '__main__':
    plot_figure4(**FIGURE4_PARAMS)