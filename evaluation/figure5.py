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
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
sys.path.append('./')
import cartopy.feature as cfeature
import evaluation.evaluators as ev
from matplotlib.gridspec import GridSpec
from figure1 import get_year
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath
import matplotlib.dates as mdates
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import cartopy.feature as cft 
import logging
from drift import linear_fit
import matplotlib.colors as mcolors 
logging.basicConfig(level=logging.INFO)
from monsoon import monsoon_hovmoller,dlesm_open,isccp_open, calculate_monsoon_index
from taylor_diagram_monsoon import hovmoller_taylor_diagrams

# this script contains code for generating Figure 5 from https://arxiv.org/pdf/2409.16247

# custom colormap for precipitation accumulation
def get_precip_cmap(cmap_base='Blues', white_levels=32, start_color=.15):

    # Create a colormap that goes from grey to white
    cmap1 = mcolors.LinearSegmentedColormap.from_list("mycmap", ['whitesmoke', 'whitesmoke'])

    # Get the 'YlOrRd' colormap
    cmap2 = plt.get_cmap(cmap_base)

    # Combine the two colormaps
    colors = np.vstack((cmap1(np.linspace(0, 1, white_levels)), cmap2(np.linspace(start_color, 1, 128))))
    cmap_combined = mcolors.LinearSegmentedColormap.from_list('colormap', colors)
    return cmap_combined

FIGURE5_PARAMS = dict(
    plot_ts_params = dict(
        verification_file="path/to/olr_ll/file/isccp_1deg_1d_HPX64_1983_2017_olr_ll.nc",
        forecast_file="path/to/forecast/file/remapped_forecast_olr_ll.nc",
        comp_time_period=slice('2017-01-01', '2017-06-30'),
        verif_climo_time_period=slice('1985-01-01','2014-12-31'),
        fcst_climo_time_period=slice('2085-01-01','2114-12-31'),
        open_func_verif=isccp_open,
        open_func_fcst=dlesm_open,
        caching_dir="path/to/cache_dir/monsoon_cache",
    ), 
    plot_climo_obs_params = dict(
        hov_params = dict(
            verification_file="path/to/olr_ll/file/isccp_1deg_1d_HPX64_1983_2017_olr_ll.nc",
            time_period=slice('1985-01-01','2014-12-31'),
            forecast=False,
            open_func=isccp_open,
            climatology=True,
            caching_dir="path/to/cache_dir/monsoon_cache",
        ),
        levels=np.arange(180, 305+20, 10),
        contour_levels=[245,],
    ),
    plot_climo_dlesm_params = dict(
        hov_params = dict(
            verification_file="path/to/olr_ll/file/dlesm_1deg_1d_HPX64_1983_2017_olr_ll.nc",
            time_period=slice('2085-01-01','2114-12-31'),
            forecast=True,
            open_func=dlesm_open,
            climatology=True,
            caching_dir="path/to/cache_dir/monsoon_cache",
        ),
        levels=np.arange(180, 305+20, 10),
        contour_levels=[245,],
    ),     
    taylor_params = dict(
        simulation_dicts = [
            {'model_name': 'DL$\it{ESy}$M', 'file_path': "path/to/cache_dir/monsoon_cache/dlesm_2085-2114_climo_hovmoller_scaled.nc",'cmip':False, 'var_name': 'olr',
                'marker_specs': {"labelColor": "k","symbol": "o", "size": 11,"faceColor": "r","edgeColor": "r",}},
            {'model_name': 'CESM2', 'file_path': "path/to/cache_dir/monsoon_cache/cesm_1985-2014_climo_hovmoller.nc",'cmip':True, 'var_name': 'olr',
                'marker_specs': {"labelColor": "k","symbol": "D", "size": 11,"faceColor": "w","edgeColor": "b",}},
            {'model_name': 'GFDL', 'file_path': "path/to/cache_dir/monsoon_cache/gfdl_1985-2014_climo_hovmoller.nc",'cmip':True, 'var_name': 'olr',
                'marker_specs': {"labelColor": "k","symbol": "o", "size": 11,"faceColor": "w","edgeColor": "g",}},
            {'model_name': 'HadGEM3', 'file_path': "path/to/cache_dir/monsoon_cache/hadgem_1985-2014_climo_hovmoller.nc",'cmip':True, 'var_name': 'olr',
                'marker_specs': {"labelColor": "k","symbol": ">", "size": 11,"faceColor": "w","edgeColor": "m",}},
            {'model_name': 'MPI', 'file_path': "path/to/cache_dir/monsoon_cache/mpi_1985-2014_climo_hovmoller.nc",'cmip':True, 'var_name': 'olr',
                'marker_specs': {"labelColor": "k","symbol": "<", "size": 11,"faceColor": "w","edgeColor": "c",}},
        ],
        ref_file="path/to/cache_dir/monsoon_cache/isccp_1985-2014_climo_hovmoller_scaled.nc",
        ylim=[0, 1.05],
        ylabel='STD Ratio',
        legend=False,
    ),
    plot_precip_accumulation_params_gt = dict(
        input_file='path/to/verif/precip/ll_truth_monsoon.nc',
        forecast=False,
        time='2017-06-12',
        levels=np.arange(0, 301, 30),
        data_cache="path/to/cache_dir/monsoon_cache/obs_precip_accumulation_map_2017.06.12.nc",
    ),
    plot_precip_accumulation_params_dlesm = dict(
        input_file='path/to/verif/precip/ll_dlesm_monsoon.nc',
        forecast=True,
        time='2017-06-12',
        levels=np.arange(0, 301, 30),
        data_cache="path/to/cache_dir/monsoon_cache/dlesm_precip_accumulation_map_2017.06.12.nc",
    ),
    savefig_params = dict(
        fname='/home/disk/brume/nacc/WeeklyNotebooks/2025.02.17/figure5.png',
        dpi=400,
        bbox_inches='tight',
    ),
)

def plot_hovmoller(fig, ax, hov_params, levels, contour_levels, return_quad=False, no_yticks=False, no_xticks=False):

    # calculate hovmoller pattern
    olr, time, lat = monsoon_hovmoller(
        **hov_params
    )
    # plot hovmoller
    cs = ax.contourf(lat,time,olr, cmap='RdBu_r', levels=levels)
    ax.contour(lat,time,olr, colors='black', levels=contour_levels, extend='both')
    ax.set_xlim(lat[0], lat[-1])
    date_format = mdates.DateFormatter('%b')
    ax.yaxis.set_major_formatter(date_format)
    ax.yaxis.set_major_locator(mdates.MonthLocator())
    ax.set_xticks(np.arange(lat[0], lat[-1]+1, 5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(axis='y', labelrotation=30)
    if no_yticks:
        ax.set_yticklabels([])
    if no_xticks:
        ax.set_xticklabels([])
    if return_quad:
        return fig, ax, cs
    else:
        return fig, ax

def plot_timeseries(fig, ax, verification_file, forecast_file, comp_time_period, verif_climo_time_period, fcst_climo_time_period, open_func_verif, open_func_fcst, caching_dir):

    # obs single year monsoon index
    obs_index, obs_index_time = calculate_monsoon_index(
        file=verification_file,
        forecast=False,
        time_range=comp_time_period,
        monsoon_index_cache=os.path.join(caching_dir, "obs_monsoon_index_single_year.nc"),
        detrend=False,
        smoothing=10,
        open_func=open_func_verif,
    )
    ax.plot(obs_index_time, obs_index, label='ISCCP', color='black', linewidth=2)
    # dlesm single year monsoon index
    fcst_index, fcst_index_time = calculate_monsoon_index(
        file=forecast_file,
        forecast=True,
        time_range=comp_time_period,
        monsoon_index_cache=os.path.join(caching_dir, "forecast_monsoon_index_single_year.nc"),
        detrend=False,
        smoothing=10,
        open_func=open_func_fcst,
    )
    ax.plot(fcst_index_time, fcst_index, label='DL$\it{ESy}$M', color='red',linewidth=2)
    # obs climo
    obs_index_climo, obs_index_climo_time = calculate_monsoon_index(
        file=verification_file,
        forecast=False,
        time_range=verif_climo_time_period,
        monsoon_index_cache=os.path.join(caching_dir, "obs_monsoon_index_climo.nc"),
        climatology=True,
        detrend=False,
        smoothing=10,
        open_func=open_func_verif,
    )
    ax.plot(obs_index_climo_time, obs_index_climo, label='ISCCP Clima', color='black',linewidth=2,linestyle='--')
    # dlesm climo
    fcst_index_climo, fcst_index_climo_time = calculate_monsoon_index(
        file=forecast_file,
        forecast=True,
        time_range=fcst_climo_time_period,
        monsoon_index_cache=os.path.join(caching_dir, "forecast_monsoon_index_climo.nc"),
        climatology=True,
        detrend=False,
        smoothing=10,
        open_func=open_func_fcst,
    )
    ax.plot(fcst_index_climo_time, fcst_index_climo, label='DL$\it{ESy}$M Clima', color='red',linewidth=2,linestyle='--')
    # style index plot
    def add_sim_subpanel(ax):
        ax = fig.add_axes([0.135, 0.555, 0.12, 0.12], projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE,lw=0.5,color='grey')
        box = [70, 100, 10, 25]
        ax.set_extent([box[0]-5, box[1]+5, box[2]-5, box[3]+5], crs=ccrs.PlateCarree())
        rect = patches.Rectangle((box[0], box[2]), box[1]-box[0], box[3]-box[2],
                                linewidth=1, edgecolor='r', facecolor='r', alpha=0.2, transform=ccrs.PlateCarree())
        ax.add_patch(rect)
    date_format = mdates.DateFormatter('%b')
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    # ax.tick_params(axis='y', labelrotation=30)
    ax.grid()
    ax.legend(loc=4, fontsize=9)
    add_sim_subpanel(ax)
    ax.set_title('2017 Indian Summer Monsoon',fontsize=15)
    ax.set_ylabel('ISM Index', fontsize=14)

    return ax

def plot_average_maps(fig, gs, input_file, forecast, times, levels, data_cache, open_func=xr.open_dataarray, return_quad=False, no_yticks=False, no_xticks=False):
    
    # calculate average map
    if not os.path.exists(data_cache):
        logging.info(f"Calculating average map for {input_file} and saving to {data_cache}")
        # open dataset
        olr = open_func(input_file)
        # calculate average
        if forecast:
            # if file is a forecast, fix the time dimension
            new_step = olr.time.values + olr.step.values
            olr = olr.squeeze().drop('time')
            olr = olr.assign_coords({'step': new_step}).rename({'step': 'time'})
        olr = olr.sel(time=times).mean('time')
        # cache 
        olr.to_netcdf(data_cache)
        

    logging.info(f"Loading average map from {data_cache}")
    olr = xr.open_dataarray(data_cache)

    def init_map(gs):
        ax = fig.add_subplot(gs, projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, lw=0.5, color='black')
        ax.set_extent([70, 100, 10, 25], crs=ccrs.PlateCarree())
        return ax
    
    ax = init_map(gs)
    # plot average map
    cs = ax.contourf(olr.lon, olr.lat, olr, cmap='RdBu_r', levels=levels, extend='both')

    if return_quad:
        return fig, ax, cs
    else:
        return fig, ax
    
def plot_precip_accumulation(fig, gs, input_file, forecast, time, levels, data_cache, open_func=xr.open_dataarray, return_quad=False, no_yticks=False, no_xticks=False):
    
    logging.info(f"Loading average map from {input_file}")
    precip = open_func(input_file).sel(time=time).squeeze()

    def init_map(gs):
        ax = fig.add_subplot(gs, projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, lw=0.5, color='black')
        ax.set_extent([70, 100, 10, 25], crs=ccrs.PlateCarree())
        return ax
    
    ax = init_map(gs)
    # plot average map
    cs = ax.contourf(precip.lon, precip.lat, precip, cmap=get_precip_cmap(), levels=levels, extend='both')

    if return_quad:
        return fig, ax, cs
    else:
        return fig, ax

def add_shared_colorbar(fig, ax1, ax2, im, label):

    # create colorbar
    cbar = fig.colorbar(im, ax=[ax1, ax2], orientation='vertical', pad=0.05, aspect=30, shrink=0.8)
    cbar.set_label(label)
    return cbar

def add_aligned_colorbar(fig, other_cbar, im, y_offset=0, x_offset=0, width_offset=0, label=None):
    other_cbar_pos = other_cbar.ax.get_position()
    new_x = other_cbar_pos.x0 + x_offset
    new_y = other_cbar_pos.y0 + y_offset
    new_width = other_cbar_pos.width + width_offset
    new_height = other_cbar_pos.height
    
    # Create a new position for the colorbar
    new_pos = [new_x, new_y, new_width, new_height]
    
    # Add the new colorbar to the figure
    cbar_ax = fig.add_axes(new_pos)
    cbar = fig.colorbar(im, cax=cbar_ax)
    if label:
        cbar.set_label(label)
    return cbar

def add_shared_title(fig, axl,axr, title):

    # add title
    ax_l_pos = axl.get_position()
    ax_r_pos = axr.get_position()

    new_x = (ax_l_pos.x1 + ax_r_pos.x0) / 2

    fig.text(new_x, ax_l_pos.y1+.01, title, va='bottom', ha='center', fontsize=14)

def add_shared_xlabel(fig, axl, axr, label):
    
    # add title
    ax_l_pos = axl.get_position()
    ax_r_pos = axr.get_position()

    new_x = (ax_l_pos.x1 + ax_r_pos.x0) / 2

    fig.text(new_x, ax_l_pos.y0-.042, label, va='bottom', ha='center', fontsize=14)
    
def plot_figure5(
        plot_ts_params: dict, 
        plot_climo_obs_params: dict,
        plot_climo_dlesm_params: dict,
        taylor_params: dict,
        plot_precip_accumulation_params_gt: dict,
        plot_precip_accumulation_params_dlesm: dict,
        # plot_average_map_isccp_params: dict,
        # plot_average_map_dlesm_params: dict,
        savefig_params: dict,
):
    def add_subplot_label(label, ax, x=0.01, y=.93):
        ax.text(x, y, label, transform=ax.transAxes, fontsize=18, fontweight='bold')
    def stretch_down(ax, offset, stretch):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0-offset, pos.width*stretch, pos.height*stretch])
        return ax
    def shift_right(ax, offset):
        pos = ax.get_position()
        ax.set_position([pos.x0+offset, pos.y0, pos.width, pos.height])
        return ax
    def shift_down(ax, offset, stretch):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0-offset, pos.width*stretch, pos.height*stretch])
        return ax
    def align_average_maps(upper_ax_left, upper_ax_right, ax):
        left_limit = upper_ax_left.get_position().x0
        right_limit = upper_ax_right.get_position().x1
        width = right_limit - left_limit
        pos = ax.get_position()
        scale = width / pos.width
        height = pos.height * scale
        ax.set_position([left_limit, pos.y0, width, height])
        return ax
    def show_legend(ax):
        legend = ax.get_legend()
        legend.set_visible(True)
        legend.set_alpha(1)
        legend.set_bbox_to_anchor((0.3, .9))
        for text in legend.get_texts():
            text.set_fontsize(14)
        return ax
    def add_panel_label(ax, label, x=0.64, y=0.027):
        ax.text(x, y, label, transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='black'))
        return ax
    def place_line(ax, loc=1.05):
        ax.axhline(loc, xmax=.713,color='black', linewidth=1.5)
        return ax
    
    # create figure 
    fig = plt.figure(figsize=(13, 10))
    gs = GridSpec(7, 4, figure=fig)

    # initiailzie timeseries 
    ax_ts = fig.add_subplot(gs[0:3, 0:2])

    # plot timeseries
    ax_ts = plot_timeseries(fig, ax_ts, **plot_ts_params)
    add_subplot_label('A', ax_ts)

    # climo obs hovmoller 
    ax_climo_obs_hov = fig.add_subplot(gs[0:3, 2])
    fig, ax_climo_obs_hov = plot_hovmoller(fig, ax_climo_obs_hov, **plot_climo_obs_params, no_xticks=False)
    ax_climo_obs_hov = shift_right(ax_climo_obs_hov, 0.051)
    ax_climo_obs_hov = add_panel_label(ax_climo_obs_hov, 'ISCCP',x=0.735, y=0.027)
    add_subplot_label('B', ax_climo_obs_hov)

    # climo dlesm hovmoller
    ax_climo_dlesm_hov = fig.add_subplot(gs[0:3, 3])
    fig, ax_climo_dlesm_hov, im_climo = plot_hovmoller(fig, ax_climo_dlesm_hov, **plot_climo_dlesm_params, return_quad=True, no_yticks=True, no_xticks=False)
    ax_climo_dlesm_hov = shift_right(ax_climo_dlesm_hov, 0.05)
    ax_climo_dlesm_hov = add_panel_label(ax_climo_dlesm_hov, 'DL$\it{ESy}$M', y=0.033)
    add_subplot_label('C', ax_climo_dlesm_hov)

    # climo colorbar and title
    ax_cbar_climo = add_shared_colorbar(fig, ax_climo_obs_hov, ax_climo_dlesm_hov, im_climo, 'OLR (W/m^2)')
    add_shared_title(fig, ax_climo_obs_hov, ax_climo_dlesm_hov, 'ISM Climatology')
    add_shared_xlabel(fig, ax_climo_obs_hov, ax_climo_dlesm_hov, 'Latitude')

    # initiliaze taylor diagram 
    ax_td = fig.add_subplot(gs[3:7, 0:2])
    ax_td = hovmoller_taylor_diagrams(**taylor_params, ax=ax_td)
    # plot legend and add top boarder
    ax_td = place_line(show_legend(ax_td))
    ax_td = shift_down(ax_td, 0.04, 1.1)
    add_subplot_label('D', ax_td)

    # initialize average map obs 
    # fig, ax_average_map_obs = plot_average_maps(fig, gs[3:5, 2:4], **plot_average_map_isccp_params)
    fig, ax_precip_obs = plot_precip_accumulation(fig, gs[3:5, 2:4], **plot_precip_accumulation_params_gt)
    ax_precip_obs = shift_down(align_average_maps(ax_climo_obs_hov, ax_climo_dlesm_hov, ax_precip_obs), 0.03, 1)
    add_panel_label(ax_precip_obs, 'ERA5',x=0.8886, y=0.045)
    add_subplot_label('E', ax_precip_obs, y=0.88)


    # intilaize average map dlesm
    # fig, ax_average_map_dlesm, map_im = plot_average_maps(fig, gs[5:7, 2:4], **plot_average_map_dlesm_params, return_quad=True)
    fig, ax_precip_dlesm, map_im = plot_precip_accumulation(fig, gs[5:7, 2:4], **plot_precip_accumulation_params_dlesm, return_quad=True)
    ax_precip_dlesm = shift_down(align_average_maps(ax_climo_obs_hov, ax_climo_dlesm_hov, ax_precip_dlesm), 0.002, 1)
    add_panel_label(ax_precip_dlesm, 'DL$\it{ESy}$M', x=0.835, y=0.055)
    add_subplot_label('F', ax_precip_dlesm, y=0.88)

    # average map colorbar
    ax_cbar_map = add_aligned_colorbar(fig, ax_cbar_climo, map_im, y_offset=-.42, label='14-day Accumulation (mm)', width_offset=-0.047)

    def _format_axes(fig, axes):
        return fig, axes
    
    # format axes position 
    fig, axs = _format_axes(fig, (ax_ts, ax_precip_obs, ax_precip_dlesm, ax_td, ax_climo_obs_hov, ax_climo_dlesm_hov))

    # save figure
    logging.info(f"Saving figure to {savefig_params['fname']}")
    fig.savefig(**savefig_params)

def plot_figure5_v2(
        plot_ts_params: dict, 
        plot_climo_obs_params: dict,
        plot_climo_dlesm_params: dict,
        taylor_params: dict,
        plot_precip_accumulation_params_gt: dict,
        plot_precip_accumulation_params_dlesm: dict,
        # plot_average_map_isccp_params: dict,
        # plot_average_map_dlesm_params: dict,
        savefig_params: dict,
):
    def add_subplot_label(label, ax, x=0.01, y=.93):
        ax.text(x, y, label, transform=ax.transAxes, fontsize=18, fontweight='bold')
    def stretch_down(ax, offset, stretch):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0-offset, pos.width*stretch, pos.height*stretch])
        return ax
    def shift_right(ax, offset):
        pos = ax.get_position()
        ax.set_position([pos.x0+offset, pos.y0, pos.width, pos.height])
        return ax
    def shift_down(ax, offset, stretch):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0-offset, pos.width*stretch, pos.height*stretch])
        return ax
    def align_average_maps(upper_ax_left, upper_ax_right, ax):
        left_limit = upper_ax_left.get_position().x0
        right_limit = upper_ax_right.get_position().x1
        width = right_limit - left_limit
        pos = ax.get_position()
        scale = width / pos.width
        height = pos.height * scale
        ax.set_position([left_limit, pos.y0, width, height])
        return ax
    def show_legend(ax):
        legend = ax.get_legend()
        legend.set_visible(True)
        legend.set_alpha(1)
        legend.set_bbox_to_anchor((0.3, .9))
        for text in legend.get_texts():
            text.set_fontsize(14)
        return ax
    def add_panel_label(ax, label, x=0.64, y=0.027):
        ax.text(x, y, label, transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='black'))
        return ax
    def place_line(ax, loc=1.05):
        ax.axhline(loc, xmax=.713,color='black', linewidth=1.5)
        return ax
    
    # create figure 
    fig = plt.figure(figsize=(13, 10))
    gs = GridSpec(7, 4, figure=fig)

    # initiailzie timeseries 
    ax_ts = fig.add_subplot(gs[0:3, 0:2])

    # plot timeseries
    ax_ts = plot_timeseries(fig, ax_ts, **plot_ts_params)

    # climo obs hovmoller 
    ax_climo_obs_hov = fig.add_subplot(gs[0:3, 2])
    fig, ax_climo_obs_hov = plot_hovmoller(fig, ax_climo_obs_hov, **plot_climo_obs_params, no_xticks=False)
    ax_climo_obs_hov = shift_right(ax_climo_obs_hov, 0.051)
    ax_climo_obs_hov = add_panel_label(ax_climo_obs_hov, 'ISCCP',x=0.735, y=0.027)

    # climo dlesm hovmoller
    ax_climo_dlesm_hov = fig.add_subplot(gs[0:3, 3])
    fig, ax_climo_dlesm_hov, im_climo = plot_hovmoller(fig, ax_climo_dlesm_hov, **plot_climo_dlesm_params, return_quad=True, no_yticks=True, no_xticks=False)
    ax_climo_dlesm_hov = shift_right(ax_climo_dlesm_hov, 0.05)
    ax_climo_dlesm_hov = add_panel_label(ax_climo_dlesm_hov, 'DL$\it{ESy}$M', y=0.033)

    # climo colorbar and title
    ax_cbar_climo = add_shared_colorbar(fig, ax_climo_obs_hov, ax_climo_dlesm_hov, im_climo, 'OLR (W/m^2)')
    add_shared_title(fig, ax_climo_obs_hov, ax_climo_dlesm_hov, 'ISM Climatology')
    add_shared_xlabel(fig, ax_climo_obs_hov, ax_climo_dlesm_hov, 'Latitude')

    # initiliaze taylor diagram 
    ax_td = fig.add_subplot(gs[3:7, 0:2])
    ax_td = hovmoller_taylor_diagrams(**taylor_params, ax=ax_td)
    # plot legend and add top boarder
    if taylor_params['legend']:
        ax_td = show_legend(ax_td)
    ax_td = place_line(ax_td)
    ax_td = shift_down(ax_td, 0.04, 1.1)

    # initialize average map obs 
    # fig, ax_average_map_obs = plot_average_maps(fig, gs[3:5, 2:4], **plot_average_map_isccp_params)
    fig, ax_precip_obs = plot_precip_accumulation(fig, gs[3:5, 2:4], **plot_precip_accumulation_params_gt)
    ax_precip_obs = shift_down(align_average_maps(ax_climo_obs_hov, ax_climo_dlesm_hov, ax_precip_obs), 0.03, 1)
    add_panel_label(ax_precip_obs, 'ERA5',x=0.8886, y=0.045)

    # intilaize average map dlesm
    # fig, ax_average_map_dlesm, map_im = plot_average_maps(fig, gs[5:7, 2:4], **plot_average_map_dlesm_params, return_quad=True)
    fig, ax_precip_dlesm, map_im = plot_precip_accumulation(fig, gs[5:7, 2:4], **plot_precip_accumulation_params_dlesm, return_quad=True)
    ax_precip_dlesm = shift_down(align_average_maps(ax_climo_obs_hov, ax_climo_dlesm_hov, ax_precip_dlesm), 0.002, 1)
    add_panel_label(ax_precip_dlesm, 'DL$\it{ESy}$M', x=0.835, y=0.055)

    # average map colorbar
    ax_cbar_map = add_aligned_colorbar(fig, ax_cbar_climo, map_im, y_offset=-.42, label='14-day Accumulation (mm)', width_offset=-0.047)

    def _format_axes(fig, axes):
        return fig, axes
    
    # format axes position 
    fig, axs = _format_axes(fig, (ax_ts, ax_precip_obs, ax_precip_dlesm, ax_td, ax_climo_obs_hov, ax_climo_dlesm_hov))

    # save figure
    logging.info(f"Saving figure to {savefig_params['fname']}")
    fig.savefig(**savefig_params)

if __name__ == '__main__':
    plot_figure5(**FIGURE5_PARAMS)