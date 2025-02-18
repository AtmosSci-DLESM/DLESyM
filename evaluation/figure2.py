import xarray as xr 
import os
import numpy as np
import sys
import copy
import warnings
from tqdm import tqdm
from PIL import Image
from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import cartopy.feature as cft 
sys.path.append('./')
from evaluation.evaluators import EvaluatorHPX
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors 
import glob
import matplotlib.path as mpath
import pandas as pd 

# This script creates plots of example ET cyclones in the forecast and verification data. 

PARAMS_100th_year_storm_hpx64 = {
    'forecast_file': 'path/to/100yr/forecast/100yr_forecast_filename', # path to forecast file
    'forecast_time' : '2116-02-07T18:00', # timestamp for forecasted storm
    'z500_verification_file' : '/path/to/verif/era5_z500.nc', # path to verification z500 data
    'z1000_verification_file' : '/path/to/verif/era5_z1000.nc', # path to verification z1000 data
    'ws10_verification_file' : '/path/to/verif/era5_ws10.nc', # path to verification ws10 data
    'output_plot_file' : './storm_hpx64_100yr.png',
}
PARAMS_1000th_year_storm_hpx64 = {
    'forecast_file': 'path/to/1000yr/forecast/1000yr_forecast_filename', # path to forecast file
    'forecast_time' : '2116-01-22T06:00', # timestamp used for forecasted 1000 yr storm. Here we used the year 2116 to substitute 3116 for compatibility with datetime
    'z500_verification_file' : '/path/to/verif/era5_z500.nc', # path to verification z500 data
    'z1000_verification_file' : '/path/to/verif/era5_z1000.nc', # path to verification z1000 data
    'ws10_verification_file' : '/path/to/verif/era5_ws10.nc', # path to verification ws10 data
    'output_plot_file' : './storm_hpx64_1000yr.png',
    'obs_storm_time' : '2018-03-13T12:00',
}

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

def plot_storm(
        forecast_file,
        forecast_time,
        z500_verification_file,
        z1000_verification_file,
        ws_verification_file,
        output_plot_file,
        plot_extent = [-100,20,15,75],
        obs_storm_time = None,
):
    #####    DATA PREPARATION    #####

    # initialize evaluator for z500 forecast
    fcst_z500 = EvaluatorHPX(
        forecast_path = f'{forecast_file}.nc',
        verification_path = z500_verification_file,
        eval_variable = 'z500',
        on_latlon = True,
        poolsize = 20,
        ll_file=f'{forecast_file}_z500_ll.nc'
    )
    # initialize an evaluator for z1000 forecast
    fcst_z1000 = EvaluatorHPX(
        forecast_path = f'{forecast_file}.nc',
        verification_path = z1000_verification_file,
        eval_variable = 'z1000',
        on_latlon = True,
        poolsize = 20,
        ll_file=f'{forecast_file}_z1000_ll.nc'
    )
    # initialize an evaluator for ws forecast
    fcst_ws = EvaluatorHPX(
        forecast_path = f'{forecast_file}.nc',
        verification_path = ws_verification_file,
        eval_variable = 'ws10',
        on_latlon = True,
        poolsize = 20,
        ll_file=f'{forecast_file}_ws10_ll.nc'
    )

    # extract forecast dataarrays, fix time dimensions, grab storm forecasted time, eliminate singleton dimensions, scale heights to dekameters
    fcst_storm_z500 = fcst_z500.forecast_da.assign_coords(step = fcst_z500.forecast_da.time.values + fcst_z500.forecast_da.step).sel(step=forecast_time).squeeze() * 1/98.1
    fcst_storm_z1000 = fcst_z1000.forecast_da.assign_coords(step = fcst_z1000.forecast_da.time.values + fcst_z1000.forecast_da.step).sel(step=forecast_time).squeeze() * 1/98.1
    fcst_storm_ws = fcst_ws.forecast_da.assign_coords(step = fcst_ws.forecast_da.time.values + fcst_ws.forecast_da.step).sel(step=forecast_time).squeeze()
    

    if obs_storm_time is not None:
        obs_z500 = xr.open_dataset(z500_verification_file)['z'].sel(time=obs_storm_time).rename({'longitude':'lon','latitude':'lat'}).squeeze() * 1/98.1
        obs_z1000 = xr.open_dataset(z1000_verification_file)['z'].sel(time=obs_storm_time).rename({'longitude':'lon','latitude':'lat'}).squeeze() * 1/98.1
        obs_ws = xr.open_dataset(ws_verification_file)['ws10'].sel(time=obs_storm_time).rename({'longitude':'lon','latitude':'lat'}).squeeze()

    ######    PLOTTING CODE    ######

    print('generating plot...')

    # function to initialize figure 
    def init_figure(projection=ccrs.LambertConformal):
        
        # calcualate number of rows and initialize figure
        fig = plt.figure(figsize = (10,5))
        
        # initialize cartopy axes for forecasted field and error field
        parallels = (20,25) 
        axs=[
            plt.subplot(1,2,1, projection=projection(central_longitude=320,central_latitude=42, standard_parallels=parallels)),
            plt.subplot(1,2,2, projection=projection(central_longitude=320,central_latitude=42, standard_parallels=parallels)),
        ]
        for ax in axs: ax.coastlines('50m',linewidth=1.2)
        def draw_boundary(ax):
            n = 20
            aoi = mpath.Path(
                list(zip(np.linspace(-100,20, n), np.full(n, 80))) + \
                list(zip(np.full(n, 20), np.linspace(80, 15, n))) + \
                list(zip(np.linspace(20, -100, n), np.full(n, 15))) + \
                list(zip(np.full(n, -100), np.linspace(20, 80, n)))
            )
            ax.set_boundary(aoi, transform=ccrs.PlateCarree())
            ax.set_extent(plot_extent, crs=ccrs.PlateCarree())
        for ax in axs: draw_boundary(ax)

        # fig.subplots_adjust(wspace=0.1, hspace=0.12)

        return fig, np.asarray(axs) # forecast on right
    
    fig, axs = init_figure()

    def plot_z500(fig, ax, da, return_colors=False):
      
        # add cyclic point 
        da_cyc, lon = add_cyclic_point(da.values, coord=da.lon.values)

        # plot z500 in forecast
        im = ax.contourf(lon,da.lat,da_cyc,transform=ccrs.PlateCarree(),
                                cmap='Spectral_r',
                                levels=np.linspace(490,591,20),
                                extend='both',
                                )
        # Plot using pcolormesh
        # im = ax.pcolormesh(lon, da.lat, da_cyc, transform=ccrs.PlateCarree(),
        #            cmap='Spectral_r',
        #            shading='auto')
        if return_colors:
            return fig, ax, im
        else:
            return fig, ax
    
    # plot z500 in forecast
    fig, axs[0], im_z500 = plot_z500(fig, axs[0], fcst_storm_z500, True)

    def plot_z1000(fig, ax, da, return_contours=False):
        data,lon = add_cyclic_point(da.values, coord=da.lon.values)
        cont_pos = ax.contour(lon,da.lat,data,transform=ccrs.PlateCarree(),
                              colors='k',
                              levels=np.arange(4,50,4),
                              linewidths=.5,
                              zorder=10
                              )
        # negative heights
        cont_neg = ax.contour(lon,da.lat,data,transform=ccrs.PlateCarree(),
                              colors='k',
                              levels=np.arange(-32,1,4),
                              linewidths=.5,
                              linestyles='dashed',
                              zorder=10
                              )  
        
        if return_contours:
            return fig, ax, (cont_pos, cont_neg)
        else:
            return fig, ax
    
    # plot z1000 in forecast
    fig, axs[0] = plot_z1000(fig, axs[0], fcst_storm_z1000)
    fig, axs[1] = plot_z1000(fig, axs[1], fcst_storm_z1000)

    def plot_ws(fig, ax, da, return_colormesh=False):

        # add cyclic point
        da_cyc, lon = add_cyclic_point(da.values, coord=da.lon.values)

        # plot ws10 in forecast
        im = ax.contourf(lon,da.lat,da_cyc,transform=ccrs.PlateCarree(),
                        cmap=get_custom_cmap('hot_r', white_levels=10, start_color=.1),
                        levels=np.linspace(8,24,20),
                        extend='both',
                        )
        # Plot using pcolormesh
        # im = ax.pcolormesh(lon, da.lat, da_cyc, transform=ccrs.PlateCarree(),
        #                 cmap='hot_r',
        #                 shading='auto')
        if return_colormesh:
            return fig, ax, im
        else:
            return fig, ax

    # plot ws10 in forecast
    fig, axs[1], im_ws = plot_ws(fig, axs[1], fcst_storm_ws, True)

    # add colorbars
    fig.subplots_adjust(bottom=0.008)

    # scoot ax[0] to right
    pos = axs[0].get_position()
    axs[0].set_position([pos.x0+.05, pos.y0, pos.width, pos.height]) 

    cbar_axs = [fig.add_axes([.124,.1,.355,.025]), fig.add_axes([.545,.1,.355,.025])]
    z500_cbar = fig.colorbar(im_z500, cax=cbar_axs[0], orientation='horizontal', ticks=np.arange(490,596,20),label='Z$_{500}$ (dam)')
    ws_cbar = fig.colorbar(im_ws, cax=cbar_axs[1], orientation='horizontal', ticks=np.arange(8,25,2),label='WS$_{10}$ (m/s)')
    
    # scoot corresponding colorbar to right
    pos = cbar_axs[0].get_position()
    cbar_axs[0].set_position([pos.x0+.05, pos.y0, pos.width, pos.height])

    # tighten and save
    print(f'...finished. Saving to {output_plot_file}')
    fig.savefig(output_plot_file, dpi=400)

    if obs_storm_time is not None:

        print('generating observed plot...')
        obs_fig, obs_axs = init_figure()

        # plot observed z500
        obs_fig, obs_axs[0], im_z500 = plot_z500(obs_fig, obs_axs[0], obs_z500, True)
        # plot observed z1000
        obs_fig, obs_axs[0] = plot_z1000(obs_fig, obs_axs[0], obs_z1000)
        # plot observed ws10
        obs_fig, obs_axs[1], im_ws = plot_ws(obs_fig, obs_axs[1], obs_ws, True)
        # plot observed z1000
        obs_fig, obs_axs[1] = plot_z1000(obs_fig, obs_axs[1], obs_z1000)

        # add colorbars
        obs_fig.subplots_adjust(bottom=0.008)
        cbar_obs_axs = [obs_fig.add_axes([.124,.1,.355,.025]), obs_fig.add_axes([.545,.1,.355,.025])]
        z500_cbar = obs_fig.colorbar(im_z500, cax=cbar_obs_axs[0], orientation='horizontal', ticks=np.arange(490,596,20),label='Z$_{500}$ (dam)')
        ws_cbar = obs_fig.colorbar(im_ws, cax=cbar_obs_axs[1], orientation='horizontal', ticks=np.arange(8,25,2),label='WS$_{10}$ (m/s)')

        # scoot ax and corresponding colorbar to right
        pos = obs_axs[0].get_position()
        obs_axs[0].set_position([pos.x0+.05, pos.y0, pos.width, pos.height])
        pos = cbar_obs_axs[0].get_position()
        cbar_obs_axs[0].set_position([pos.x0+.05, pos.y0, pos.width, pos.height])
        
        # tighten and save
        if output_plot_file.endswith('.png'):
            output_obs_plot_file = output_plot_file.replace('.png',f'_obs_{str(obs_storm_time)[:13]}.png')
        elif output_plot_file.endswith('.pdf'):
            output_obs_plot_file = output_plot_file.replace('.pdf',f'_obs_{str(obs_storm_time)[:13]}.pdf')
        elif output_plot_file.endswith('.svg'):
            output_obs_plot_file = output_plot_file.replace('.svg',f'_obs_{str(obs_storm_time)[:13]}.svg')
        print(f'...finished. Saving to {output_obs_plot_file}')
        obs_fig.savefig(output_obs_plot_file, dpi=400)

def plot_blocking(
        forecast_file,
        forecast_time,
        z500_verification_file,
        z1000_verification_file,
        ws_verification_file,
        output_plot_file,
        plot_extent = [-100,20,15,75],
        obs_storm_time = None,
):
    #####    DATA PREPARATION    #####

    # initialize evaluator for z500 forecast
    fcst_z500 = EvaluatorHPX(
        forecast_path = f'{forecast_file}.nc',
        verification_path = z500_verification_file,
        eval_variable = 'z500',
        on_latlon = True,
        poolsize = 20,
        ll_file=f'{forecast_file}_z500_ll.nc'
    )
    # # initialize an evaluator for z1000 forecast
    # fcst_z1000 = EvaluatorHPX(
    #     forecast_path = f'{forecast_file}.nc',
    #     verification_path = z1000_verification_file,
    #     eval_variable = 'z1000',
    #     on_latlon = True,
    #     poolsize = 20,
    #     ll_file=f'{forecast_file}_z1000_ll.nc'
    # )
    # # initialize an evaluator for ws forecast
    # fcst_ws = EvaluatorHPX(
    #     forecast_path = f'{forecast_file}.nc',
    #     verification_path = ws_verification_file,
    #     eval_variable = 'ws10',
    #     on_latlon = True,
    #     poolsize = 20,
    #     ll_file=f'{forecast_file}_ws10_ll.nc'
    # )

    # extract forecast dataarrays, fix time dimensions, grab storm forecasted time, eliminate singleton dimensions, scale heights to dekameters
    fcst_storm_z500 = fcst_z500.forecast_da.assign_coords(step = fcst_z500.forecast_da.time.values + fcst_z500.forecast_da.step).sel(step=forecast_time).squeeze() * 1/98.1
    # fcst_storm_z1000 = fcst_z1000.forecast_da.assign_coords(step = fcst_z1000.forecast_da.time.values + fcst_z1000.forecast_da.step).sel(step=forecast_time).squeeze() * 1/98.1
    # fcst_storm_ws = fcst_ws.forecast_da.assign_coords(step = fcst_ws.forecast_da.time.values + fcst_ws.forecast_da.step).sel(step=forecast_time).squeeze()
    

    if obs_storm_time is not None:
        obs_z500 = xr.open_dataset(z500_verification_file)['z'].sel(time=obs_storm_time).rename({'longitude':'lon','latitude':'lat'}).squeeze() * 1/98.1
        # obs_z1000 = xr.open_dataset(z1000_verification_file)['z'].sel(time=obs_storm_time).rename({'longitude':'lon','latitude':'lat'}).squeeze() * 1/98.1
        # obs_ws = xr.open_dataset(ws_verification_file)['ws10'].sel(time=obs_storm_time).rename({'longitude':'lon','latitude':'lat'}).squeeze()

    ######    PLOTTING CODE    ######

    print('generating plot...')

    # function to initialize figure 
    def init_figure(projection=ccrs.LambertConformal):
        
        # calcualate number of rows and initialize figure
        fig = plt.figure(figsize = (10,5))
        
        # initialize cartopy axes for forecasted field and error field
        parallels = (20,25) 
        axs=[
            plt.subplot(1,2,1, projection=projection(central_longitude=360,central_latitude=42, standard_parallels=parallels)),
            plt.subplot(1,2,2, projection=projection(central_longitude=360,central_latitude=42, standard_parallels=parallels)),
        ]
        for ax in axs: ax.coastlines('50m',linewidth=1.2)
        def draw_boundary(ax):
            n = 20
            max_lat = 80
            min_lat = 15
            max_lon = 60
            min_lon = -60
            aoi = mpath.Path(
                list(zip(np.linspace(min_lon,max_lon, n), np.full(n, max_lat))) + \
                list(zip(np.full(n, max_lon), np.linspace(max_lat, min_lat, n))) + \
                list(zip(np.linspace(max_lon, min_lon, n), np.full(n, min_lat))) + \
                list(zip(np.full(n, min_lon), np.linspace(min_lat, max_lat, n)))
            )
            # aoi = mpath.Path(
            #     list(zip(np.linspace(-60,60, n), np.full(n, 80))) + \
            #     list(zip(np.full(n, 60), np.linspace(80, 15, n))) + \
            #     list(zip(np.linspace(20, -100, n), np.full(n, 15))) + \
            #     list(zip(np.full(n, -100), np.linspace(20, 80, n)))
            # )
            ax.set_boundary(aoi, transform=ccrs.PlateCarree())
            ax.set_extent(plot_extent, crs=ccrs.PlateCarree())
        for ax in axs: draw_boundary(ax)

        # fig.subplots_adjust(wspace=0.1, hspace=0.12)

        return fig, np.asarray(axs) # forecast on right
    
    fig, axs = init_figure()

    def plot_z500(fig, ax, da, return_colors=False):
      
        # add cyclic point 
        da_cyc, lon = add_cyclic_point(da.values, coord=da.lon.values)

        # plot z500 in forecast
        im = ax.contourf(lon,da.lat,da_cyc,transform=ccrs.PlateCarree(),
                                cmap='Spectral_r',
                                levels=np.linspace(490,591,20),
                                extend='both',
                                )
        _ = ax.contour(lon,da.lat,da_cyc,transform=ccrs.PlateCarree(),
                                colors='black',
                                levels=np.linspace(490,591,20),
                                extend='both',
                                linewidths=.2,
                                )
        # Plot using pcolormesh
        # im = ax.pcolormesh(lon, da.lat, da_cyc, transform=ccrs.PlateCarree(),
        #            cmap='Spectral_r',
        #            shading='auto')
        if return_colors:
            return fig, ax, im
        else:
            return fig, ax
    
    # plot z500 in forecast
    fig, axs[1], im_z500 = plot_z500(fig, axs[1], fcst_storm_z500, True)
    fig, axs[0] = plot_z500(fig, axs[0], obs_z500)

    # add colorbars
    fig.subplots_adjust(bottom=0.008)

    # scoot ax[0] to right
    pos = axs[0].get_position()
    axs[0].set_position([pos.x0-.1, pos.y0, pos.width, pos.height]) 
    pos = axs[1].get_position()
    axs[1].set_position([pos.x0-.14, pos.y0, pos.width, pos.height])

    cbar_ax = fig.add_axes([.25,.1,.5,.025])
    z500_cbar = fig.colorbar(im_z500, cax=cbar_ax, orientation='horizontal', ticks=np.arange(490,596,20),label='Z$_{500}$ (dam)')
    # ws_cbar = fig.colorbar(im_ws, cax=cbar_axs[1], orientation='horizontal', ticks=np.arange(8,25,2),label='WS$_{10}$ (m/s)')

    # tighten and save
    print(f'...finished. Saving to {output_plot_file}')
    fig.savefig(output_plot_file, dpi=400)

def compare_storms(
        forecast_file,
        verification_time,
        forecast_time,
        z500_verification_file,
        z1000_verification_file,
        ws_verification_file,
        output_plot_file,
        plot_extent = [-100,20,15,75],
):


    #####    DATA PREPARATION    #####

    # initialize evaluator for z500 forecast
    fcst_z500 = EvaluatorHPX(
        forecast_path = f'{forecast_file}.nc',
        verification_path = z500_verification_file,
        eval_variable = 'z500',
        on_latlon = True,
        poolsize = 20,
        ll_file=f'{forecast_file}_z500_ll.nc'
    )
    # initialize an evaluator for z1000 forecast
    fcst_z1000 = EvaluatorHPX(
        forecast_path = f'{forecast_file}.nc',
        verification_path = z1000_verification_file,
        eval_variable = 'z1000',
        on_latlon = True,
        poolsize = 20,
        ll_file=f'{forecast_file}_z1000_ll.nc'
    )
    # initialize an evaluator for ws forecast
    fcst_ws = EvaluatorHPX(
        forecast_path = f'{forecast_file}.nc',
        verification_path = ws_verification_file,
        eval_variable = 'ws10',
        on_latlon = True,
        poolsize = 20,
        ll_file=f'{forecast_file}_ws10_ll.nc'
    )

    # extract forecast dataarrays, fix time dimensions, grab storm forecasted time, eliminate singleton dimensions, scale heights to dekameters
    fcst_storm_z500 = fcst_z500.forecast_da.assign_coords(step = fcst_z500.forecast_da.time.values + fcst_z500.forecast_da.step).sel(step=forecast_time).squeeze() * 1/98.1
    fcst_storm_z1000 = fcst_z1000.forecast_da.assign_coords(step = fcst_z1000.forecast_da.time.values + fcst_z1000.forecast_da.step).sel(step=forecast_time).squeeze() * 1/98.1
    fcst_storm_ws = fcst_ws.forecast_da.assign_coords(step = fcst_ws.forecast_da.time.values + fcst_ws.forecast_da.step).sel(step=forecast_time).squeeze()

    # do the same for the observed comparison
    verif_storm_z500 = xr.open_dataset(z500_verification_file)['z'].sel(time=verification_time).rename({'longitude':'lon','latitude':'lat'}).squeeze() * 1/98.1
    verif_storm_z1000 = xr.open_dataset(z1000_verification_file)['z'].sel(time=verification_time).rename({'longitude':'lon','latitude':'lat'}).squeeze() * 1/98.1
    verif_storm_ws = xr.open_dataset(ws_verification_file)['ws10'].sel(time=verification_time).rename({'longitude':'lon','latitude':'lat'}).squeeze()

    ######    PLOTTING CODE    ######

    print('generating plot...')

    # function to initialize figure 
    def init_figure(projection=ccrs.LambertConformal):
        
        # calcualate number of rows and initialize figure
        fig = plt.figure(figsize = (10,6))
        
        # initialize cartopy axes for forecasted field and error field
        parallels = (25,35) 
        axs=[
            plt.subplot(2,2,1, projection=projection(central_longitude=320,central_latitude=42, standard_parallels=parallels)),
            plt.subplot(2,2,2, projection=projection(central_longitude=320,central_latitude=42, standard_parallels=parallels)),
            plt.subplot(2,2,3, projection=projection(central_longitude=320,central_latitude=42, standard_parallels=parallels)),
            plt.subplot(2,2,4, projection=projection(central_longitude=320,central_latitude=42, standard_parallels=parallels)),
        ]
        for ax in axs: ax.coastlines('50m',linewidth=0.6)
        def draw_boundary(ax):
            n = 20
            aoi = mpath.Path(
                list(zip(np.linspace(-100,25, n), np.full(n, 80))) + \
                list(zip(np.full(n, 25), np.linspace(80, 15, n))) + \
                list(zip(np.linspace(25, -100, n), np.full(n, 15))) + \
                list(zip(np.full(n, -100), np.linspace(15, 80, n)))
            )
            ax.set_boundary(aoi, transform=ccrs.PlateCarree())
            ax.set_extent(plot_extent, crs=ccrs.PlateCarree())
        for ax in axs: draw_boundary(ax)

        fig.subplots_adjust(wspace=0, hspace=0.15)

        return fig, np.asarray(axs) # forecast on right, verification on left
        
    # initialize figure
    fig, axs = init_figure()

    def plot_z500(fig, ax, da, return_colormesh=False):
      
        # z500 norm 
        z500_norm = mcolors.BoundaryNorm(np.linspace(490,590,30),
                                ncolors=getattr(plt.cm,'Spectral').N, 
                                clip=True)
        # plot z500 in forecast
        pcol = ax.pcolormesh(da.lon,da.lat,da,transform=ccrs.PlateCarree(),
                                cmap='Spectral_r',
                                norm=z500_norm,
                                shading='auto',
                                )
        if return_colormesh:
            return fig, ax, pcol
        else:
            return fig, ax
        
    # plot z500 in forecast and verif
    fig, axs[0] = plot_z500(fig, axs[0], fcst_storm_z500)
    fig, axs[2], z500_pcol = plot_z500(fig, axs[2], verif_storm_z500, return_colormesh=True)

    def plot_ws(fig, ax, da, return_colormesh=False):
        # ws norm 
        ws_norm = mcolors.BoundaryNorm(np.linspace(8,24,30),
                                ncolors=getattr(plt.cm,'hot_r').N, 
                                clip=True)
        # plot z500 in forecast
        pcol = ax.pcolormesh(da.lon,da.lat,da,transform=ccrs.PlateCarree(),
                             cmap='hot_r',
                             norm=ws_norm,
                             shading='auto',
                             )
        if return_colormesh:
            return fig, ax, pcol
        else:
            return fig, ax
    
    # plot ws in forecast and verif
    fig, axs[1] = plot_ws(fig, axs[1], fcst_storm_ws)
    fig, axs[3], ws_pcol = plot_ws(fig, axs[3], verif_storm_ws, return_colormesh=True)

    def plot_z1000(fig, ax, da, return_contours=False):
        data,lon = add_cyclic_point(da.values, coord=da.lon.values)
        cont_pos = ax.contour(lon,da.lat,data,transform=ccrs.PlateCarree(),
                              colors='k',
                              levels=np.arange(4,50,4),
                              linewidths=.5,
                              zorder=10
                              )
        # negative heights
        cont_neg = ax.contour(lon,da.lat,data,transform=ccrs.PlateCarree(),
                              colors='k',
                              levels=np.arange(-32,1,4),
                              linewidths=.5,
                              linestyles='dashed',
                              zorder=10
                              )  
        
        if return_contours:
            return fig, ax, (cont_pos, cont_neg)
        else:
            return fig, ax

    # plot forecast and verification z1000
    fig, axs[0] = plot_z1000(fig, axs[0], fcst_storm_z1000)
    fig, axs[2] = plot_z1000(fig, axs[2], verif_storm_z1000)
    fig, axs[1] = plot_z1000(fig, axs[1], fcst_storm_z1000)
    fig, axs[3] = plot_z1000(fig, axs[3], verif_storm_z1000)

    # add colorbars
    cbar_axs = [fig.add_axes([.1325,.04,.337,.02]), fig.add_axes([.5565,.04,.337,.02])]
    cbar_axs[0].set_title('Z$_{500}$ (dkm)', fontsize=10)
    cbar_axs[1].set_title('WS$_{10}$ (m/s)', fontsize=10)
    z500_cbar = fig.colorbar(z500_pcol, cax=cbar_axs[0], orientation='horizontal', ticks=np.arange(490,596,20))
    ws_cbar = fig.colorbar(ws_pcol, cax=cbar_axs[1], orientation='horizontal', ticks=np.arange(8,25,2))

    # add labels
    axs[0].text(-.26,.3125,'Forecast', rotation=90, fontsize=15, transform=axs[0].transAxes)
    axs[2].text(-.26,.3125,'Observed', rotation=90, fontsize=15, transform=axs[2].transAxes)

    # tighten and save
    print(f'...finished. Saving to {output_plot_file}')
    fig.savefig(output_plot_file, dpi=300)

if __name__ == "__main__":

    plot_storm(**PARAMS_100th_year_storm_hpx64)
    plot_storm(**PARAMS_1000th_year_storm_hpx64)
