import os 
import pandas as pd
import xarray as xr
import numpy as np
import pprint
import matplotlib.pyplot as plt
import sys 
import argparse
import evaluation.evaluators as ev
import logging
from tqdm import tqdm
import data_processing.remap.healpix as hpx 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PARAMS_Z500_control = {
    # forecast file 
    'forecast_file': '/home/disk/rhodium/WEB/DLESyM_AGU-Advances/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit.nc',
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
        'fname' : './forecast_seasonal_cycle_z500_control.png',
        'dpi' : 300,
    },
    'verif_title':'6-year ERA5 Seasonal Cycle',
}


def main(params):

    def plot_cycle(da, 
                   rolling_params,
                   scale_factor,
                   cmap,
                   levels,
                   ref_line,
                   colorbar_label,
                   title,
                   savefig_params,
                   xticks,
                   xtick_labels,
                   xlabel,
                   xlim
    ):
            
        # use time metadata to create calendar of forecast 
        months = [pd.Timedelta(d).days/30 for d in da.step.values]
            # claculated seasonal cycle using xarray rolling. The "construct" operation stacks the rolling windows as a new dimension and 
        # and allows for calculating the mean accross the window while ignoring NaNs. https://github.com/pydata/xarray/issues/4278
        #                                                       -> this is probably what jonathan did for figure 8 in the 2021 paper
        zonal_da = da.mean(dim='lon').squeeze().transpose().rolling(**rolling_params).construct('new').mean('new',skipna=True)
        
        # initialize figure and start plotting! 
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(18,4))

        # forecast 
        im = ax.contourf(months,da.lat,zonal_da/scale_factor, cmap=cmap, levels = levels, extend='both')  

        # add ref line from forecast and verif 
        if ref_line is not None:
            # again use the rolling->construct method to smooth data, this time over a 15 day window (15 days * 4 steps per day = 60) 
            ref_fcst = da.mean(dim='lon').squeeze().transpose().rolling(step=60,center=True).construct('new').mean('new',skipna=True)
            ref_months = [pd.Timedelta(d).days/30 for d in da.step.values]
            ax.contour(months,da.lat,ref_fcst/scale_factor,levels=[ref_line],colors='white')

        # labels and ticks 
        if xticks is not None:
            ax.set_xticks(xticks)
        else:
            ax.set_xticks(np.arange(np.min(months),np.max(months),12,dtype='int'))
        if xtick_labels is not None:
            ax.set_xticklabels(xtick_labels)
        ax.yaxis.set_tick_params(labelright=True, right=True) # yticks on right side
        ax.set_title(title,fontsize=15)
        ax.set_xlabel('Forecast Months' if xlabel is None else xlabel, fontsize=12)
        ax.set_ylabel('Latitude',fontsize=12)
        if xlim is not None:
            ax.set_xlim(xlim)

        # tighten 
        fig.tight_layout()

        # colorbar 
        fig.subplots_adjust(right=.92)
        cbax = fig.add_axes([.95,.1,.015,.8])
        fig.colorbar(im,cax=cbax,label=colorbar_label)

        #save 
        print(f'saving fig as {savefig_params}')
        fig.savefig(**savefig_params)

    fcst = xr.open_dataset(f'{params["forecast_file"]}')[params['eval_variable']].sel(step=params['time'])  
    # initialize the remopper 
    mapper = hpx.HEALPixRemap(
            latitudes=181,
            longitudes=360,
            nside=64,
        )
    
    logger.info(f'Remapping forecast data from healpix to lat lon grid')
    # buffer for remaped data
    remap_buffer = np.zeros((len(fcst.time), len(fcst.step), 181, 360), dtype=np.float32)   
    # remap the forecast data to lat lon grid
    for i, t in tqdm(
        enumerate(fcst.time),
        desc="Time",
        unit="step",
        total=len(fcst.time), 
    ):
        for j, s in tqdm(
            enumerate(fcst.step),
            desc=f"Step (t={i})",
            unit="step",
            total=len(fcst.step),
            leave=False,
        ):
            remap_buffer[i, j] = mapper.hpx2ll(fcst.sel(time=t, step=s).values)
    
    # create remap buffer
    logger.info(f'Remapping forecast data from healpix to lat lon grid')
    # create new xarray DataArray with remapped data
    fcst_ll = xr.DataArray(
        remap_buffer,
        dims=['time', 'step', 'lat', 'lon'],
        coords={
            'time': fcst.time,
            'step': fcst.step,
            'lat': np.arange(90, -90.1, -1),
            'lon': np.arange(0, 360, 1)
        }
    )

    print('making seasonal cycle...')
    plot_cycle(da=fcst_ll,
               rolling_params=params['rolling_params'],
               scale_factor=params['scale_factor'],
               cmap=params['cmap'],
               levels=params['levels'],
               ref_line=params['ref_line'],
               colorbar_label=params['colorbar_label'],
               title=params['title'],
               savefig_params=params['savefig_params'],
               xticks = getattr(argparse.Namespace(**params),'xticks',None),
               xtick_labels = getattr(argparse.Namespace(**params),'xtick_labels',None),
               xlabel = getattr(argparse.Namespace(**params),'xlabel',None),
               xlim=getattr(argparse.Namespace(**params),'xlim',None),
    )
    return

if __name__=="__main__":
 
    main(PARAMS_Z500_control)
    main(PARAMS_Z500_1in1out_24AR)

