import os 
import pandas as pd
import xarray as xr
import numpy as np
import pprint
import matplotlib.pyplot as plt
import sys 
import argparse
sys.path.append('/home/disk/brume/nacc/dlesm/zephyr')
import evaluation.evaluators as ev


PARAMS_Z500 = {
    # forecast file 
    'forecast_file': '/home/disk/rhodium/WEB/DLESyM_AGU-Advances/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit',
    # params for evaluator initialization  
    'eval_variable' : 'z500',
    'verification_path' : '/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_z500.nc',
    # params for seasonal cycle calculation 
    'levels' : np.arange(490,591,10),
    'scale_factor':98.1, # transform geopotential to deka meters
    'time': slice(pd.Timedelta(0,'D'),pd.Timedelta(364,'D')),
    'init_index' : 1, # corresponds to july initialization 
    'add_verif_ref':True,
    'rolling_params': {'dim':{'step':int(12)},
                       'center':True},
    'ref_line':560,
    'cmap':'Spectral_r',
    'colorbar_label':'Z$_{500}$ (dkm)',
    'title':'10-year Forecast Initialized 2017-01-01',
    'savefig_params': {
        'fname' : './forecast_seasonal_cycle_z500.png',
        'dpi' : 300,
    },
    'verif_title':'6-year ERA5 Seasonal Cycle',
    'make_verif_seasonal_cycle':True,
    'savefig_params_verif': {
        'fname' : './verif_seasonal_cycle_z500.png',
        'dpi' : 300,
    },
}

def main(params):

    def plot_cycle(da, 
                   verif_da,
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
            if params['add_verif_ref']:
                ref_verif = verif_da.mean(dim='lon').squeeze().transpose().rolling(step=60,center=True).construct('new').mean('new',skipna=True)
                ax.contour(months,da.lat,ref_verif/scale_factor,levels=[ref_line],colors='black')

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

    # initialize evaluator around atmos forecast file and remap to lat lon 
    remapped_file=f'{params["forecast_file"]}_{params["eval_variable"]}_ll.nc'
    fcst = ev.EvaluatorHPX(
               forecast_path = f'{params["forecast_file"]}'+'.nc',
               verification_path = params['verification_path'],
               eval_variable = params['eval_variable'],
               remap_config = None,
               on_latlon = True,
               times = params['times'] if 'times' in params else "2017-01-01--2018-12-31",
               poolsize = 30,
               verbose = True,
               ll_file=remapped_file,
    ) 

    # load datasets and grab indicated initialization if provided 
    # da = xr.open_dataset(remapped_file)[params['eval_variable']].sel(step=params['time'])
    da = fcst.forecast_da

    # load verification data if indicated
    if params['add_verif_ref']:
        fcst.generate_verification(
            verification_path = params['verification_path'],
            defined_verif_only = False,
        )
        verif_da = fcst.verification_da

    print('making seasonal cycle...')
    plot_cycle(da=da,
               verif_da=verif_da if params['add_verif_ref'] else None,
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
    # make verif seasonal cycle if indicated 
    if getattr(argparse.Namespace(**params),'make_verif_seasonal_cycle',False):
        print('making verif seasonal cycle...')
        plot_cycle(da=verif_da,
                   verif_da=verif_da,
                   rolling_params=params['rolling_params'],
                   scale_factor=params['scale_factor'],
                   cmap=params['cmap'],
                   levels=params['levels'],
                   ref_line=params['ref_line'],
                   colorbar_label=params['colorbar_label'],
                   title=params['verif_title'],
                   savefig_params=params['savefig_params_verif'],
                   xticks = getattr(argparse.Namespace(**params),'xticks',None),
                   xtick_labels = getattr(argparse.Namespace(**params),'xtick_labels',None),
                   xlabel = getattr(argparse.Namespace(**params),'xlabel',None),
                   xlim=getattr(argparse.Namespace(**params),'xlim',None),
        )
    return

if __name__=="__main__":
 
    # main(PARAMS_Z500_ATMOS_ONLY_JULY_INIT)
    # main(PARAMS_Z500_ATMOS_ONLY)
    main(PARAMS_Z500)

