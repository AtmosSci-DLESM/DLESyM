import numpy as np
import pandas as pd
import xarray as xr
import pprint
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
import logging
from drift import linear_fit
logging.basicConfig(level=logging.INFO)


FIGURE1_PARAMS = dict(
    broken_ts_params = {
        # 1000 year forecast file
        'forecast_file': 'path/to/1000yr/forecast/forecast_filename',
        # file containing first 5 years of the 1000 year forecast. Must be seperate netcdf file, subdivided before analysis
        'first_5_forecast_file': 'path/to/first5yrs/of/1000yr/forecast/forecast_filename',
        # file containing last 5 years of the 1000 year forecast. Must be seperate netcdf file, subdivided before analysis
        'last_5_forecast_file': 'path/to/last5yrs/of/1000yr/forecast/forecast_filename',
        # params for evaluator initialization  
        'eval_variable' : 'z500', # variable name in file
        'verification_path' : '/path/to/verif/era5_z500.nc', # path to verification data
        # params for seasonal cycle calculation 
        'levels' : np.arange(490,591,10), # levels for contour plot
        'scale_factor':98.1, # transform geopotential to deka meters
        'time': slice(pd.Timedelta(0,'D'),pd.Timedelta(36512,'D')), # time slice for seasonal cycle calculation
        'resampled_dates_first5': pd.date_range('2017-02-01','2022-02-01',freq='6H'), # resampled dates for first 5 years
        'resampled_dates_last5': pd.date_range('2112-02-01','2117-02-01',freq='6H'), # resampled dates for last 5 years, useful if you've reassigned time dimension to datetime obj
        'init_index' : 1, # corresponds to july initialization 
        # parameters for plotting contours
        'add_verif_ref':True,
        'rolling_params_color': {'dim':{'step':int(12)},
                                        'center':True}, 
        'rolling_params_contour': {'dim':{'step':int(60)},
                                        'center':True},
        'ref_line':560,
        'cmap':'Spectral_r',
        'colorbar_label':'Z$_{500}$ (dam)',
        'forecast_init': pd.Timestamp('2017-01-01'),
        'xticks_left':pd.date_range('2017-01-01','2022-01-01',freq='YS'),
        'xtick_labels_left':[],#pd.date_range('2017-01-01','2022-01-01',freq='YS').year,
        'xticks_right':pd.date_range('2112-01-01','2117-01-01',freq='YS'),
        'xtick_labels_right':np.arange(3012,3018,dtype=int),
        'xlabel':None,
    },
    first5_verif_params = {
        # forecast files 
        'forecast_file': 'path/to/1000yr/forecast/forecast_filename',
        'first_5_forecast_file': 'path/to/first5yrs/of/1000yr/forecast/forecast_filename',
        # params for evaluator initialization  
        'eval_variable' : 'z500',
        'verification_path' : '/path/to/verif/era5_z500.nc', # path to verification data
        # params for seasonal cycle calculation 
        'levels' : np.arange(490,591,10),
        'scale_factor':98.1, # transform geopotential to deka meters
        'time': slice(pd.Timedelta(0,'D'),pd.Timedelta(36512,'D')),
        'init_index' : 1, # corresponds to july initialization 
        'add_verif_ref':True,
        'rolling_params_color': {'dim':{'step':int(12)},
                                        'center':True},
        'rolling_params_contour': {'dim':{'step':int(60)},
                                        'center':True},
        'ref_line':560,
        'cmap':'Spectral_r',
        'colorbar_label':'Z$_{500}$ (dkm)',
        'forecast_init': pd.Timestamp('2017-01-01'),
        'xticks':pd.date_range('2017-01-01','2022-01-01',freq='YS'),
        'xtick_labels':pd.date_range('2017-01-01','2022-01-01',freq='YS').year,
    },
    t2m_drift_params = dict(
        file='path/to/1000yr/forecast/forecast_filename.nc',
        var='t2m0',
        cache_dir='place/to/cache/drift_cache',
        indexing=dict(year=slice(2017, 3016)),
        smoothing_func=None,
        linear_fit_params=dict(
            years=slice(2017, 3017),
        ),
        ylim=(287.5, 288.1),
        yticks=np.arange(287.5, 288.15, 0.2),
        xticks=np.arange(2017, 3018, 200),
        xtick_labels=[],
        xlabel=None,
        ylabel='T$_{2m}$ (K)',
    ),
    sst_drift_params = dict(
        file='path/to/1000yr/forecast/forecast_filename.nc',
        var='sst',
        cache_dir='place/to/cache/drift_cache',
        indexing=dict(year=slice(2017, 3016)),
        smoothing_func=None,
        linear_fit_params=dict(
            years=slice(2017, 3016),
        ),
        ylim=(291.5, 291.9),
        yticks=np.arange(291.5, 292, 0.1),
        xticks=np.arange(2017, 3018, 200),
        xtick_labels=np.arange(2017, 3018, 200),
        xlabel='Year',
        ylabel='SST (K)',
        mask='./example_data/hpx64_1varCoupledOcean-z1000-ws10-olr.zarr',
    ),
    savefig_params = {
        # 'fname': './figure1.pdf',
        'fname': './figure1.png',
        'dpi': 300,
        'bbox_inches': 'tight',
    },
)

# helper function which takes datetime objects and selects associated forecast data, caches selected slices
def get_year(params):
    def get_dir(file):
        return os.path.dirname(file)
    os.makedirs(get_dir(params['output_file']),exist_ok=True)
    
    ds = xr.open_dataset(params['forecast_file'],chunks={'step':10})
    # if ds 'step' is already in datetime format, we can directly select the desired forecast
    if isinstance(ds.step.values[0],np.datetime64):
        ds = ds.sel(step=params['resample_daterange'])
    else:
        dates_to_timedelta = params['resample_daterange'] - params['forecast_init']
        ds = ds.sel(step=dates_to_timedelta)
    print(f'Writing to {params["output_file"]}')
    with ProgressBar():
        ds.to_netcdf(params['output_file'])
    ds.close()
    print('Done.')
    print(xr.open_dataset(params['output_file']))

# creates a linear fit of the data and returns the fit and the slope in percent per century
def linear_fit(ts, years):

    # Ensure the time series has a 'time' coordinate
    if 'year' not in ts.coords:
        raise ValueError("The input time series must have a 'year' coordinate.")
    ts=ts.sel(year=years)
    
    # Extract time and data values
    time_values = ts['year'].values.astype(float)  # Convert to float for polyfit
    data_values = ts.values
    
    # Perform linear regression
    slope, intercept = np.polyfit(time_values, data_values, 1)
    
    # Calculate the linear fit
    linear_fit_values = slope * time_values + intercept
    
    # Create a DataArray for the linear fit
    linear_fit_da = xr.DataArray(linear_fit_values, coords=[ts['year']], dims=['year'])
    
    return linear_fit_da, slope*100

#region Broken TS
def broken_ts(fig, ax_first5, ax_last5, params):

    if not os.path.exists(f"{params['first_5_forecast_file']}.nc"):
        if 'resampled_dates_first5' not in params.keys():
            resampled_dates = pd.date_range('2017-01-01','2022-01-01',freq='2D' if params['eval_variable'] == 'sst' else '6h') # backwards compatible. from before we were experimenting with different inits 
        else:
            resampled_dates = params['resampled_dates_first5']
        PARAMS_first_5= {
            'forecast_file': f"{params['forecast_file']}.nc",
            'forecast_init': params['forecast_init'], 
            'resample_daterange': resampled_dates,
            'output_file': f"{params['first_5_forecast_file']}.nc",
        }
        get_year(PARAMS_first_5)
    if not os.path.exists(f"{params['last_5_forecast_file']}.nc"):
        if 'resampled_dates_last5' not in params.keys():
            resampled_dates = pd.date_range('2112-01-02','2117-01-01',freq='2D') if params['eval_variable'] == 'sst' \
                else pd.date_range('2112-01-01','2117-01-01',freq='6h'), # 100th year for 48h forecast starts on jan 2
        else:
            resampled_dates = params['resampled_dates_last5']
        PARAMS_last_5= {
            'forecast_file': f"{params['forecast_file']}.nc",
            'forecast_init': params['forecast_init'], 
            'resample_daterange': resampled_dates,
            'output_file': f"{params['last_5_forecast_file']}.nc",
        }
        get_year(PARAMS_last_5)

    def plot_cycle(fig,
                   ax,
                   da, 
                   verif_da,
                   rolling_params,
                   rolling_params_ref,
                   scale_factor,
                   cmap,
                   levels,
                   ref_line,
                   add_verif_ref,
                   colorbar_label,
                   xticks,
                   xtick_labels,
                   xlabel,
                   processed_fcst,
                   processed_fcst_ref=None,
                   processed_verif=None,
                   right_plot=False,
                   init=None
    ):
            
        # claculated seasonal cycle using xarray rolling. The "construct" operation stacks the rolling windows as a new dimension and 
        # and allows for calculating the mean accross the window while ignoring NaNs. https://github.com/pydata/xarray/issues/4278
        #                                                       -> this is probably what jonathan did for figure 8 in the 2021 paper
        if os.path.exists(processed_fcst):
            print(f'loading processed forecast from {processed_fcst}')
            zonal_da = getattr(xr.open_dataset(processed_fcst),params['eval_variable'])
        else:
            print(f'processing forecast and saving to {processed_fcst}')
            # mask provided for sst, weight zonal average by ocean fraction
            if 'mask' in params.keys():
                # change dimension names, for valid combination with sst
                mask = 1 - xr.open_dataset(params['mask'])['lsm'].rename({'longitude':'lon','latitude':'lat'})
                mask['lon'] = mask['lon'].astype(int)
                mask['lat'] = mask['lat'].astype(int)
                zonal_da = (da * mask).sum(dim='lon') / mask.sum(dim='lon') # weight average by ocean fraction
                zonal_da.name = params['eval_variable'] # for proper combination with data
                zonal_da = zonal_da.squeeze().transpose().rolling(**rolling_params).construct('new').mean('new',skipna=True)
            else:
                zonal_da = da.mean(dim='lon').squeeze().transpose().rolling(**rolling_params).construct('new').mean('new',skipna=True)
            zonal_da.to_netcdf(processed_fcst)

        # plot everything in seconds 
        seconds = da.step.values / 1e9

        # forecast 
        im = ax.contourf(seconds,da.lat,zonal_da/scale_factor, cmap=cmap, levels = levels, extend='both')  

        # add ref line from forecast and verif 
        if ref_line is not None:
            # again use the rolling->construct method to smooth data, this time over a 15 day window (15 days * 4 steps per day = 60) 
            if processed_fcst_ref is not None:
                if os.path.exists(processed_fcst_ref):
                    print(f'loading processed forecast from {processed_fcst_ref}')
                    ref_fcst = getattr(xr.open_dataset(processed_fcst_ref),params['eval_variable'])
                else:
                    print(f'processing forecast and saving to {processed_fcst_ref}')
                    if 'mask' in params.keys():
                        # change dimension names, for valid combination with sst
                        mask = 1 - xr.open_dataset(params['mask'])['lsm'].rename({'longitude':'lon','latitude':'lat'})
                        mask['lon'] = mask['lon'].astype(int)
                        mask['lat'] = mask['lat'].astype(int)
                        ref_fcst = (da * mask).sum(dim='lon') / mask.sum(dim='lon') # weight average by ocean fraction
                        ref_fcst.name = params['eval_variable'] # for proper combination with data
                        ref_fcst = zonal_da.squeeze().transpose().rolling(**rolling_params_ref).construct('new').mean('new',skipna=True)
                    else:
                        ref_fcst = da.mean(dim='lon').squeeze().transpose().rolling(**rolling_params_ref).construct('new').mean('new',skipna=True)
                    ref_fcst.to_netcdf(processed_fcst_ref)
            else:
                if 'mask' in params.keys():
                    # change dimension names, for valid combination with sst
                    mask = 1 - xr.open_dataset(params['mask'])['lsm'].rename({'longitude':'lon','latitude':'lat'})
                    mask['lon'] = mask['lon'].astype(int)
                    mask['lat'] = mask['lat'].astype(int)
                    ref_fcst = (da * mask).sum(dim='lon') / mask.sum(dim='lon') # weight average by ocean fraction
                    ref_fcst.name = params['eval_variable'] # for proper combination with data
                    ref_fcst = zonal_da.squeeze().transpose().rolling(**rolling_params_ref).construct('new').mean('new',skipna=True)
                else:
                    ref_fcst = da.mean(dim='lon').squeeze().transpose().rolling(**rolling_params_ref).construct('new').mean('new',skipna=True)
            ref_months = [pd.Timedelta(d).days/30 for d in da.step.values]
            # ensure that lat is the first dimension of ref
            ref_fcst = ref_fcst.transpose('lat','step')
            if params['eval_variable'] == 'sst':
                ax.contour(seconds,da.lat,ref_fcst/scale_factor,levels=[ref_line],colors='grey')
            else:
                ax.contour(seconds,da.lat,ref_fcst/scale_factor,levels=[ref_line],colors='white')
            if add_verif_ref:
                if processed_verif is not None:
                    if os.path.exists(processed_verif):
                        ref_verif = getattr(xr.open_dataset(processed_verif),params['eval_variable'])
                    else:
                        print(f'processing verif and saving to {processed_verif}')
                        if 'mask' in params.keys():
                            # change dimension names, for valid combination with sst
                            mask = 1 - xr.open_dataset(params['mask'])['lsm'].rename({'longitude':'lon','latitude':'lat'})
                            mask['lon'] = mask['lon'].astype(int)
                            mask['lat'] = mask['lat'].astype(int)
                            ref_verif = (verif_da * mask).sum(dim='lon') / mask.sum(dim='lon')
                            ref_verif.name = params['eval_variable'] # for proper combination with data
                            ref_verif = ref_verif.squeeze().transpose().rolling(step=60,center=True).construct('new').mean('new',skipna=True)
                        else:
                            ref_verif = verif_da.mean(dim='lon').squeeze().transpose().rolling(step=60,center=True).construct('new').mean('new',skipna=True)
                        ref_verif.to_netcdf(processed_verif)
                else:
                    if 'mask' in params.keys():
                        # change dimension names, including variable name, for valid combination with sst
                        mask = 1 - xr.open_dataset(params['mask'])['lsm'].rename({'longitude':'lon','latitude':'lat'})
                        mask['lon'] = mask['lon'].astype(int)
                        mask['lat'] = mask['lat'].astype(int)
                        ref_verif = (verif_da * mask).sum(dim='lon') / mask.sum(dim='lon')
                        ref_verif.name = params['eval_variable'] # for proper combination with data
                        ref_verif = ref_verif.squeeze().transpose().rolling(**rolling_params_ref).construct('new').mean('new',skipna=True)
                    else:
                        ref_verif = verif_da.mean(dim='lon').squeeze().transpose().rolling(**rolling_params_ref).construct('new').mean('new',skipna=True)
                ax.contour(seconds,da.lat,ref_verif/scale_factor,levels=[ref_line],colors='black')

        # labels and ticks
        ax.set_xticks(xticks)
        if right_plot:
            ax.set_xticklabels(xtick_labels, color='red')
            ax.yaxis.set_tick_params(labelleft=False, left=False)
            ax.yaxis.set_tick_params(labelright=True, right=True)
            pos = ax.get_position()
            ax.spines['left'].set_visible(False)
        else:
            ax.set_xticklabels(xtick_labels)
            ax.yaxis.set_tick_params(labelleft=True, left=True)
            ax.yaxis.set_tick_params(labelright=False, right=False) # yticks on right side
            ax.set_ylabel('Latitude',fontsize=12)
            pos = ax.get_position()
            ax.spines['right'].set_visible(False)        

        # colorbar 
        fig.subplots_adjust(right=.885)
        cbax = fig.add_axes([.92,.525,.01,.3595])
        fig.colorbar(im,cax=cbax,label=colorbar_label)

        # return 
        return fig,ax

    # initialize evaluator around atmos forecast file and remap to lat lon 
    fcst_first5 = ev.EvaluatorHPX(
               forecast_path = f'{params["first_5_forecast_file"]}'+'.nc',
               verification_path = params['verification_path'],
               eval_variable = params['eval_variable'],
               remap_config = None,
               on_latlon = True,
               times = None,
               poolsize = 30,
               verbose = True,
               ll_file=f'{params["first_5_forecast_file"]}_{params["eval_variable"]}_ll.nc',
    ) 
    # load verification data if indicated
    verif_da = None
    if params['add_verif_ref'] and not os.path.exists(f'{params["first_5_forecast_file"]}_{params["eval_variable"]}_ll_ProcessedVerif.nc'):
        fcst_first5.generate_verification(
            verification_path = params['verification_path'],
            defined_verif_only = False,
        )
        verif_da = fcst_first5.verification_da
    logging.info('making seasonal cycle first 5...')
    fig,ax_first5 = plot_cycle(fig,
        ax_first5,
        da=fcst_first5.forecast_da,
        verif_da=verif_da,
        rolling_params=params['rolling_params_color'],
        rolling_params_ref=params['rolling_params_contour'],
        scale_factor=params['scale_factor'],
        cmap=params['cmap'],
        levels=params['levels'],
        ref_line=params['ref_line'],
        add_verif_ref=params['add_verif_ref'],
        colorbar_label=params['colorbar_label'],
        xticks = [float((xt-params['forecast_init']).total_seconds()) for xt in params['xticks_left']],
        xtick_labels = params['xtick_labels_left'],
        xlabel = None,
        processed_fcst = f'{params["first_5_forecast_file"]}_{params["eval_variable"]}_ll_ProcessedFcst.nc',
        processed_fcst_ref = f'{params["first_5_forecast_file"]}_{params["eval_variable"]}_ll_ProcessedFcstRef.nc',
        processed_verif = f'{params["first_5_forecast_file"]}_{params["eval_variable"]}_ll_ProcessedVerif.nc', 
    )
    fcst_last5 = ev.EvaluatorHPX(
        forecast_path = f'{params["last_5_forecast_file"]}'+'.nc',
        verification_path = params['verification_path'],
        eval_variable = params['eval_variable'],
        remap_config = None,
        on_latlon = True,
        times = None,
        poolsize = 30,
        verbose = True,
        ll_file=f'{params["last_5_forecast_file"]}_{params["eval_variable"]}_ll.nc',
    )
    fig, ax_last5 = plot_cycle(fig,
        ax_last5,
        da=fcst_last5.forecast_da,
        verif_da=verif_da,
        rolling_params=params['rolling_params_color'],
        rolling_params_ref=params['rolling_params_contour'],
        scale_factor=params['scale_factor'],
        cmap=params['cmap'],
        levels=params['levels'],
        ref_line=params['ref_line'],
        add_verif_ref=False,
        colorbar_label=params['colorbar_label'],
        xticks = [float((xt-params['forecast_init']).total_seconds()) for xt in params['xticks_right']],
        xtick_labels = params['xtick_labels_right'],
        xlabel = None,
        processed_fcst = f'{params["last_5_forecast_file"]}_{params["eval_variable"]}_ll_ProcessedFcst.nc',
        processed_fcst_ref = f'{params["last_5_forecast_file"]}_{params["eval_variable"]}_ll_ProcessedFcstRef.nc',
        processed_verif = f'{params["last_5_forecast_file"]}_{params["eval_variable"]}_ll_ProcessedVerif.nc',
        right_plot=True, 
    )

    def add_axis_break(fig):
        # Create a new Axes that spans the entire figure
        ax = fig.add_axes([0, 0, 1, 1])

        # Hide everything except the line we're going to add
        ax.axis('off')

        # Create a line from (x1, x2) to (y1, y2) in figure coordinates
        lines = []

        top = ax_first5.get_position().bounds[1] + ax_first5.get_position().bounds[3]
        bottom = ax_last5.get_position().bounds[1]
        left = ax_first5.get_position().bounds[0]
        right = ax_last5.get_position().bounds[0] + ax_first5.get_position().bounds[2]

        # top line break
        lines.append(Line2D([0.497, 0.501], [0.86, 0.9], transform=fig.transFigure, linewidth=.75, color='k', zorder=11))
        lines.append(Line2D([0.506, 0.51], [0.86, 0.9], transform=fig.transFigure, linewidth=.75, color='k', zorder=11))
        lines.append(Line2D([0.1255, 0.4985], [0.88, 0.88], transform=fig.transFigure, linewidth=1, color='k', zorder=9))
        lines.append(Line2D([0.5085, 0.8845], [0.88, 0.88], transform=fig.transFigure, linewidth=1, color='k', zorder=9))

        # bottom line break
        lines.append(Line2D([0.497, 0.501], [0.51, 0.55], transform=fig.transFigure, linewidth=.75, color='k', zorder=11))
        lines.append(Line2D([0.506, 0.51], [0.51, 0.55], transform=fig.transFigure, linewidth=.75, color='k', zorder=11))
        lines.append(Line2D([0.1255, 0.4985], [0.53, 0.53], transform=fig.transFigure, linewidth=1, color='k', zorder=9))
        lines.append(Line2D([0.5085, 0.8845], [0.53, 0.53], transform=fig.transFigure, linewidth=1, color='k', zorder=9))

        # Add the line to the Axes
        for line in lines: ax.add_line(line)
        # Force a redraw of the figure
        fig.canvas.draw()

        return fig
    fig = add_axis_break(fig)
    return fig, ax_first5, ax_last5
#endregion

#region Verif TS
def verif_ts(fig, ax_verif, params):

    if not os.path.exists(f"{params['first_5_forecast_file']}.nc"):
        PARAMS_first_5= {
            'forecast_file': f"{params['forecast_file']}.nc",
            'forecast_init': params['forecast_init'], 
            'resample_daterange': pd.date_range('2017-01-01','2022-01-01',freq='2D' if params['eval_variable'] == 'sst' else '6h'),
            'output_file': f"{params['first_5_forecast_file']}.nc",
        }
        get_year(PARAMS_first_5)

    def plot_cycle(fig,
                   ax,
                   da, 
                   verif_da,
                   rolling_params,
                   rolling_params_ref,
                   scale_factor,
                   cmap,
                   levels,
                   ref_line,
                   add_verif_ref,
                   colorbar_label,
                   xticks,
                   xtick_labels,
                   xlabel,
                   processed_fcst,
                   processed_fcst_ref=None,
                   processed_verif=None,
                   title='',
                   init=None,
                   seconds=None,
                   ref_line_kwargs={'color':'white'},
    ):

        # plot everything in seconds 
        if seconds is None:
            seconds = da.step.values / 1e9

        # claculated seasonal cycle using xarray rolling. The "construct" operation stacks the rolling windows as a new dimension and 
        # and allows for calculating the mean accross the window while ignoring NaNs. https://github.com/pydata/xarray/issues/4278
        #                                                       -> this is probably what jonathan did for figure 8 in the 2021 paper
        if os.path.exists(processed_fcst):
            logging.info(f'loading processed forecast from {processed_fcst}')
            zonal_da = getattr(xr.open_dataset(processed_fcst),params['eval_variable'])
        else:
            logging.info(f'processing forecast and saving to {processed_fcst}')
            # mask provided for sst, weight zonal average by ocean fraction
            if 'mask' in params.keys():
                # change dimension names, for valid combination with sst
                mask = 1 - xr.open_dataset(params['mask'])['lsm'].rename({'longitude':'lon','latitude':'lat'})
                mask['lon'] = mask['lon'].astype(int)
                mask['lat'] = mask['lat'].astype(int)
                zonal_da = (da * mask).sum(dim='lon') / mask.sum(dim='lon') # weight average by ocean fraction
                zonal_da.name = params['eval_variable'] # for proper combination with data
                zonal_da = zonal_da.squeeze().transpose().rolling(**rolling_params).construct('new').mean('new',skipna=True)
            else:
                zonal_da = da.mean(dim='lon').squeeze().transpose().rolling(**rolling_params).construct('new').mean('new',skipna=True)
            zonal_da.to_netcdf(processed_fcst)

        # forecast 
        im = ax.contourf(seconds,zonal_da.lat,zonal_da/scale_factor, cmap=cmap, levels = levels, extend='both')  

        # add ref line from forecast and verif 
        if ref_line is not None:
            # again use the rolling->construct method to smooth data, this time over a 15 day window (15 days * 4 steps per day = 60) 
            if processed_fcst_ref is not None:
                if os.path.exists(processed_fcst_ref):
                    logging.info(f'loading processed forecast from {processed_fcst_ref}')
                    ref_fcst = getattr(xr.open_dataset(processed_fcst_ref),params['eval_variable'])
                else:
                    logging.info(f'processing forecast and saving to {processed_fcst_ref}')
                    if 'mask' in params.keys():
                        # change dimension names, for valid combination with sst
                        mask = 1 - xr.open_dataset(params['mask'])['lsm'].rename({'longitude':'lon','latitude':'lat'})
                        mask['lon'] = mask['lon'].astype(int)
                        mask['lat'] = mask['lat'].astype(int)
                        ref_fcst = (da * mask).sum(dim='lon') / mask.sum(dim='lon') # weight average by ocean fraction
                        ref_fcst.name = params['eval_variable'] # for proper combination with data
                        ref_fcst = zonal_da.squeeze().transpose().rolling(**rolling_params_ref).construct('new').mean('new',skipna=True)
                    else:
                        print(rolling_params_ref)
                        ref_fcst = da.mean(dim='lon').squeeze().transpose().rolling(**rolling_params_ref).construct('new').mean('new',skipna=True)
                    ref_fcst.to_netcdf(processed_fcst_ref)
            else:
                if 'mask' in params.keys(): # applying mask indicates sst
                    # change dimension names, for valid combination with sst
                    mask = 1 - xr.open_dataset(params['mask'])['lsm'].rename({'longitude':'lon','latitude':'lat'})
                    mask['lon'] = mask['lon'].astype(int)
                    mask['lat'] = mask['lat'].astype(int)
                    ref_fcst = (da * mask).sum(dim='lon') / mask.sum(dim='lon') # weight average by ocean fraction
                    ref_fcst.name = params['eval_variable'] # for proper combination with data
                    ref_fcst = zonal_da.squeeze().transpose().rolling(**rolling_params_ref).construct('new').mean('new',skipna=True)
                else:
                    ref_fcst = da.mean(dim='lon').squeeze().transpose().rolling(step=60,center=True).construct('new').mean('new',skipna=True)
            ref_months = [pd.Timedelta(d).days/30 for d in zonal_da.step.values]
            # ensure that lat is the first dimension of ref
            ref_fcst = ref_fcst.transpose('lat','step')
            if params['eval_variable'] == 'sst':
                ax.contour(seconds,zonal_da.lat,ref_fcst/scale_factor,levels=[ref_line],**ref_line_kwargs)
            else:
                ax.contour(seconds,zonal_da.lat,ref_fcst/scale_factor,levels=[ref_line],**ref_line_kwargs)

        # labels and ticks
        ax.set_xticks(xticks)
        ax.set_xlabel(xlabel,fontsize=14)
        ax.set_xticklabels(xtick_labels)
        ax.yaxis.set_tick_params(labelleft=True, left=True)
        ax.yaxis.set_tick_params(labelright=False, right=False) # yticks on right side
        ax.set_ylabel('Latitude',fontsize=12)

        # return 
        return fig,ax

    # initialize evaluator around atmos forecast file and remap to lat lon 
    fcst_first5 = ev.EvaluatorHPX(
               forecast_path = f'{params["first_5_forecast_file"]}'+'.nc',
               verification_path = params['verification_path'],
               eval_variable = params['eval_variable'],
               remap_config = None,
               on_latlon = True,
               times = None,
               poolsize = 30,
               verbose = True,
               ll_file=f'{params["first_5_forecast_file"]}_{params["eval_variable"]}_ll.nc',
    ) 
    # load verification data if indicated
    verif_da = None
    if params['add_verif_ref'] and not (os.path.exists(f'{params["first_5_forecast_file"]}_{params["eval_variable"]}_ll_ProcessedVerif.nc') and os.path.exists(f'{params["first_5_forecast_file"]}_{params["eval_variable"]}_ll_ProcessedERA5.nc')):
        fcst_first5.generate_verification(
            verification_path = params['verification_path'],
            defined_verif_only = False,
        )
        verif_da = fcst_first5.verification_da
    print('making seasonal cycle first 5...')
    fig,ax_verif = plot_cycle(fig,
        ax_verif,
        da=verif_da,
        verif_da=verif_da,
        rolling_params=params['rolling_params_color'],
        rolling_params_ref=params['rolling_params_contour'],
        scale_factor=params['scale_factor'],
        cmap=params['cmap'],
        levels=params['levels'],
        ref_line=params['ref_line'],
        add_verif_ref=False,
        colorbar_label=params['colorbar_label'],
        xticks = [float((xt-params['forecast_init']).total_seconds()) for xt in params['xticks']],
        xtick_labels = params['xtick_labels'],
        xlabel = 'Year',
        processed_fcst = f'{params["first_5_forecast_file"]}_{params["eval_variable"]}_ll_ProcessedERA5.nc',
        processed_fcst_ref = f'{params["first_5_forecast_file"]}_{params["eval_variable"]}_ll_ProcessedERA5Ref.nc',
        processed_verif = None, 
        title='ERA5',
        seconds=fcst_first5.forecast_da.step.values / 1e9,
        ref_line_kwargs = {'colors':'black'},
    )

    # fig.tight_layout()
    return fig, ax_verif
#endregion

#region Drift TS
def drift_ts(
        file,
        var,
        fig,
        ax, 
        cache_dir = './',
        indexing = None,
        smoothing_func = None,
        linear_fit_params = None,
        ylim = (287.6, 288.0),
        yticks = None,
        xticks = None,
        xtick_labels = None,
        xlabel = None,
        ylabel = None,
        mask = None,
):
    
    # check for cache 
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f'{cache_dir}/{var}_annual.nc'

    if os.path.exists(cache_file):
        logging.info(f'Cache found, loading global annual average for {var} from {file}')
    else:
        
        logging.info(f'Cache not found, calculating global annual average for {var} in {file}')
        # Open the input file
        if mask is not None:
                logging.info(f'Applying mask lsm to {file}')
                ds = xr.open_dataset(file, chunks={'step': 10})[var]
                # load land sea mask and calculat weighted mean
                mask = 1 - xr.open_dataset(mask, engine='zarr')['constants'].sel(channel_c='lsm')
                ds= (ds * mask.values).sum(dim=('face','height','width')) / mask.sum(dim=('face','height','width')) # weight average by ocean fraction
                ds.name = var # for proper combination with data
                ds = ds.squeeze()
        else:
            ds = xr.open_dataset(file, chunks={'step': 10})[var].mean(dim=('face','height','width')).squeeze()

        # change the step dimension to a datetime
        # Extract the year from the time dimension
        years = ds['step'].dt.year

        # Assign the new coordinate and calculate annual average
        annual = ds.assign_coords(year=('step', years.values)).groupby('year').mean('step')

        # save to cache 
        logging.info(f'Saving cache to {cache_file}')
        with ProgressBar():
            annual.to_netcdf(cache_file)

    # load cached time series 
    ds = xr.open_dataset(cache_file)[var]
    if indexing is not None:
        ds = ds.sel(indexing)
    if smoothing_func is not None:
        ds = smoothing_func(ds)

    # get linear fit
    if linear_fit_params is not None:
        linear_fit_da, slope = linear_fit(ds, **linear_fit_params)
    
    # plot time series
    ax.plot(ds.year, ds, linewidth=0.75, color='k')
    ax.set_xlabel(xlabel,fontsize=14)
    if ylabel is not None:
        ax.set_ylabel(ylabel,fontsize=14)
    else:
        ax.set_ylabel(f'{var} (K)',fontsize=14)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_xlim(ds.year.min(), ds.year.max())
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)  
    ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.grid()
    if linear_fit_params is not None:
        label_str = f'Trend: {slope:.1e} K 100 yr$^{{-1}}$'
        label_str = label_str.replace('e-0',r'$\times$10$^{-').replace(' K','}$ K')
        ax.plot(linear_fit_da.year, linear_fit_da, color='r', linewidth=2, label=r'{}'.format(label_str))
        ax.legend(fontsize=12,loc='lower left') 
    return fig, ax
#endregion

#region Plotting
def plot_figure1(
    params,
):
    
    def stretch_left(ax, stretch=0.03):
        pos = ax.get_position()
        new_pos = [pos.x0 - stretch, pos.y0, pos.width+stretch, pos.height]
        ax.set_position(new_pos)
        return ax

    def scootch_up(ax, scootch=0.05):
        pos = ax.get_position()
        new_pos = [pos.x0, pos.y0+scootch, pos.width, pos.height]
        ax.set_position(new_pos)
        return ax
    def add_center_text(text,axis):
        axis.text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='center', transform=axis.transAxes, fontsize=16, color='white', fontweight='bold')
        return axis
    def add_subplot_label(label, axis, x=0.03, y=0.91):
        axis.text(x, y, label, horizontalalignment='center', verticalalignment='center', transform=axis.transAxes, fontsize=19, color='black', fontweight='bold')
        return axis

    # Initialize figure
    fig = plt.figure(figsize=(13*1.2, 7*1.2))
    
    # Create a GridSpec with 1 row and 2 columns
    gs = GridSpec(nrows=4, ncols=2, figure=fig)

    # Adjust the spacing between rows
    fig.subplots_adjust(hspace=0.5, wspace=0.2)

    # initilaize broken_ts axes 
    ax_first5 = fig.add_subplot(gs[0:2, 0])
    ax_last5 = fig.add_subplot(gs[0:2, 1])
    fig, ax_first5, ax_last5 = broken_ts(fig, ax_first5, ax_last5, params['broken_ts_params']) 
    # add center text and subplot label
    add_center_text('Simulated: 2017-2021',ax_first5)
    add_center_text('Simulated: 3012-3016',ax_last5)

    # create first 5 verif ts 
    ax_obs_first5 = fig.add_subplot(gs[2:4, 0])
    fig, ax_obs_first5 = verif_ts(fig, ax_obs_first5, params['first5_verif_params'])
    ax_obs_first5 = scootch_up(ax_obs_first5)
    add_center_text('Observed: 2017-2021',ax_obs_first5)


    # initiailize t2m ts acis 
    ax_t2m = fig.add_subplot(gs[2, 1])
    fig, ax_t2m = drift_ts(**params['t2m_drift_params'], fig=fig, ax=ax_t2m)
    # position axes
    ax_t2m = scootch_up(stretch_left(ax_t2m), scootch=0.01)
    # add subplot label
    # add_subplot_label('C',ax_t2m, y=0.82)

    # initialize sst ts axis
    ax_sst = fig.add_subplot(gs[3, 1])
    fig, ax_sst = drift_ts(**params['sst_drift_params'], fig=fig, ax=ax_sst)
    ax_sst = scootch_up(stretch_left(ax_sst))
    # add_subplot_label('D',ax_sst, y=0.82)

    # save 
    logging.info(f'Saving figure to {params["savefig_params"]["fname"]}')
    # fig.tight_layout()
    fig.savefig(**params['savefig_params'])
#endregion

if __name__ == '__main__':
    plot_figure1(FIGURE1_PARAMS)