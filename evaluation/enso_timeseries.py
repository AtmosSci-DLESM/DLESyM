import numpy as np
import os
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
import sys 
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.dates as date2num
import pandas as pd
from evaluation.evaluators import EvaluatorHPX

def filter_121(ts):
    # Define the 1-2-1 filter
    filter_121 = np.array([1, 2, 1]) / 4

    # Apply the filter to the time series
    ts_sm = np.convolve(ts, filter_121, mode='same')

    return ts_sm

def rolling_12(ts):

    # Apply the filter to the time series
    return pd.DataFrame(ts).rolling(window=12, center=True).mean().values.flatten()

def rolling_6(ts):

    # Apply the filter to the time series
    return pd.DataFrame(ts).rolling(window=6, center=True).mean().values.flatten()

def plot_enso_time_series(
    forecast_file,
    verification_file,
    verif_climo_file,
    enso_region,
    output_directory,
    cache_dir,
    prefix,
    verif_plotting_range=slice('1950-01-01','2020-01-01'),
    fcst_plotting_range=slice('2066-01-01','2116-12-31'),
    seasonal_cycle_range=(26.0,28.5),
    anomaly_range_fcst=(-3.5,3.5),
    anomaly_range_verif=(-3.5,3.5),
    absolute_sst_range=(25.5,28.5),
    plot_file='enso_suite',
    extra_suffix='',
    filter_function=filter_121,
):
    
    # make caching and output directories if they do not exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # initialize evaluator object around forecast 
    fcst_ev = EvaluatorHPX(
        forecast_path = forecast_file,
        verification_path = verification_file,
        eval_variable = 'sst',
        on_latlon = True,
        poolsize = 20,
        ll_file=f'{forecast_file[:-3]}_sst_ll.nc'
    )

    fcst = fcst_ev.forecast_da

    # fix time dimension to correspond target time of the forecast step
    fcst = fcst.assign_coords(step = fcst.time.values + fcst.step)
    fcst = fcst.drop('time')
    
    # select region
    fcst = fcst.sel(enso_region)

    # resample data to monthly, cache
    averages_cache_file = f'{cache_dir}/{prefix}_fcst_monthly_averages_sst.nc'
    if averages_cache_file is not None and os.path.exists(averages_cache_file):
        print(f'Loading cached averages from {averages_cache_file}')
        fcst = xr.open_dataarray(averages_cache_file)
    else:
        print(f'Resampling to monthly and caching to {averages_cache_file}')
        fcst = fcst.resample(step='1M', 
                             label='left', 
                             loffset='15D').mean().sel(step=fcst_plotting_range)
        fcst.to_netcdf(averages_cache_file)
    # resample verif data to monthly, cache
    verif_average_cache_file = f'{cache_dir}/{prefix}_verif_monthly_averages_sst.nc'
    if verif_average_cache_file is not None and os.path.exists(verif_average_cache_file):
        print(f'Loading cached averages from {verif_average_cache_file}')
        verif = xr.open_dataarray(verif_average_cache_file)
    else:
        print(f'Resampling to monthly and caching to {verif_average_cache_file}')
        verif = xr.open_dataset(verification_file)['sst'].rename({'longitude':'lon','latitude':'lat'})
        verif = verif.sel(**enso_region).mean(dim=['lat','lon'])
        verif = verif.resample(time='1M', 
                             label='left', 
                             loffset='15D').mean().sel(time=verif_plotting_range)
        verif.to_netcdf(verif_average_cache_file)

    # fetch climatology
    verif_climo = xr.open_dataset(verif_climo_file)['sst']
    verif_climo_nino = verif_climo.sel(**enso_region)

    # change dimension of forecast to be the same as climo
    fcst['month'] = fcst['step'].dt.month
    fcst_climo = fcst.groupby('month').mean().squeeze()
    # add month to era5 dims
    verif['month'] = verif['time'].dt.month

    # calculate monthly anomalies versus era5 climo and versus forecast climo
    anom_era5 = (verif.groupby('month') - verif_climo_nino).mean(dim=['lat','lon']).squeeze()
    anom_fcst = (fcst.groupby('month') - fcst_climo).mean(dim=['lat','lon']).squeeze()
    abso = fcst.groupby('month').mean(dim=['lat','lon']).squeeze()

    #resample to desired date range
    anom_era5 = anom_era5.sel(time=verif_plotting_range)
    anom_fcst = anom_fcst.sel(step=fcst_plotting_range)
    abso = abso.sel(step=fcst_plotting_range)

    # resample data to monthly, cache
    std_cache_file = f'{cache_dir}/{prefix}_fcst_monthly_std_sst.nc'
    if std_cache_file is not None and os.path.exists(std_cache_file):
        print(f'Loading cached averages from {std_cache_file}')
        fcst_std = xr.open_dataarray(std_cache_file).squeeze()
    else:
        print(f'Resampling to monthly andd caching to {std_cache_file}')

        # initialize evaluator object around forecast 
        fcst_ev = EvaluatorHPX(
            forecast_path = forecast_file,
            verification_path = verification_file,
            eval_variable = 'sst',
            on_latlon = True,
            poolsize = 20,
            ll_file=f'{forecast_file[:-3]}_sst_ll.nc'
        )
        fcst = fcst_ev.forecast_da
        
        # fix time dimension to correspond target time of the forecast step
        fcst = fcst.assign_coords(step = fcst.time.values + fcst.step)
        fcst = fcst.drop('time')
        
        # select region and resample
        fcst = fcst.sel(enso_region).mean(dim=['lat','lon'])
        fcst = fcst.resample(step='1M', 
                             label='left', 
                             loffset='15D').mean().sel(step=fcst_plotting_range)
        fcst_std = fcst.groupby('step.month').std()
        fcst_std.to_netcdf(std_cache_file)
    
    # resample data to monthly, cache
    verif_std_cache_file = f'{cache_dir}/{prefix}_verif_monthly_std_sst.nc'
    if verif_std_cache_file is not None and os.path.exists(verif_std_cache_file):
        print(f'Loading cached averages from {verif_std_cache_file}')
        verif_std = xr.open_dataarray(verif_std_cache_file).squeeze()
    else:
        print(f'Resampling to monthly andd caching to {verif_std_cache_file}')
        # get era5 verification
        verif = xr.open_dataset(verification_file)['sst'].rename({'longitude':'lon','latitude':'lat'})
        verif = verif.sel(**enso_region).mean(dim=['lat','lon'])
        verif = verif.resample(time='1M', 
                             label='left', 
                             loffset='15D').mean().sel(time=verif_plotting_range)
        verif_std = verif.groupby('time.month').std()
        verif_std.to_netcdf(verif_std_cache_file)


    ##################  CLIMO CYCLE PLOTTING ##################
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(fcst_climo.month.values, fcst_climo.mean(dim=['lat','lon']).squeeze().values-273.15, color='b', label='Forecast Climatology')
    ax.plot(fcst_climo.month.values, verif_climo_nino.mean(dim=['lat','lon']).squeeze().values-273.15, color='k', label='ERA5 Climatology')
    ax.set_xticks(fcst_climo.month.values)
    ax.set_ylim(seasonal_cycle_range)
    ax.grid()
    ax.set_xlabel('Month')
    ax.set_ylabel('Nino SST (C)')
    ax.legend()

    def setup_anomaly_plot(ax, xlim, ylim):
        # format y axis
        ax.set_ylabel('SST Anomaly (C)')
        if type(ylim)==np.ndarray:
            ax.set_yticks(ylim)
            ax.set_ylim(ylim[0],ylim[-1])
        else:
            ax.set_ylim(ylim)
        # format x axis
        ax.set_xlabel('Year')
        ax.set_xlim(pd.Timestamp(xlim.start), pd.Timestamp(xlim.stop))
        ax.grid()

        # reference lines and regions
        ax.axhline(y=0, color='grey', linestyle='-',linewidth=1)
        return ax
    
    # tighten and save 
    plot_file = f'{output_directory}/{prefix}_sst_annual_cycle'
    print(f'Saving plot to {plot_file}')
    fig.tight_layout()
    fig.savefig(fname=plot_file+'.png', dpi=300)
    fig.savefig(fname=plot_file+'.pdf', dpi=300)
    plt.close(fig)
    
    ##################  ERA5 ANOMALY PLOTTING ##################

    # create figure, set up axes 
    fig, ax = plt.subplots(figsize=(7,3))
    ax = setup_anomaly_plot(ax, xlim=verif_plotting_range, ylim=anomaly_range_verif)

    # Get time values and filtered anomalies
    time_values = anom_era5.time.values
    filtered_values = filter_function(anom_era5.values)

    # plot anomalies, shade +-
    ax.fill_between(time_values, filtered_values, where=(filtered_values >= 0), interpolate=True, color='red', alpha=0.3)
    ax.fill_between(time_values, filtered_values, where=(filtered_values <= 0), interpolate=True, color='blue', alpha=0.3)
    ax.plot(anom_era5.time.values, filtered_values, color='k')

    # 10 year ticks
    ax.set_xticks([pd.Timestamp(t) for t in pd.date_range(start=verif_plotting_range.start, end=verif_plotting_range.stop, freq='10YS')])
    ax.set_xticklabels([str(t.year) for t in pd.date_range(start=verif_plotting_range.start, end=verif_plotting_range.stop, freq='10YS')])

    # tighten and save
    plot_file = f'{output_directory}/{prefix}_verif_anomaly_ts'
    print(f'Saving plot to {plot_file}')
    fig.tight_layout()
    fig.savefig(fname=plot_file+'.png', dpi=300)
    fig.savefig(fname=plot_file+'.pdf', dpi=300)
    plt.close(fig)

    ##################  FCST ANOMALY PLOTTING ##################

    # create figure, set up axes
    fig, ax = plt.subplots(figsize=(7,3))
    ax = setup_anomaly_plot(ax, xlim=fcst_plotting_range, ylim=anomaly_range_fcst)

    # Get time values and filtered values
    time_values = anom_fcst.step.values
    filtered_values = filter_function(anom_fcst.values)

    # plot anomalies, shade +-
    ax.fill_between(time_values, filtered_values, where=(filtered_values >= 0), interpolate=True, color='red', alpha=0.3)
    ax.fill_between(time_values, filtered_values, where=(filtered_values <= 0), interpolate=True, color='blue', alpha=0.3)
    ax.plot(time_values,filtered_values, color='k')

    # y axis labels on right side 
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')

    # 10 year ticks
    ax.set_xticks([pd.Timestamp(t) for t in pd.date_range(start=fcst_plotting_range.start, end=fcst_plotting_range.stop, freq='10YS')])
    ax.set_xticklabels([str(t.year) for t in pd.date_range(start=fcst_plotting_range.start, end=fcst_plotting_range.stop, freq='10YS')])

    # tighten and save
    plot_file = f'{output_directory}/{prefix}_fcst_anomaly_ts'
    print(f'Saving plot to {plot_file}')
    fig.tight_layout()
    fig.savefig(fname=plot_file+'.png', dpi=300)
    fig.savefig(fname=plot_file+'.pdf', dpi=300)
    plt.close(fig)

    ##################  FCST ABSOLUTE PLOTTING ##################

    # create figure, set up axes
    fig, ax = plt.subplots(figsize=(7,3))
    ax.grid()

    # format y axis
    ax.set_ylabel('Nino Absolute SST (C)')
    # format x axis
    ax.set_xlabel('Year')
    ax.set_xlim(pd.Timestamp(fcst_plotting_range.start),
                pd.Timestamp(fcst_plotting_range.stop))
    ax.set_ylim(absolute_sst_range)

    # plot 
    ax.plot(abso.step.values, abso.values-273.15, color='k')

    # tighten and save
    plot_file = f'{output_directory}/{prefix}_fcst_absolute_ts'
    print(f'Saving plot to {plot_file}')
    fig.tight_layout()
    fig.savefig(fname=plot_file+'.png', dpi=300)
    fig.savefig(fname=plot_file+'.pdf', dpi=300)
    plt.close(fig)

    ##################  STD SEASONALITY PLOTTING ##################

    # create figure
    fig, ax = plt.subplots(figsize=(7,3))

    # plot stds
    ax.bar(fcst_std['month'].values, fcst_std.values.squeeze(), color='b', alpha=.5, label='Forecasted')
    ax.bar(verif_std['month'].values, verif_std.values.squeeze(), color='k', alpha=.2, label='ERA5')

    # format plot
    ax.set_ylim(0,1.5)
    ax.set_xlabel('Month')
    ax.set_ylabel('Standard deviation')
    ax.legend()

    # tighten and save figure
    plot_file = f'{output_directory}/{prefix}_std_seasonality'
    print(f'Saving plot to {plot_file}')
    fig.tight_layout()
    fig.savefig(fname=plot_file+'.png', dpi=300)
    fig.savefig(fname=plot_file+'.pdf', dpi=300)
    plt.close(fig)

if __name__ == '__main__':


    plot_enso_time_series(
        forecast_file = '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/ocean_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit.nc',
        verification_file = '/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_sst.nc',
        verif_climo_file = '~/WeeklyNotebooks/2023.02.20/monthly_sst_climatology_1deg.nc',
        output_directory='enso_timeseries_analysis',
        cache_dir='cache',
        fcst_plotting_range = slice('2017-01-01','2117-01-01'),
        verif_plotting_range = slice('1970-01-01','2020-01-01'),
        enso_region={
            'lat':slice(5,-5),
            'lon':slice(360-170,360-120)
        }, 
        seasonal_cycle_range=(26.0,28.5),
        anomaly_range_fcst=np.arange(-.75,.76,.25),
        anomaly_range_verif=(-3.5,3.5),
        absolute_sst_range=(26.25,28.25),
        prefix='100yr_hpx64_6month_rolling',
        filter_function= rolling_6,
    )