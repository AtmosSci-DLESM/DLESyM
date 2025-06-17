import os
import numpy as np 
import xarray as xr 
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import pandas as pd
import scipy.stats as stats
from scipy.signal import detrend

def _detrend_no_annual(data):

    mean = data.mean(dim='time')
    monthly_clim = data.groupby("time.month").mean("time")
    da_anom = data.groupby("time.month") - monthly_clim
    # remove linear trend
    

    # reinstate time 
    da_anom['time'] = data['time']
    return detrend(da_anom.values, type='linear')

def main(
        forecast_file: str,
        forced_forecast_file: str,
        reference_file: str,
        output_file: str,
        cache_dir: str,
        overwrite_cache: bool = False,
        var='t2m0',
        var_name=r"T$_{2m}$"
):

    # caching data to avoid recomputing
    cache_file_forecast = f"{cache_dir}/forecast_monthly_{var}.nc"
    cache_file_forced = f"{cache_dir}/forced-forecast_monthly_{var}.nc"
    cache_file_reference = f"{cache_dir}/era5_monthly_{var}.nc"


    # check caches, overwrite flag
    if overwrite_cache or not (os.path.exists(cache_file_forecast) and \
                               os.path.exists(cache_file_forced) and \
                               os.path.exists(cache_file_reference)):
        
        # open the forecast and reference files
        forecast = xr.open_dataset(forecast_file, chunks='auto')[var]
        forced_forecast = xr.open_dataset(forced_forecast_file, chunks='auto')[var]
        reference = xr.open_zarr(reference_file, chunks='auto').targets.sel(channel_out=var)

        # replace the step coordinate in the forecast with valid_time
        valid_time = forecast['step'].values + forecast['time'].values
        forecast = forecast.rename({'step': 'valid_time'})
        forecast['valid_time'] = valid_time

        # replace the step coordinate in the forced forecast with valid_time,
        # for uniform forecast processing
        forced_forecast = forced_forecast.rename({'step': 'valid_time'})

        # select the last 35 years of forecast and reference data
        # we're also going to only select every 6 hours of the reference data
        # this probably isn't necessary but will most appropriate comparison
        forecast = forecast.sel(valid_time=slice('2017-01-01', '2116-12-31T2300'))
        forced_forecast = forced_forecast.sel(valid_time=pd.date_range('1984-01-01', '2016-12-31T2300', freq='6H'))
        reference = reference.sel(time=pd.date_range('1984-01-01', '2016-12-31T2300', freq='6H'))

        # calculate monthly averages
        forecast = forecast.resample(valid_time='1M').mean().mean(dim=('face','height','width'))
        forced_forecast = forced_forecast.resample(valid_time='1M').mean().mean(dim=('face','height','width'))
        reference = reference.resample(time='1M').mean().mean(dim=('face','height','width'))

        # clean up data, common dimensions, coordinates, and names
        forecast = forecast.squeeze().drop('time').rename({'valid_time': 'time'})
        forced_forecast = forced_forecast.squeeze().drop('time').rename({'valid_time': 'time'})
        reference = reference.squeeze().drop(['channel_out','level']).rename(var)

        # save the data to cache in chunks
        forecast = forecast.chunk({'time': 10000})
        forced_forecast = forced_forecast.chunk({'time': 10000})
        reference = reference.chunk({'time': 10000})           

        with ProgressBar():
            forecast.to_netcdf(cache_file_forecast, mode='w', compute=True)
            forced_forecast.to_netcdf(cache_file_forced, mode='w', compute=True)
            reference.to_netcdf(cache_file_reference, mode='w', compute=True)
    
    forecast_t = xr.open_dataset(cache_file_forecast)[var]
    forced_forecast_t = xr.open_dataset(cache_file_forced)[var]
    reference_t = xr.open_dataset(cache_file_reference)[var]

    # shorthand for saving multiple formats
    def save_fig(fig, output_file):
        fig.tight_layout()
        print(f"Saving figure to {output_file}.png and {output_file}.pdf")
        fig.savefig(output_file + '.png', dpi=300)
        fig.savefig(output_file + '.pdf', dpi=300)
        plt.close(fig)


    # first, let's plot time series detrended with the annual cycle removed
    # and the KDE of the anomalies
    era5_vals = _detrend_no_annual(reference_t)
    dlesym_vals = _detrend_no_annual(forecast_t)
    dlesym_forced_vals = _detrend_no_annual(forced_forecast_t)

    # Create the figure and layout
    fig = plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 13], wspace=0.05)

    # KDE plot on the left
    ax_kde = fig.add_subplot(gs[0])
    sns.kdeplot(era5_vals, ax=ax_kde, color='black', vertical=True)
    sns.kdeplot(dlesym_vals, ax=ax_kde, color='red', vertical=True)
    sns.kdeplot(dlesym_forced_vals, ax=ax_kde, color='red',linestyle='dashed', vertical=True)

    ax_kde.set_xlabel('Density')
    ax_kde.set_ylabel('')  # Let main plot handle y-label
    ax_kde.set_xticks([])
    # ax_kde.tick_params(left=False,right=True)
    ax_kde.set_yticks(np.arange(-.4, .5, .2))
    ax_kde.yaxis.set_ticks_position('left')
    ax_kde.yaxis.set_label_position('left')
    ax_kde.set_ylabel(var_name + ' Anomaly (K)', fontsize=14)
    ax_kde.invert_xaxis()  # Put it on the left

    # Time series plot on the right
    ax_ts = fig.add_subplot(gs[1], sharey=ax_kde)
    ax_ts.set_xticks(pd.date_range('1984-01-01', '2116-12-31', freq='20Y'))
    ax_ts.set_xlim(pd.Timestamp('1984-01-01'), pd.Timestamp('2116-12-31'))
    ax_ts.set_ylim(-.45, .45)
    ax_ts.yaxis.set_ticks_position('right')
    ax_ts.yaxis.set_label_position('right')
    ax_ts.set_ylabel(var_name + ' Anomaly (K)', fontsize=14)
    ax_ts.set_xlabel('Year', fontsize=14)
    ax_ts.axhline(0, color='grey', linewidth=1.5)
    ax_ts.set_xticks(pd.date_range('1985-01-01', '2116-12-31', freq='20Y'))
    ax_ts.set_xticklabels(pd.date_range('1985-01-01', '2116-12-31', freq='20Y').strftime('%Y'))

    # Plot the time series
    ax_ts.plot(reference_t['time'], _detrend_no_annual(reference_t), label='ERA5', color='black', linewidth=1)
    ax_ts.plot(forecast_t['time'], _detrend_no_annual(forecast_t), label='DL$ESy$M', color='red', linewidth=1)
    ax_ts.plot(forced_forecast_t['time'], _detrend_no_annual(forced_forecast_t), label='DL$ESy$M forced-SST', color='red', linestyle='dashed', linewidth=1)

    # Legend and save
    fig.legend(loc=(.54, .12), fontsize=12, ncol=3)

    # save
    save_fig(fig, output_file)

    
    # calculate forced and reference correlation, print 
    corr_ref_forced, pval = stats.pearsonr(
        _detrend_no_annual(reference_t),
        _detrend_no_annual(forced_forecast_t)
    )
    print(f"Correlation between forced forecast and reference: {corr_ref_forced:.2f} (p-value: {pval:.2e})")





if __name__ == "__main__":
    
    # t2m
    var_name = r"T$_{2m}$"
    main(
        forecast_file='/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit.nc',
        forced_forecast_file='/home/disk/rhodium/nacc/forecasts/testing_dlesym/forced_atmos_dlesym_1983-2017.nc',
        reference_file='/home/disk/rhodium/dlwp/data/HPX64/hpx64_1983-2017_3h_9varCoupledAtmos-sst.zarr',
        output_file='global_temperature_variability',
        cache_dir='/home/disk/brume/nacc/DLESyM/evaluation/cache',
        overwrite_cache=False,
    )
    # t850 
    main(
        forecast_file='/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit.nc',
        forced_forecast_file='/home/disk/rhodium/nacc/forecasts/testing_dlesym/forced_atmos_dlesym_1983-2017.nc',
        reference_file='/home/disk/rhodium/dlwp/data/HPX64/hpx64_1983-2017_3h_9varCoupledAtmos-sst.zarr',
        output_file='global_t850_variability',
        cache_dir='/home/disk/brume/nacc/DLESyM/evaluation/cache',
        overwrite_cache=False,
        var='t850',
        var_name = r"T$_{850}$",
    )

