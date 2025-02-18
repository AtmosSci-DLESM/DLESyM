
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as patches
import matplotlib.colors as mcolors 
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import os
import logging
logging.basicConfig(level=logging.INFO)


def monsoon_hovmoller(
        file,
        time_period,
        forecast=False,
        lon_range=slice(70, 100),
        lat_range=slice(10, 25),
        open_func=None,
        climatology=False,
        hovmoller_cache=None, 
):
    """
    Extract a hovmoller diagram of the south indian monsoon.

    Args:
        file (str): Path to the file containing the monsoon index.
        time_period (str): Time period to plot.
        forecast (bool): If True, expect forecast format.
        lon_range (slice): Longitude range to select. Default to SIM region.
        lat_range (slice): Latitude range to select. Default to SIM region.
        open_func (function): If not None, apply this function to open the file.
        climatology (bool): If True, calculate the dayofyear climatology.
        hovmoller_cache (str): Path to cache the hovmoller diagram.
    """

    # check cache is empty before calculating
    if (hovmoller_cache is not None) and os.path.exists(hovmoller_cache):
        logging.info(f"Loading hovmoller diagram from cache: {hovmoller_cache}")
        return xr.open_dataset(hovmoller_cache)['olr'], xr.open_dataset(hovmoller_cache)['olr'].time, xr.open_dataset(hovmoller_cache)['lat']
    
    # open file, select monsoon region and filter time
    if not forecast:
        olr = xr.open_dataset(file)['olr'] if open_func is None else open_func(file)
        olr = olr[:,::-1,:].sel(time=time_period, lon=slice(70, 100), lat=slice(10, 25))
    else:
        olr = xr.open_dataset(file)['olr'] if open_func is None else open_func(file)
        # fix time dimension of forecast
        olr = olr.assign_coords(step=olr.step.values+olr.time.values).drop('time').squeeze().rename(step='time')[:,::-1,:]
        # select monsoon region and filter time
        olr = olr.sel(time=time_period, lon=slice(70, 100), lat=slice(10, 25))

    # resample to daily and take zonal average
    olr = olr.resample(time='1D').mean().rolling(time=10, center=True, min_periods=1).mean().mean(dim='lon').squeeze()

    if climatology:
        # calculate the dayofyear climatology
        olr = olr.groupby('time.dayofyear').mean(dim='time')
        olr = olr.rename(dayofyear='time')
        olr = olr.assign_coords(time=[pd.Timestamp(f"2000-01-01") + pd.Timedelta(days=int(dayofyear)-1) for dayofyear in olr.time.values])

    # cache the result
    if hovmoller_cache is not None:
        logging.info(f"Caching hovmoller diagram to: {hovmoller_cache}")
        olr.to_netcdf(hovmoller_cache)

    return olr, olr.time, olr.lat

def calculate_monsoon_index(
        file,
        time_range,
        forecast=False,
        lon_range=slice(70, 100),
        lat_range=slice(10, 25),
        monsoon_index_cache=None,
        climatology=False,
        climatology_time_dim=None,
        detrend=False,
        open_func=None,
        smoothing=None,
):
    """
    Calculate the monsoon index.

    Args:
        file (str): Path to the file containing the monsoon index.
        time_range (str): Time period to calculate the monsoon index.
        forecast (bool): If True, expect forecast format.
        lon_range (slice): Longitude range to select. Default to SIM region.
        lat_range (slice): Latitude range to select. Default to SIM region.
        monsoon_index_cache (str): Path to cache the monsoon index.
        climatology (bool): If True, calculate the dayofyear climatology.
        climatology_time_dim (str): If not None, rename the time dimension to this value.
        detrend (bool): If True, remove linear trend in the monsoon index.
        open_func (function): If not None, apply this function to open the file.
        smoothing (int): If not None, apply a rolling mean to the monsoon index.
    """

    # check cache is empty before calculating
    if (monsoon_index_cache is not None) and os.path.exists(monsoon_index_cache):
        logging.info(f"Loading monsoon index from cache: {monsoon_index_cache}")
        return xr.open_dataset(monsoon_index_cache)['olr'], xr.open_dataset(monsoon_index_cache)['olr'].time
    
    # open file, select monsoon region and filter time 
    if not forecast:
        olr = xr.open_dataset(file)['olr'] if open_func is None else open_func(file)
        olr = olr[:,::-1,:].sel(lon=lon_range, lat=lat_range, time=time_range)
    else:
        olr = xr.open_dataset(file)['olr'] if open_func is None else open_func(file)
        olr = olr.assign_coords(step=olr.step.values+olr.time.values).drop('time').squeeze().rename(step='time') # fix time dimension
        olr = olr[:,::-1,:] # enforce ascending latitude
        olr = olr.sel(lon=lon_range, lat=lat_range, time=time_range)

    # daily means of regional average OLR
    olr = olr.resample(time='1D').mean(skipna=True).mean(dim=['lat', 'lon'], skipna=True)
    # smooth if indicated 
    if smoothing is not None:
        olr = olr.rolling(time=smoothing, center=True, min_periods=1).mean()
    # calculate climatology if indicated
    if climatology:
        # calculate the dayofyear climatology
        olr = olr.groupby('time.dayofyear').mean(dim='time', skipna=True)
        olr = olr.rename(dayofyear='time')
        olr = olr.assign_coords(time=[pd.Timestamp(f"2017-01-01") + pd.Timedelta(days=int(dayofyear)-1) for dayofyear in olr.time.values] if climatology_time_dim is None else climatology_time_dim)

    # detrend the monsoon index
    if detrend:
        logging.info(f"Detrending monsoon index from {file} over {time_range}")
        trend = olr.polyfit('time', 1).polyfit_coefficients.sel(degree=1) * (olr.time-olr.time[0]).astype('float')
        olr = olr - trend.values
        olr.name = 'olr' # rename after detrending

    # cache the result
    if monsoon_index_cache is not None:
        logging.info(f"Caching monsoon index from {file} over {time_range} to: {monsoon_index_cache}")
        olr.to_netcdf(monsoon_index_cache)

    return olr, olr.time

def monsoon_index_anomaly(
        file,
        time_range,
        forecast=False,
        lon_range=slice(70, 100),
        lat_range=slice(10, 25),
        monsoon_index_anom_cache=None,
        months = [6, 7, 8, 9],
        index_calculation_args={},
        mean_months=True,
):
    """
    Calculate the anomaly of the monsoon index.

    Args:
        file (str): Path to the file containing the monsoon index.
        time_range (str): Time period to calculate the monsoon index.
        forecast (bool): If True, expect forecast format.
        lon_range (slice): Longitude range to select. Default to SIM region.
        lat_range (slice): Latitude range to select. Default to SIM region.
        monsoon_index_anom_cache (str): Path to cache the monsoon index anomaly.
        months (list): Months to calculate the anomaly.
        index_calculation_args (dict): Arguments to pass to calculate_monsoon_index.
        mean_months (bool): If True, average the anomaly over the indicated months.
    """

    # check cache is empty before calculating
    if (monsoon_index_anom_cache is not None) and os.path.exists(monsoon_index_anom_cache):
        logging.info(f"Loading monsoon index anomaly from cache: {monsoon_index_anom_cache}")
        return xr.open_dataset(monsoon_index_anom_cache)['olr'], xr.open_dataset(monsoon_index_anom_cache)['olr'].time
    
    logging.info(f"Calculating monsoon index anomaly for {file} over {time_range}")
    # first calcaulte monsoon index
    olr, time = calculate_monsoon_index(
        file=file,
        time_range=time_range,
        forecast=forecast,
        lon_range=lon_range,
        lat_range=lat_range,
        **index_calculation_args,
    )

    # resample olr to monthly
    olr = olr.resample(time='1M', loffset=pd.DateOffset(days=-15)).mean()

    # calculate monthly anomaly
    monthly_climo = olr.groupby('time.month').mean(dim='time')
    olr_monthly_anomaly = olr.groupby('time.month') - monthly_climo

    # select only months in indicated months list
    olr_seasonal_anomaly = olr_monthly_anomaly.where(olr_monthly_anomaly['time.month'].isin(months), drop=True)
    # if annual mean is indicated, average over years
    if mean_months:
        olr_seasonal_anomaly=olr_seasonal_anomaly.groupby('time.year').mean(dim='time')
        # fix time dimension lables and convert back to datetime 
        olr_seasonal_anomaly = olr_seasonal_anomaly.rename(year='time')
        olr_seasonal_anomaly = olr_seasonal_anomaly.assign_coords(time=[pd.Timestamp(f"{int(year)}-08-01") for year in olr_seasonal_anomaly.time.values])

    # cache the result
    if monsoon_index_anom_cache is not None:
        logging.info(f"Caching monsoon index anomaly to: {monsoon_index_anom_cache}")
        olr_seasonal_anomaly.to_netcdf(monsoon_index_anom_cache)

    return olr_seasonal_anomaly, olr_seasonal_anomaly.time

def main(
        forecast_file, 
        verification_file, 
        comparison_time_period,
        forecast_climatology_time_period,
        verification_climatology_time_period,
        forecast_anomaly_time_period,
        verification_anomaly_time_period,
        caching_dir,
        plot_file,
):
        """
        Compile South Indian monsoon analysis into a single figure for publication.

        Args:
            forecast_file (str): Path to the forecast file.
            verification_file (str): Path to the verification file.
            comparison_time_period (str): overlap time period to compare forecast and verification.
            forecast_climatology_time_period (str): Time period to calculate the forecast climatology.
            verification_climatology_time_period (str): Time period to calculate the verification climatology.
            forecast_variabilty_time_period (str): Time period to calculate the forecast anomaly.
            verification_variability_time_period (str): Time period to calculate the verification anomaly.
            caching_dir (str): Path to the directory where to cache plotted values. 
            plot_file (str): Path to save the plot.
        """

        # create cache directory if it does not exist
        os.makedirs(caching_dir, exist_ok=True)

        # Create a figure
        fig = plt.figure(figsize=(14, 8))

        # Create a GridSpec object
        gs = gridspec.GridSpec(4, 4)

        # Create subplots with custom sizes
        ax1 = fig.add_subplot(gs[0:2, 0])
        ax2 = fig.add_subplot(gs[0:2, 1])
        ax3 = fig.add_subplot(gs[2:4, 0])
        ax4 = fig.add_subplot(gs[2:4, 1])
        ax5 = fig.add_subplot(gs[0:2, 2:4])
        ax_ts_obs = fig.add_subplot(gs[2, 2:4])  
        ax_ts_obs_anom = ax_ts_obs.twinx()
        ax_ts_fcst = fig.add_subplot(gs[3, 2:4])
        ax_ts_fcst_anom = ax_ts_fcst.twinx()

        def plot_hovmoller(x, y, z, color_levels, vs, contour_levels, ax):
            # plot hovmoller
            cs = ax.contourf(x, y, z, cmap='RdBu_r', levels=color_levels, **vs)
            ax.contour(x, y, z, colors='black', levels=contour_levels)
            ax.set_xlim(x[0], x[-1])
            date_format = mdates.DateFormatter('%b')
            ax.yaxis.set_major_formatter(date_format)
            ax.yaxis.set_major_locator(mdates.MonthLocator())
            ax.set_xticks(np.arange(x[0], x[-1]+1, 5))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
            ax.tick_params(axis='y', labelrotation=30)
            return cs

        # plot 2017 hovmoller of obs and forecast
        #region
        # obs_2017
        obs_hovmoller_2017, obs_hovmoller_time_2017, hovmoller_lat = monsoon_hovmoller(
            file=verification_file,
            time_period=comparison_time_period,
            forecast=False,
            hovmoller_cache=os.path.join(caching_dir, "obs_hovmoller.nc"),
        )
        plot_hovmoller(
            hovmoller_lat,
            obs_hovmoller_time_2017,
            obs_hovmoller_2017,
            color_levels=np.arange(180, 460+25, 25),
            vs=dict(vmin=180-25, vmax=485+25),
            contour_levels=[330,],
            ax=ax1,
        )
        ax1.set_title('2017 ISCCP')
        ax1.set_xlabel('Latitude')

        # fcst_2017
        fcst_hovmoller_2017, fcst_hovmoller_time_2017, hovmoller_lat = monsoon_hovmoller(
            file=forecast_file,
            time_period=comparison_time_period,
            forecast=True,
            hovmoller_cache=os.path.join(caching_dir, "forecast_hovmoller.nc"),
        )
        plot_hovmoller(
            hovmoller_lat,
            fcst_hovmoller_time_2017,
            fcst_hovmoller_2017,
            color_levels=np.arange(180, 460+25, 25),
            vs=dict(vmin=180-25, vmax=485+25),
            contour_levels=[330,],
            ax=ax2,
        )
        ax2.set_title('2017 DLESM')
        ax2.set_xlabel('Latitude')
        #endregion

        # plot hovmoller of climatology of obs and forecast
        #region
        # obs_climatology
        obs_hovmoller_climo, obs_hovmoller_time_climo, hovmoller_lat = monsoon_hovmoller(
            file=verification_file,
            time_period=verification_climatology_time_period,
            forecast=False,
            climatology=True,
            hovmoller_cache=os.path.join(caching_dir, "obs_hovmoller_climo.nc"),
        )
        plot_hovmoller(
            hovmoller_lat,
            obs_hovmoller_time_climo,
            obs_hovmoller_climo,
            color_levels=np.arange(200, 440+20, 20),
            vs=dict(vmin=180, vmax=480),
            contour_levels=[330,],
            ax=ax3,
        )
        ax3.set_title('ISCCP Climatology')
        ax3.set_xlabel('Latitude')
        # fcst_climatology
        fcst_hovmoller_climo, fcst_hovmoller_time_climo, hovmoller_lat = monsoon_hovmoller(
            file=forecast_file,
            time_period=forecast_climatology_time_period,
            forecast=True,
            climatology=True,
            hovmoller_cache=os.path.join(caching_dir, "forecast_hovmoller_climo.nc"),
        )
        plot_hovmoller(
            hovmoller_lat,
            fcst_hovmoller_time_climo,
            fcst_hovmoller_climo,
            color_levels=np.arange(200, 440+20, 20),
            vs=dict(vmin=180, vmax=480),
            contour_levels=[330,],
            ax=ax4,
        )
        ax4.set_title('DLESM Climatology')
        ax4.set_xlabel('Latitude')
        #endregion

        # plot 2017 and climo SIM index 
        #region

        # obs 2017
        obs_index, obs_index_time = calculate_monsoon_index(
            file=verification_file,
            forecast=False,
            time_range=comparison_time_period,
            monsoon_index_cache=os.path.join(caching_dir, "obs_monsoon_index_2017.nc"),
            detrend=False,
            smoothing=10,
        )
        ax5.plot(obs_index_time, obs_index, label='ISCCP', color='black',linewidth=2)
        # dlesm 2017
        fcst_index, fcst_index_time = calculate_monsoon_index(
            file=forecast_file,
            forecast=True,
            time_range=comparison_time_period,
            monsoon_index_cache=os.path.join(caching_dir, "forecast_monsoon_index_2017.nc"),
            detrend=False,
            smoothing=10,
        )
        ax5.plot(fcst_index_time, fcst_index, label='DLESM', color='red',linewidth=2)
        # obs climo
        obs_index_climo, obs_index_climo_time = calculate_monsoon_index(
            file=verification_file,
            forecast=False,
            time_range=verification_climatology_time_period,
            monsoon_index_cache=os.path.join(caching_dir, "obs_monsoon_index_climo.nc"),
            climatology=True,
            detrend=False,
            smoothing=10,
        )
        ax5.plot(obs_index_climo_time, obs_index_climo, label='ISCCP climo', color='black',linewidth=2,linestyle='--')
        # dlesm climo
        fcst_index_climo, fcst_index_climo_time = calculate_monsoon_index(
            file=forecast_file,
            forecast=True,
            time_range=forecast_climatology_time_period,
            monsoon_index_cache=os.path.join(caching_dir, "forecast_monsoon_index_climo.nc"),
            climatology=True,
            detrend=False,
            smoothing=10,
        )
        ax5.plot(fcst_index_climo_time, fcst_index_climo, label='DLESM climo', color='red',linewidth=2,linestyle='--')
        # style index plot
        def add_sim_subpanel(ax):
            ax = fig.add_axes([0.53, 0.57, 0.15, 0.15], projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE,lw=0.5,color='grey')
            box = [70, 100, 10, 25]
            ax.set_extent([box[0]-5, box[1]+5, box[2]-5, box[3]+5], crs=ccrs.PlateCarree())
            rect = patches.Rectangle((box[0], box[2]), box[1]-box[0], box[3]-box[2],
                                    linewidth=1, edgecolor='r', facecolor='r', alpha=0.2, transform=ccrs.PlateCarree())
            ax.add_patch(rect)
        ax5.grid()
        ax5.legend(loc=4)
        add_sim_subpanel(ax5)
        ax5.set_title('2017 Indian Summer Monsoon Index',fontsize=15)

        #endregion

        # plot time series and anomaly of obs
        #region
        obs_monthly_anom, obs_monthly_anom_time = monsoon_index_anomaly(
            file=verification_file,
            forecast=False,
            time_range=verification_climatology_time_period,
            monsoon_index_anom_cache=os.path.join(caching_dir, "obs_monsoon_index_anom_monthly.nc"),
            months=[1,2,3,4,5,6,7,8,9,10,11,12],
            index_calculation_args = dict(
                monsoon_index_cache=os.path.join(caching_dir, "obs_monsoon_index_monthly.nc"),
                detrend=True,
            ),
            mean_months=False,
        )
        ax_ts_obs.plot(obs_monthly_anom_time, obs_monthly_anom, label='obs', color='black',linewidth=.5,alpha=1)
        obs_index_anom, obs_index_anom_time = monsoon_index_anomaly(
            file=verification_file,
            forecast=False,
            time_range=verification_climatology_time_period,
            monsoon_index_anom_cache=os.path.join(caching_dir, "obs_monsoon_index_anom.nc"),
            index_calculation_args = dict(
                monsoon_index_cache=os.path.join(caching_dir, "obs_monsoon_index.nc"),
                detrend=True,
            )
        )
        ax_ts_obs.set_ylim(-40, 40)
        ax_ts_obs_anom.bar(obs_index_anom_time, obs_index_anom, label='obs', color='red',width=pd.Timedelta(120, 'D'),alpha=.7)
        ax_ts_obs_anom.set_ylim(-20, 20)
        # style
        ax_ts_obs_anom.set_xlim(obs_index_anom_time[0], obs_index_anom_time[-1]+1)
        ax_ts_obs.xaxis.set_major_locator(mdates.YearLocator(5))
        # ax_ts_obs.set_yticklabels([])
        ax_ts_obs_anom.tick_params(axis='y', colors='red') 
        ax_ts_obs.set_title('ISM Anomaly (ISCCP)',fontsize=12)
        ax_ts_obs.grid()
        #endregion

        # plot time series and anomaly of forecast
        #region
        fcst_index_monthly_anom, fcst_index_monthly_anom_time = monsoon_index_anomaly(
            file=forecast_file,
            forecast=True,
            time_range=forecast_climatology_time_period,
            monsoon_index_anom_cache=os.path.join(caching_dir, "forecast_monsoon_index_anom_monthly.nc"),
            months=[1,2,3,4,5,6,7,8,9,10,11,12],
            index_calculation_args = dict(
                monsoon_index_cache=os.path.join(caching_dir, "forecast_monsoon_index_monthly.nc"),
                detrend=True,
            ),
            mean_months=False,
        )
        ax_ts_fcst.plot(fcst_index_monthly_anom_time, fcst_index_monthly_anom, label='DLESM', color='black',linewidth=.5,alpha=1)
        # ax_ts_fcst.plot(fcst_index_time, fcst_index, label='DLESM', color='black',linewidth=.5,alpha=.3)
        # ax_ts_fcst.set_ylim(0, 600)
        fcst_index_anom, fcst_index_anom_time = monsoon_index_anomaly(
            file=forecast_file,
            forecast=True,
            time_range=forecast_climatology_time_period,
            monsoon_index_anom_cache=os.path.join(caching_dir, "forecast_monsoon_index_anom.nc"),
            index_calculation_args = dict(
                monsoon_index_cache=os.path.join(caching_dir, "forecast_monsoon_index.nc"),
            )
        )
        ax_ts_fcst_anom.bar(fcst_index_anom_time, fcst_index_anom, label='DLESM', color='red',width=pd.Timedelta(120, 'D'))
        ax_ts_fcst_anom.set_ylim(-20, 20)
        ax_ts_fcst.set_ylim(-40, 40)
        # set xtick params
        ax_ts_fcst.xaxis.set_major_locator(mdates.YearLocator(5))
        ax_ts_fcst_anom.set_xlim(fcst_index_anom_time[0], fcst_index_anom_time[-1]+1)
        ax_ts_fcst_anom.tick_params(axis='y', colors='red') 
        ax_ts_fcst.set_title('ISM Anomaly (DLESM)',fontsize=12)
        ax_ts_fcst.grid()
        #endregion

        # tighten and save
        fig.tight_layout()
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')

def plot_monsoon(
        verification_file,
        plotting_time,
        climo_time,
        plot_file,
        index_args={},
):
    
    # Create a figure
    fig, ax = plt.subplots(1, 1, figsize=(8,8))

    # calculat index 
    index, time = calculate_monsoon_index(
        file=verification_file,
        time_range=plotting_time,
        forecast=False,
        lon_range=slice(70, 100),
        lat_range=slice(10, 25),
        monsoon_index_cache=None,
        climatology=False,
        **index_args,
    )
    ax.plot(time, index, label='ISCCP', color='blue',linewidth=2)
    ax.set_title('Indian Summer Monsoon Index',fontsize=15)
    ax.invert_yaxis()
    ax.set_ylim(300, 150)
    ax.set_xlim(time[0], time[-1])

    # calculate climo 
    climo, _ = calculate_monsoon_index(
        file=verification_file,
        time_range=climo_time,
        forecast=False,
        lon_range=slice(70, 100),
        lat_range=slice(10, 25),
        monsoon_index_cache=None,
        climatology=True,
        **index_args,
    )
    climo = climo.sel(time=plotting_time)
    ax.plot(climo.time, climo, label='ISCCP climo', color='black',linewidth=3,linestyle='-')
    ax.grid()

    # calculate std of cycle 
    index_std, time_std = calculate_monsoon_index(
        file=verification_file,
        time_range=climo_time,
        forecast=False,
        lon_range=slice(70, 100),
        lat_range=slice(10, 25),
        monsoon_index_cache=None,
        climatology=False,
        **index_args,
    )
    index_std = index_std.groupby('time.dayofyear').std(dim='time')
    index_std = index_std.rename(dayofyear='time')
    index_std = index_std.assign_coords(time=[pd.Timestamp(f"2000-01-01") + pd.Timedelta(days=int(dayofyear)-1) for dayofyear in index_std.time.values])
    index_std = index_std.sel(time=plotting_time)
    ax.fill_between(index_std.time, climo-index_std, climo+index_std, color='grey', alpha=0.4)

    # 
    fig.tight_layout()
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')

def isccp_open(file):

    ds = xr.open_dataset(file)['olr']
    ds = ds*0.548+59.892
    return ds

def dlesm_open(file):

    ds = xr.open_dataset(file)['olr']
    ds = ds*0.548+59.892
    return ds 

def cesm_open(file):

    ds = xr.open_mfdataset(file, engine='netcdf4', combine='by_coords')
    # Change the time dimension from CFTime 360-day year to Gregorian calendar
    ds = ds.convert_calendar(calendar='gregorian',align_on='date')
    ds = ds.chunk({'time': -1})
    ds = ds.interpolate_na(dim='time',fill_value="extrapolate")

    ds = ds.rename(rlut='olr',time='step')['olr']
    
    ds = ds.expand_dims({'time':[ds.step.values[0]]})
    ds = ds.assign_coords({'step':ds.step.values-ds.step.values[0]})[:,:,::-1,:]   
    return ds

def gfdl_open(file):

    ds = xr.open_mfdataset(file, engine='netcdf4', combine='by_coords')

    # Change the time dimension from CFTime 360-day year to Gregorian calendar
    ds = ds.convert_calendar(calendar='gregorian',align_on='date')
    ds = ds.chunk({'time': -1})
    ds = ds.interpolate_na(dim='time',fill_value="extrapolate")

    ds = ds.rename(rlut='olr',time='step')['olr']
    
    ds = ds.expand_dims({'time':[ds.step.values[0]]})
    ds = ds.assign_coords({'step':ds.step.values-ds.step.values[0]})[:,:,::-1,:]    
    return ds

def mpi_open(file):

    ds = xr.open_mfdataset(file, engine='netcdf4', combine='by_coords')
    ds = ds.rename(rlut='olr',time='step')['olr']
    
    ds = ds.expand_dims({'time':[ds.step.values[0]]})
    ds = ds.assign_coords({'step':ds.step.values-ds.step.values[0]})[:,:,::-1,:]    
    return ds

def hadgem_open(file):

    ds = xr.open_mfdataset(file, engine='netcdf4', combine='by_coords')

    # Change the time dimension from CFTime 360-day year to Gregorian calendar
    ds = ds.convert_calendar(calendar='gregorian',align_on='date')
    ds = ds.chunk({'time': -1})
    ds = ds.interpolate_na(dim='time',fill_value="extrapolate")

    ds = ds.rename(rlut='olr',time='step')['olr']
    
    ds = ds.expand_dims({'time':[ds.step.values[0]]})
    ds = ds.assign_coords({'step':ds.step.values-ds.step.values[0]})[:,:,::-1,:]    
    return ds

def plot_multiple_climo(
        verification_file: str,
        simulations: list,
        lat_range: slice = slice(10, 25),
        lon_range: slice = slice(70, 100),
        plot_file: str = None,
):
    
    # open verification file
    climos = []
    kwargs = []
    verif_index, _ = calculate_monsoon_index(
        file=verification_file,
        time_range=slice('1985-01-01','2014-12-31'),
        forecast=False,
        open_func=isccp_open,
        lon_range=lon_range,
        lat_range=lat_range,
        monsoon_index_cache="/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/FigureScripts/monsoon_cache/isccp_2085-2114_climo_fitted.nc",
        climatology=True,
        detrend=False,
        smoothing=10,
    )
    climos.append(verif_index)
    kwargs.append({'label': 'ISCCP', 'color': 'black', 'linestyle': '--'})

    for sim in simulations:
        # open simulation file
        index,_ = calculate_monsoon_index(
            file=sim['filename'],
            time_range=sim['time_range'],
            forecast=True,
            lon_range=lon_range,
            lat_range=lat_range,
            monsoon_index_cache=sim['climo_cache'],
            open_func=sim.get('open_func', None),
            climatology=True,
            detrend=False,
            smoothing=10,
        )
        climos.append(index)
        kwargs.append(sim['plotting_kwargs'])
    
    # plot climos
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    for i, climo in enumerate(climos):
        ax.plot(climo.time, climo, **kwargs[i])

    # style
    ax.grid()
    ax.legend()

    fig.tight_layout()
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')

def plot_single_panel_hovmoller(
    simulation_file,   
    plot_file,
    forecast=True,
    source_name=None,
    time_period=slice('2085-01-01','2114-12-31'),
    open_func=None,
    cache=None,
    contourf_levels=np.arange(200, 440+20, 20),
    contour_levels=[330,],
    vs=dict(vmin=180, vmax=480),
    label_y=True,
):
    def plot_hovmoller(x, y, z, color_levels, vs, contour_levels, ax):
        # plot hovmoller
        cs = ax.contourf(x, y, z, cmap='RdBu_r', levels=color_levels, **vs,extend='both')  
        ax.contour(x, y, z, colors='black', levels=contour_levels)
        ax.set_xlim(x[0], x[-1])
        if label_y:
            date_format = mdates.DateFormatter('%b')
            ax.yaxis.set_major_formatter(date_format)
            ax.yaxis.set_major_locator(mdates.MonthLocator())
            ax.set_xticks(np.arange(15,21, 5))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
            ax.tick_params(axis='y', labelrotation=30, labelsize=12)
        else:
            ax.set_yticklabels([])
        return cs
    
    obs_hovmoller_climo, obs_hovmoller_time_climo, hovmoller_lat = monsoon_hovmoller(
        file=simulation_file,
        time_period=time_period,
        forecast=forecast,
        climatology=True,
        hovmoller_cache=cache,
        open_func=open_func,
    )
    fig, ax = plt.subplots(1, 1, figsize=(2.5,6))
    im = plot_hovmoller(
        hovmoller_lat,
        obs_hovmoller_time_climo,
        obs_hovmoller_climo,
        color_levels=contourf_levels,
        vs=vs,
        contour_levels=contour_levels,
        ax=ax,
    )
    fig.colorbar(im, ax=ax, orientation='horizontal', label='OLR (W/m^2)',aspect=50)
    ax.set_title(f'{source_name}', fontsize=15)
    ax.set_xlabel('Latitude', fontsize=12)
    fig.tight_layout()
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    if False:
        main(
            forecast_file="/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit_olr_ll.nc",
            verification_file="/home/disk/rhodium/bowenliu/remap/era5_1deg_1d_HPX64_1983_2017_olr_ll.nc",
            comparison_time_period=slice('2017-01-01', '2017-06-30'),
            forecast_climatology_time_period= slice('2085-01-01','2114-12-31'),
            verification_climatology_time_period=slice('1985-01-01','2014-12-31'),
            forecast_anomaly_time_period=slice('2085-01-01','2114-12-31'),
            verification_anomaly_time_period=slice('1985-01-01','2014-12-31'),
            caching_dir="/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/FigureScripts/monsoon_cache",
            plot_file="/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/monsoon_figure.png",
        )
    if False:
        plot_monsoon(
            verification_file="/home/disk/rhodium/bowenliu/remap/era5_1deg_1d_HPX64_1983_2017_olr_ll.nc",
            plotting_time=slice('2000-04-01', '2000-10-31'),
            climo_time=slice('1991-01-01','2017-06-30'),
            plot_file="/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/monsoon_findex_testing.png",
            index_args=dict(
                detrend=False,
                smoothing=7,
                climatology_time_dim=pd.date_range('2000-01-01','2000-12-31',freq='D'),
                open_func=isccp_open,
            )
        )
    if False:
        plot_multiple_climo(
            verification_file="/home/disk/rhodium/bowenliu/remap/era5_1deg_1d_HPX64_1983_2017_olr_ll.nc",
            simulations=[
                {
                    'filename': "/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit_olr_ll.nc",
                    'time_range': slice('2085-01-01','2114-12-31'),
                    'open_func': dlesm_open,
                    'climo_cache': "/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/FigureScripts/monsoon_cache/dlesm_2085-2114_climo_fitted.nc",
                    'plotting_kwargs':{
                        'label': 'DLESM',
                        'color': 'red',
                        'linestyle': '-',
                    },
                },
                {
                    'filename': "/home/disk/mercury5/dlwp/cmip6/CESM2/rlut/rlut_day_CESM2_historical_r1i1p1f1_gn_*.nc",
                    'time_range': slice('1985-01-01','2014-12-31'),
                    'open_func': cesm_open,
                    'climo_cache': "/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/FigureScripts/monsoon_cache/cesm2_1985-2014_climo.nc",
                    'plotting_kwargs':{
                        'label': 'CESM2',
                        'color': 'blue',
                        'linestyle': '-',
                    },
                },
                {
                    'filename': "/home/disk/mercury5/dlwp/cmip6/MPI-ESM1-2-HR/rlut/rlut_day_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_*.nc",
                    'time_range': slice('1985-01-01','2014-12-31'),
                    'open_func': mpi_open,
                    'climo_cache': "/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/FigureScripts/monsoon_cache/mpi_1985-2014_climo.nc",
                    'plotting_kwargs':{
                        'label': 'MPI-ESM1-2-HR',
                        'color': 'orange',
                        'linestyle': '-',
                    },
                },
                {
                    'filename': "/home/disk/mercury5/dlwp/cmip6/HadGEM3-GC31-LL/rlut/rlut_day_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_*.nc",
                    'time_range': slice('1985-01-01','2014-12-31'),
                    'open_func': hadgem_open,
                    'climo_cache': "/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/FigureScripts/monsoon_cache/hadgem_1985-2014_climo.nc",
                    'plotting_kwargs':{
                        'label': 'HadGEM3-GC31',
                        'color': 'green',
                        'linestyle': '-',
                    },
                },
                {
                    'filename': "/home/disk/mercury5/dlwp/cmip6/GFDL-CM4/rlut/rlut_day_GFDL-CM4_historical_r1i1p1f1_gr2_*.nc",
                    'time_range': slice('1985-01-01','2014-12-31'),
                    'open_func': gfdl_open,
                    'climo_cache': "/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/FigureScripts/monsoon_cache/gfdl_1985-2014_climo.nc",
                    'plotting_kwargs':{
                        'label': 'GFDL-CM4',
                        'color': 'purple',
                        'linestyle': '-',
                    },
                },
            ],
            plot_file="/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/monsoon_climo_comparison.png",
        )
    if True:
        plot_single_panel_hovmoller(
            simulation_file="/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit_olr_ll.nc",
            plot_file="/home/disk/brume/nacc/WeeklyNotebooks/2024.03.18/FigureScripts/cmip5_monsoon/dlesm_climo_hovmoller.svg",
            forecast=True,
            open_func=dlesm_open,
            time_period=slice('2085-01-01','2114-12-31'),
            source_name='DL$\it{ESy}$M',
            cache="/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/FigureScripts/monsoon_cache/dlesm_2085-2114_climo_hovmoller_fitted.nc",
            contourf_levels=np.arange(180, 305+20, 10),
            vs=dict(vmin=180, vmax=330),
            contour_levels=[260,],
        )
        plot_single_panel_hovmoller(
            simulation_file="/home/disk/rhodium/bowenliu/remap/era5_1deg_1d_HPX64_1983_2017_olr_ll.nc",
            plot_file="/home/disk/brume/nacc/WeeklyNotebooks/2024.03.18/FigureScripts/cmip5_monsoon/isccp_climo_hovmoller.svg",
            forecast=False,
            open_func=isccp_open,
            source_name='ISCCP',
            time_period=slice('1985-01-01','2014-12-31'),
            cache="/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/FigureScripts/monsoon_cache/isccp_1985-2014_climo_hovmoller_fitted.nc",
            contourf_levels=np.arange(180, 305+20, 10),
            vs=dict(vmin=180, vmax=330),
            contour_levels=[260,],
        )
        plot_single_panel_hovmoller(
            simulation_file="/home/disk/mercury5/dlwp/cmip6/HadGEM3-GC31-LL/rlut/rlut_day_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_*.nc",
            plot_file="/home/disk/brume/nacc/WeeklyNotebooks/2024.03.18/FigureScripts/cmip5_monsoon/hadgem_climo_hovmoller.svg",
            forecast=True,
            source_name='HadGEM3',
            time_period=slice('1985-01-01','2014-12-31'),
            open_func=hadgem_open,
            cache="/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/FigureScripts/monsoon_cache/hadgem_1985-2014_climo_hovmoller.nc",
            contourf_levels=np.arange(180, 305+20, 10),
            vs=dict(vmin=180, vmax=330),
            contour_levels=[260,],
        )
        plot_single_panel_hovmoller(
            simulation_file="/home/disk/mercury5/dlwp/cmip6/GFDL-CM4/rlut/rlut_day_GFDL-CM4_historical_r1i1p1f1_gr2_*.nc",
            plot_file="/home/disk/brume/nacc/WeeklyNotebooks/2024.03.18/FigureScripts/cmip5_monsoon/gfdl_climo_hovmoller.svg",
            forecast=True,
            source_name='GFDL',
            time_period=slice('1985-01-01','2014-12-31'),
            open_func=gfdl_open,
            cache="/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/FigureScripts/monsoon_cache/gfdl_1985-2014_climo_hovmoller.nc",
            contourf_levels=np.arange(180, 305+20, 10),
            vs=dict(vmin=180, vmax=330),
            contour_levels=[260,],
        )
        plot_single_panel_hovmoller(
            simulation_file="/home/disk/mercury5/dlwp/cmip6/MPI-ESM1-2-HR/rlut/rlut_day_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_*.nc",
            plot_file="/home/disk/brume/nacc/WeeklyNotebooks/2024.03.18/FigureScripts/cmip5_monsoon/mpi_climo_hovmoller.svg",
            forecast=True,
            source_name='MPI',
            time_period=slice('1985-01-01','2014-12-31'),
            open_func=mpi_open,
            cache="/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/FigureScripts/monsoon_cache/mpi_1985-2014_climo_hovmoller.nc",
            contourf_levels=np.arange(180, 305+20, 10),
            vs=dict(vmin=180, vmax=330),
            contour_levels=[260,],
        )
        plot_single_panel_hovmoller(
            simulation_file="/home/disk/mercury5/dlwp/cmip6/CESM2/rlut/rlut_day_CESM2_historical_r1i1p1f1_gn_*.nc",
            plot_file="/home/disk/brume/nacc/WeeklyNotebooks/2024.03.18/FigureScripts/cmip5_monsoon/cesm_climo_hovmoller.svg",
            forecast=True,
            source_name='CESM2',
            time_period=slice('1985-01-01','2014-12-31'),
            open_func=cesm_open,
            cache="/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/FigureScripts/monsoon_cache/cesm_1985-2014_climo_hovmoller.nc",
            contourf_levels=np.arange(180, 305+20, 10),
            vs=dict(vmin=180, vmax=330),
            contour_levels=[260,],
        )