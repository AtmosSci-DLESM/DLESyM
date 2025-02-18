import xarray as xr 
import numpy as np
import os
import cftime 
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt

def rolling_mean_100yr(ds):
    return ds.rolling(year=100, center=True).mean()

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

def scale_geopotential(ts): return ts / 9.81

def weighted_mean(ds):
    weights = np.cos(np.deg2rad(ds.latitude))
    print(weights)
    weights = weights / (weights.sum()*len(ds.longitude))
    print(weights)
    return (ds * weights).sum(dim=('latitude', 'longitude'))

PARAMS_1000yr = dict(
    file='/home/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_1000year.nc',
    var='t2m0',
    output_file='/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/FigureScripts/1000yr_t2m_drift.png',
    cache_dir='/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/FigureScripts/drift_cache',
    indexing=dict(year=slice(2017, 3016)),
    smoothing_func=None,
    linear_fit_params=dict(
        years=slice(2017, 3017),
    ),
    ylim=(287.5, 288.1),
    yticks=np.arange(287.5, 288.15, 0.1),
    xticks=np.arange(2017, 3018, 200),
    xlabel='Year',
    ylabel='T$_{2m}$ (K)',
)
PARAMS_1000yr_sst = dict(
    file='/home/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/ocean_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_1000year.nc',
    var='sst',
    output_file='/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/FigureScripts/1000yr_sst_drift.png',
    cache_dir='/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/FigureScripts/drift_cache',
    indexing=dict(year=slice(2017, 3016)),
    smoothing_func=None,
    linear_fit_params=dict(
        years=slice(2017, 3016),
    ),
    ylim=(290.6,291),
    yticks=np.arange(290.6, 291.05, 0.1),
    xticks=np.arange(2017, 3018, 200),
    xlabel='Year',
    ylabel='SST (K)',
)
PARAMS_1000yr_accumulaion_z500 = dict(
    file='/home/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_1000year.nc',
    var='z500',
    output_file='/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/FigureScripts/1000yr_z500_conservation.png',
    cache_dir='/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/FigureScripts/drift_cache',
    indexing=dict(year=slice(2017, 3016)),
    additional_processing=scale_geopotential,
    linear_fit_params=[
        dict(
            years=slice(2017, 2516),
        ),
        dict(
            years=slice(2267, 2766),
        ),
        dict(
            years=slice(2517, 3016),
        ),
    ],
    unit='m',
)

PARAMS_era5_t2m = dict(
    era5_file='/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_2m_temperature.nc',
    var='t2m',
    output_file='/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/FigureScripts/era5_t2m_global_annual_area_weighted.png',
    cache_dir='/home/disk/brume/nacc/WeeklyNotebooks/2024.08.05/FigureScripts/drift_cache',
    unit='K',
    averaging_func=weighted_mean,
    cache_suffix='_area_weighted',
)

def conservation_ts(
        file,
        var,
        output_file,
        cache_dir = './',
        indexing = None,
        additional_processing = None,
        linear_fit_params = None,
        unit = 'k',
):
    
    # check for cache 
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f'{cache_dir}/{var}_annual_mass.nc'

    if os.path.exists(cache_file):
        print(f'Loading {cache_file}')
    else:
        
        print(f'Cache not found, calculating global mass average for {var} in {file}')
        # Open the input file
        ds = xr.open_dataset(file, chunks={'step': 10})[var].sum(dim=('face','height','width')).squeeze()

        # change the step dimension to a datetime
        # Extract the year from the time dimension
        years = ds['step'].dt.year

        # Assign the new coordinate and calculate annual average
        annual = ds.assign_coords(year=('step', years.values)).groupby('year').mean('step')

        # save to cache 
        print(f'Saving cache to {cache_file}')
        with ProgressBar():
            annual.to_netcdf(cache_file)
        print('Done!')

    # load cached time series 
    ds = xr.open_dataset(cache_file)[var]
    if indexing is not None:
        ds = ds.sel(indexing)
    if additional_processing is not None:
        ds = additional_processing(ds)

    # get linear fit
    if linear_fit_params is not None:
        if isinstance(linear_fit_params, list):
            linear_fit_da = []
            slope = []
            relative_slope = []
            for params in linear_fit_params:
                linear_fit_da_, slope_ = linear_fit(ds, **params)
                linear_fit_da.append(linear_fit_da_)
                slope.append(slope_)
                relative_slope.append(slope_ / ds.mean() * 100)
        else:
            linear_fit_da, slope = linear_fit(ds, **linear_fit_params)
            relative_slope = slope / ds.mean() * 100
    
    # plot time series
    print(f'Plotting time series to {output_file}')
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(ds.year, ds, linewidth=0.5, color='k')
    ax.set_xlabel('Year')
    ax.set_ylabel(f'{var} ({unit})')
    ax.set_xlim(ds.year.min(), ds.year.max())
    ax.grid()
    if linear_fit_params is not None:
        if isinstance(linear_fit_params, list):
            for linear_fit_da_, slope_, relative_slope_ in zip(linear_fit_da, slope, relative_slope):
                ax.plot(linear_fit_da_.year, linear_fit_da_, label=f'Trend: {slope_:.4f} {unit} 100yr$^{-1}$ ({relative_slope_:.4f}%)')
        else:
            ax.plot(linear_fit_da.year, linear_fit_da, color='r', label=f'Trend: {slope:.4f} {unit} 100yr$^{-1}$ ({relative_slope:.4f}%)')
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    print('Done!')

def drift_ts(
        file,
        var,
        output_file,
        cache_dir = './',
        indexing = None,
        smoothing_func = None,
        linear_fit_params = None,
        ylim = (287.6, 288.0),
        yticks = None,
        xticks = None,
        xlabel = None,
        ylabel = None,
):
    
    # check for cache 
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f'{cache_dir}/{var}_annual.nc'

    if os.path.exists(cache_file):
        print(f'Loading {cache_file}')
    else:
        
        print(f'Cache not found, calculating global annual average for {var} in {file}')
        # Open the input file
        ds = xr.open_dataset(file, chunks={'step': 10})[var].mean(dim=('face','height','width')).squeeze()

        # change the step dimension to a datetime
        # Extract the year from the time dimension
        years = ds['step'].dt.year

        # Assign the new coordinate and calculate annual average
        annual = ds.assign_coords(year=('step', years.values)).groupby('year').mean('step')

        # save to cache 
        print(f'Saving cache to {cache_file}')
        with ProgressBar():
            annual.to_netcdf(cache_file)
        print('Done!')

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
    print(f'Plotting time series to {output_file}')
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(ds.year, ds, linewidth=0.75, color='k')
    ax.set_xlabel('Year',fontsize=14)
    if ylabel is not None:
        ax.set_ylabel(ylabel,fontsize=14)
    else:
        ax.set_ylabel(f'{var} (K)',fontsize=14)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_xlim(ds.year.min(), ds.year.max())
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.grid()
    if linear_fit_params is not None:
        ax.plot(linear_fit_da.year, linear_fit_da, color='r', linewidth=2, label=f'Trend: {slope:.1e} K 100 yr$^{{-1}}$')
        ax.legend(fontsize=16,loc='upper left') 
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    print('Done!')

def era5_ts(
        era5_file,
        var,
        output_file,
        cache_dir = './',
        indexing = None,
        time_slice = None,
        linear_fit_params = None,
        unit = 'K',
        averaging_func = None,
        cache_suffix = '',
):
        
    # check for cache 
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f'{cache_dir}/{var}_era5_annual{cache_suffix}.nc'

    if os.path.exists(cache_file):
        print(f'Loading {cache_file}')
    else:
        
        print(f'Cache not found, calculating global annual average for {var} in {era5_file}')
        # Open the input file
        if averaging_func is not None:
            ds = averaging_func(xr.open_dataset(era5_file, chunks={'time': 10})[var].squeeze())
        else:
            ds = xr.open_dataset(era5_file, chunks={'time': 10})[var].mean(dim=('latitude','longitude')).squeeze()

        # change the step dimension to a datetime
        # Extract the year from the time dimension
        years = ds['time'].dt.year

        # Assign the new coordinate and calculate annual average
        annual = ds.assign_coords(year=('time', years.values)).groupby('year').mean('time')

        # save to cache 
        print(f'Saving cache to {cache_file}')
        with ProgressBar():
            annual.to_netcdf(cache_file)
        print('Done!')

    # load cached time series 
    ds = xr.open_dataarray(cache_file)
    if indexing is not None:
        ds = ds.sel(indexing)
    if time_slice is not None:
        ds = ds.sel(year=time_slice)

    # get linear fit
    if linear_fit_params is not None:
        linear_fit_da, slope = linear_fit(ds, **linear_fit_params)
    
    # plot time series
    print(f'Plotting time series to {output_file}')
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(ds.year, ds, linewidth=0.5, color='k')
    ax.set_xlabel('Year')
    ax.set_ylabel(f'{var}')
    ax.set_xlim(ds.year.min(), ds.year.max())
    ax.grid()
    if linear_fit_params is not None:
        ax.plot(linear_fit_da.year, linear_fit_da, color='r', label=f'Trend: {slope:.3f}')
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    print('Done!')
    

if __name__ == '__main__':
    drift_ts(**PARAMS_1000yr)
    drift_ts(**PARAMS_1000yr_sst)
    # conservation_ts(**PARAMS_1000yr_accumulaion_z500)
    # era5_ts(**PARAMS_era5_t2m)
