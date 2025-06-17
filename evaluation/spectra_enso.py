import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
from scipy.io import loadmat
import pandas as pd
import scipy.signal as sig
import xarray as xr
import netCDF4 as nc
import warnings
from scipy.signal import detrend, welch
from scipy.stats import f as f_dist
from dask.diagnostics import ProgressBar

#  red noise spectrum
def red_noise_spectrum(r, f):
    return (1 - r**2) / (1 + r**2 - 2 * r * np.cos(2 * np.pi * f))

# Helper: compute lag-1 autocorrelation
def estimate_ar1(x):
    x = x - np.mean(x)
    return np.corrcoef(x[1:], x[:-1])[0, 1]

def _detrend_no_annual(data):

    # Remove monthly climatology
    monthly_clim = data.groupby("time.month").mean("time")
    da_anom = data.groupby("time.month") - monthly_clim

    # Detrend in time while keeping xarray structure
    da_detrended = xr.apply_ufunc(
        detrend,
        da_anom,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        kwargs={"type": "linear", "axis": 0},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[da_anom.dtype],
    )

    return da_detrended

def main(
        forecast_file: str,
        reference_file: str,
        cache_dir: str,
        window_size: int,
        spectral_fraction_plot: float,
        output_file: str,
        normalize_variance: bool = True,
        overwrite_cache: bool = False,
):
    cache_file_forecast = f"{cache_dir}/forecast_monthly_nino34.nc"
    cache_file_reference = f"{cache_dir}/reference_monthly_nino34.nc"

    # computationally bulky part of analysis, caching useful for quick re-runs
    if overwrite_cache or not (os.path.exists(cache_file_forecast) and \
                                os.path.exists(cache_file_reference)):
        
        print(f'Calculating Nino 3.4 region SST from forecast and reference data and caching to {cache_dir}')
        # open sst from forecast and reference
        forecast_sst = xr.open_dataset(forecast_file).sst
        reference_sst = xr.open_zarr(reference_file).inputs.sel(channel_in='sst')

        # get lat-lon encoding from reference
        lat = reference_sst.lat
        lon = reference_sst.lon

        # get face, height, width coordinates of Nino 3.4 region from lat, lon arrays
        nino_34_coords = lat >= -5.0
        nino_34_coords &= lat <= 5.0
        nino_34_coords &= lon >= 190.0
        nino_34_coords &= lon <= 240.0
        # clean up the coordinates
        nino_34_coords = nino_34_coords.squeeze().drop_vars(['lat', 'lon','level','channel_in'])

        # turn step dimension into valid time
        forecast_sst = forecast_sst.assign_coords(
            step=forecast_sst.step.values + forecast_sst.time.values
        ).squeeze().drop('time').rename({'step': 'time'})
        # index reference to match sampling of forecast
        reference_sst = reference_sst.sel(
            time=pd.date_range(reference_sst.time.values[0],
                            reference_sst.time.values[-1],
                            freq='2D')
        ).squeeze()

        # resample to monthly means
        forecast_sst = forecast_sst.resample(time='1M').mean()
        reference_sst = reference_sst.resample(time='1M').mean()

        # select Nino 3.4 region
        forecast_sst = forecast_sst.where(nino_34_coords, drop=True)
        reference_sst = reference_sst.where(nino_34_coords, drop=True)

        # enforce chunking for dask
        forecast_sst = forecast_sst.chunk({'time': 100, 'face': -1, 'height': -1, 'width': -1})
        reference_sst = reference_sst.chunk({'time': 100, 'face': -1, 'height': -1, 'width': -1})

        # final cleanup of reference sst dimensions
        reference_sst = reference_sst.squeeze().drop_vars(['channel_in', 'level', 'lat', 'lon'])

        # save to cache
        with ProgressBar():
            forecast_sst.to_netcdf(cache_file_forecast, mode='w', compute=True)
            reference_sst.to_netcdf(cache_file_reference, mode='w', compute=True)


    # load cached data
    print(f'Loading cached data from {cache_file_forecast} and {cache_file_reference}')
    forecast_nino34 = xr.open_dataarray(cache_file_forecast)
    reference_nino34 = xr.open_dataarray(cache_file_reference)

    # detrend and remove annual cycle
    forecast_nino34 = _detrend_no_annual(forecast_nino34.mean(dim=['face', 'height', 'width']))
    reference_nino34 = _detrend_no_annual(reference_nino34.mean(dim=['face', 'height', 'width']))
    
    # calculate spectra using Welch's method, we want 50% overlap 
    f, Pxx = sig.welch(forecast_nino34, nperseg=window_size, noverlap=window_size//2, detrend='linear') #  calculate spectra over same time period
    f, Pyy = sig.welch(reference_nino34, nperseg=window_size, noverlap=window_size//2, detrend='linear')

    if normalize_variance:
        # Normalize the spectra to have unit variance
        print("Normalizing spectra to unit variance")
        Pxx /= np.sum(Pxx)
        Pyy /= np.sum(Pyy)

    # Estimating AR1 correlation using our reference
    r_y = estimate_ar1(reference_nino34.values)
    P_red_y = red_noise_spectrum(r_y, f)
    # P_red_x = P_red_x * np.mean(Pxx)
    P_red_y = P_red_y * np.mean(Pyy)

    def plot_power_spectra(ax, f, p, red, label):
        iend = int(len(f) * spectral_fraction_plot)
        ax.plot(f[0:iend], p[0:iend], label=label)
        if red is not None:
            ax.plot(f[0:iend], red[0:iend], '--', label=r'Red noise ($r=$'+str(np.around(r_y,3))+')', color='r')
        ax.set_xlabel('Cycles per month', fontsize=12)
        ax.set_ylabel('Normalized Power' if normalize_variance else 'Power', fontsize=12)
        return ax

    fig, ax = plt.subplots(figsize=(5, 4))
    ax = plot_power_spectra(ax, f, Pxx, None, f"DL$ESy$M")
    ax = plot_power_spectra(ax, f, Pyy, P_red_y, f"ERA5")
    ax.axvspan(0.04167,0.0119, color='k', alpha=.1, label='2-7 Year Period')
    ax.set_title(r'Ni$\tilde{n}$o 3.4 Spectral Variance',fontsize=14)
    ax.set_ylim(0, 1.2 * np.max([np.max(Pxx), np.max(Pyy)]))
    ax.set_xlim(0, np.max(f[int(len(f) * spectral_fraction_plot)-1]))
    ax.legend()
    fig.tight_layout()
    print(f"Saving figure to {output_file}.png and {output_file}.pdf")
    fig.savefig(f"{output_file}.pdf", dpi=300)
    fig.savefig(f"{output_file}.png", dpi=300)


if __name__ == "__main__":
    
    # spectral comparison of DLESYM and ERA5
    main(
        forecast_file='/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/ocean_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit.nc',
        reference_file='/home/disk/rhodium/dlwp/data/HPX64/hpx64_1983-2017_3h_9varCoupledAtmos-sst.zarr',
        cache_dir='/home/disk/brume/nacc/DLESyM/evaluation/cache',
        window_size=128,
        spectral_fraction_plot=.4,
        output_file='enso_spectra_comparison',
        normalize_variance=True,
        overwrite_cache=False,
    )
    # unnormalized version for comparison with the paper
    main(
        forecast_file='/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/ocean_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit.nc',
        reference_file='/home/disk/rhodium/dlwp/data/HPX64/hpx64_1983-2017_3h_9varCoupledAtmos-sst.zarr',
        cache_dir='/home/disk/brume/nacc/DLESyM/evaluation/cache',
        window_size=128,
        spectral_fraction_plot=.4,
        output_file='enso_spectra_comparison_unnormalized',
        normalize_variance=False,
        overwrite_cache=False,
    )