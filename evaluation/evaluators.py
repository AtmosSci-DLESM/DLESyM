import os
import yaml
import multiprocessing

import numpy as np
import torch as th
import pandas as pd
import xarray as xr
import einops as eo
from scipy.ndimage import rotate

from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm/57364423#57364423
from .istarmap import istarmap
from data_processing.remap import HEALPixRemap

def load_da_parallel(
        ds_path: str,
        vname: str,
        ds_sel_dict: dict,
        time_dim_name: str = 'time',
        defined_verif_only: bool = True,
        ) -> np.array:
    """
    Helper function to load entries specified by ds_sel_dict and vname from a .nc dataset in parallel

    :param ds_path: Path to the dataset
    :param vname: The variable to load from the dataset
    :param ds_sel_dict: A dictionary specifying what entries and dimensions are loaded
    :param time_dim_name: The name of the time dimension in the dataset
    :param defined_verif_only: Whether to only use the valid indices of the forecast file. If false, verification file will correspond to forecast file exactly.
    :return: A numpy array with the corresponding entries
    """

    # routine to get all available times and populate rest with nans
    if not defined_verif_only:
        ds = xr.open_dataset(ds_path)[vname]
        # Get the coordinates from the actual dataset
        coords = {dim: ds.coords[dim] for dim in ds.dims if dim != time_dim_name}
        coords[time_dim_name] = ds_sel_dict[time_dim_name]
        # create corresponding dataset of NaNs 
        full_ds = xr.Dataset({vname: (tuple(coords.keys()), np.full([len(coords[dim]) for dim in coords.keys()], np.nan))}, coords=coords)
        # enforce proper order of dimensions
        full_ds = full_ds.transpose(*ds.dims)
        # find available times in original da 
        available_times = []
        for t in ds[time_dim_name].values: 
            if t in full_ds[time_dim_name].values: available_times.append(t)
        # create result array merged array if values can be filled in 
        if len(available_times) == 0: result = full_ds[vname]
        else: result = xr.merge([full_ds,xr.open_dataset(ds_path, chunks='auto')[vname].sel({time_dim_name:available_times})])
        # return 
        return result[vname].sel(ds_sel_dict).values.squeeze()
    else:
        return xr.open_dataset(ds_path)[vname].sel(ds_sel_dict).values.squeeze()

        
class EvaluatorBase(object):
    """
    EvaluatorBase object is designed to make forecast evaluation easier by providing methods for analysis of a single
    field within a forecast. Support for information sharing accross object instances will allow for relatively 
    painless comparison of forecasts and evaluation of ensembles.

    This is an abstract base class (interface) which provides the basic functionality for different geometry types,
    such as latitude-longitude (LL), cubed sphere (CS), or HEALPix (HPX). Each specific geometry must implement its own
    Evaluator class, e.g., EvaluatorCS, by inheriting this abstract class to provide according mapping methods
    from the geometry of interest to LL.
    """

    # Library of units and other for each varlev used in DLWP use latex notation for plotting with matplotlib
    global variable_metas
    variable_metas = {
        "z500": {
            "unit": r"m",
            "fname_era5": "geopotential_500",
            "vname_era5": "z",
            "vname_long": "Geopotential 500hPa",
            "cmap": "viridis",
            "conversion":1/9.81,
            "plot_label":r"Z$_{500}$",
            },
        "z1000": {
            "unit": r"m",
            "fname_era5": "geopotential_1000",
            "vname_era5": "z",
            "vname_long": "Geopotential 1000hPa",
            "plot_label":r"Z$_{1000}$",
            "conversion":1/9.81,
            "cmap": "viridis",
            },
        "z250": {
            "unit": r"m",
            "fname_era5": "geopotential_250",
            "vname_era5": "z",
            "vname_long": "Geopotential 250hPa",
            "plot_label":r"Z$_{250}$",
            "conversion":1/9.81,
            "cmap": "viridis",
            },
        "t850": {
            "unit": r"$K$",
            "fname_era5": "temperature_850",
            "vname_era5": "t",
            "vname_long": "Air temperature 850hPa",
            "plot_label":r"T$_{850}$",
            "cmap": "coolwarm"
            },
        "t2m0": {
            "unit": r"$K$",
            "fname_era5": "2m_temperature",
            "vname_era5": "t2m",
            "vname_long": "Air temperature 2m",
            "plot_label":r"T$_{2m}$",
            "cmap": "coolwarm"
            },
        "tau300-700": {
            "unit": r"$m^2 s^{-2}$",
            "fname_era5": "tau_300-700",
            "vname_era5": "z",
            "vname_long": "Geopotential thickness 300-700hPa",
            "cmap": "viridis"
            },
        "tcwv0": {
            "unit": r"$kg m^{-2}$",
            "fname_era5": "total_column_water_vapour",
            "vname_era5": "tcwv",
            "vname_long": "Total column water vapour",
            "cmap": "summer"
            },
        "q850": {
            "unit": r"$kg kg^{-1}$",
            "fname_era5": "?",
            "vname_era5": "?",
            "vname_long": "? 850hpa",
            "cmap": "Blues"
            },
        "w10m": {
            "unit": r"$m s^{-1}$",
            "fname_era5": "?",
            "vname_era5": "w10m",
            "vname_long": "Wind strength 10m",
            "cmap": "Oranges"
            },
        "ws10m": {
            "unit": r"$m s^{-1}$",
            "fname_era5": "?",
            "vname_era5": "ws10",
            "vname_long": "Wind strength 10m",
            "cmap": "Oranges"
            },
        "u10m": {
            "unit": r"$m s^{-1}$",
            "fname_era5": "?",
            "vname_era5": "u10m",
            "vname_long": "U-component of wind 10m",
            "cmap": "Oranges"
            },
        "v10m": {
            "unit": r"$m s^{-1}$",
            "fname_era5": "?",
            "vname_era5": "v10m",
            "vname_long": "V-component of wind 10m",
            "cmap": "Oranges"
            },
        "sst": {
            "unit": r"$K$",
            "fname_era5": "sea_surface_temperature",
            "vname_era5": "sst",
            "vname_long": "Sea surface temperature",
            "cmap": "coolwarm"
            },
        "ttr1h": {
            "unit": r"$J m^{-2}$",
            "fname_era5": "ttr1h",
            "vname_era5": "ttr1h",
            "vname_long": "top net thermal radiation",
            "cmap": "greys_r"
            }
        }
    
    def __init__(
            self,
            forecast_path=None,
            eval_variable=None,
            verbose=True,
            on_latlon: bool = True,
            poolsize: int = 30
            ):
        
        """
        Initialize EvaluatorBase object. 
        
        :param forecast_path: string: 
               Path to forcast file. Evaluator will attempt to initialize around
               the given forecast_path. If file does not already exist at this location, 
               any generated forecasts will be saved there. If None, path to generated
               forecast will be created automatically. 
        :param eval_variable: string:
              variable used to calculate evaluation metrics      
        :param on_latlon: instructions whether to map forecast and predictor to LL
        :param poolsize: Number of processes used for parallelization
        """

        # General initialization
        self.variable_metas = variable_metas
        self.on_latlon = on_latlon
        self.poolsize = poolsize
        self.verbose = verbose
        self.is_latitude_evaluator = False
 
        # Initialize verification attributes 
        self.verification_path = None
        self.verification_da = None
        self.eval_variable = eval_variable
        self._valid_inits = None 
        self._scale_verif = False
        self._scaled = False

        # Initialize forecast attributes and configuration variables
        self.forecast_path = forecast_path
        self.forecast_da = None
        self.init_times = None
        self.forecast_range = None
        self.num_forecast_steps = None
        self.forecast_steps = None
        self.forecast_dt = None

        # Initialize climatology attributes
        self.climatology_da = None

    def convert_to_ll(self) -> None:
        """
        Convert the given forecast from its specific geometry to LL format. Must be implemented by all classes that
        implement this EvaluatorBase class.
        """
        raise NotImplementedError(self.__class__.__name__)

    def generate_climatology(
            self,
            verification_path: str,
            netcdf_dst_path: str = None,
            ) -> None:
        """
        Generate a climatology DataArray from a given verification DataArray.

        :param verification_path: The path to the verification DataArray to generate the climatology from
        :param netcdf_dst_path: If not None, writes climatology to this path
        """
        if self.verbose: print(f"Generating climatology from {verification_path}")

        if netcdf_dst_path is not None and os.path.isfile(netcdf_dst_path):
            if self.verbose: print(f"Loading climatology from {netcdf_dst_path}")
            climatology_ds = xr.open_dataarray(netcdf_dst_path)
            self.climatology_da = climatology_ds
            return
        elif self.on_latlon:
            climatology_ds = self._daily_climo_time_series(
                verif_ds=xr.open_dataset(verification_path)#.sel({"time": slice(self.forecast_da.time.values[0], self.forecast_da.time.values[-1] + self.forecast_da.step[-1])}),
                )
            # Transpose coordinates to match forecast and verification da 
            climatology_ds = climatology_ds.rename({'latitude':'lat','longitude':'lon'})
            climatology_ds = climatology_ds.transpose('time','step','lat','lon')
            self.climatology_da = climatology_ds.to_array().isel({"variable": 0})
        else:
            verif_ds = xr.open_dataset(self.verification_path)#.rename(
            if "predictors" in list(verif_ds.keys()):
                verif_ds = verif_ds.rename({"sample": "time", "predictors": self.eval_variable})
            else:
                verif_ds = verif_ds.rename({"sample": "time"})
            for vname in ["lat", "lon", "mean", "std"]:
                try:
                    verif_ds = verif_ds.drop_vars(vname)
                except ValueError:
                    pass
            climatology_ds = self._daily_climo_time_series(verif_ds)

            # Bring climatology in compatible shape with verification
            if "varlev" in climatology_ds.coords: climatology_ds = climatology_ds.isel(varlev=0)
            climatology_ds = climatology_ds.transpose("time", "step", "face", "height", "width")
            self.climatology_da = climatology_ds[self.eval_variable]
        
        # Optionally write climatology to file
        if netcdf_dst_path is not None:
            os.makedirs(os.path.dirname(netcdf_dst_path), exist_ok=True)
            climatology_ds.to_netcdf(netcdf_dst_path)
            if self.verbose: print(f"Wrote climatology data array to {netcdf_dst_path}")

    def generate_verification(
            self,
            verification_path: str,
            level: int = None,
            interpolation_method: str = "linears",
            defined_verif_only: bool = True,
            ) -> None:
        """
        Generate a verification data array from a given path.

        :param verification_path: path and name of verification dataset in netcdf form
        :param level: level of evaluation variable
        :param interpolation method: method used to interpolate era5 data to same mesh as forecast
        :param defined_verif_only: Whether to only use the valid indices of the forecast file. If false, verification file will correspond to forecast file exactly.
            forecasted time not contained in verification will be reported as nans
        """
        # Assign file used in verification
        if self.verification_path is not None and self.verbose:
            print(f"Replacing old verification path: {self.verification_path}")
        self.verification_path = verification_path
        
        # Find init dates that correspond to forecasted times sampled in the given predictor file
        # unless instructed to load all forecasted init times
        time_dim_name = "time" if self.on_latlon else "sample"
        if defined_verif_only:
            valid_inits = self._find_valid_inits(xr.open_dataset(verification_path)[time_dim_name].values)
        else: 
            valid_inits = self.forecast_da.time.values

        # Initialize array to hold verification data in forecast format
        verif = xr.zeros_like(self.forecast_da.sel(time=valid_inits))*np.nan
        if self.verbose: print(f"Generating verification dataset from era5_file_path: {verification_path}")

        # Prepare arguments to load verification data in parallel
        arguments = []
        if self.on_latlon:
            vname = variable_metas[self.eval_variable]["vname_era5"]
        else:
            ds = xr.open_dataset(self.verification_path)
            vname = "predictors" if "predictors" in list(ds.keys()) else self.eval_variable
        for valid_init in valid_inits:
            # create array of dates corresponding to forecast hours
            samples = pd.date_range(start=valid_init,
                                    end=valid_init + (self.num_forecast_steps - 1) * self.forecast_dt,
                                    freq=pd.Timedelta(self.forecast_dt))
            ds_sel_dict = {time_dim_name: samples}
            if self.is_latitude_evaluator: ds_sel_dict["latitude"] = self.forecast_da.coords["lat"]
            if level is not None: ds_sel_dict["level"] = level
            arguments.append([verification_path, vname, ds_sel_dict, time_dim_name, defined_verif_only])

        # Populate verif array with samples from date array above, using parallel processes
        with multiprocessing.Pool(self.poolsize) as pool:
            if self.verbose:
                print(f"Loading verification time steps with {self.poolsize} processes in parallel")
                verif.values = np.array(list(tqdm(pool.istarmap(load_da_parallel, arguments), total=len(arguments))))
            else:
                verif.values = pool.starmap(load_da_parallel, arguments)
            pool.terminate()
            pool.join()

        # Scale the newly assigned verification file if the old has been scaled too
        if self._scaled and self._scale_verif: verif = (verif*self._scale_std) + self._scale_mean
        self.verification_da = verif

        # Only use the valid indices in the forecast file
        self.forecast_da = self.forecast_da.sel(time=valid_inits)

    def get_acc(
            self,
            climatology_path: str = None,
            mean: bool = True,
            ):
        """
        Return the ACC of the forecast under consideration of a climatology file

        :param climatology_path: The path to the climatology file
        :param mean: Whether to return a single value or an array of length forecast steps
        """
        # Ensure climatology file exists
        if self.climatology_da is None:
            if climatology_path is None: 
                self.generate_climatology(verification_path=self.verification_path)
            else:
                if os.path.isfile(climatology_path):
                    vname = variable_metas[self.eval_variable]["vname_era5"] if self.on_latlon else self.eval_variable
                    self.climatology_da = xr.open_dataset(climatology_path)[vname]
                else:
                    self.generate_climatology(verification_path=self.verification_path,netcdf_dst_path=climatology_path)
        if self.on_latlon:
            axis_mean = (0, 1, 2, 3) if mean else (0, 2, 3)
        else:
            axis_mean = (0, 1, 2, 3, 4) if mean else (0, 2, 3, 4)
        return self._compute_acc(axis_mean=axis_mean)

    def get_mse(
            self,
            mean: bool = True
            ):
        """
        Return the MSE of the forecast against the verification on the Cubed sphere

        :param mean: Whether to return a single value or an array of length forecast steps
        """
        if self.on_latlon:
            print('computing MSE on lat lon...')
            # Calculate the weights to apply to the (latitude-weighted) MSE
            weights = np.cos(np.deg2rad(self.verification_da.lat.values))
            weights /= weights.mean()        
            # Reshape weights to be compatible with verif array
            weights = np.expand_dims(np.expand_dims(np.expand_dims(weights, axis=1), axis=0), axis=0)

            # Enforce proper order of dimensions in data arrays to avoid incorrect calculation
            # support latitude and longitude as well as lat and lon  
            forec = self.forecast_da.transpose("time", "step", "lat", "lon")
            verif = self.verification_da.transpose("time", "step", "lat", "lon")
            if isinstance(mean,bool):
                axis_mean = (0, 1, 2, 3) if mean else (0, 2, 3)
            elif isinstance(mean,tuple):
                axis_mean = mean
            else:
                raise ValueError('mean must be bool or tuple')
        else:
            print('computing MSE on native...')
            weights = None  # No latitude weighting on the native mesh
            # Enforce aligning dimensions
            forec = self.forecast_da.transpose("time", "step", "face", "height", "width")
            verif = self.verification_da.transpose("time", "step", "face", "height", "width")
            if isinstance(mean,bool):
                axis_mean = (0, 1, 2, 3, 4) if mean else (0, 2, 3, 4)
            elif isinstance(mean,tuple):
                axis_mean = mean
            else:
                raise ValueError('mean must be bool or tuple')
        return self._compute_mse(forec=forec, verif=verif, axis_mean=axis_mean, weights=weights)
        
    def get_rmse(
            self,
            mean=True
            ):
        """
        return the RMSE of forecast against verification
        """
        return np.sqrt(self.get_mse(mean=mean))

    def get_ssim(
            self,
            mean=True,
            scale_file_path: str = "training/configs/data/scaling/classic.yaml"
            ):
        """
        Return the structural similarity index measure (SSIM) of forecast against verification

        :param mean: Whether to return the mean SSIM or a time series over forecast step
        """
        with th.no_grad():
            ssim_module = SSIM(time_series_forecasting=True)
            forec = th.tensor(self.forecast_da.values, dtype=th.float32)
            verif = th.tensor(self.verification_da.values, dtype=th.float32)
            
            # Enforce shape of [B, T, C=1, F, H, W]
            if self.on_latlon:
                forec = eo.rearrange(forec, "t s h w -> t s 1 1 h w")
                verif = eo.rearrange(verif, "t s h w -> t s 1 1 h w")
            else:
                forec = eo.rearrange(forec, "t s f h w -> t s 1 f h w")
                verif = eo.rearrange(verif, "t s f h w -> t s 1 f h w")

            # Normalize forecast and verification as requirement for the SSIM calculation
            with open(scale_file_path, "r") as file:
                scale_dict = yaml.safe_load(file)
                mu, sigma = scale_dict[self.eval_variable]["mean"], scale_dict[self.eval_variable]["std"]
            forec = (forec - mu) / sigma
            verif = (verif - mu) / sigma

            if mean:
                return ssim_module(img1=forec, img2=verif).cpu().numpy()
            else:
                # Iterate over steps and compute SSIM sequentially
                ssims = list()
                for s_idx in range(self.num_forecast_steps):
                    ssim = ssim_module(img1=forec[:, s_idx:s_idx+1], img2=verif[:, s_idx:s_idx+1]).cpu().numpy()
                    ssims.append(ssim)
                return np.array(ssims)

    def get_forecast_hours(self) -> np.array:
        """
        return an array with the leadtime of the forecast in hours 
        """
        f_hour = self.forecast_da.step/(3600*1e9)  # convert from nanoseconds to hours
        f_hour = np.array(f_hour, dtype=float)
        return f_hour

    def set_climatology(
            self,
            climatology_da: xr.DataArray
            ) -> None:
        """
        Set the climatology path from incoming da.

        :param climatology-da: Climatology data array
        """
        self.climatology_da = climatology_da

    def set_verification(
            self,
            verification_da: xr.DataArray
            ) -> None:
        """
        Set native verification from incoming da. This is useful if comparing several forecasts from different models;
        you want to avoid generating the verification from data everytime.

        :param:  xarray.DataArray: verification_da:
            dataarray of formated verification 
        """
        self.verification_da = verification_da
        
    def scale_das(
            self,
            scale_file_path: str = "training/configs/data/scaling/classic.yaml",
            scale_verif: bool = True
            ) -> None:
        """
        Scale the incoming DataArray with mean and std. This function will change as we update scaling conventions.

        :param scale_file_path: The path to the scale file
        :param scale_verif: Boolean indicating whether to scale the verification file as well
        """
        if self._scaled is True:
            if self.verbose: print("DataArrays have already been scaled")
            return
        else:
            if self.verbose: print(f"Loading scaling statistics from default file at {scale_file_path}")
            with open(scale_file_path, "r") as file:
                scale_dict = yaml.safe_load(file)
            mean, std = scale_dict[self.eval_variable]["mean"], scale_dict[self.eval_variable]["std"]
            #mean = xr.open_dataset(self.verification_path)['mean'].values[0]
            #std = xr.open_dataset(self.verification_path)['std'].values[0]
        if self.verbose: print(f"Scaling forecast_da by mean: {mean} and std: {std}")
        self.forecast_da = (self.forecast_da*std) + mean
        if scale_verif:
            if self.verbose: print(f"Scaling verification_da by mean: {mean} and std: {std}")
            self.verification_da = (self.verification_da*std) + mean
        self._scaled = True
        self._scale_verif = scale_verif
        self._scale_mean = mean
        self._scale_std = std

    def _daily_climo_time_series(
            self,
            verif_ds: xr.Dataset,
            ) -> xr.DataArray:
        """
        Generate a time series of daily climatology values from a climatology Dataset or DataArray and the specific
        desired times.
        
        :param verif_ds: The verification (ground truth) dataset to generate the climatology from
        :return: DataArray with a 'time' dimension corresponding to daily climatologies for those times
        """
        if self._scaled and self._scale_verif: verif_ds = (verif_ds*self._scale_std) + self._scale_mean
        valid_inits = self._find_valid_inits(verif_ds.time.values)
        # Load verification for the desired days and compute their daily climatolory
        if "level" in verif_ds.dims: verif_ds = verif_ds.isel({"level": 0})
        climatology = verif_ds.groupby('time.dayofyear').mean()
        # Create dataset containing the daily climatology for each time step of interest
        result = []
        for step in tqdm(self.forecast_steps):
            doy = [pd.Timestamp(t + np.array(step).astype('timedelta64[ns]')).dayofyear for t in valid_inits]
            result.append(climatology.sel(dayofyear=doy).rename({'dayofyear': 'time'}).assign_coords(time=valid_inits))
        result = xr.concat(result, dim='step').assign_coords(step=self.forecast_steps)
        return result

    def _get_metadata_from_da(
            self,
            da: xr.DataArray
            ) -> None:
        """
        Attempt to extract metatdata from passed dataset

        :param da: Data array to extract metadata from
        """ 
        self.init_times = da.time.values
        self.forecast_steps = da.step
        self.forecast_range = da.step[-1].values
        self.num_forecast_steps = len(da.step)
        self.forecast_dt = da.step[1].values-da.step[0].values

    def _compute_acc(
            self,
            axis_mean: tuple
            ) -> np.array:
        """
        Computes and returns the ACC score for given forecast, verification, and climatology data arrays.

        :param axis_mean: Tuple indicating over which axes to compute the mean
        :return: ACC over all variables, or numpy array of length forecast steps
        """
        if self.verbose: print("Computing ACC")
        forec = self.forecast_da
        verif = self.verification_da
        climat = self.climatology_da
        if self.eval_variable == 't2m0':
            forec['f_day'] = xr.DataArray(np.floor((forec.step.values.astype('float')-1)/8.64e13 + 1), dims='step')
            verif['f_day'] = xr.DataArray(np.floor((verif.step.values.astype('float')-1)/8.64e13 + 1), dims='step')
            climat['f_day'] = xr.DataArray(np.floor((climat.step.values.astype('float')-1)/8.64e13 + 1), dims='step')
            forec = forec.groupby('f_day').mean()
            verif = verif.groupby('f_day').mean()
            climat = climat.groupby('f_day').mean()
        
        return (np.nanmean((verif - climat)*(forec - climat), axis=axis_mean)
                / np.sqrt(np.nanmean((verif - climat)**2., axis=axis_mean)
                          * np.nanmean((forec - climat)**2., axis=axis_mean)))

    def _compute_mse(
            self,
            forec: xr.DataArray,
            verif: xr.DataArray,
            axis_mean: tuple,
            weights: np.array = None,
            ) -> np.array:
        """
        :param forec: The forecast data array
        :param verif: The verification (ground truth) data array
        :param axis_mean: Tuple indicating over which axes to compute the mean
        :param weights: The (optional) weights for a latitude-weighted MSE calculation
        :return: MSE over all variables, or numpy array of length forecast steps
        """
        if self.verbose: print("Computing MSE")
        if self.eval_variable == 't2m0':
            forec['f_day'] = xr.DataArray(np.floor((forec.step.values.astype('float')-1)/8.64e13 + 1), dims='step')
            verif['f_day'] = xr.DataArray(np.floor((verif.step.values.astype('float')-1)/8.64e13 + 1), dims='step')
            forec = forec.groupby('f_day').mean()
            verif = verif.groupby('f_day').mean()
        if weights is None:
            return np.nanmean((verif.values-forec.sel(time=verif.time).values)**2., axis=axis_mean)
        
        else:
            return np.nanmean((verif.values-forec.sel(time=verif.time).values)**2*weights, axis=axis_mean)

    def _find_valid_inits(
            self,
            sample_array: np.array
            ) -> np.array:
        """
        Find initialization dates whose associated forecast samples are a subset of sample_array
        
        :param sample_array: np.array: array of datetime64 objects corresponding to samples 
        :return: An array containing all valid datatime64 initialization times
        """
        valid_inits = []
        for i in range(len(self.init_times)):
            samples = pd.date_range(start=self.init_times[i],
                                    end=self.init_times[i]+(self.num_forecast_steps-1) *
                                    self.forecast_dt,
                                    freq=pd.Timedelta(self.forecast_dt))

            if np.all(np.in1d(samples, sample_array)):
                valid_inits.append(self.init_times[i])
            else: 
                if self.verbose: print(f"forecast initialized at {self.init_times[i]} cannot be verified with this "
                                       "predictor file; omitting.")
        return valid_inits

    def plot_acc(
            self,
            data: np.array,
            ax: plt.axis = None,
            model_name: str = None,
            kwargs = {},
            ) -> plt.axis:
        """
        Plots the provided ACC data into an optionally given axis and returns it.

        :param data: The ACC data time series
        :param ax: The matplotlib axis instance to plot the data into. If None, creates a new instance.
        :param model_name: The name of the model for the legend
        :param kwargs: kwargs to pass to plot method 
        :return: The axis instance updated with the provided ACC data
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.set_title(variable_metas[self.eval_variable]["vname_long"])
            ax.set_xlabel("Forecast day")
            ax.set_ylabel(rf"ACC")
            ax.grid()
        x = self.get_forecast_hours()
        steps_per_day = 24//np.array((self.forecast_dt/3600)*1e-9, dtype=int)
        if self.eval_variable == "t2m0": x = x[::steps_per_day]
        ax.plot(x/24, data, label=model_name,**kwargs)
        return ax

    def plot_rmse(
            self,
            data: np.array,
            ax: plt.axis = None,
            model_name: str = None,
            kwargs = {},
            ):
        """
        Plots the provided RMSE data into an optionally gived axis and returns it.

        :param data: The RMSE data time series
        :param ax: The matplotlib axis instance to plot the data into. If None, creates a new instance.
        :param model_name: The name of the model for the legend
        :param kwargs: kwargs to pass to plot method 
        :return: The axis instance updated with the provided RMSE data
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.set_title(variable_metas[self.eval_variable]["vname_long"])
            ax.set_xlabel("Forecast day")
            ax.set_ylabel(rf"RMSE [{variable_metas[self.eval_variable]['unit']}]")
            ax.grid()
        # apply conversion of convention is established in variable metas 
        ax.plot(self.get_forecast_hours()/24,variable_metas[self.eval_variable]["conversion"]*data if "conversion" in variable_metas[self.eval_variable].keys() else data, label=model_name, **kwargs)
        return ax

    def plot_ssim(
            self,
            data: np.array,
            ax: plt.axis = None,
            model_name: str = None
            ):
        """
        Plots the provided SSIM data into an optionally gived axis and returns it.

        :param data: The SSIM data time series
        :param ax: The matplotlib axis instance to plot the data into. If None, creates a new instance.
        :param model_name: The name of the model for the legend
        :return: The axis instance updated with the provided RMSE data
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.set_title(variable_metas[self.eval_variable]["vname_long"])
            ax.set_xlabel("Forecast day")
            ax.set_ylabel(rf"SSIM")
            ax.set_ylim([-0.025, 1.025])
            ax.grid()
        ax.plot(self.get_forecast_hours()/24, data, label=model_name)
        return ax

    def plot_rmse_per_face(
            self,
            model_name: str,
            plot_path: str,
            add_std: bool = True,
            ncol: int = 2
            ):
        """
        Plots the error for each individual face over forecast hours.

        :param model_name: Name of the model to name the plot file appropriately
        :param plot_path: The path where the plot is written to
        :param add_std: Whether to add standard deviations to each RMSE per face into the plot
        :param ncol: The number of columns for the legend. Recommendation for CS is 1, for HPX is 2
        """
        if self.verbose: print("Plotting errors per face")
        plot_path = os.path.join(plot_path, model_name.lower().replace(" ", "_"))
        os.makedirs(plot_path, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        forecast_days = self.get_forecast_hours()/24
        diff = ((self.forecast_da-self.verification_da)**2).values

        # Compute and plot errors per face
        for face_idx in range(len(self.face_names)):
            #rmse_label = np.round(np.sqrt(np.nanmean(diff, axis=(0, 1, 3, 4))), 2)
            #std_label = np.round(np.sqrt(np.nanstd(diff, axis=(0, 1, 3, 4))), 2)
            rmse_plot, std_plot = np.sqrt(np.nanmean(diff, axis=(0, 3, 4))), np.sqrt(np.nanstd(diff, axis=(0, 3, 4)))
            #ax.plot(forecast_days, rmse_plot[:, face_idx],
            #        label=fr"{self.face_names[face_idx]}, RSME={rmse_label[face_idx]}$\pm${std_label[face_idx]}")
            ax.plot(forecast_days, rmse_plot[:, face_idx], label=fr"{self.face_names[face_idx]}")
            if add_std:
                ax.fill_between(forecast_days, rmse_plot[:, face_idx], rmse_plot[:, face_idx]+std_plot[:, face_idx],
                                alpha=0.2)
        
        # Compute and plot overall error
        #rmse_label, std_label = np.round(np.sqrt(np.nanmean(diff)), 2), np.round(np.sqrt(np.nanstd(diff)), 2)
        rmse_plot = np.sqrt(np.nanmean(diff, axis=(0, 2, 3, 4)))
        #ax.plot(forecast_days, rmse_plot, color="k", ls="--", label=fr"Overall, RSME={rmse_label}$\pm${std_label}")
        ax.plot(forecast_days, rmse_plot, color="k", ls="--", label=fr"Overall")
        ax.grid()

        ax.set_title(self.eval_variable)
        ax.set_xlabel("Forecast day")
        ax.set_ylabel(rf"RMSE [{variable_metas[self.eval_variable]['unit']}]")
        ax.legend(loc="upper left", ncol=ncol)#, fontsize="xx-small")
        fig.tight_layout()
        
        plot_name = f"{self.eval_variable}_rmse_per_face.pdf"
        fig.savefig(os.path.join(plot_path, plot_name), format="pdf")
        plt.close(fig)

    def plot_image_series_native(
            self,
            model_name: str,
            plot_path: str,
            f_idx: int = 0,
            colormap_discretization: int = 8
            ):
        """
        Plot exemplary image series of forecast, verification, and their difference evolving over forecast steps.

        :param model_name: Name of the model to name the plot file appropriately
        :param plot_path: The path where the plot is written to
        :param f_idx: The index of the forecast that will be visualized
        :param colormap_discretization: The number of distinct colors in the colormap
        """
        if self.verbose: print(f"Plotting image series on native grid")
        plot_path = os.path.join(plot_path, model_name.lower().replace(" ", "_"), self.eval_variable)
        os.makedirs(plot_path, exist_ok=True)

        forec = self.forecast_da.isel({"time": f_idx}).load()
        verif = self.verification_da.isel({"time": f_idx}).load()

        vname_long = variable_metas[self.eval_variable]["vname_long"]
        unit = variable_metas[self.eval_variable]["unit"]
        date = pd.Timestamp(forec.time.values)
        cmap = variable_metas[self.eval_variable]["cmap"]

        def plot_image(fig: plt.figure, ax: plt.axes, data: np.array, vmin: float, vmax: float, title: str,
                       cmap: str = cmap):
            """
            Plots given image data with title and a colorbar
            """
            im = ax.imshow(data, origin=self.plot_origin, vmin=vmin, vmax=vmax,
                           cmap=mpl.cm.get_cmap(cmap, colormap_discretization))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, label=f"{vname_long} [{unit}]")
            ax.set_title(title)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)

        # Plot the data
        vmin, vmax = verif.min(), verif.max()
        vmin_diff = np.sqrt((verif-forec)**2).min()
        vmax_diff = np.sqrt((verif-forec)**2).max()
        for idx, step in enumerate(self.get_forecast_hours().astype(int)):
            fig, ax = plt.subplots(1, 3, figsize=(15, 3))
            fig.suptitle(f"{vname_long}, {date} +{step}h")

            # Map face data to a 2D array
            forec_image = self.faces2image(data=forec[idx])
            verif_image = self.faces2image(data=verif[idx])

            # Plot sphere data
            plot_image(fig=fig, ax=ax[0], data=forec_image, vmin=vmin, vmax=vmax, title="Forecast")
            plot_image(fig=fig, ax=ax[1], data=verif_image, vmin=vmin, vmax=vmax, title="Verification")
            plot_image(fig=fig, ax=ax[2], data=np.sqrt((forec_image-verif_image)**2), vmin=vmin_diff, vmax=vmax_diff,
                       title="RMSE", cmap="PuRd")

            #plt.subplots_adjust(left=0.05, bottom=0.0, right=0.95, top=1.0, wspace=0.15, hspace=0.15)
            fig.tight_layout()
            fig.savefig(os.path.join(plot_path, f"step_{str(int(step)).zfill(4)}h.png"), format="png")
            plt.close(fig)


class EvaluatorLL(EvaluatorBase):
    
    """
    EvaluatorLL class implementing the abstract EvaluatorBase for use cases where a model was trained on plain LatLon.
    """
    
    def __init__(
            self,
            forecast_path: str = None,
            verification_path: str = None,
            eval_variable: str = None,
            on_latlon: bool = True,
            times: xr.DataArray = None,
            poolsize: int = 30,
            verbose: bool = True
            ):
        
        """
        Initialize EvaluatorCS object. 
        
        :param forecast_path: string: 
               Path to forcast file. EvaluatorCS will attempt to initialize around
               the given forecast_path. If file does not already exist at this location, 
               any generated forecasts will be saved there. If None, path to generated
               forecast will be created automatically. 
        :param verification_path: string:
               Path to the according ground truth file.
        :param eval_variable: string:
              variable used to calculate evaluation metrics      
        :param on_latlon: bool: instructions whether to map CS forecast and predictor to LL
        :param times: An xarray DataArray of desired time steps; or compatible, e.g., slice(start, stop)
        """
        super().__init__(
            forecast_path=forecast_path,
            eval_variable=eval_variable,
            on_latlon=on_latlon,
            poolsize=poolsize,
            verbose=verbose,
            )

        self.face_names = ["Earth"]
        self.plot_origin = "lower"
        self.is_latitude_evaluator = True

        # initialize forecast around file or configuration if given
        if self.forecast_path is not None:
            if os.path.isfile(self.forecast_path):
                self.forecast_da = xr.open_dataset(forecast_path)[self.eval_variable]
                #if times is not None: self.forecast_da = self.forecast_da.sel({"time": times})
                self._get_metadata_from_da(self.forecast_da)
                if self.verbose:
                    print(f"Initialized EvaluatorCS around file {self.forecast_path} for {self.eval_variable}")
            else: 
                raise ValueError(f"{forecast_path} was not found. Evaluator was not able to initialize around a forecast.")        

    def faces2image(
        self,
        data: np.array
        ) -> np.array:
        """
        Nothing to do since the data of the LatLon evaluator only has one single face.

        :param data: The data array of shape [height, width]
        :return: Numpy array of size [height, width].
        """
        return data


class EvaluatorCS(EvaluatorBase):
    
    """
    EvaluatorCS class implementing the abstract EvaluatorBase class to specify the remapping from cubed sphere to
    latlon.
    """
    
    def __init__(
            self,
            forecast_path: str = None,
            verification_path: str = None,
            eval_variable: str = None,
            remap_config: dict = None,
            on_latlon: bool = True,
            times: str = None,
            poolsize: int = 30,
            verbose: bool = True
            ):
        
        """
        Initialize EvaluatorCS object. 
        
        :param forecast_path: string: 
               Path to forcast file. EvaluatorCS will attempt to initialize around
               the given forecast_path. If file does not already exist at this location, 
               any generated forecasts will be saved there. If None, path to generated
               forecast will be created automatically. 
        :param verification_path: string:
               Path to the according ground truth file.
        :param eval_variable: string:
              variable used to calculate evaluation metrics      
        :param remap_config: dict:
               a dictionary that configures the remap of the forecast and verification
               from the CubeSphere
        :param on_latlon: bool: instructions whether to map CS forecast and predictor to LL
        :param times: A string indicating start and end date, e.g., "2016-01-01--2018-12-31"
        :param poolsize: Number of processes used for parallelization
        """
        super().__init__(
            forecast_path=forecast_path,
            eval_variable=eval_variable,
            on_latlon=on_latlon,
            poolsize=poolsize,
            verbose=verbose,
            )

        self.face_names = ["Equator 1", "Equator 2", "Equator 3", "Equator 4", "South pole", "North pole"]
        self.plot_origin = "lower"

        # initialize configuration of CubeSphereRemap
        if remap_config is None:
            self.cs_config = {'path_to_remapper': '/home/disk/brume/nacc/tempestremap/bin',
                              'map_files': ('/home/disk/brass/nacc/map_files/O2/map_LL121x240_CS64.nc',
                                            '/home/disk/brass/nacc/map_files/O2/map_CS64_LL121x240.nc')}
        else:
            self.cs_config = remap_config

        if times is not None:
            t_start, t_stop = times.split("--")
            times = slice(np.datetime64(t_start), np.datetime64(t_stop))

        # initialize forecast around file or configuration if given
        if self.forecast_path is not None:
            if os.path.isfile(self.forecast_path):
                self.forecast_da = xr.open_dataset(forecast_path)[self.eval_variable]
                if times is not None: self.forecast_da = self.forecast_da.sel({"time": times})
                if self.on_latlon is True:
                    if self.verbose: print("Mapping forecast to LatLon mesh for evaluation")
                    # map to lat lon mesh and add proper coordinates
                    self.forecast_da = self.convert_to_ll(
                        da=self.forecast_da,
                        path_to_remapper=self.cs_config['path_to_remapper'],
                        map_files=self.cs_config['map_files'],
                        var_name=self.eval_variable
                        ).assign_coords(time=self.forecast_da.time, step=self.forecast_da.step)
                else: 
                    if self.verbose:
                        print("Instructed to not remap forecast into lat-lon mesh. Keeping native format.")
                self._get_metadata_from_da(self.forecast_da)
                if self.verbose:
                    print(f"Initialized EvaluatorCS around file {self.forecast_path} for {self.eval_variable}")
            else: 
                raise ValueError(f"{forecast_path} was not found. Evaluator was not able to initialize around a forecast.")
    
    def convert_to_ll(
            self,
            da,
            path_to_remapper,
            map_files,
            var_name
            ):
        """
        Map ds from CS to LL mesh

        param da: DataArray:
             array to be converted to Lat-Lon mesh  
        param path_to_remapper: str:
             location of tempest remap utilities
        param map_files: list:
             list of map files to use in remapping 
        param var_name: str:
             variable name to use when applying inverse map. This is the 
             variable name in the ds 
        """
        # initialize CSR object 
        csr = CubeSphereRemap(path_to_remapper=path_to_remapper)
        csr.assign_maps(*map_files)

        # create netcdf from ds to go into remapper
        da.load().to_netcdf('cs_data.nc')

        # map CS data to intermediate file
        csr.convert_from_faces('cs_data.nc', 'tmp_data.nc')
        # map to LL
        csr.inverse_remap('tmp_data.nc', 'll_data.nc')

        ll_da = xr.open_dataset('ll_data.nc')[var_name].load()
        
        # clear working directory 
        os.remove('cs_data.nc')
        os.remove('tmp_data.nc')
        os.remove('ll_data.nc')

        return ll_da

    def faces2image(
        self,
        data: np.array
        ) -> np.array:
        """
        Unifies the six cubed sphere faces into one array

        :param data: The data array of shape [face=6, height, width]
        :return: Numpy array of size [3*height, 4*width] containing the data of all faces
        """
        # Split the faces into four equator, one south, and one north face, respectively
        e0, e1, e2, e3, s, n = data
        
        # Initialize sphere with nan
        height, width = data.shape[-2:]
        sphere = np.full((3*height, 4*width), np.nan)

        # Fill sphere with faces
        sphere[1*height:2*height] = np.concatenate((e0, e1, e2, e3), axis=1)
        sphere[0*height:1*height, 0*width:1*width] = s  # South face
        sphere[2*height:3*height, 0*width:1*width] = n  # North face

        return sphere


class EvaluatorHPX(EvaluatorBase):
    
    """
    EvaluatorHPX class implementing the abstract EvaluatorBase class to specify the remapping from HEALPix to
    latlon.
    """
    
    def __init__(
            self,
            forecast_path: str,
            verification_path: str,
            eval_variable: str = "z500",
            remap_config: dict = None,
            on_latlon: bool = True,
            times: str = None,
            poolsize: int = 30,
            verbose: bool = True,
            ll_file: str = None,
            ):
        
        """
        Initialize EvaluatorHPX object. 
        
        :param forecast_path: string: 
               Path to forcast file. Evaluator will attempt to initialize around
               the given forecast_path. If file does not already exist at this location, 
               any generated forecasts will be saved there. If None, path to generated
               forecast will be created automatically.
        :param verification_path: string:
               Path to the according ground truth file.
        :param eval_variable: string:
               variable used to calculate evaluation metrics      
        :param remap_config: dict:
               a dictionary that configures the remap of the forecast and verification
               from the HEALPix
        :param on_latlon: bool: instructions whether to map HPX forecast and predictor to LL
        :param times: A string indicating start and end date, e.g., "2016-01-01--2018-12-31"
        :param poolsize: Number of processes used for parallelization
        """
        super().__init__(
            forecast_path=forecast_path,
            eval_variable=eval_variable,
            on_latlon=on_latlon,
            poolsize=poolsize,
            verbose=verbose,
            )

        self.face_names = ["N1", "N2", "N3", "N4", "E1", "E2", "E3", "E4", "S1", "S2", "S3", "S4"]
        self.plot_origin = "upper"

        if ".nc" not in verification_path:
            verification_path = verification_path + variable_metas[eval_variable]["fname_era5"] + ".nc"

        
        if times is not None:
            t_start, t_stop = times.split("--")
            self.times = slice(np.datetime64(t_start), np.datetime64(t_stop))
        else:
            self.times = times

        # initialize forecast around file or configuration if given
        if self.forecast_path is not None:
            if os.path.isfile(self.forecast_path):
                self.forecast_da = xr.open_dataset(forecast_path)[self.eval_variable]
                if self.times is not None: self.forecast_da = self.forecast_da.sel({"time": self.times})
                if self.on_latlon is True:
                    if self.verbose: print('Mapping forecast to LatLon mesh for evaluation')
                    # Map to lat lon mesh
                    self.forecast_da = self.convert_to_ll(
                        forecast_path=forecast_path,
                        verification_path=verification_path,
                        hpx_config=remap_config,
                        var_name=self.eval_variable,
                        times=self.times,
                        ll_file=ll_file
                        )
                else:
                    if self.verbose:
                        print("Instructed to not remap forecast into lat-lon mesh. Keeping native format.")
                self._get_metadata_from_da(self.forecast_da)
                if self.verbose:
                    print(f"Initialized EvaluatorHPX around file {self.forecast_path} for {self.eval_variable}")
            else: 
                raise ValueError(f"{forecast_path} was not found. Evaluator was not able to initialize around a forecast.")

    def convert_to_ll(
            self,
            forecast_path: str,
            verification_path: str,
            hpx_config: dict,
            var_name: str,
            ll_file: str,
            times: xr.DataArray = None,
            ):
        """
        Map ds from HPX to LL mesh

        param forecast_path: str:
             String holding the path to the forecast dataset
        param verification_path: str:
             String holding the path to the ground truth dataset
        param hpx_config: dict:
             Dictionary with configurations for the HEALPix mapping
        param var_name: str:
             variable name to use when applying inverse map. This is the 
             variable name in the ds 
        param ll_file: a path for lat-lon file, if provided converted forecast will be saved here.
             if provided and file already exists at location, loading file will be perferred to remap
        param times: An xarray DataArray of desired time steps; or compatible, e.g., slice(start, stop)
        """

        def get_hpx_remapper(forecast_ds_name,gt_dataset_name, hpx_config):

            if hpx_config is None:
                # Build a default HEALPix configuration
                fc_ds = xr.open_dataset(forecast_path)
                gt_ds = xr.open_dataset(verification_path)
                hpx_config = {
                    "latitudes": gt_ds.dims["latitude"],
                    "longitudes": gt_ds.dims["longitude"],
                    "nside": fc_ds.dims["height"],
                    "order": "bilinear",
                    "resolution_factor": 1.0,
                    "prefix": "forecast",
                    "model_name": "model-name",
                    "poolsize": self.poolsize,
                    "to_netcdf": False,
                    "verbose": True
                }
                fc_ds.close()
                gt_ds.close()

            hpx_mapper = HEALPixRemap(
                latitudes=hpx_config["latitudes"],
                longitudes=hpx_config["longitudes"],
                nside=hpx_config["nside"],
                order=hpx_config["order"],
                resolution_factor=hpx_config["resolution_factor"],
                verbose=self.verbose
                )
            return hpx_mapper, hpx_config
        
        if ll_file is None: 
            hpx_mapper, hpx_config = get_hpx_remapper(forecast_path,verification_path, hpx_config)
            ll_ds = hpx_mapper.inverse_remap(
                forecast_path=forecast_path,
                verification_path=verification_path,
                prefix=hpx_config["prefix"],
                model_name=hpx_config["model_name"],
                vname=var_name,
                poolsize=hpx_config["poolsize"],
                to_netcdf=hpx_config["to_netcdf"],
                times=self.times
                )
        else:
            if os.path.isfile(ll_file):
                if self.verbose:
                    print(f'Openning lat-lon file from {ll_file}.')
                ll_ds = xr.open_dataset(ll_file)
            else:
                hpx_mapper, hpx_config = get_hpx_remapper(forecast_path,verification_path, hpx_config)
                ll_ds = hpx_mapper.inverse_remap(
                    forecast_path=forecast_path,
                    verification_path=verification_path,
                    prefix=hpx_config["prefix"],
                    model_name=hpx_config["model_name"],
                    vname=var_name,
                    poolsize=hpx_config["poolsize"],
                    to_netcdf=hpx_config["to_netcdf"],
                    )
                if self.verbose:
                    print(f'Writing lat-lon file to {ll_file}.')
                ll_ds.to_netcdf(ll_file)
        return ll_ds[var_name].sel(time=self.times) if self.times is not None else ll_ds[var_name]

    def faces2image(
        self,
        data: np.array
        ) -> np.array:
        """
        Unifies the six cubed sphere faces into one array

        :param data: The data array of shape [face=6, height, width]
        :return: Numpy array of size [3*height, 4*width] containing the data of all faces
        """
        s = 1e12  # Discrimiator value for nans

        # Split the twelve HPX faces into four north, equator, and sourth faces, respectively
        f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11 = data

        # Concatenate the faces in a HEALPix-like diamond structure
        nans = np.ones_like(f0)*s
        row0 = np.concatenate((nans, nans, nans, f3, nans), axis=1)
        row1 = np.concatenate((nans, nans, f2, f7, f11), axis=1)
        row2 = np.concatenate((nans, f1, f6, f10, nans), axis=1)
        row3 = np.concatenate((f0, f5, f9, nans, nans), axis=1)
        row4 = np.concatenate((f4, f8, nans, nans, nans), axis=1)
        data = np.concatenate((row0, row1, row2, row3, row4), axis=0)

        # Create mask and set all masked data points to zero (necessary for successfull rotation)
        mask = np.ones_like(data, dtype=np.int32)*(-s)
        mask[data==s] = s
        data[mask==s] = 0.0

        # Rotate data and mask and apply mask to rotated data
        data = rotate(data, angle=-45, reshape=True)
        mask = rotate(mask, angle=-45, reshape=True)
        mask[mask==0.0] = s
        data[mask > s/2] = np.nan

        h, w = data.shape
        return data[int(h/3.3):h-int(h/3.3), :int(w*0.91)]


if __name__ == "__main__":

    #
    # Example of how to generate climatology files on the LatLon mesh and write them to file
    #

    from datetime import datetime as dt

    vmetas = EvaluatorBase().variable_metas
    
    nside = 32
    eval_variable = "z1000"
    src_path = f"/home/disk/quicksilver/nacc/Data/ERA5/1deg/1979-2022_era5_1deg_3h_" + vmetas[eval_variable]["fname_era5"] + ".nc"
    dst_path = f"/home/disk/brume/karlbam/Data/DLWP/LL181x360/climatology/2016-2018_era5_1deg_3h_climatology_{eval_variable}.nc"
    start, stop = dt(year=2018, month=12, day=2), dt(year=2018, month=12, day=31)

    fc_file_path = "/home/disk/brume/karlbam/DELUSioN/evaluation/forecasts/forecast_06-02-02_hpx_u3p_c360-180-90_bs64_lr4e-4_vNone.nc"
    hpx_config = {
        "latitudes": 181,
        "longitudes": 360,
        "nside": nside,
        "order": "bilinear",
        "resolution_factor": 1.0,
        "prefix": "forecast",
        "model_name": "model-name",
        "poolsize": 30,
        "to_netcdf": False
    }
    evaluator = EvaluatorHPX(
        forecast_path=fc_file_path,
        verification_path=src_path,
        eval_variable=eval_variable,
        remap_config=hpx_config,
        #times=slice(start, stop)
        )

    evaluator.generate_climatology(verification_path=src_path, netcdf_dst_path=dst_path)
    print("Done.")
