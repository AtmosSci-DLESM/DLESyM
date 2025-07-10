# imports 
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
import logging 
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('log')

# library with variable specific constants
# IMPORTANT: This dictionary should be updated with the scale_factor and units for each new variable used in the analysis
variable_metas = {
    'z500': {
        'scale_factor':1/9.81, # Scale factor to convert source units to desired plotting physical units, e.g. 1/9.81 for geopotential height to meters
        'units':'m',           # Physical units of the variable to be used in RMSE label, e.g. 'm' for geopotential height
        'level' : 500.,        # Level of the variable in hPa, e.g. 500 for geopotential height
        'era5_name' : 'z',     # Name of the variable in the ERA5 dataset, e.g. 'z' for geopotential height
        'rename_func' : lambda x: x.rename({'geopotential': 'z500'}).drop('level').squeeze(), # Function to reformat era5 dims to match forecast dims
    },
}

def format_verif(verif, forecast, variable):
    """
    Takes a dataset in the DLWP-style zarr format (channel_c, channel_in, channel_out, face, heigh, width, time) and 
    extracts desired variable and converts to format useful for analysis (face, height, width, time, step)

    Parameters:
    verif (xarray.Dataset): The verification data in DLWP-style zarr format.
    forecast (xarray.Dataset): The forecast data in DLWP-output-style netcdf format.
    variable (str): The variable to extract from the verification data.

    Returns:
    xarray.DataArray: The verification data in aligned with forecast.
    """

    # extract variable from verification data
    verif = verif['targets'].sel(channel_out=variable).squeeze()

    # align verification data with forecast data
    verif_aligned = xr.full_like(forecast, fill_value=np.nan)
    logger.info("Aligning verification data with forecast data.")
    for t in tqdm(forecast.time):
        for s in tqdm(forecast.step, leave=False):
            try:
                verif_aligned.loc[{'time': t, 'step': s}] = verif.sel(time=t + s).values
            except KeyError:
                logger.debug(f"Verification data missing for time {(t + s).values}.")
                verif_aligned.loc[{'time': t, 'step': s}] = np.nan

    return verif_aligned

def daily(data):
    """
    Calculates daily mean of skill scores.
    Parameters:
    data (xarray.DataArray): The skill scores to calculate daily mean for.
    Returns:
    xarray.DataArray: The daily mean of the skill scores.
    """

    return data.resample(step='1D').mean(dim='step', skipna=True)

def rmse(
        forecast_params,
        variable,
        plot_file=None,
        xlim=None,
        return_rmse=False,
        spatial_weights=None,
        daily_mean=False,
): 
    """
    Calculates the Root Mean Square Error (RMSE) for a set of forecasts and plots and/or the results.

    Parameters:
    forecast_params (list of dict): A list of dictionaries with instrcutions for handling each forecast. Each may contain: 
        'file' (str): The file path to the forecast data.
        'verification_file' (str): The file path to the verification data to be used for that forecast. 
        'rmse_cache' (str, optional): The file path where the calculated RMSE should be cached. If None, the RMSE is not cached.
        'plot_kwargs' (dict, optional): The keyword arguments to be passed to the plot function for that forecast.
    variable (str): The variable for which the RMSE is to be calculated.
    plot_file (str, optional): The file path where the plot should be saved. If None, the plot is not saved. Defaults to None.
    xlim (dict, optional): The x-axis limits for the plot. Defaults to None.
    return_rmse (bool, optional): If True, the function returns the calculated RMSE. Defaults to False.
    spatial_weights (np.ndarray, optional): A spatial weight to be applied to the RMSE and ACC calculation. Defaults to None which weights all points uniformly.
    daily_mean (bool, optional): If True, the RMSE and ACC are calculated as a daily mean. Defaults to False.

    Returns:
    list of xarray.DataArray: A list of RMSE values for each forecast. Only returned if return_rmse is True.
    """
    rmse = []
    # iterate through forecats and obtain rmse 
    for forecast_param in forecast_params:
        if os.path.isfile(forecast_param.get('rmse_cache','')):
            logger.info(f"Loading RMSE from {forecast_param['rmse_cache']}.")
            rmse.append(xr.open_dataarray(forecast_param['rmse_cache']))
        else: 
            logger.info(f"Calculating RMSE for {forecast_param['file']}.")
        
            # open forecast and verification data
            forecast = xr.open_dataset(forecast_param['file'])[variable]

            # print(f'shape of spatial weights: {spatial_weights.shape}')
            # print(f'shape of forecast: {forecast.shape}')

            # print(f'shape of spatial weights expanded: {np.stack((np.stack((spatial_weights,)*len(forecast.step), axis=0),)*len(forecast.time), axis=0).shape}')
            # exit()


            verif = format_verif(xr.open_dataset(forecast_param['verification_file'], engine='zarr'),
                                 forecast,
                                 variable)

            # calculate rmse 
            if spatial_weights is not None:
                # create weights dataarray
                expanded_weights = xr.DataArray(
                    np.stack((np.stack((spatial_weights,)*len(forecast.step), axis=0),)*len(forecast.time), axis=0), # we have to expand the weights here to match the forecast dims
                    dims=['time', 'step', 'face', 'height', 'width'], 
                    coords={'time': forecast.time, 'step': forecast.step, 'face': forecast.face, 'height': forecast.height, 'width': forecast.width})
                # calculate rmse weighted by spatial weights
                rmse.append(np.sqrt( (((forecast - verif) ** 2) * expanded_weights).sum(dim=['time', 'face', 'height', 'width'], skipna=True) / expanded_weights.sum(dim=['time', 'face', 'height', 'width'], skipna=True) ) )
            else:
                rmse.append(np.sqrt(((forecast - verif) ** 2).mean(dim=['time', 'face', 'height','width'], skipna=True)))

            # cache rmse if requested
            if forecast_param.get('rmse_cache',None) is not None:
                logger.info(f"Caching RMSE to {forecast_param['rmse_cache']}.")
                # create directory if it doesn't exist
                os.makedirs(os.path.dirname(forecast_param['rmse_cache']), exist_ok=True)
                rmse[-1].to_netcdf(forecast_param['rmse_cache'])

    # plot RMSE if requested
    if plot_file is not None:
        fig, ax = plt.subplots()
        for skill, plot_kwargs in zip(rmse, [forecast_param['plot_kwargs'] for forecast_param in forecast_params]):
            ax.plot([s / np.timedelta64(1, 'D') for s in skill.step.values], # plot in days 
                    daily(skill * variable_metas[variable]['scale_factor']) if daily_mean else skill * variable_metas[variable]['scale_factor'], # scale to physical units
                    **plot_kwargs) # style curve and label

        # style plot
        ax.set_xlabel('Forecast Days')
        ax.set_ylabel(f'RMSE [{variable_metas[variable]["units"]}]')
        # calculate y_max for plot
        y_max = max(max(arr.values.flatten()) for arr in rmse) * variable_metas[variable]['scale_factor'] * 1.1
        ax.grid()
        ax.legend()
        ax.set_xlim(**{'left':0, 'right':max([t.step[-1].values / np.timedelta64(1, 'D') for t in rmse])} if xlim is None else xlim)
        ax.set_ylim(bottom=0, top=y_max)
        logger.info(f"Saving plot to {plot_file}.")
        fig.savefig(plot_file,dpi=200)

    # return rmse if requested
    if return_rmse:
        return [daily(skill) for skill in rmse] if daily_mean else rmse
    else:
        return
    
def acc(
        forecast_params,
        variable,
        plot_file=None,
        xlim=None,
        return_acc=False,
        spatial_weights=None,
        daily_mean=False,

): 
    """
    Calculates the anomaly correlation coefficeint (ACC) for a set of forecasts and plots and/or returns the results.

    Parameters:
    forecast_params (list of dict): A list of dictionaries with instrcutions for handling each forecast. Each may contain: 
        'file' (str): The file path to the forecast data.
        'verification_file' (str): The file path to the verification data to compare to the forecast. ALso used for calculating the climatology.
        'climatology_file' (str, optional): The file path to the climatology data. If None, the climatology is calculated from the verification data.
            if not None, but file doesn't exist, the climatology is calculated from the verification data and cached to the file.
        'acc_cache' (str, optional): The file path where the calculated ACC should be cached. If None, the ACC is not cached.
        'plot_kwargs' (dict, optional): The keyword arguments to be passed to the plot function for that forecast.
    variable (str): The variable for which the ACC is to be calculated.
    plot_file (str, optional): The file path where the plot should be saved. If None, the plot is not saved. Defaults to None.
    xlim (dict, optional): The x-axis limits for the plot. Defaults to None.
    return_acc (bool, optional): If True, the function returns the calculated ACC. Defaults to False.
    spatial_weights (np.ndarray, optional): A spatial weight to be applied to the RMSE and ACC calculation. Defaults to None which weights all points uniformly.
    daily_mean (bool, optional): If True, the RMSE and ACC are calculated as a daily mean. Defaults to False.
    Returns:
    list of xarray.DataArray: A list of ACC values for each forecast. Only returned if return_acc is True.
    """

    acc = []
    # iterate through forecats and obtain acc 
    for forecast_param in forecast_params:
        # if acc is cached already, load it
        if os.path.isfile(forecast_param.get('acc_cache','')):
            logger.info(f"Loading ACC from {forecast_param['acc_cache']}.")
            acc.append(xr.open_dataarray(forecast_param['acc_cache']))
        else:
            logger.info(f"Calculating ACC for {forecast_param['file']}.")
        
            # open forecast
            forecast = xr.open_dataset(forecast_param['file'])[variable]
            verif = format_verif(xr.open_dataset(forecast_param['verification_file'], engine='zarr'),
                        forecast,
                        variable)

            # calculate climatology
            if forecast_param.get('climatology_file',None) is not None:
                if os.path.isfile(forecast_param['climatology_file']):
                    logger.info(f"Loading climatology from {forecast_param['climatology_file']}")
                    climo_raw = xr.open_dataset(forecast_param['climatology_file'])['targets']
                else:
                    logger.info(f"Calculating climatology from {forecast_param['verification_file']} and caching to {forecast_param['climatology_file']}.")
                    
                    # load verification_data, calculate climo
                    climo_raw = xr.open_dataset(forecast_param['verification_file'],
                                                engine='zarr')['targets'].sel(channel_out=variable).groupby('time.dayofyear').mean(dim='time')
                    # create directory if it doesn't exist
                    os.makedirs(os.path.dirname(forecast_param['climatology_file']), exist_ok=True)
                    climo_raw.to_netcdf(forecast_param['climatology_file'])
            else: 
                logger.info(f"Calculating climatology from {forecast_param['verification_file']}.")
                climo_raw = xr.open_dataset(forecast_param['verification_file'],
                                            engine='zarr')['targets'].sel(channel_out=variable).groupby('time.dayofyear').mean(dim='time')

            # align climo data with forecast data
            logger.info("Aligning climatology with forecast data.")
            climo = xr.full_like(forecast, fill_value=np.nan)
            for time in forecast.time:
                for step in forecast.step:
                    climo.loc[{'time': time, 'step': step}] = climo_raw.sel(dayofyear=(time + step).dt.dayofyear).values

            # calculate anomalies 
            forec_anom = forecast - climo
            verif_anom = verif - climo
            
            # calculate acc
            if spatial_weights is not None:
                # create weights dataarray
                expanded_weights = xr.DataArray(
                    np.stack((np.stack((spatial_weights,)*len(forecast.step), axis=0),)*len(forecast.time), axis=0), # we have to expand the weights here to match the forecast dims
                    dims=['time', 'step', 'face', 'height', 'width'], 
                    coords={'time': forecast.time, 'step': forecast.step, 'face': forecast.face, 'height': forecast.height, 'width': forecast.width})
                axis_mean = ['time', 'face', 'height','width']
                acc.append((verif_anom * forec_anom * expanded_weights).mean(dim=axis_mean, skipna=True)
                    / np.sqrt((expanded_weights * verif_anom**2).mean(dim=axis_mean, skipna=True) *
                            (expanded_weights * forec_anom**2).mean(dim=axis_mean, skipna=True)))
            else:
                axis_mean = ['time', 'face', 'height','width']
                acc.append((verif_anom * forec_anom).mean(dim=axis_mean, skipna=True)
                    / np.sqrt((verif_anom**2).mean(dim=axis_mean, skipna=True) *
                            (forec_anom**2).mean(dim=axis_mean, skipna=True)))

            # cache acc if requested
            if forecast_param.get('acc_cache',None) is not None:
                logger.info(f"Caching ACC to {forecast_param['acc_cache']}.")
                # create directory if it doesn't exist
                os.makedirs(os.path.dirname(forecast_param['acc_cache']), exist_ok=True)
                acc[-1].to_netcdf(forecast_param['acc_cache'])

    # plot acc if requested
    if plot_file is not None:
        fig, ax = plt.subplots()
        for skill, plot_kwargs in zip(acc, [forecast_param['plot_kwargs'] for forecast_param in forecast_params]):
            ax.plot([s / np.timedelta64(1, 'D') for s in skill.step.values], # plot in days 
                    daily(skill) if daily_mean else skill, # scale to physical units
                    **plot_kwargs) # style curve and label

        # style plot
        ax.set_xlabel('Forecast Days')
        ax.set_ylabel(f'ACC')
        ax.grid()
        ax.legend()
        ax.set_xlim(**{'left':0, 'right':max([t.step[-1].values / np.timedelta64(1, 'D') for t in acc])} if xlim is None else xlim)
        ax.set_ylim(bottom=0, top=1)
        logger.info(f"Saving plot to {plot_file}.")
        fig.savefig(plot_file,dpi=200)

    # return acc if requested
    if return_acc:
        return [daily(skill) for skill in acc] if daily_mean else acc
    else:
        return

def plot_baseline_metrics(
        forecast_params,
        variable,
        plot_file=None,
        xlim=None,
        ymax=None,
        spatial_weights=None,
        daily_mean=False,
        overwrite_cache=False,
):
    """
    Uses rmse and acc function to create a two panel baseline metrics plot 

    Parameters:
    forecast_params (list of dict): A list of dictionaries with instrcutions for handling each forecast. Each may contain: 
        'file' (str): The file path to the forecast data.
        'verification_file' (str): The file path to the verification data to compare to the forecast. ALso used for calculating the climatology. Assumes zarr format.
        'climatology_file' (str, optional): The file path to the climatology data. If None, the climatology is calculated from the verification data.
            if not None, but file doesn't exist, the climatology is calculated from the verification data and cached to the file.
        'rmse_cache' (str, optional): The file path where the calculated RMSE should be cached. If None, the RMSE is not cached.
        'acc_cache' (str, optional): The file path where the calculated ACC should be cached. If None, the ACC is not cached.
        'plot_kwargs' (dict, optional): The keyword arguments to be passed to the plot function for that forecast.
    variable (str): The variable for which the ACC is to be calculated.
    plot_file (str, optional): The file path where the plot should be saved. If None, the plot is not saved. Defaults to None.
    xlim (dict, optional): The x-axis limits for the plot. Defaults to max leadtime in forecasts. Example: {'left':0,'right':10}
    ymax (float, optional): The y-axis maximum for the RMSE plot. If None, the maximum is calculated from the data. Defaults to None.
    spatial_weights (np.ndarray, optional): A spatial weight to be applied to the RMSE and ACC calculation. Defaults to None which weights all points uniformly. 
        If calculating skill for ocean forecasts, use an ocean mask. This can be derived from the inverse of 'lsm' within our training data. To do this use the 
        following as the argument for 'spatial_weights': '1-xr.open_dataset('your/trainset/path.zarr',engine='zarr').constants.sel(channel_c='lsm').squeeze().values'.
        Make sure to update the path to point to the correct location of training data.
    daily_mean (bool, optional): If True, the RMSE and ACC are calculated as a daily mean. Defaults to False.
    overwrite_cache (bool, optional): If True, the function will overwrite any existing cached RMSE or ACC files. Defaults to False.

    Returns:
    None
    """

    # configure plot 
    fig, axs = plt.subplots(1,2, figsize=(12,4))
    
    # remove cache if requested
    if overwrite_cache:
        for forecast_param in forecast_params:
            if os.path.isfile(forecast_param.get('rmse_cache','')):
                logger.info(f"Removing cached RMSE from {forecast_param['rmse_cache']}.")
                os.remove(forecast_param['rmse_cache'])
            if os.path.isfile(forecast_param.get('acc_cache','')):
                logger.info(f"Removing cached ACC from {forecast_param['acc_cache']}.")
                os.remove(forecast_param['acc_cache'])

    # get rmses
    rmses = rmse(
        forecast_params=forecast_params,
        variable=variable,
        return_rmse=True,
        spatial_weights=spatial_weights,
        daily_mean=daily_mean,
    )

    # get accs
    accs = acc(
        forecast_params=forecast_params,
        variable=variable,
        return_acc=True,
        spatial_weights=spatial_weights,
        daily_mean=daily_mean,
    )

    # plot rmse and acc, save 
    logger.info(f"Plotting metrics and saving to {plot_file}.")
    for i in range(len(forecast_params)):
        # rmse
        axs[0].plot([s / np.timedelta64(1, 'D') for s in rmses[i].step.values], # plot in days 
                    rmses[i] * variable_metas[variable]['scale_factor'], # scale to physical units
                    **forecast_params[i]['plot_kwargs']) # convey kwargs
        # acc
        axs[1].plot([s / np.timedelta64(1, 'D') for s in accs[i].step.values], # plot in days 
                    accs[i],
                    **forecast_params[i]['plot_kwargs'])
        # add indicator 
        if 'indicator' in forecast_params[i]:
            print(f"Adding indicator for {forecast_params[i]['indicator']} to plot.")
            # add rmse indicator
            axs[0].plot(rmses[i][forecast_params[i]['indicator']].step.values / np.timedelta64(1, 'D'),
                        rmses[i][forecast_params[i]['indicator']] * variable_metas[variable]['scale_factor'],
                        color=forecast_params[i]['plot_kwargs']['color'], marker='D', markersize=7,)
            # add acc indicator
            axs[1].plot(accs[i][forecast_params[i]['indicator']].step.values / np.timedelta64(1, 'D'),
                        accs[i][forecast_params[i]['indicator']],
                        color=forecast_params[i]['plot_kwargs']['color'], marker='D', markersize=5,)

    # style rmse plot 
    axs[0].set_xlabel('Forecast Days', fontsize=10)
    axs[0].set_ylabel(f'RMSE [{variable_metas[variable]["units"]}]', fontsize=10)
    y_max = max(max(arr.values.flatten()) for arr in rmses) * variable_metas[variable]['scale_factor'] * 1.1 if ymax is None else ymax # calculate y_max for plot
    axs[0].grid()
    axs[0].set_xlim(**{'left':0, 'right':max([t.step[-1].values / np.timedelta64(1, 'D') for t in rmses])} if xlim is None else xlim)
    axs[0].set_ylim(bottom=0, top=y_max)
    # set ticklabels to have fontsize 10
    axs[0].tick_params(axis='both', which='major', labelsize=10)

    # style acc plot
    axs[1].set_xlabel('Forecast Days', fontsize=10)
    axs[1].set_ylabel(f'ACC', fontsize=10)
    axs[1].grid()
    axs[1].legend(loc='lower left', fontsize=10)
    axs[1].set_xlim(**{'left':0, 'right':max([t.step[-1].values / np.timedelta64(1, 'D') for t in accs])} if xlim is None else xlim)
    axs[1].set_ylim(bottom=0, top=1.05)
    # set ticklabels to have fontsize 10
    axs[1].tick_params(axis='both', which='major', labelsize=10)

    
    fig.savefig(plot_file + '.png',dpi=400)
    fig.savefig(plot_file + '.pdf',dpi=400)

# example call to plot_baseline_metrics
if __name__ == '__main__':

    # DLESyM skill
    plot_baseline_metrics(
        forecast_params=[
            {
                'file':'dlesym_zenodo/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_SkillForecast.nc',
                'verification_file':'dlesym_zenodo/hpx64_1983-2017_3h_9varCoupledAtmos-sst.zarr',
                'climatology_file':'dlesym_zenodo/analysis_cache/climatologies/hpx64_1979-2021_3h_9Atmos-ttr_Coupled-sst-swvl1_z500-clima.nc',
                'rmse_cache':'dlesym_zenodo/analysis_cache/skill_caches/rmse_cache_z500_dlesym.nc',
                'acc_cache':'dlesym_zenodo/analysis_cache/skill_caches/acc_cache_z500_dlesym.nc',
                'plot_kwargs':{'label':'DL$ESy$M','color':'red','linewidth':2, 'linestyle':'-'},  
                'indicator': 8, # index of indicator
            },
            {
                'file':None, # we're using a precalculated, cached RMSE, only parameters needed are caches
                'verification_file':None,
                'climatology_file':None,
                'rmse_cache':'dlesym_zenodo/analysis_cache/skill_caches/rmse_z500_Climatology.nc',
                'acc_cache':'dlesym_zenodo/analysis_cache/skill_caches/acc_cache_z500_dlesym.nc',
                'plot_kwargs':{'label':'Climatology','color':'grey','linewidth':2, 'linestyle':':'},  
            },
            {
                'file':None,
                'verification_file':None,
                'climatology_file':None,
                'rmse_cache':'dlesym_zenodo/analysis_cache/skill_caches/rmse_z500_GraphCast.nc',
                'acc_cache':'dlesym_zenodo/analysis_cache/skill_caches/acc_z500_GraphCast.nc',
                'plot_kwargs':{'label':'GraphCast','color':'purple','linewidth':2, 'linestyle':'dashdot'},  
                'indicator': 8, # index of indicator

            },
            {
                'file':None,
                'verification_file':None,
                'climatology_file':None,
                'rmse_cache':'dlesym_zenodo/analysis_cache/skill_caches/rmse_z500_ECMWF_IFS_S2S.nc',
                'acc_cache':'dlesym_zenodo/analysis_cache/skill_caches/acc_z500_ECMWF_IFS_S2S.nc',
                'plot_kwargs':{'label':'IFS-S2S','color':'hotpink','linewidth':2, 'linestyle':'dashdot'},
                'indicator': 2, # index of indicator
            }
        ],
        variable='z500',
        plot_file='z500_skill_comparison',
        xlim={'left':0, 'right':10}, # set xlim to 10 days
        ymax=100,

    )