import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys
import skill_metrics as sm


sys.path.append('./evaluation')
from taylor_diagrams import TaylorDiagram, weighted_lat, interp_cmip
# Dependency available in this repo: https://github.com/PeterRochford/SkillMetrics/tree/master
sys.path.append('path/to/SkillMetrics')

def interp_cmip_hovmoller(input_var, target):

    # Interpolate only along the latitude dimension
    var_high = input_var.interp(latitude=target.latitude, kwargs={'fill_value': 'extrapolate'})

    # Ensure the resulting DataArray has the correct coordinates and dimensions
    var_high = xr.DataArray(var_high, coords=[input_var.time, target.latitude], dims=['time', 'latitude'])

    return var_high

def hovmoller_taylor_diagrams(
        simulation_dicts: list,
        ref_file: str,
        ax: plt.Axes,
        legend: bool = True,
        ylim: tuple = (0, 1.5),
        ylabel: str = '',
):
    
    # # load the reference data and format dimension names 
    ref = xr.open_dataset(ref_file).squeeze().rename({'lat':'latitude'})

    # reformat coord dimensions in the model simulations 
    for sim in simulation_dicts:
        sim['data'] = xr.open_dataarray(sim['file_path']).squeeze().rename({'lat':'latitude'})

    # weight data by lat, check for interpolating cmip data
    for sim in simulation_dicts:
        if sim['cmip']:
            sim['data'] = interp_cmip_hovmoller(sim['data'], ref['olr'])
        sim['data'] = weighted_lat(sim['data']).values.flatten()

    # calculate the taylor statistics
    taylor_stats = [sm.taylor_statistics(sim['data'], ref['olr'].values.flatten()) for sim in simulation_dicts]

    # extract the statistics for the simulations and reference data
    sdev = np.array([taylor_stats[0]['sdev'][0]]+[taylor_stat['sdev'][1] for taylor_stat in taylor_stats])
    crmsd = np.array([taylor_stats[0]['crmsd'][0]]+[taylor_stat['crmsd'][1] for taylor_stat in taylor_stats])
    ccoef = np.array([taylor_stats[0]['ccoef'][0]]+[taylor_stat['ccoef'][1] for taylor_stat in taylor_stats])
    ref_sdev = np.array([taylor_stats[0]['sdev'][0]]+[taylor_stat['sdev'][0] for taylor_stat in taylor_stats])

    # normalization
    sdev = sdev/ref_sdev
    crmsd = crmsd/ref_sdev

    # create marker type 
    markers = {}
    for sim in simulation_dicts:
        markers[sim['model_name']] = sim['marker_specs']

    sm.taylor_diagram(
        ax, sdev,crmsd,ccoef, 
        markerLabel = ['ISCCP']+[sim['model_name'] for sim in simulation_dicts],
        markerSize = 11,
        markerLegend = 'on',                
        tickRMS= [0.25, 0.5, 1.0,], 
        axismax = 1.5,
        tickRMSangle = 110.0, 
        titleRMS = 'off',
        titleCOR = 'off',
        colRMS = 'grey', styleRMS = ':', widthRMS = 2.0,
        tickSTD = [0.5,1.5],  
        colSTD = 'k', styleSTD = '--', widthSTD = 0.8,
        colCOR = 'k', styleCOR = '-.', widthCOR = 0.8,
        tickCOR = [0.8, 0.9, 0.95, 0.99, 1.0],
        styleOBS = '--', 
        colOBS = 'k', 
        markerobs = 'o',
        titleOBS = 'ISCCP',
        widthobs = 2.0,
        markers = markers,
    ) 

    ax.set_ylabel(ylabel, fontsize=14, fontweight='normal')
    ax.set_ylim(ylim)

    if not legend:
        print('legend is being turned off...')
        ax.get_legend().set_visible(False)

    return ax