import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys

# Dependency available in this repo: https://github.com/PeterRochford/SkillMetrics/tree/master
sys.path.append('path/to/SkillMetrics')
import skill_metrics as sm

def interp_cmip(input_var, target):
    # var_high = np.zeros(target.shape)
    rbs = RectBivariateSpline(input_var.latitude, input_var.longitude, input_var)
    var_high = rbs(target.latitude, target.longitude)
    var_high = xr.DataArray(var_high, coords=[target.latitude, target.longitude], dims=['latitude', 'longitude'])
    return var_high

def weighted_lat(input_var):
    # weight by latitude
    weight_lat = input_var.latitude.values
    weights = np.sqrt(np.cos(np.deg2rad(weight_lat)))
    ds_weights = xr.DataArray(weights, coords=[input_var.latitude], dims=['latitude'])
    weighted_var = input_var.weighted(ds_weights.fillna(0))
    return weighted_var.obj

def normalize_to_ref(data, target):

    # Calculate mean and standard deviation of the data
    mean_data = np.mean(data)
    std_data = np.std(data)
    
    # Calculate mean and standard deviation of the target
    mean_target = np.mean(target)
    std_target = np.std(target)
    
    # Scale the data
    return ((data - mean_data) / std_data) * std_target + mean_target
    
def TaylorDiagram(
        simulation_dicts: list,
        ref_file: str,
        ax: plt.Axes,
        ref_var_name: str,
        legend: bool = False,
        ylim: tuple = (0, 1.05),
        ylabel: str = '',
):
    
    # load the reference data and format dimension names 
    ref = xr.open_dataset(ref_file).squeeze()[ref_var_name]
    if 'latitude' not in ref.coords:
        ref = ref.rename({'lat':'latitude','lon':'longitude'}) 

    # reformat coord dimensions in the model simulations 
    for sim in simulation_dicts:
        sim['data'] = xr.open_dataset(sim['file_path']).squeeze()[sim['var_name']]
        if 'latitude' not in sim['data'].coords:
            sim['data'] = sim['data'].rename({'lat':'latitude','lon':'longitude'})

    # weight data by lat, check for interpolating cmip data
    for sim in simulation_dicts:
        if sim['cmip']:
            sim['data'] = interp_cmip(sim['data'], ref)
        sim['data'] = weighted_lat(sim['data']).values.flatten()

    # calculate the taylor statistics
    taylor_stats = [sm.taylor_statistics(sim['data'], ref.values.flatten()) for sim in simulation_dicts]

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
        markerLabel = ['ERA5']+[sim['model_name'] for sim in simulation_dicts],
        markerSize = 11,
        markerLegend = 'on',                
        tickRMS= [0.25, 0.5, 1.0], axismax = 1.5,
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
        markerobs = 'D',
        titleOBS = 'ERA5',
        widthobs = 2.0,
        markers = markers,
    ) 

    ax.set_ylabel(ylabel, fontsize=14, fontweight='normal')
    ax.set_ylim(ylim)
    
    if not legend:
        ax.get_legend().set_visible(False)

    return ax
  
if __name__ == '__main__':
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax = TaylorDiagram(
        simulation_dicts = [
            {'model_name': 'HPX64', 'file_path': '/home/disk/brume/nacc/WeeklyNotebooks/2024.05.06/FigureScripts/blocking_cache/blocking_freq_nh_map_40yr.nc','cmip':False},
            {'model_name': 'CESM2', 'file_path': '/home/disk/brume/nacc/WeeklyNotebooks/2024.05.06/FigureScripts/blocking_cache/blocking_freq_cesm_nh_map_40yr.nc','cmip':True},
            {'model_name': 'GFDL-CM4', 'file_path': '/home/disk/brume/nacc/WeeklyNotebooks/2024.05.06/FigureScripts/blocking_cache/blocking_freq_gfdl_nh_map_40yr.nc','cmip':True},
            {'model_name': 'HadGEM3-GC31-LL', 'file_path': '/home/disk/brume/nacc/WeeklyNotebooks/2024.05.06/FigureScripts/blocking_cache/blocking_freq_hadgem_nh_map_40yr.nc','cmip':True},
            {'model_name': 'MPI-ESM1-2-HR', 'file_path': '/home/disk/brume/nacc/WeeklyNotebooks/2024.05.06/FigureScripts/blocking_cache/blocking_freq_mpi-esp1-2-hr_nh_map_40yr.nc','cmip':True},
        ],
        ref_file = '/home/disk/brume/nacc/WeeklyNotebooks/2024.06.10/FigureScripts/blocking_cache/blocking_freq_nh_map-verif.nc',
        ax = ax,
    )
    legend = ax.get_legend()
    if legend:
        legend.set_bbox_to_anchor((1.05, 1.1))
        legend.labelspacing = 10.0
        legend.labelspacing = 10  
        plt.setp(legend.get_texts(), fontsize=14) 
    plt.tight_layout()
    fig.savefig('taylor_diagram_blocking_40yr.png')
