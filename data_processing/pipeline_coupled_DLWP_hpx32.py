from utils import era5_retrieval, data_imputation, map2hpx, windspeed, trailing_average
from training.dlwp.data import data_loading as dl
import numpy as np
from omegaconf import OmegaConf

era5_requests = [
    # lsm
    {'constant':True,
     'single_level_variable':True,
     'variable_name':'land_sea_mask',
     'grid':[1.,1.],
     'target_file':'/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_lsm.nc'},
    # u 10m wind component
    {'constant':False,
     'single_level_variable':True,
     'variable_name':'10u',
     'pressure_level':'1000',
     'grid':[1.,1.],
     'year': [y for y in range(1950,2023)],
     'month':[month+1 for month in range(0,12)],
     'day': [d+1 for d in range(0,31)],
     'time': np.arange(0,24,3).tolist(),
     'target_file':'/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_u10m.nc'},
    # v 10m wind component
    {'constant':False,
     'single_level_variable':True,
     'variable_name':'10v',
     'pressure_level':'1000',
     'grid':[1.,1.],
     'year': [y for y in range(1950,2023)],
     'month':[month+1 for month in range(0,12)],
     'day': [d+1 for d in range(0,31)],
     'time': np.arange(0,24,3).tolist(),
     'target_file':'/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_v10m.nc'},
    # z1000 
    {'constant':False,
     'single_level_variable':False,
     'variable_name':'z',
     'pressure_level':'1000',
     'grid':[1.,1.],
     'year': [y for y in range(1950,2023)],
     'month':[month+1 for month in range(0,12)],
     'day': [d+1 for d in range(0,31)],
     'time': np.arange(0,24,3).tolist(),
     'target_file':'/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_z1000.nc'},
    # t2m
    {'constant':False,
     'single_level_variable':True,
     'variable_name':'2m_temperature',
     'pressure_level':'1000',
     'grid':[1.,1.],
     'year': [y for y in range(1950,2023)],
     'month':[month+1 for month in range(0,12)],
     'day': [d+1 for d in range(0,31)],
     'time': np.arange(0,24,3).tolist(),
     'target_file':'/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_t2m.nc'},
    # t850
    {'constant':False,
     'single_level_variable':False,
     'variable_name':'temperature',
     'pressure_level':'850',
     'grid':[1.,1.],
     'year': [y for y in range(1950,2023)],
     'month':[month+1 for month in range(0,12)],
     'day': [d+1 for d in range(0,31)],
     'time': np.arange(0,24,3).tolist(),
     'target_file':'/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_t850.nc'},
    # z500 
    {'constant':False,
     'single_level_variable':False,
     'variable_name':'z',
     'pressure_level':'500',
     'grid':[1.,1.],
     'year': [y for y in range(1950,2023)],
     'month':[month+1 for month in range(0,12)],
     'day': [d+1 for d in range(0,31)],
     'time': np.arange(0,24,3).tolist(),
     'target_file':'/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_z500.nc'},
    # z700 for tau  
    {'constant':False,
     'single_level_variable':False,
     'variable_name':'z',
     'pressure_level':'700',
     'grid':[1.,1.],
     'year': [y for y in range(1950,2023)],
     'month':[month+1 for month in range(0,12)],
     'day': [d+1 for d in range(0,31)],
     'time': np.arange(0,24,3).tolist(),
     'target_file':'/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_z700.nc'},
    # z300 for tau  
    {'constant':False,
     'single_level_variable':False,
     'variable_name':'z',
     'pressure_level':'300',
     'grid':[1.,1.],
     'year': [y for y in range(1950,2023)],
     'month':[month+1 for month in range(0,12)],
     'day': [d+1 for d in range(0,31)],
     'time': np.arange(0,24,3).tolist(),
     'target_file':'/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_z300.nc'},
    # z250 
    {'constant':False,
     'single_level_variable':False,
     'variable_name':'z',
     'pressure_level':'250',
     'grid':[1.,1.],
     'year': [y for y in range(1950,2023)],
     'month':[month+1 for month in range(0,12)],
     'day': [d+1 for d in range(0,31)],
     'time': np.arange(0,24,3).tolist(),
     'target_file':'/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_z250.nc'},
    # tcwv 
    {'constant':False,
     'single_level_variable':True,
     'variable_name':'total_column_water_vapour',
     'pressure_level':'1000',
     'grid':[1.,1.],
     'year': [y for y in range(1950,2023)],
     'month':[month+1 for month in range(0,12)],
     'day': [d+1 for d in range(0,31)],
     'time': np.arange(0,24,3).tolist(),
     'target_file':'/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_tcwv.nc'},
    # sst 
    {'constant':False,
     'single_level_variable':True,
     'variable_name':'sst',
     'pressure_level':'1000',
     'grid':[1.,1.],
     'year': [y for y in range(1950,2023)],
     'month':[month+1 for month in range(0,12)],
     'day': [d+1 for d in range(0,31)],
     'time': np.arange(0,24,3).tolist(),
     'target_file':'/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_sst.nc'},
]
impute_params = {
    'filename':'/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_sst.nc',
    'variable':'sst',
    'chunks':{'time':10000},
    'imputed_file':'/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_sst-ti.nc'
}
windspeed_params = {
    'u_file':'/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_u10m.nc',
    'v_file':'/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_v10m.nc',
    'target_file':'/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_windspeed.nc',
    'chunks':{'time':10},
}
hpx_params = [
    {'file_name' : '/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_sst-ti.nc',
     'target_variable_name' : 'sst', 
     'file_variable_name' : 'sst', 
     'prefix' : '/home/disk/rhodium/dlwp/data/HPX32/era5_1deg_3h_HPX32_1950-2022_',
     'nside' : 32,
     'order' : 'bilinear', 
     'resolution_factor' : 1.0,
     'visualize':False},
    {'file_name' : '/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_windspeed.nc',
     'target_variable_name' : 'ws10', 
     'file_variable_name' : 'ws10', 
     'prefix' : '/home/disk/rhodium/dlwp/data/HPX32/era5_1deg_3h_HPX32_1950-2022_',
     'nside' : 32,
     'order' : 'bilinear', 
     'resolution_factor' : 1.0,
     'visualize':False},
    {'file_name' : '/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_z1000.nc',
     'target_variable_name' : 'z1000', 
     'file_variable_name' : 'z', 
     'prefix' : '/home/disk/rhodium/dlwp/data/HPX32/era5_1deg_3h_HPX32_1950-2022_',
     'nside' : 32,
     'order' : 'bilinear', 
     'resolution_factor' : 1.0,
     'visualize':False},
    {'file_name' : '/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_lsm.nc',
     'target_variable_name' : 'lsm', 
     'file_variable_name' : 'lsm', 
     'prefix' : '/home/disk/rhodium/dlwp/data/HPX32/era5_1deg_3h_HPX32_1950-2022_',
     'nside' : 32,
     'order' : 'bilinear', 
     'resolution_factor' : 1.0,
     'visualize':False},
]
zarr_params = {
    'src_directory' : '/home/disk/rhodium/dlwp/data/HPX32/',
    'dst_directory' : '/home/disk/rhodium/dlwp/data/HPX32/',
    'dataset_name' : 'hpx32_1950-2022_3h_sst_coupled_test',
    'input_variables' : [
       'sst',
       'ws10',
       'z1000',
    ],
    'output_variables' : [
        'sst',
    ],
    'constants': {
        'lsm':'lsm'
    },
    'prefix' : 'era5_1deg_3h_HPX32_1950-2022_',
    'batch_size': 16,
    'scaling' : OmegaConf.load('/home/disk/quicksilver/nacc/dlesm/zephyr/training/configs/data/scaling/hpx32.yaml'),
    'overwrite' : False,
}
# Retrive raw data
for request in era5_requests:
    era5_retrieval.main(request) 

# Impute ocean data 
data_imputation.triple_interp(impute_params)
# windspeed calculation  
windspeed.main(windspeed_params)
# Remap data to HPX mesh 
for hpx_param in hpx_params:
    map2hpx.main(hpx_param)
# 48 hour trailing average of atmospheric fields 
for trailing_average_param in trailing_average_params:
    trailing_average.main(trailing_average_param)
# create zarr file for optimized training 
dl.create_time_series_dataset_classic(**zarr_params)
