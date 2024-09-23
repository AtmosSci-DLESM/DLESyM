from dask.diagnostics import ProgressBar
import xarray as xr
import copy
import os
from scripts.coupled_forecast import coupled_inference

EXAMPLE_PARAMS = {
    'atmos_models' : [
        {'model_dir':'/home/disk/quicksilver/nacc/dlesm/zephyr/outputs/hpx32_coupled-dlwp_agu_seed2', 'checkpoint':None},
        {'model_dir':'/home/disk/quicksilver/nacc/dlesm/zephyr/outputs/hpx32_coupled-dlwp_agu_seed3', 'checkpoint':None},
        {'model_dir':'/home/disk/quicksilver/nacc/dlesm/zephyr/outputs/hpx32_coupled-dlwp_agu_seed4', 'checkpoint':None},
        {'model_dir':'/home/disk/quicksilver/nacc/dlesm/zephyr/outputs/hpx32_coupled-dlwp_agu_seed5', 'checkpoint':None},
    ],
    'ocean_models' : [
        {'model_dir':'/home/disk/quicksilver/nacc/dlesm/zephyr/outputs/hpx32_coupled-dlom_agu_seed20_bs128_large-test', 'checkpoint':None},
        {'model_dir':'/home/disk/quicksilver/nacc/dlesm/zephyr/outputs/hpx32_coupled-dlom_agu_seed21_bs128_large-test', 'checkpoint':None},
        {'model_dir':'/home/disk/quicksilver/nacc/dlesm/zephyr/outputs/hpx32_coupled-dlom_agu_seed22_bs128_large-test', 'checkpoint':None},
        {'model_dir':'/home/disk/quicksilver/nacc/dlesm/zephyr/outputs/hpx32_coupled-dlom_agu_seed23_bs128_large-test', 'checkpoint':None},
        {'model_dir':'/home/disk/quicksilver/nacc/dlesm/zephyr/outputs/hpx32_coupled-dlom_agu_seed24_bs128_large-test', 'checkpoint':None},
        {'model_dir':'/home/disk/quicksilver/nacc/dlesm/zephyr/outputs/hpx32_coupled-dlom_agu_seed25_bs128_large-test', 'checkpoint':None},
        {'model_dir':'/home/disk/quicksilver/nacc/dlesm/zephyr/outputs/hpx32_coupled-dlom_agu_seed26_bs128_large-test', 'checkpoint':None},
        {'model_dir':'/home/disk/quicksilver/nacc/dlesm/zephyr/outputs/hpx32_coupled-dlom_agu_seed27_bs128_large-test', 'checkpoint':None},
    ],
    'forecast_params' : {
        'non_strict' : True,
        'lead_time' : 1152,
        'forecast_init_start': '2017-01-02',
        'forecast_init_end': '2018-12-30',
        'freq' : 'biweekly',
        'batch_size' : None,
        'output_directory' : '/home/disk/brass/nacc/forecasts/agu_coupled_ensemble',
        'atmos_output_filename' : None,
        'ocean_output_filename' : None,
        'encode_int' : False,
        'to_zarr' : False,
        'data_directory' : '/home/disk/mercury4/nacc/data/HPX32/',
        'data_prefix' : None,
        'data_suffix' : None,
        'gpu' : 0,
    },
    'overwrite' : False,
    'mix_match' : True,
}
# helper class to create attributable object from dictionary 
# this is necessary for inference treatment of parameters 
class ParamDict():
    def __init__(self, param_dict):
        os.chdir('/home/disk/quicksilver/nacc/dlesm/zephyr')
        param_dict['atmos_hydra_path'] = os.path.relpath(param_dict['atmos_model_path'], os.path.join(os.getcwd(), 'hydra'))
        param_dict['ocean_hydra_path'] = os.path.relpath(param_dict['ocean_model_path'], os.path.join(os.getcwd(), 'hydra'))
        self.__dict__= param_dict

def ensemble_inference(params):
    """
    This function will create an ensemble coupled forecast from the model-checkpoint combinations listed 
    in models. 

    params: 
    - 'atmos_models': A list of dictionaries, each representing an atmospheric model. Each dictionary has:
        - 'model_dir': The directory where the model's output is stored.
        - 'checkpoint': The checkpoint for the model. If None, no checkpoint is used.

    - 'ocean_models': A list of dictionaries, each representing an oceanic model. Each dictionary has the same structure as the atmospheric models.

    - 'forecast_params': A dictionary of parameters for the forecast. It includes:
        - 'non_strict': If True, the forecast is not strict.
        - 'lead_time': The lead time for the forecast.
        - 'forecast_init_start': The start date for the forecast initialization.
        - 'forecast_init_end': The end date for the forecast initialization.
        - 'freq': The frequency of the forecast. 'biweekly' means every two weeks.
        - 'batch_size': The batch size for the forecast. If None, no batch size is set.
        - 'output_directory': The directory where the forecast output is stored.
        - 'atmos_output_filename': The filename for the atmospheric forecast output. If None, no filename is set.
        - 'ocean_output_filename': The filename for the oceanic forecast output. If None, no filename is set.
        - 'encode_int': If True, integers are encoded.
        - 'to_zarr': If True, the output is converted to Zarr format.
        - 'data_directory': The directory where the data is stored.
        - 'data_prefix': The prefix for the data files. If None, no prefix is used.
        - 'data_suffix': The suffix for the data files. If None, no suffix is used.
        - 'gpu': The GPU to use for the forecast. 0 means the first GPU.
        - 'mix_match': If True the ocean and atmosphere models are mixed and matched to form an ensemble size n_ocean_models * n_atmos_models. 
                If false the ocean and atmosphere models are paired to form an ensemble size max(n_ocean_models, n_atmos_models), with the 
                smaller ensemble size being cycled through.
    - 'overwrite': If True, the existing forecasts are overwritten. If False, pass.
    """
    import json 
    #pretty=json.dumps(params,indent=2)
    #print(pretty)

    def get_model_name(model_dict):
        name = model_dict['model_dir'].split('/')[-1]
        ckpt = model_dict['checkpoint']

        if ckpt is None:
            return f"{name}-best"
        else:
            return f"{name}-{str(ckpt)}"
    
    # list of params to send to inference 
    param_list = []
    if params['mix_match']:
        # populate params list with model specific information
        for atmos_model in params['atmos_models']: 
            for ocean_model in params['ocean_models']:
                model_params = copy.deepcopy(params['forecast_params'])
                model_params['atmos_model_path'] = atmos_model['model_dir']
                model_params['atmos_model_checkpoint'] = atmos_model['checkpoint']
                model_params['ocean_model_path'] = ocean_model['model_dir']
                model_params['ocean_model_checkpoint'] = ocean_model['checkpoint']
                model_params['atmos_output_filename'] = f"atmos_{params['forecast_params']['lead_time']}h_{get_model_name(atmos_model)}+{get_model_name(ocean_model)}"
                model_params['ocean_output_filename'] = f"ocean_{params['forecast_params']['lead_time']}h_{get_model_name(atmos_model)}+{get_model_name(ocean_model)}"
                param_list.append(model_params)
                del model_params
    else:
        # populate params list with model specific information
        for i in range(max(len(params['atmos_models']), len(params['ocean_models']))):
            model_params = copy.deepcopy(params['forecast_params'])
            model_params['atmos_model_path'] = params['atmos_models'][i % len(params['atmos_models'])]['model_dir']
            model_params['atmos_model_checkpoint'] = params['atmos_models'][i % len(params['atmos_models'])]['checkpoint']
            model_params['ocean_model_path'] = params['ocean_models'][i % len(params['ocean_models'])]['model_dir']
            model_params['ocean_model_checkpoint'] = params['ocean_models'][i % len(params['ocean_models'])]['checkpoint']
            model_params['atmos_output_filename'] = f"atmos_{params['forecast_params']['lead_time']}h_{get_model_name(params['atmos_models'][i % len(params['atmos_models'])])}+{get_model_name(params['ocean_models'][i % len(params['ocean_models'])])}"
            model_params['ocean_output_filename'] = f"ocean_{params['forecast_params']['lead_time']}h_{get_model_name(params['atmos_models'][i % len(params['atmos_models'])])}+{get_model_name(params['ocean_models'][i % len(params['ocean_models'])])}"
            param_list.append(model_params)
            del model_params
          
    # create member forecasts 
    for model_param in param_list:
        forecast_exist = (os.path.isfile(f'{model_param["output_directory"]}/{model_param["atmos_output_filename"]}.nc') or os.path.isfile(f'{model_param["output_directory"]}/{model_param["ocean_output_filename"]}.nc'))
        if forecast_exist and not params['overwrite']:
            print(f'One or more of [{model_param["output_directory"]}/{model_param["atmos_output_filename"]}.nc, {model_param["output_directory"]}/{model_param["ocean_output_filename"]} exists.')
            print(f'To replace existing file pass "overwrite=True"')
        else:
            # perform inference
            coupled_inference(ParamDict(model_param))
        # import json 
        # pretty=json.dumps(model_param,indent=2)
        # print(pretty)

if __name__ == "__main__" :

    ensemble_inference(EXAMPLE_PARAMS)

    # ensemble_inference(MODELS, FORECAST_PARAMS, ENSEMBLE_PARAMS, OVERWRITE)
     
    
