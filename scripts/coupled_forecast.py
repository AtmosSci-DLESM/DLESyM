import logging
import argparse
import os
from pathlib import Path
import time

from hydra import initialize, compose
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate
import dask.array
import numpy as np
import pandas as pd
import torch as th
import xarray as xr
from tqdm import tqdm
import pprint
from torchinfo import summary
from dask.diagnostics import ProgressBar

from training.dlwp.utils import to_chunked_dataset, encode_variables_as_int, configure_logging, get_best_checkpoint_path
from scripts.forecast import _convert_time_step, get_forecast_dates, read_forecast_dates_from_file, get_latest_version

logger = logging.getLogger(__name__)
logging.getLogger('cfgrib').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

def get_coupled_time_dim(atmos_cfg, ocean_cfg):
    
    atmos_step = pd.Timedelta(atmos_cfg.data.gap)+(atmos_cfg.data.input_time_dim-1)*pd.Timedelta(atmos_cfg.data.time_step)
    ocean_step = pd.Timedelta(ocean_cfg.data.gap)+(ocean_cfg.data.input_time_dim-1)*pd.Timedelta(ocean_cfg.data.time_step)
    intersection = pd.Timedelta(np.lcm(atmos_step.delta, ocean_step.delta),units='ns')
    
    ocean_output_time_dim = (intersection // ocean_step)*ocean_cfg.data.input_time_dim
    atmos_output_time_dim = (intersection // atmos_step)*atmos_cfg.data.input_time_dim
    
    return atmos_output_time_dim, ocean_output_time_dim
  
def coupled_inference(args: argparse.Namespace):
    forecast_dates = get_forecast_dates(args.forecast_init_start, args.forecast_init_end, args.freq)
    os.makedirs(args.output_directory, exist_ok=True)

    if args.gpu == -1:
        device = th.device('cpu')
    else:
        device = th.device(f'cuda:{args.gpu}' if th.cuda.is_available() else 'cpu')
    with initialize(config_path=os.path.join(args.atmos_hydra_path, '.hydra'), version_base=None):
        atmos_cfg = compose('config.yaml')
    with initialize(config_path=os.path.join(args.ocean_hydra_path, '.hydra'), version_base=None):
        ocean_cfg = compose('config.yaml')

    batch_size = 1
    # override config params for forecasting 
    atmos_cfg.num_workers = 0
    atmos_cfg.batch_size = batch_size
    atmos_cfg.data.prebuilt_dataset = True
    ocean_cfg.num_workers = 0
    ocean_cfg.batch_size = batch_size
    ocean_cfg.data.prebuilt_dataset = True

    # some models do not have custom cuda healpix padding flags in config, instead they assume default behavior of model class
    # here we ensure that this default behaviopr is overridden fore forecasting 
    if not hasattr(atmos_cfg.model,'enable_healpixpad'):
        OmegaConf.set_struct(atmos_cfg,True)
        with open_dict(atmos_cfg):
            atmos_cfg.model.enable_healpixpad = False
    else:
        atmos_cfg.model.enable_healpixpad = False
    if not hasattr(ocean_cfg.model,'enable_healpixpad'):
        OmegaConf.set_struct(ocean_cfg,True)
        with open_dict(ocean_cfg):
            ocean_cfg.model.enable_healpixpad = False
    else:
        ocean_cfg.model.enable_healpixpad = False
 
    # Set up data module with some overrides for inference. Compute expected output time dimension.
    atmos_output_lead_times = np.arange(
        _convert_time_step(atmos_cfg.data.gap),
        _convert_time_step(args.lead_time) + pd.Timedelta(seconds=1),
        _convert_time_step(atmos_cfg.data.time_step)
    )
    ocean_output_lead_times = np.arange(
        _convert_time_step(ocean_cfg.data.gap),
        _convert_time_step(args.lead_time) + pd.Timedelta(seconds=1),
        _convert_time_step(ocean_cfg.data.time_step)
    )
    # figure concurrent forecasting time variables 
    atmos_coupled_time_dim, ocean_coupled_time_dim = get_coupled_time_dim(atmos_cfg, ocean_cfg)
    # The number of times each model will be called. Should be the same whether ocean or atmos 
    # is used to make calculation 
    forecast_integrations = len(ocean_output_lead_times) // ocean_coupled_time_dim
    # set up couplers for forecasting 
    try:
        nc = len(atmos_cfg.data.module.couplings)
        for i in range(nc):
            atmos_cfg.data.module.couplings[i]['params']['output_time_dim'] = atmos_coupled_time_dim 
    except AttributeError:
        print(f'model {args.atmos_model_path} is not interpreted as a coupled model, cannot perform coupled forecast. Aborting.')
    try:
        nc = len(ocean_cfg.data.module.couplings)
        for i in range(nc):
            ocean_cfg.data.module.couplings[i]['params']['output_time_dim'] = ocean_coupled_time_dim 
    except AttributeError:
        print(f'model {args.atmos_model_path} is not interpreted as a coupled model, cannot perform coupled forecast. Aborting.')
    
    optional_kwargs = {k: v for k, v in {
        'dst_directory': args.data_directory,
        'prefix': args.data_prefix,
        'suffix': args.data_suffix
    }.items() if v is not None}
    
    # instantiate data modules 
    atmos_data_module = instantiate(
        atmos_cfg.data.module,
        output_time_dim=atmos_coupled_time_dim,
        forecast_init_times=forecast_dates,
        shuffle=False,
        **optional_kwargs
    )
    atmos_loader, _ = atmos_data_module.test_dataloader()
    ocean_data_module = instantiate(
        ocean_cfg.data.module,
        output_time_dim=ocean_coupled_time_dim,
        forecast_init_times=forecast_dates,
        shuffle=False,
        **optional_kwargs
    )
    ocean_loader, _ = ocean_data_module.test_dataloader()
    
    # checks to make sure timeing and lead time line up 
    if forecast_integrations*ocean_coupled_time_dim*pd.Timedelta(ocean_data_module.time_step) != \
        forecast_integrations*atmos_coupled_time_dim*pd.Timedelta(atmos_data_module.time_step):
        raise ValueError('Lead times of atmos and ocean models does not align.')
    if forecast_integrations*ocean_coupled_time_dim*pd.Timedelta(ocean_data_module.time_step) != _convert_time_step(args.lead_time):
        raise ValueError(f'Requested leadtime ({_convert_time_step(args.lead_time)}) and coupled integration ({forecast_integrations*ocean_coupled_time_dim*pd.Timedelta(ocean_data_module.time_step)}) are not the same. Make sure lead time is compatible with component model intersections.')


    # Set output_time_dim param override.
    atmos_input_channels = len(atmos_cfg.data.input_variables)
    atmos_output_channels = len(atmos_cfg.data.output_variables) if atmos_cfg.data.output_variables is not None else atmos_input_channels
    atmos_constants_arr = atmos_data_module.constants
    atmos_n_constants = 0 if atmos_constants_arr is None else len(atmos_constants_arr.keys()) # previously was 0 but with new format it is 1
    ocean_input_channels = len(ocean_cfg.data.input_variables)
    ocean_output_channels = len(ocean_cfg.data.output_variables) if ocean_cfg.data.output_variables is not None else ocean_input_channels
    ocean_constants_arr = ocean_data_module.constants
    ocean_n_constants = 0 if ocean_constants_arr is None else len(ocean_constants_arr.keys()) # previously was 0 but with new format it is 1

    atmos_decoder_input_channels = int(atmos_cfg.data.get('add_insolation', 0))
    atmos_cfg.model['input_channels'] = atmos_input_channels
    atmos_cfg.model['output_channels'] = atmos_output_channels
    atmos_cfg.model['n_constants'] = atmos_n_constants
    atmos_cfg.model['decoder_input_channels'] = atmos_decoder_input_channels
    ocean_decoder_input_channels = int(ocean_cfg.data.get('add_insolation', 0))
    ocean_cfg.model['input_channels'] = ocean_input_channels
    ocean_cfg.model['output_channels'] = ocean_output_channels
    ocean_cfg.model['n_constants'] = ocean_n_constants
    ocean_cfg.model['decoder_input_channels'] = ocean_decoder_input_channels
    
    # instantiate models and find best checkpoint 
    atmos_model = instantiate(atmos_cfg.model, output_time_dim=atmos_coupled_time_dim)
    atmos_model_name = Path(args.atmos_model_path).name
    atmos_checkpoint_basepath = os.path.join(args.atmos_model_path, "tensorboard", "checkpoints")
    if args.atmos_model_checkpoint is None:
        atmos_checkpoint_path = get_best_checkpoint_path(path=atmos_checkpoint_basepath)
    else:
        atmos_checkpoint_path = os.path.join(atmos_checkpoint_basepath, args.atmos_model_checkpoint)
    logger.info("load model checkpoint %s", atmos_checkpoint_path)
    ocean_model = instantiate(ocean_cfg.model, output_time_dim=ocean_coupled_time_dim)
    ocean_model_name = Path(args.ocean_model_path).name
    ocean_checkpoint_basepath = os.path.join(args.ocean_model_path, "tensorboard", "checkpoints")
    if args.ocean_model_checkpoint is None:
        ocean_checkpoint_path = get_best_checkpoint_path(path=ocean_checkpoint_basepath)
    else:
        ocean_checkpoint_path = os.path.join(ocean_checkpoint_basepath, args.ocean_model_checkpoint)
    logger.info("load model checkpoint %s", ocean_checkpoint_path)

    # load state dicts and print summary 
    atmos_checkpoint = th.load(atmos_checkpoint_path, map_location=device)
    atmos_model_state_dict = atmos_checkpoint["model_state_dict"]
    atmos_model.load_state_dict(atmos_model_state_dict)
    atmos_model = atmos_model.to(device)
    atmos_model.eval()
    print(f'Atmos Model Summary:')
    summary(atmos_model)
    ocean_checkpoint = th.load(ocean_checkpoint_path, map_location=device)
    ocean_model_state_dict = ocean_checkpoint["model_state_dict"]
    ocean_model.load_state_dict(ocean_model_state_dict)
    ocean_model = ocean_model.to(device)
    ocean_model.eval()
    print(f'Ocean Model Summary:')
    summary(ocean_model)

    # Allocate giant array. One extra time step for the init state.
    logger.info("allocating prediction array. If this fails due to OOM consider reducing lead times or "
                "number of forecasts.")
    atmos_prediction = np.empty((len(forecast_dates),
                                len(atmos_output_lead_times) + 1,
                                len(atmos_data_module.output_variables)) + atmos_data_module.test_dataset.spatial_dims,
                                dtype='float32')
    ocean_prediction = np.empty((len(forecast_dates),
                                len(ocean_output_lead_times) + 1,
                                len(ocean_data_module.output_variables)) + ocean_data_module.test_dataset.spatial_dims,
                                dtype='float32')

    # get references to the atmos and ocean couplers and iteratable loaders 
    atmos_coupler = atmos_data_module.test_dataset.couplings[0]
    atmos_coupler.setup_coupling(ocean_data_module)
    atmos_loader_iter = iter(atmos_loader)
    ocean_coupler = ocean_data_module.test_dataset.couplings[0]
    ocean_coupler.setup_coupling(atmos_data_module)
    ocean_loader_iter = iter(ocean_loader)
    
    # integrations 
    # Initialize progress bar 
    pbar = tqdm(forecast_dates)
    # buffers for updating inputs after each integration 
    atmos_constants = None 
    ocean_constants = None

    # dummy models that produce ground truth values are used for debugging coupling. These models need information 
    # about forecast dates and integration_time_dim. Set them here.
    for model, data_module in [[atmos_model,atmos_data_module], [ocean_model, ocean_data_module]]: 
        if getattr(model,'debugging_model', False):
            model.set_output(forecast_dates, 
                             forecast_integrations,
                             data_module,)
   
    # loop through initializations and produce forecasts 
    for i,init in enumerate(pbar):
        # update progress bar 
        pbar.postfix = pd.Timestamp(forecast_dates[i]).strftime('init %Y-%m-%d %HZ')
        pbar.update()
        new_init = True
        # reset_couplers for new initialization 
        atmos_coupler.reset_coupler()
        ocean_coupler.reset_coupler()

        for j in range(forecast_integrations):
            if j==0:
                # Get input field and forecast with atmos model
                atmos_input = [k.to(device) for k in next(atmos_loader_iter)]
                if atmos_constants is None:
                    atmos_constants = atmos_input[2]
                with th.no_grad():
                    atmos_output = atmos_model(atmos_input)
                
                # Repeat with ocean model, use atmos output to set forcing
                ocean_coupler.set_coupled_fields(atmos_output.cpu())
                ocean_input = [k.to(device) for k in next(ocean_loader_iter)]
                if ocean_constants is None:
                    ocean_constants = ocean_input[2]
                with th.no_grad():
                    ocean_output = ocean_model(ocean_input)
            else:
                atmos_coupler.set_coupled_fields(ocean_output)
                atmos_input = [k.to(device) for k in atmos_data_module.test_dataset.next_integration(
                                                         atmos_output, 
                                                         constants = atmos_constants,
                                                     )]

                with th.no_grad():
                    atmos_output = atmos_model(atmos_input)
       
                ocean_coupler.set_coupled_fields(atmos_output.cpu())
                ocean_input = [k.to(device) for k in ocean_data_module.test_dataset.next_integration(
                                                         ocean_output, 
                                                         constants = ocean_constants,
                                                     )]
                with th.no_grad():
                    ocean_output = ocean_model(ocean_input)

            # populate first timestep with initialization data 
            if new_init:
                ocean_prediction[i*batch_size:(i+1)*batch_size][:, 0] = ocean_input[0].permute(0, 2, 3, 1, 4, 5)[:, -1].cpu().numpy()
                atmos_prediction[i*batch_size:(i+1)*batch_size][:, 0] = atmos_input[0].permute(0, 2, 3, 1, 4, 5)[:, -1].cpu().numpy()
                new_init = False

            # fill rest of integration step with model output  
            ocean_prediction[i*batch_size:(i+1)*batch_size][:,slice(1+j*ocean_coupled_time_dim,(j+1)*ocean_coupled_time_dim+1)] = ocean_output.permute(0, 2, 3, 1, 4, 5).cpu().numpy()
            atmos_prediction[i*batch_size:(i+1)*batch_size][:,slice(1+j*atmos_coupled_time_dim,(j+1)*atmos_coupled_time_dim+1)] = atmos_output.permute(0, 2, 3, 1, 4, 5).cpu().numpy()

    # Generate dataarray with coordinates
    ocean_meta_ds = ocean_data_module.test_dataset.ds
    atmos_meta_ds = atmos_data_module.test_dataset.ds
    if args.to_zarr:
        ocean_prediction = dask.array.from_array(ocean_prediction, chunks=(1,) + ocean_prediction.shape[1:])
        atmos_prediction = dask.array.from_array(atmos_prediction, chunks=(1,) + atmos_prediction.shape[1:])
    ocean_prediction_da = xr.DataArray(
        ocean_prediction,
        dims=['time', 'step', 'channel_out', 'face', 'height', 'width'],
        coords={
            'time': forecast_dates,
            'step': [pd.Timedelta(hours=0)] + list(ocean_output_lead_times),
            'channel_out': ocean_cfg.data.output_variables or ocean_cfg.data.input_variables,
            'face': ocean_meta_ds.face,
            'height': ocean_meta_ds.height,
            'width': ocean_meta_ds.width
        }
    )
    atmos_prediction_da = xr.DataArray(
        atmos_prediction,
        dims=['time', 'step', 'channel_out', 'face', 'height', 'width'],
        coords={
            'time': forecast_dates,
            'step': [pd.Timedelta(hours=0)] + list(atmos_output_lead_times),
            'channel_out': atmos_cfg.data.output_variables or atmos_cfg.data.input_variables,
            'face': atmos_meta_ds.face,
            'height': atmos_meta_ds.height,
            'width': atmos_meta_ds.width
        }
    )
    # Re-scale prediction
    ocean_prediction_da[:] *= ocean_data_module.test_dataset.target_scaling['std']
    ocean_prediction_da[:] += ocean_data_module.test_dataset.target_scaling['mean']
    ocean_prediction_ds = ocean_prediction_da.to_dataset(dim='channel_out')
    for variable in ocean_prediction_ds.data_vars:
        if ocean_cfg.data.scaling[variable].get('log_epsilon', None) is not None:
            ocean_prediction_ds[variable] = np.exp(
                ocean_prediction_ds[variable] + np.log(ocean_cfg.data.scaling[variable]['log_epsilon'])
            ) - ocean_cfg.data.scaling[variable]['log_epsilon']
    atmos_prediction_da[:] *= atmos_data_module.test_dataset.target_scaling['std']
    atmos_prediction_da[:] += atmos_data_module.test_dataset.target_scaling['mean']
    atmos_prediction_ds = atmos_prediction_da.to_dataset(dim='channel_out')
    for variable in atmos_prediction_ds.data_vars:
        if atmos_cfg.data.scaling[variable].get('log_epsilon', None) is not None:
            atmos_prediction_ds[variable] = np.exp(
                atmos_prediction_ds[variable] + np.log(atmos_cfg.data.scaling[variable]['log_epsilon'])
            ) - atmos_cfg.data.scaling[variable]['log_epsilon']

    # Export dataset
    write_time = time.time()
    ocean_prediction_ds = to_chunked_dataset(ocean_prediction_ds, {'time': 8})
    atmos_prediction_ds = to_chunked_dataset(atmos_prediction_ds, {'time': 8})
    if args.encode_int:
        ocean_prediction_ds = encode_variables_as_int(ocean_prediction_ds, compress=1)
        atmos_prediction_ds = encode_variables_as_int(atmos_prediction_ds, compress=1)

    if getattr(args,'ocean_output_filename',None) is not None:
        ocean_output_file = os.path.join(args.output_directory, f"{args.ocean_output_filename}.{'zarr' if args.to_zarr else 'nc'}")
    if getattr(args,'atmos_output_filename',None) is not None:
        atmos_output_file = os.path.join(args.output_directory, f"{args.atmos_output_filename}.{'zarr' if args.to_zarr else 'nc'}")
    else:
        ocean_output_file = os.path.join(args.output_directory, f"forecast_{ocean_model_name}.{'zarr' if args.to_zarr else 'nc'}")
        atmos_output_file = os.path.join(args.output_directory, f"forecast_{atmos_model_name}.{'zarr' if args.to_zarr else 'nc'}")
    logger.info(f"writing forecasts to {atmos_output_file} and {ocean_output_file}")
    if args.to_zarr:
        ocean_write_job = ocean_prediction_ds.to_zarr(ocean_output_file, compute=False)
        atmos_write_job = atmos_prediction_ds.to_zarr(atmos_output_file, compute=False)
    else:
        ocean_write_job = ocean_prediction_ds.to_netcdf(ocean_output_file, compute=False)
        atmos_write_job = atmos_prediction_ds.to_netcdf(atmos_output_file, compute=False)
    with ProgressBar():
        ocean_write_job.compute()
        atmos_write_job.compute()
    logger.debug("wrote file in %0.1f s", time.time() - write_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Produce forecasts from a DLWP model.')
    parser.add_argument('--atmos-model-path', type=str, required=True,
                        help="Path to model training outputs directory for the coupled atmos model")
    parser.add_argument('--ocean-model-path', type=str, required=True,
                        help="Path to model training outputs directory for the coupled ocean model")
    parser.add_argument('--atmos-model-checkpoint', type=str, default=None,
                        help="Atmos Model checkpoint file name (include ending). Set 'last.ckpt' to use last checkpoint. If "
                             "None, the best will be chosen (according to validation error).")
    parser.add_argument('--ocean-model-checkpoint', type=str, default=None,
                        help="Ocean Model checkpoint file name (include ending). Set 'last.ckpt' to use last checkpoint. If "
                             "None, the best will be chosen (according to validation error).")
    parser.add_argument('--non-strict', action='store_true',
                        help="Disable strict mode for model checkpoint loading")
    parser.add_argument('-l', '--lead-time', type=int, default=168,
                        help="Maximum forecast lead time to predict, in integer hours")
    parser.add_argument('-s', '--forecast-init-start', type=str, default='2017-01-02',
                        help="")
    parser.add_argument('-e', '--forecast-init-end', type=str, default='2018-12-30',
                        help="")
    parser.add_argument('-f', '--freq', type=str, default='biweekly',
                        help="Frequency of forecast initialization. There is a special case, 'biweekly', which will "
                             "follow the ECMWF standard of two forecasts per week, with a 3- followed by 4-day gap. "
                             "Otherwise, interpretable by pandas.")
    parser.add_argument('-b', '--batch-size', type=int, default=None,
                        help="The batch size that is used to generate the forecast.")
    parser.add_argument('-o', '--output-directory', type=str, default='forecasts/',
                        help="Directory in which to save output forecast")
    parser.add_argument('--atmos-output-filename', type=str, default=None,
                        help="Filename used to save atmos output forecast")
    parser.add_argument('--ocean-output-filename', type=str, default=None,
                        help="Filename used to save ocean output forecast")
    parser.add_argument('--encode-int', action='store_true',
                        help="Encode data variables as int16 type (may not be compatible with tempest-remap)")
    parser.add_argument('--to-zarr', action='store_true',
                        help="Export data in zarr format")
    parser.add_argument('-d', '--data-directory', type=str, default=None,
                        help="Path to test data, if different from files used for model training")
    parser.add_argument('--data-prefix', type=str, default=None,
                        help="Prefix for test data files")
    parser.add_argument('--data-suffix', type=str, default=None,
                        help="Suffix for test data files")
    parser.add_argument('--gpu', type=int, default=0,
                        help="Index of GPU device on which to run model. If -1 forecast will be done on CPU")

    configure_logging(2)
    run_args = parser.parse_args()

    # Hydra requires a relative (not absolute) path to working config directory. It also works in a sub-directory of
    # current python working directory.
    run_args.atmos_hydra_path = os.path.relpath(run_args.atmos_model_path, os.path.join(os.getcwd(), 'hydra'))
    run_args.ocean_hydra_path = os.path.relpath(run_args.ocean_model_path, os.path.join(os.getcwd(), 'hydra'))
    logger.debug("model paths: %s", (run_args.atmos_model_path, run_args.ocean_model_path))
    logger.debug("python working dir: %s", os.getcwd())
    logger.debug("hydra paths: %s", (run_args.atmos_hydra_path, run_args.ocean_hydra_path))
    coupled_inference(run_args)
