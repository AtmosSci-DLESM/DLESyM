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
from torchinfo import summary
from dask.diagnostics import ProgressBar

from training.dlwp.utils import to_chunked_dataset, encode_variables_as_int, configure_logging, get_best_checkpoint_path

logger = logging.getLogger(__name__)
logging.getLogger('cfgrib').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)


def _convert_time_step(dt):  # pylint: disable=invalid-name
    return pd.Timedelta(hours=dt) if isinstance(dt, int) else pd.Timedelta(dt)


def get_forecast_dates(start, end, freq):
    # Get the dates parameters
    if freq == 'biweekly':
        dates_1 = pd.date_range(start, end, freq='7D')
        dates_2 = pd.date_range(pd.Timestamp(start) + pd.Timedelta(days=3), end, freq='7D')
        dates = dates_1.append(dates_2).sort_values().to_numpy()
    else:
        dates = pd.date_range(start, end, freq=freq).to_numpy()
    return dates


def read_forecast_dates_from_file(path):
    import glob
    file_paths = glob.glob(os.path.join(path, "*.txt"))
    all_dates = []
    for file_path in file_paths:
        with open(file_path) as file:
            dates = file.read().splitlines()
            for date in dates:
                if not np.datetime64("2000-01-01") < np.datetime64(date) < np.datetime64("2021-01-01"): continue
                # Create numpy datetime64 at t-8, t-7, t-6, t-5, t-4 days
                all_dates += list(pd.date_range(start=pd.Timestamp(date)-pd.Timedelta(8, "D"),
                                                end=pd.Timestamp(date)-pd.Timedelta(4, "D"),
                                                freq=pd.Timedelta(1, "D")).to_numpy())
    return np.unique(np.sort(np.array(all_dates)))


def get_latest_version(directory):
    all_versions = [os.path.join(directory, v) for v in os.listdir(directory)]
    all_versions = [v for v in all_versions if os.path.isdir(v)]
    latest_version = max(all_versions, key=os.path.getmtime)
    return Path(latest_version).name


def inference(args: argparse.Namespace):
    forecast_dates = get_forecast_dates(args.forecast_init_start, args.forecast_init_end, args.freq)
    os.makedirs(args.output_directory, exist_ok=True)

    device = th.device(f'cuda:{args.gpu}' if th.cuda.is_available() else 'cpu')
    with initialize(config_path=os.path.join(args.hydra_path, '.hydra'), version_base=None):
        cfg = compose('config.yaml')
    batch_size = cfg.batch_size if args.batch_size is None else args.batch_size

    cfg.num_workers = 0
    batch_size = 1
    cfg.batch_size = batch_size
    cfg.data.prebuilt_dataset = True
    # some models do not have custom cuda healpix padding flags in config, instead they assume default behavior of model class
    # here we ensure that this default behaviopr is overridden fore forecasting 
    if not hasattr(cfg.model,'enable_healpixpad'):
        OmegaConf.set_struct(cfg,True)
        with open_dict(cfg):
            cfg.model.enable_healpixpad = False
    else:
        cfg.model.enable_healpixpad = False
 


    # Set up data module with some overrides for inference. Compute expected output time dimension.
    output_lead_times = np.arange(
        _convert_time_step(cfg.data.gap),
        _convert_time_step(args.lead_time) + pd.Timedelta(seconds=1),
        _convert_time_step(cfg.data.time_step)
    )
    output_time_dim = len(output_lead_times)
    # if coupling is included, set up coupler for detached forecasting 
    if hasattr(cfg.data.module,'couplings'):
        nc = len(cfg.data.module.couplings)
        for i in range(nc):
            cfg.data.module.couplings[i]['params']['output_time_dim'] = len(output_lead_times) 
    
    optional_kwargs = {k: v for k, v in {
        'dst_directory': args.data_directory,
        'dataset_name': args.dataset_name,
        'prefix': args.data_prefix,
        'suffix': args.data_suffix
    }.items() if v is not None}
    data_module = instantiate(
        cfg.data.module,
        output_time_dim=output_time_dim,
        forecast_init_times=forecast_dates,
        shuffle=False,
#        batch_size=batch_size,
        **optional_kwargs
    )
    loader, _ = data_module.test_dataloader()

    # Load the model checkpoint. Set output_time_dim param override.
    input_channels = len(cfg.data.input_variables)
    output_channels = len(cfg.data.output_variables) if cfg.data.output_variables is not None else input_channels
    constants_arr = data_module.constants
    n_constants = 0 if constants_arr is None else len(constants_arr.keys()) # previously was 0 but with new format it is 1

    decoder_input_channels = int(cfg.data.get('add_insolation', 0))
    cfg.model['input_channels'] = input_channels
    cfg.model['output_channels'] = output_channels
    cfg.model['n_constants'] = n_constants
    cfg.model['decoder_input_channels'] = decoder_input_channels

    #constants_arr = data_module.get_constants()
    #model = instantiate(cfg.model, constants=constants_arr, output_time_dim=output_time_dim)
    model = instantiate(cfg.model, output_time_dim=output_time_dim)
    model_name = Path(args.model_path).name
    checkpoint_basepath = os.path.join(args.model_path, "tensorboard", "checkpoints")
    if args.model_checkpoint is None:
        checkpoint_path = get_best_checkpoint_path(path=checkpoint_basepath)
    else:
        checkpoint_path = os.path.join(checkpoint_basepath, args.model_checkpoint)
    logger.info("load model checkpoint %s", checkpoint_path)

    checkpoint = th.load(checkpoint_path, map_location=device)
    #model_state_dict = {key.replace("module.", ""): checkpoint["model_state_dict"][key] for key in checkpoint["model_state_dict"].keys()}
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()
    summary(model)

    # Allocate giant array. One extra time step for the init state.
    logger.info("allocating prediction array. If this fails due to OOM consider reducing lead times or "
                "number of forecasts.")
    prediction = np.empty((len(forecast_dates),
                           output_time_dim + 1,
                           len(data_module.output_variables)) + data_module.test_dataset.spatial_dims,
                           dtype='float32')

    # Iterate over model predictions
    pbar = tqdm(loader)
    for j, inputs in enumerate(pbar):
        pbar.postfix = pd.Timestamp(forecast_dates[j]).strftime('init %Y-%m-%d %HZ')
        pbar.update()
        # Last input time step for init state
        prediction[j*batch_size:(j+1)*batch_size][:, 0] = inputs[0].permute(0, 2, 3, 1, 4, 5)[:, -1]
        inputs = [i.to(device) for i in inputs]
        with th.no_grad():
            prediction[j*batch_size:(j+1)*batch_size][:, 1:] = model(inputs).permute(0, 2, 3, 1, 4, 5).cpu().numpy()

    # Generate dataarray with coordinates
    meta_ds = data_module.test_dataset.ds
    if args.to_zarr:
        prediction = dask.array.from_array(prediction, chunks=(1,) + prediction.shape[1:])
    prediction_da = xr.DataArray(
        prediction,
        dims=['time', 'step', 'channel_out', 'face', 'height', 'width'],
        coords={
            'time': forecast_dates,
            'step': [pd.Timedelta(hours=0)] + list(output_lead_times),
            'channel_out': cfg.data.output_variables or cfg.data.input_variables,
            'face': meta_ds.face,
            'height': meta_ds.height,
            'width': meta_ds.width
        }
    )
    # Re-scale prediction
    prediction_da[:] *= data_module.test_dataset.target_scaling['std']
    prediction_da[:] += data_module.test_dataset.target_scaling['mean']
    prediction_ds = prediction_da.to_dataset(dim='channel_out')
    for variable in prediction_ds.data_vars:
        if cfg.data.scaling[variable].get('log_epsilon', None) is not None:
            prediction_ds[variable] = np.exp(
                prediction_ds[variable] + np.log(cfg.data.scaling[variable]['log_epsilon'])
            ) - cfg.data.scaling[variable]['log_epsilon']

    # Export dataset
    write_time = time.time()
    prediction_ds = to_chunked_dataset(prediction_ds, {'time': 8})
    if args.encode_int:
        prediction_ds = encode_variables_as_int(prediction_ds, compress=1)

    if args.output_filename is None:
        output_file = os.path.join(args.output_directory, f"forecast_{model_name}.{'zarr' if args.to_zarr else 'nc'}")
    else:
        output_file = os.path.join(args.output_directory, f"{args.output_filename}.{'zarr' if args.to_zarr else 'nc'}")
    logger.info(f"writing forecasts to {output_file}")
    if args.to_zarr:
        write_job = prediction_ds.to_zarr(output_file, compute=False)
    else:
        write_job = prediction_ds.to_netcdf(output_file, compute=False)
    with ProgressBar():
        write_job.compute()
    logger.debug("wrote file in %0.1f s", time.time() - write_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Produce forecasts from a DLWP model.')
    parser.add_argument('-m', '--model-path', type=str, required=True,
                        help="Path to model training outputs directory")
    parser.add_argument('-c', '--model-checkpoint', type=str, default=None,
                        help="Model checkpoint file name (include ending). Set 'last.ckpt' to use last checkpoint. If "
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
    parser.add_argument('--output-filename', type=str, default=None,
                        help="Name of file to hold forecast")
    parser.add_argument('--encode-int', action='store_true',
                        help="Encode data variables as int16 type (may not be compatible with tempest-remap)")
    parser.add_argument('--to-zarr', action='store_true',
                        help="Export data in zarr format")
    parser.add_argument('-d', '--data-directory', type=str, default=None,
                        help="Path to test data, if different from files used for model training")
    parser.add_argument('--dataset-name', type=str, default=None,
                        help="Name of init dataset, if different from files used for model training")  
    parser.add_argument('--data-prefix', type=str, default=None,
                        help="Prefix for test data files")
    parser.add_argument('--data-suffix', type=str, default=None,
                        help="Suffix for test data files")
    parser.add_argument('--gpu', type=int, default=0,
                        help="Index of GPU device on which to run model")

    configure_logging(2)
    run_args = parser.parse_args()

    # Hydra requires a relative (not absolute) path to working config directory. It also works in a sub-directory of
    # current python working directory.
    run_args.hydra_path = os.path.relpath(run_args.model_path, os.path.join(os.getcwd(), 'hydra'))
    logger.debug("model path: %s", run_args.model_path)
    logger.debug("python working dir: %s", os.getcwd())
    logger.debug("hydra path: %s", run_args.hydra_path)
    inference(run_args)
