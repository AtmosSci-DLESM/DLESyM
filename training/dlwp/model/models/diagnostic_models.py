import logging
import os
from typing import Any, Dict, Optional, Sequence, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
import xarray as xr 
import torch as th
from dask.diagnostics import ProgressBar
import pandas as pd

from training.dlwp.model.modules.healpix import HEALPixPadding, HEALPixLayer
from training.dlwp.model.modules.encoder import UNetEncoder, UNet3Encoder
from training.dlwp.model.modules.decoder import UNetDecoder, UNet3Decoder
from training.dlwp.model.modules.blocks import FoldFaces, UnfoldFaces
from training.dlwp.model.modules.losses import LossOnStep
from training.dlwp.model.modules.utils import Interpolate
from training.dlwp.model.models.unet3plus import HEALPixUNet3Plus

logger = logging.getLogger(__name__)

class ocean_gt_model(HEALPixUNet3Plus):
    def __init__(
            self,
            encoder: DictConfig,
            decoder: DictConfig,
            input_channels: int,
            output_channels: int,
            n_constants: int,
            decoder_input_channels: int,
            input_time_dim: int,
            output_time_dim: int,
            nside: int = 32,
            n_coupled_inputs: int = 0,
            enable_healpixpad = True,
            enable_nhwc = False,
            couplings: list = [],
            gt_dataset: str = None,
    ):
        """
        Ground truth model that produces the ground truth values for the ocean state.

        :param encoder: dictionary of instantiable parameters for the U-net encoder (see UnetEncoder docs)
        :param decoder: dictionary of instantiable parameters for the U-net decoder (see UnetDecoder docs)
        :param input_channels: number of input channels expected in the input array schema. Note this should be the
            number of input variables in the data, NOT including data reshaping for the encoder part.
        :param output_channels: number of output channels expected in the output array schema, or output variables
        :param n_constants: number of optional constants expected in the input arrays. If this is zero, no constants
            should be provided as inputs to `forward`.
        :param decoder_input_channels: number of optional prescribed variables expected in the decoder input array
            for both inputs and outputs. If this is zero, no decoder inputs should be provided as inputs to `forward`.
        :param input_time_dim: number of time steps in the input array
        :param output_time_dim: number of time steps in the output array
        :param nside: number of points on the side of a HEALPix face
        :param n_coupled_inputs: Number of channels model will receive from another coupled model. Default 0 
            assumes no coupling and performs similarly to traditional HEALPixUnet 
        :param couplings: sequence of dictionaries that describe coupling mechanisms
        """
        # Attribute declares this as a debuggin model for forecasting methods 
        self.debugging_model = True

        super().__init__(
            encoder,
            decoder,
            input_channels,
            output_channels,
            n_constants,
            decoder_input_channels,
            input_time_dim,
            output_time_dim,
            nside,
            n_coupled_inputs,
            enable_healpixpad,
            enable_nhwc,
            couplings,
        )

        # params used for constructing ground truth dataset 
        self.gt_dataset = xr.open_dataset(gt_dataset,engine='zarr')
        self.forecast_dates = None 
        self.integration_time_dim = None
        self.integration_counter = 0 
        self.initialization_counter = 0

    def set_output(self, forecast_dates, forecast_integrations, data_module):

        # set fields necessary for gt forecasting 
        self.forecast_dates = forecast_dates 
        self.forecast_integrations = forecast_integrations
        self.mean = data_module.test_dataset.target_scaling['mean'].transpose(0,2,1,3,4)
        self.std = data_module.test_dataset.target_scaling['std'].transpose(0,2,1,3,4)
        self.output_vars = data_module.test_dataset.output_variables 
        self.delta_t = pd.Timedelta(data_module.time_step)
        
    def forward(self, input):

        # check if we're on a new initialization 
        if self.integration_counter == self.forecast_integrations:
            self.initialization_counter+=1
            self.integration_counter=0

        dt = self.delta_t # abbreviation 

        # output array buffer. hard coded for hpx32. This will do for now. issues in the future will 
        # fail loudly
        output_array = th.empty([1, 12, self.output_time_dim, self.output_channels, 32, 32])

        for i in range(0,self.output_time_dim):
        
            # print(f'appending timestep: {(self.integration_counter*self.output_time_dim+(i+1))*dt + self.forecast_dates[self.initialization_counter]}')
            output_array[:,:,i,:,:,:]=th.tensor(self.gt_dataset.targets.sel(channel_out=self.output_vars,time=(self.integration_counter*self.output_time_dim+(i+1))*dt + self.forecast_dates[self.initialization_counter]).values.transpose([1,0,2,3]))

        # increment integration counter as appropriate
        self.integration_counter+=1 

        # scale and return output
        return (output_array - self.mean) / self.std
    
class ocean_climo_model(HEALPixUNet3Plus):
    def __init__(
            self,
            encoder: DictConfig,
            decoder: DictConfig,
            input_channels: int,
            output_channels: int,
            n_constants: int,
            decoder_input_channels: int,
            input_time_dim: int,
            output_time_dim: int,
            nside: int = 32,
            n_coupled_inputs: int = 0,
            enable_healpixpad = True,
            enable_nhwc = False,
            couplings: list = [],
            gt_dataset: str = None,
            climo_dataset: str = None,
    ):
        """
        Climo model that produces daily climatology for appropriate date with each call

        :param encoder: dictionary of instantiable parameters for the U-net encoder (see UnetEncoder docs)
        :param decoder: dictionary of instantiable parameters for the U-net decoder (see UnetDecoder docs)
        :param input_channels: number of input channels expected in the input array schema. Note this should be the
            number of input variables in the data, NOT including data reshaping for the encoder part.
        :param output_channels: number of output channels expected in the output array schema, or output variables
        :param n_constants: number of optional constants expected in the input arrays. If this is zero, no constants
            should be provided as inputs to `forward`.
        :param decoder_input_channels: number of optional prescribed variables expected in the decoder input array
            for both inputs and outputs. If this is zero, no decoder inputs should be provided as inputs to `forward`.
        :param input_time_dim: number of time steps in the input array
        :param output_time_dim: number of time steps in the output array
        :param nside: number of points on the side of a HEALPix face
        :param n_coupled_inputs: Number of channels model will receive from another coupled model. Default 0 
            assumes no coupling and performs similarly to traditional HEALPixUnet 
        :param couplings: sequence of dictionaries that describe coupling mechanisms
        """
        # Attribute declares this as a debuggin model for forecasting methods 
        self.debugging_model = True

        super().__init__(
            encoder,
            decoder,
            input_channels,
            output_channels,
            n_constants,
            decoder_input_channels,
            input_time_dim,
            output_time_dim,
            nside,
            n_coupled_inputs,
            enable_healpixpad,
            enable_nhwc,
            couplings,
        )

        
        self.gt_dataset = xr.open_dataset(gt_dataset,engine='zarr')
        if os.path.exists(climo_dataset):
            logger.info(f'openning climo dataset found at {climo_dataset}')
            self.climo_dataset = xr.open_dataset(climo_dataset)
        else:
            logger.info(f'climo dataset not found at {climo_dataset}, calculating climo from {gt_dataset} and saving to {climo_dataset}')
            self.gt_dataset = self.gt_dataset.chunk('auto')
            print(f'chunked gt dataset:')
            print(self.gt_dataset)
            self.climo_dataset = self.gt_dataset.groupby('time.dayofyear').mean('time')
            with ProgressBar():
                self.climo_dataset.to_netcdf(climo_dataset, mode='w')

        # params used for constructing ground truth dataset 
        self.forecast_dates = None 
        self.integration_time_dim = None
        self.integration_counter = 0 
        self.initialization_counter = 0
        self.nside = nside

    def set_output(self, forecast_dates, forecast_integrations, data_module):

        # set fields necessary for gt forecasting 
        self.forecast_dates = forecast_dates
        self.forecast_integrations = forecast_integrations
        self.mean = data_module.test_dataset.target_scaling['mean'].transpose(0,2,1,3,4)
        self.std = data_module.test_dataset.target_scaling['std'].transpose(0,2,1,3,4)
        self.output_vars = data_module.test_dataset.output_variables
        self.delta_t = pd.Timedelta(data_module.time_step)
        
    def forward(self, input):

        # check if we're on a new initialization 
        if self.integration_counter == self.forecast_integrations:
            self.initialization_counter+=1
            self.integration_counter=0

        dt = self.delta_t # abbreviation 

        # output array buffer. hard coded for hpx32. This will do for now. issues in the future will 
        # fail loudly
        output_array = th.empty([1, 12, self.output_time_dim, self.output_channels, self.nside, self.nside])

        for i in range(0,self.output_time_dim):
          
            # calculate day of year for requested time step 
            doy = pd.Timestamp((self.integration_counter*self.output_time_dim+(i+1))*dt + self.forecast_dates[self.initialization_counter]).dayofyear
            # populate output array with daily climatology for the requested day 
            output_array[:,:,i,:,:,:]=th.tensor(self.climo_dataset.targets.sel(channel_out=self.output_vars, dayofyear=doy).values.transpose([1,0,2,3]))

        # increment integration counter as appropriate
        self.integration_counter+=1 

        # scale and return output
        return (output_array - self.mean) / self.std

class atmos_gt_model(th.nn.Module):
    def __init__(
            self,
            encoder: DictConfig,
            decoder: DictConfig,
            input_channels: int,
            output_channels: int,
            n_constants: int,
            decoder_input_channels: int,
            input_time_dim: int,
            output_time_dim: int,
            delta_time: str = "6H",
            reset_cycle: str = "24H",
            presteps: int = 1,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False,
            couplings: list = [],
            gt_dataset: str = None,
    ):
        """
        Ground truth model that produces the ground truth values for the atmos state.

        :param encoder: dictionary of instantiable parameters for the U-net encoder
        :param decoder: dictionary of instantiable parameters for the U-net decoder
        :param input_channels: number of input channels expected in the input array schema. Note this should be the
            number of input variables in the data, NOT including data reshaping for the encoder part.
        :param output_channels: number of output channels expected in the output array schema, or output variables
        :param n_constants: number of optional constants expected in the input arrays. If this is zero, no constants
            should be provided as inputs to `forward`.
        :param decoder_input_channels: number of optional prescribed variables expected in the decoder input array
            for both inputs and outputs. If this is zero, no decoder inputs should be provided as inputs to `forward`.
        :param input_time_dim: number of time steps in the input array
        :param output_time_dim: number of time steps in the output array
        :param delta_time: hours between two consecutive data points
        :param reset_cycle: hours after which the recurrent states are reset to zero and re-initialized. Set np.infty
            to never reset the hidden states.
        :param presteps: number of model steps to initialize recurrent states.
        :param enable_nhwc: Model with [N, H, W, C] instead of [N, C, H, W]
        :param enable_healpixpad: Enable CUDA HEALPixPadding if installed
        :param coupings: sequence of dictionaries that describe coupling mechanisms
        :param gt_datset: the dataset of ground truth values to populate ground through prediction 
        """

        # Attribute declares this as a debuggin model for forecasting methods 
        self.debugging_model = True

        super().__init__()
        self.channel_dim = 2  # Now 2 with [B, F, T*C, H, W]. Was 1 in old data format with [B, T*C, F, H, W]

        self.input_channels = input_channels
        # add coupled fields to input channels for model initialization 
        self.coupled_channels = self._compute_coupled_channels(couplings) 
        self.couplings = couplings 
        self.train_couplers = None
        self.output_channels = output_channels
        self.n_constants = n_constants
        self.decoder_input_channels = decoder_input_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.delta_t = int(pd.Timedelta(delta_time).total_seconds()//3600)
        self.reset_cycle = int(pd.Timedelta(reset_cycle).total_seconds()//3600)
        self.presteps = presteps
        self.enable_nhwc = enable_nhwc
        self.enable_healpixpad = enable_healpixpad
        self.gt_dataset = xr.open_dataset(gt_dataset,engine='zarr')

        # params used for constructing ground truth dataset 
        self.forecast_dates = None 
        self.integration_time_dim = None
        self.integration_counter = 0 
        self.initialization_counter = 0

        # Number of passes through the model, or a diagnostic model with only one output time
        self.is_diagnostic = self.output_time_dim == 1 and self.input_time_dim > 1
        if not self.is_diagnostic and (self.output_time_dim % self.input_time_dim != 0):
            raise ValueError(f"'output_time_dim' must be a multiple of 'input_time_dim' (got "
                             f"{self.output_time_dim} and {self.input_time_dim})")

        # Build the model layers
        # Build the model layers
        self.fold = FoldFaces()
        self.unfold = UnfoldFaces(num_faces=12)
        self.encoder = instantiate(config=encoder,
                                   input_channels=self._compute_input_channels(),
                                   enable_nhwc=self.enable_nhwc,
                                   enable_healpixpad=self.enable_healpixpad)
        self.encoder_depth = len(self.encoder.n_channels)
        self.decoder = instantiate(config=decoder,
                                   output_channels=self._compute_output_channels(),
                                   enable_nhwc = self.enable_nhwc,
                                   enable_healpixpad = self.enable_healpixpad)

    def _compute_input_channels(self) -> int:
        return self.input_time_dim * (self.input_channels + self.decoder_input_channels) \
+ self.n_constants + self.coupled_channels
    
    def _compute_coupled_channels(self, couplings):
        
        c_channels = 0
        for c in couplings:
            c_channels += len(c['params']['variables'])*len(c['params']['input_times'])
        return c_channels


    def _compute_output_channels(self) -> int:
        return (1 if self.is_diagnostic else self.input_time_dim) * self.output_channels

    def set_output(self, forecast_dates, forecast_integrations, data_module):

        # set fields necessary for gt forecasting
        self.forecast_dates = forecast_dates 
        self.forecast_integrations = forecast_integrations
        self.mean = data_module.test_dataset.target_scaling['mean'].transpose(0,2,1,3,4)
        self.std = data_module.test_dataset.target_scaling['std'].transpose(0,2,1,3,4)
        self.output_vars = data_module.test_dataset.output_variables 
        
    def forward(self, input):

        # check if we're on a new initialization 
        if self.integration_counter == self.forecast_integrations:
            self.initialization_counter+=1
            self.integration_counter=0

        # need a pdTimedelta object for time arithmatic 
        dt = pd.Timedelta(str(self.delta_t) + 'H')

        # output array buffer. hard coded for hpx32. This will do for now. issues in the future will 
        # fail loudly
        output_array = th.empty([1, 12, self.output_time_dim, self.output_channels, 32, 32])

        for i in range(0,self.output_time_dim):
        
            # print(f'appending timestep: {(self.integration_counter*self.output_time_dim+(i+1))*dt + self.forecast_dates[self.initialization_counter]}')
            output_array[:,:,i,:,:,:]=th.tensor(self.gt_dataset.targets.sel(channel_out=self.output_vars,time=(self.integration_counter*self.output_time_dim+(i+1))*dt + self.forecast_dates[self.initialization_counter]).values.transpose([1,0,2,3]))

        # increment integration counter as appropriate
        self.integration_counter+=1 

        # scale and return output
        return (output_array - self.mean) / self.std
