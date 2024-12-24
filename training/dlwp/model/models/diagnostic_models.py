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
from training.dlwp.model.models.unet import HEALPixUNet

logger = logging.getLogger(__name__)

class ocean_gt_model(HEALPixUNet):
    def __init__(
            self,
            encoder: DictConfig,
            decoder: DictConfig,
            input_channels: int,
            output_channels: int,
            n_constants: int,
            decoder_input_channels: int,
            input_time_dim: int,
            presteps: int,
            output_time_dim: int,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False,
            couplings: list = [],
            gt_dataset: str = None,
            nside: int = 32,
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
        :param couplings: sequence of dictionaries that describe coupling mechanisms
        :param gt_datset: the dataset of ground truth values
        :param nside: the nside of the healpix grid
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
            presteps,
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
        
            # print(f'appending timestep: {(self.integration_counter*self.output_time_dim+(i+1))*dt + self.forecast_dates[self.initialization_counter]}')
            output_array[:,:,i,:,:,:]=th.tensor(self.gt_dataset.targets.sel(channel_out=self.output_vars,time=(self.integration_counter*self.output_time_dim+(i+1))*dt + self.forecast_dates[self.initialization_counter]).values.transpose([1,0,2,3]))

        # increment integration counter as appropriate
        self.integration_counter+=1 

        # scale and return output
        return (output_array - self.mean) / self.std