import logging
from typing import Any, Dict, Optional, Sequence, Union

from hydra.utils import instantiate
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch as th
import pandas as pd

from training.dlwp.model.modules.healpix import HEALPixPadding, HEALPixLayer
from training.dlwp.model.modules.encoder import UNetEncoder, UNet3Encoder
from training.dlwp.model.modules.decoder import UNetDecoder, UNet3Decoder
from training.dlwp.model.modules.blocks import FoldFaces, UnfoldFaces
from training.dlwp.model.modules.losses import LossOnStep
from training.dlwp.model.modules.utils import Interpolate
from training.dlwp.model.models.unet import HEALPixUNet
from training.dlwp.model.models.unet3plus import HEALPixUNet3Plus

class CoupledUnet(th.nn.Module):
    def __init__(
        self,
        ocean_encoder : DictConfig,
        ocean_decoder : DictConfig,
        atmos_encoder : DictConfig,
        atmos_decoder : DictConfig,
        input_channels: int,
        output_channels: int,
        n_constants: int,
        decoder_input_channels: int,
        input_time_dim: int,
        output_time_dim: int,
        delta_time: str = "6H",
        reset_cycle: str = "24H",
        presteps: int = 0,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_constants = n_constants
        self.decoder_input_channels = decoder_input_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.channel_dim = 2  # Now 2 with [B, F, C*T, H, W]. Was 1 in old data format with [B, T*C, F, H, W]
        self.enable_nhwc = enable_nhwc
        self.enable_healpixpad = enable_healpixpad
        
        self.fold = FoldFaces()
        self.unfold = UnfoldFaces(num_faces=12)
        self.atmos_encoder = instantiate(config=atmos_encoder,
                                        input_channels=self._compute_input_channels(),
                                        enable_nhwc=self.enable_nhwc,
                                        enable_healpixpad=self.enable_healpixpad)
        self.atmos_encoder_depth = len(self.encoder.n_channels)
        self.atmos_decoder = instantiate(config=atmos_decoder,
                                         output_channels=self._compute_output_channels(),
                                         enable_nhwc = self.enable_nhwc,
                                         enable_healpixpad = self.enable_healpixpad)

    def _compute_input_channels(self) -> int:
        return self.input_time_dim * (self.input_channels + self.decoder_input_channels) + self.n_constants

    def _compute_output_channels(self) -> int:
        return (1 if self.is_diagnostic else self.input_time_dim) * self.output_channels


#@hydra.main(version_base=None,config_path='../../../configs', config_name="config_CoupledUnet")
#def instantiate_model(cfg: DictConfig) -> None:
#    print(OmegaConf.to_yaml(cfg))
#
#if __name__ == "__main__" :
#
#    instantiate_model()
#
#    # instantiate coupled_unet
#  
