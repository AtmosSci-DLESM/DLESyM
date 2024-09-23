import logging
from typing import Any, Dict, Optional, Sequence, Union
import time

from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
import torch
import pandas as pd

from training.dlwp.model.modules.blocks import FoldFaces, UnfoldFaces
from training.dlwp.model.modules.blocks import FoldFaces, UnfoldFaces
from training.dlwp.model.modules.losses import LossOnStep
from training.dlwp.model.modules.utils import Interpolate
from training.dlwp.model.modules.healpix import HEALPixPadding, HEALPixLayer

class HEALPixUNet3Plus(torch.nn.Module):
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
    ):
        """
        Pytorch module implementation of the Deep Learning Weather Prediction (DLWP) U-net3+ model on the
        HEALPix grid.

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
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_constants = n_constants
        self.couplings = couplings
        self.decoder_input_channels = decoder_input_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim

        # Number of passes through the model, or a diagnostic model with only one output time
        self.is_diagnostic = self.output_time_dim == 1 and self.input_time_dim > 1
        if not self.is_diagnostic and (self.output_time_dim % self.input_time_dim != 0):
            raise ValueError(f"'output_time_dim' must be a multiple of 'input_time_dim' (got "
                             f"{self.output_time_dim} and {self.input_time_dim})")

        # Make the generator
        self.generator = IterativeUnet(
            encoder=encoder,
            decoder=decoder,
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            n_constants=self.n_constants,
            decoder_input_channels=self.decoder_input_channels,
            input_time_dim=self.input_time_dim,
            output_time_dim=self.output_time_dim,
            n_coupled_inputs=n_coupled_inputs,
            couplings=self.couplings,
            enable_healpixpad=enable_healpixpad,
            enable_nhwc=enable_nhwc,
        )

    def forward(self, inputs: Sequence, output_only_last=False) -> torch.Tensor:
        
        return self.generator(inputs, output_only_last)

class IterativeUnet(torch.nn.Module):
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
            n_coupled_inputs: int,
            couplings: list,
            enable_healpixpad,
            enable_nhwc,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.coupled_channels = self._compute_coupled_channels(couplings)
        self.couplings = couplings
        self.n_constants = n_constants
        self.channel_dim = 2  # Now 2 with [B, F, C*T, H, W]. Was 1 in old data format with [B, T*C, F, H, W]
        self.n_coupled_inputs = n_coupled_inputs
        self.decoder_input_channels = decoder_input_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        # Build the model layers
        self.fold = FoldFaces()
        self.unfold = UnfoldFaces(num_faces=12)

        # Number of passes through the model, or a diagnostic model with only one output time
        self.is_diagnostic = self.output_time_dim == 1 and self.input_time_dim > 1
        if not self.is_diagnostic and (self.output_time_dim % self.input_time_dim != 0):
            raise ValueError(f"'output_time_dim' must be a multiple of 'input_time_dim' (got "
                             f"{self.output_time_dim} and {self.input_time_dim})")

        # Build the model layers
        self.encoder = instantiate(encoder, input_channels=self._compute_input_channels(), enable_healpixpad=enable_healpixpad)
        self.encoder_depth = len(self.encoder.n_channels)
        self.decoder = instantiate(decoder, input_channels=self.encoder.n_channels,
                                   output_channels=self._compute_output_channels(),enable_healpixpad=enable_healpixpad)

    @property
    def integration_steps(self):
        return max(self.output_time_dim // self.input_time_dim, 1)

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

    def _reshape_inputs(self, inputs: Sequence, step: int = 0) -> torch.Tensor:
        """
        Returns a single tensor to pass into the model encoder/decoder. Squashes the time/channel dimension and
        concatenates in constants and decoder inputs.
        :param inputs: list of expected input tensors (inputs, decoder_inputs, [coupled_inputs], constants)
        :param step: step number in the sequence of integration_steps
        :return: reshaped Tensor in expected shape for model encoder
        """

        if  len(self.couplings) > 0 :
            if not (self.n_constants > 0 or self.decoder_input_channels > 0):
               raise NotImplementedError('support for coupled models with no constant fields \
or decoder inputs (TOA insolation) is not available at this time.') 
            if self.n_constants == 0:
               raise NotImplementedError('support for coupled models with no constant fields \
or decoder inputs (TOA insolation) is not available at this time.') 
            if self.decoder_input_channels == 0:
               raise NotImplementedError('support for coupled models with no decoder input fields \
is not available at this time.') 

            result = [
                inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1),
                inputs[1][:, :, slice(step*self.input_time_dim, (step+1)*self.input_time_dim), ...].flatten(
                    start_dim=self.channel_dim, end_dim=self.channel_dim+1
                    ),  # DI
                inputs[2].expand(*tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1])),  # constants
                inputs[3].permute(0,2,1,3,4) # coupled inputs
            ]
            res = torch.cat(result, dim=self.channel_dim)

        else:
            if not (self.n_constants > 0 or self.decoder_input_channels > 0):
                return inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1)

            if self.n_constants == 0:
                result = [
                    inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1),  # inputs
                    inputs[1][:, :, slice(step*self.input_time_dim, (step+1)*self.input_time_dim), ...].flatten(
                        self.channel_dim, self.channel_dim+1
                        )  # DI
                ]
                res = torch.cat(result, dim=self.channel_dim)

                # fold faces into batch dim
                res = self.fold(res)
                
                return res

            if self.decoder_input_channels == 0:
                result = [
                    inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1),  # inputs
                    inputs[1].expand(*tuple([inputs[0].shape[0]] + len(inputs[1].shape)*[-1]))  # constants
                ]
                res = torch.cat(result, dim=self.channel_dim)

                # fold faces into batch dim
                res = self.fold(res)
                
                return res
            
            result = [
                inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1),  # inputs
                inputs[1][:, :, slice(step*self.input_time_dim, (step+1)*self.input_time_dim), ...].flatten(
                    self.channel_dim, self.channel_dim+1
                    ),  # DI
                inputs[2].expand(*tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1]))  # constants
                #torch.tile(self.constants, (inputs[0].shape[0], 1, 1, 1, 1)) # constants
            ]
            res = torch.cat(result, dim=self.channel_dim)

        # fold faces into batch dim
        res = self.fold(res)
        return res

    def _reshape_outputs(self, outputs: torch.Tensor) -> torch.Tensor:

        # unfold:
        outputs = self.unfold(outputs)
        
        # extract shape and reshape
        shape = tuple(outputs.shape)
        res = torch.reshape(outputs, shape=(shape[0], shape[1], 1 if self.is_diagnostic else self.input_time_dim, -1, *shape[3:]))
        
        return res

    def forward(self, inputs: Sequence, output_only_last=False) -> torch.Tensor:
        outputs = []
        for step in range(self.integration_steps):
            if step == 0:
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(inputs[0:3] + [inputs[3][step]], step)
                else:
                    input_tensor = self._reshape_inputs(inputs, step)
            else:
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs([outputs[-1]] + list(inputs[1:3]) + [inputs[3][step]], step)
                else:
                    input_tensor = self._reshape_inputs([outputs[-1]] + list(inputs[1:]), step)
            hidden_states = self.encoder(input_tensor)
            outputs.append(self._reshape_outputs(self.decoder(hidden_states)))
        if output_only_last:
            return outputs[-1]
        return torch.cat(outputs, dim=self.channel_dim)

class Unet3plusEncoder(torch.nn.Module):
    def __init__(
            self,
            input_channels: int = 3,
            n_channels: Sequence = (16, 32, 64),
            convolutions_per_depth: int = 2,
            kernel_size: int = 3,
            dilations: list = None,
            pooling_type: str = 'torch.nn.MaxPool2d',
            pooling: int = 2,
            activation: Optional[DictConfig] = None,
            enable_healpixpad=True,
            enable_nhwc=False,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.pooling_type =pooling_type
        self.pooling = pooling
        self.activation = activation

        if dilations is None:
            # Defaults to [1, 1, 1...] in accordance with the number of unet levels
            dilations = [1 for _ in range(len(n_channels))]

        assert input_channels >= 1
        assert convolutions_per_depth >= 1
        assert kernel_size >= 1 and kernel_size % 2 == 1
        assert pooling >= 1

        old_channels = input_channels
        self.encoder = []
        for n, curr_channel in enumerate(self.n_channels):
            modules = list()
            if n > 0 and self.pooling is not None:
                modules.append(HEALPixLayer(
                    layer=self.pooling_type,
                    kernel_size=self.pooling,
                    enable_healpixpad=enable_healpixpad ,
                    enable_nhwc=enable_nhwc
                    ))
            #convolution_steps = convolutions_per_depth if n < len(self.n_channels) - 1 else convolutions_per_depth//2  # <-- original (one conv)
            convolution_steps = convolutions_per_depth  # <-- two convs
            #convolution_steps = convolutions_per_depth if n < len(self.n_channels) - 1 else convolutions_per_depth*2  # <-- four convs (https://arxiv.org/pdf/2205.10972.pdf)
            for _ in range(convolution_steps):
                modules.append(HEALPixLayer(
                    layer='torch.nn.Conv2d',
                    in_channels=old_channels,
                    out_channels=curr_channel,
                    kernel_size=self.kernel_size,
                    dilation=dilations[n],
                    enable_healpixpad=enable_healpixpad,
                    enable_nhwc=enable_nhwc
                    ))
                if self.activation is not None:
                    modules.append(instantiate(self.activation))
                old_channels = curr_channel
            self.encoder.append(torch.nn.Sequential(*modules))

        self.encoder = torch.nn.ModuleList(self.encoder)

    def forward(self, inputs: Sequence) -> Sequence:
        outputs = []
        for layer in self.encoder:
            outputs.append(layer(inputs))
            inputs = outputs[-1]
        return outputs


class Unet3plusDecoder(torch.nn.Module):
    def __init__(
            self,
            input_channels: Sequence = (16, 32, 64),
            n_channels: Sequence = (64, 32, 16),
            output_channels: int = 1,
            convolutions_per_depth: int = 2,
            kernel_size: int = 3,
            dilations: list = None,
            pooling_type: str = 'torch.nn.MaxPool2d',
            pooling: int = 2,
            upsampling_type: str = 'interpolate',
            upsampling: int = 2,
            activation: Optional[DictConfig] = None,
            enable_healpixpad=True,
            enable_nhwc=False,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.upsampling_type = upsampling_type
        self.upsampling = upsampling
        self.activation = activation

        if dilations is None:
            # Defaults to [1, 1, 1...] according to the number of unet levels
            dilations = [1 for _ in range(len(input_channels))]
        
        assert output_channels >= 1
        assert convolutions_per_depth >= 1
        assert len(input_channels) == len(n_channels)
        assert kernel_size >= 1 and kernel_size % 2 == 1
        assert upsampling_type in ['interpolate', 'transpose']

        levels = len(n_channels)  # Depth or number of hierarchies in the unet
        pow2 = [2**x for x in range(len(input_channels))][::-1]  # i.e. [..., 8, 4, 2, 1]

        input_channels = list(input_channels[::-1])
        self.decoder = []
        for n, curr_channel in enumerate(n_channels):

            # No additional convolutions at the bottom of the u-net decoder (they are already part of the encoder)
            if n == 0:
                continue
            
            skip_modules = list()
            samp_modules = list()
            pool_modules = list()
            conv_modules = list()            

            # Skippers
            skip_dilation = 4  # According to Fig. 2 in https://arxiv.org/pdf/2205.10972.pdf
            skip_modules.append(HEALPixLayer(
                layer='torch.nn.Conv2d',
                in_channels=curr_channel,
                out_channels=curr_channel,
                kernel_size=self.kernel_size,
                dilation=skip_dilation,
                enable_healpixpad=enable_healpixpad,
                enable_nhwc=enable_nhwc
                ))
            if self.activation is not None:
                skip_modules.append(instantiate(self.activation))

            # Upsamplers
            for ch_below_idx, channels_below in enumerate(input_channels[:n]):
                samp_modules.append(UpSampler(
                    input_channels=channels_below,
                    output_channels=curr_channel,
                    upsampling_type=upsampling_type,
                    upsampling=upsampling*pow2[-n:][ch_below_idx],
                    kernel_size=kernel_size,
                    dilation=dilations[n],
                    activation=activation,
                    enable_healpixpad=enable_healpixpad,
                    enable_nhwc=enable_nhwc
                    ))

            # Downpoolers
            for ch_above_idx, channels_above in enumerate(input_channels[::-1][:len(input_channels)-1-n]):
                pool_modules.append(DownPooler(
                    pooling_type=pooling_type,
                    pooling=pooling*pow2[n+1:][ch_above_idx],
                    enable_healpixpad=enable_healpixpad,
                    enable_nhwc=enable_nhwc
                    ))

            # Convolvers
            convolution_steps = convolutions_per_depth // 2 if n == 0 else convolutions_per_depth
            # Regular convolutions. The last convolution depth is dealt with in the next segment, because we either
            # add another conv + interpolation or a transpose conv.
            for m in range(convolution_steps):
                if n == 0 and m == 0:
                    in_ch = input_channels[n]
                elif m == 0 and n > 0:
                    in_ch = curr_channel
                    # Add channels for upsamplings coming from below and downpoolings coming from above
                    for channels_below in input_channels[:n]:
                        in_ch += curr_channel  # Upsamplers convolve to curr_channel
                    for channels_above in input_channels[::-1][:len(input_channels)-1-n]:
                        in_ch += channels_above  # Downpoolers keep originial number of channels
                else:
                    in_ch = curr_channel
                conv_modules.append(HEALPixLayer(
                    layer='torch.nn.Conv2d',
                    in_channels=in_ch,
                    out_channels=curr_channel,
                    kernel_size=self.kernel_size,
                    dilation=dilations[n],
                    enable_healpixpad=enable_healpixpad,
                    enable_nhwc=enable_nhwc

                    ))
                if self.activation is not None:
                    conv_modules.append(instantiate(self.activation))

            self.decoder.append(torch.nn.ModuleDict(
                {"skips": torch.nn.Sequential(*skip_modules),
                 "samps": torch.nn.ModuleList(samp_modules),
                 "pools": torch.nn.ModuleList(pool_modules),
                 "convs": torch.nn.Sequential(*conv_modules)}
                ))

        self.decoder = torch.nn.ModuleList(self.decoder)

        # Linear Output layer
        conv_modules = list()
        conv_modules.append(HEALPixLayer(
            layer='torch.nn.Conv2d',
            in_channels=curr_channel*2,  # Residual connection
            out_channels=output_channels,
            kernel_size=3,
            dilation=dilations[-1],
            enable_healpixpad=enable_healpixpad,
            enable_nhwc=enable_nhwc
            ))
        self.output_layer = torch.nn.Sequential(*conv_modules)

    def forward(self, inputs: Sequence) -> torch.Tensor:
        outputs = [inputs[-1]]
        for n, layer in enumerate(self.decoder):
            # Skips
            skip = layer["skips"](inputs[-2 - n])

            # Upsamplings
            ups = list()
            for u_idx, upsampler in enumerate(layer["samps"]):
                ups.append(upsampler(outputs[u_idx]))
            ups = torch.cat(ups, dim=1)

            # Downpoolings
            if len(layer["pools"]) > 0:
                downs = list()
                for d_idx, downpooler in enumerate(layer["pools"]):
                    downs.append(downpooler(inputs[d_idx]))
                downs = torch.cat(downs, dim=1)
                
                # Stick upsamplings, (downpoolings), and skip together
                x = torch.cat([ups, downs, skip], dim=1)
            else:
                x = torch.cat([ups, skip], dim=1)

            # Convolutions
            x = layer["convs"](x)
            outputs.append(x)

        # Linear output residual skip connection
        x = self.output_layer(torch.cat([x, inputs[0]], dim=1))

        return x


class UpSampler(torch.nn.Module):
    def __init__(
            self,
            input_channels: int = 3,
            output_channels: int = 1,
            upsampling_type: str = 'interpolate',
            upsampling: int = 2,
            kernel_size: int = 3,
            dilation: int = 1,  # Not considered!
            activation: Optional[DictConfig] = None,
            enable_healpixpad=True,
            enable_nhwc=False,
    ):
        super().__init__()
        upsampler = []
        if upsampling_type == 'interpolate':
            # Regular conv + interpolation
            upsampler.append(HEALPixLayer(
                layer='torch.nn.Conv2d',
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                enable_healpixpad=enable_healpixpad,
                enable_nhwc=enable_nhwc
                ))
            if activation is not None:
                upsampler.append(instantiate(activation))
            upsampler.append(HEALPixLayer(
                layer=Interpolate,
                scale_factor=upsampling,
                mode='nearest',
                enable_healpixpad=enable_healpixpad,
                enable_nhwc=enable_nhwc
                ))
        else:
            # Upsample transpose conv
            upsampler.append(HEALPixLayer(
                layer='torch.nn.ConvTranspose2d',
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=upsampling,
                stride=upsampling,
                padding=0,
                enable_healpixpad=enable_healpixpad,
                enable_nhwc=enable_nhwc
                ))
            if activation is not None:
                upsampler.append(instantiate(activation))
        self.upsampler = torch.nn.Sequential(*upsampler)

    def forward(self, x):
        return self.upsampler(x)


class DownPooler(torch.nn.Module):
    def __init__(
            self,
            pooling_type: str = 'torch.nn.MaxPool2d',
            pooling: int = 2,
            enable_healpixpad=True,
            enable_nhwc=False,
    ):
        super().__init__()
        self.downpooler = HEALPixLayer(
            layer=pooling_type,
            kernel_size=pooling,
            enable_healpixpad=enable_healpixpad,
            enable_nhwc=enable_nhwc
            )

    def forward(self, x):
        return self.downpooler(x)
