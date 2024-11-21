import logging
from typing import Any, Dict, Optional, Sequence, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
import torch as th
import pandas as pd

from training.dlwp.model.modules.healpix import HEALPixPadding, HEALPixLayer
from training.dlwp.model.modules.encoder import UNetEncoder, UNet3Encoder
from training.dlwp.model.modules.decoder import UNetDecoder, UNet3Decoder
from training.dlwp.model.modules.blocks import FoldFaces, UnfoldFaces
from training.dlwp.model.modules.losses import LossOnStep
from training.dlwp.model.modules.utils import Interpolate

logger = logging.getLogger(__name__)

class HEALPixUNet(th.nn.Module):
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
            presteps: int = 0,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False,
            couplings: list = [],
    ):
        """
        Deep Learning Weather Prediction (DLWP) UNet on the HEALPix mesh.

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
        :param enable_nhwc: Model with [N, H, W, C] instead of [N, C, H, W] oder
        :param enable_healpixpad: Enable CUDA HEALPixPadding if installed
        :param coupings: sequence of dictionaries that describe coupling mechanisms
        """
        super().__init__()
        # add coupled fields to input channels for model initialization 
        self.coupled_channels = self._compute_coupled_channels(couplings) 
        self.couplings = couplings 
        self.train_couplers = None
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_constants = n_constants
        self.decoder_input_channels = decoder_input_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.channel_dim = 2  # Now 2 with [B, F, C*T, H, W]. Was 1 in old data format with [B, T*C, F, H, W]
        self.enable_nhwc = enable_nhwc
        self.enable_healpixpad = enable_healpixpad

        #self.output_dim = self.output_channels*self.input_time_dim

        # Number of passes through the model, or a diagnostic model with only one output time
        self.is_diagnostic = self.output_time_dim == 1 and self.input_time_dim > 1
        if not self.is_diagnostic and (self.output_time_dim % self.input_time_dim != 0):
            raise ValueError(f"'output_time_dim' must be a multiple of 'input_time_dim' (got "
                             f"{self.output_time_dim} and {self.input_time_dim})")

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

    def _reshape_inputs(self, inputs: Sequence, step: int = 0) -> th.Tensor:
        """
        Returns a single tensor to pass into the model encoder/decoder. Squashes the time/channel dimension and
        concatenates in constants and decoder inputs.
        :param inputs: list of expected input tensors (inputs, decoder_inputs, constants)
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
               raise NotImplementedError('support for coupled models with no constant fields \
is not available at this time.') 

            result = [
                inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1),
                inputs[1][:, :, slice(step*self.input_time_dim, (step+1)*self.input_time_dim), ...].flatten(
                    start_dim=self.channel_dim, end_dim=self.channel_dim+1
                    ),  # DI
                inputs[2].expand(*tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1])),  # constants
                inputs[3].permute(0,2,1,3,4) # coupled inputs
            ]
            res = th.cat(result, dim=self.channel_dim)
        
        else:
            if not (self.n_constants > 0 or self.decoder_input_channels > 0):
                return inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1)
            if self.n_constants == 0:
                result = [
                    inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1),  # inputs
                    inputs[1][:, :, slice(step*self.input_time_dim, (step+1)*self.input_time_dim), ...].flatten(self.channel_dim, self.channel_dim+1)  # DI
                ]
                res = th.cat(result, dim=self.channel_dim)

                # fold faces into batch dim
                res = self.fold(res)
                
                return res
            if self.decoder_input_channels == 0:
                result = [
                    inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1),  # inputs
                    inputs[1].expand(*tuple([inputs[0].shape[0]] + len(inputs[1].shape)*[-1]))  # constants
                    #th.tile(self.constants, (inputs[0].shape[0], 1, 1, 1, 1)) # constants
                ]
                res = th.cat(result, dim=self.channel_dim)

                # fold faces into batch dim
                res = self.fold(res)
                
                return res
            
            result = [
                inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1),  # inputs
                inputs[1][:, :, slice(step*self.input_time_dim, (step+1)*self.input_time_dim), ...].flatten(self.channel_dim, self.channel_dim+1),  # DI
                inputs[2].expand(*tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1]))  # constants
                #th.tile(self.constants, (inputs[0].shape[0], 1, 1, 1, 1)) # constants
            ]
            res = th.cat(result, dim=self.channel_dim)

        # fold faces into batch dim
        res = self.fold(res)
        
        return res

    def _reshape_outputs(self, outputs: th.Tensor) -> th.Tensor:

        # unfold:
        outputs = self.unfold(outputs)
        
        # extract shape and reshape
        shape = tuple(outputs.shape)
        res = th.reshape(outputs, shape=(shape[0], shape[1], 1 if self.is_diagnostic else self.input_time_dim, -1, *shape[3:]))
        
        return res

    def forward(self, inputs: Sequence, output_only_last=False) -> th.Tensor:

        outputs = []
        for step in range(self.integration_steps):
            if step == 0:
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(list(inputs[0:3])+[inputs[3][step]], step)
                else:
                    input_tensor = self._reshape_inputs(inputs, step)
            else:
                input_tensor = self._reshape_inputs([outputs[-1]] + list(inputs[1:3]) + [inputs[3][step]], step)

            encodings = self.encoder(input_tensor)
            decodings = self.decoder(encodings)

            reshaped = self._reshape_outputs(decodings)  # Absolute prediction
            #reshaped = self._reshape_outputs(input_tensor[:, :self.input_channels*self.input_time_dim] + decodings)  # Residual prediction
            outputs.append(reshaped)
            
        if output_only_last:
            res = outputs[-1]
        else:
            res = th.cat(outputs, dim=self.channel_dim)

        return res

class HEALPixRecUNet(th.nn.Module):
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
    ):
        """
        Deep Learning Weather Prediction (DLWP) recurrent UNet model on the HEALPix mesh.

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
        """

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

        # Number of passes through the model, or a diagnostic model with only one output time
        self.is_diagnostic = self.output_time_dim == 1 and self.input_time_dim > 1
        if not self.is_diagnostic and (self.output_time_dim % self.input_time_dim != 0):
            raise ValueError(f"'output_time_dim' must be a multiple of 'input_time_dim' (got "
                             f"{self.output_time_dim} and {self.input_time_dim})")

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

    @property
    def integration_steps(self):
        return max(self.output_time_dim // self.input_time_dim, 1)# + self.presteps
 
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

    def _reshape_inputs(self, inputs: Sequence, step: int = 0) -> th.Tensor:
        """
        Returns a single tensor to pass into the model encoder/decoder. Squashes the time/channel dimension and
        concatenates in constants and decoder inputs.
        :param inputs: list of expected input tensors (inputs, decoder_inputs, constants)
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
               raise NotImplementedError('support for coupled models with no constant fields \
is not available at this time.') 

            result = [
                inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1),
                inputs[1][:, :, slice(step*self.input_time_dim, (step+1)*self.input_time_dim), ...].flatten(
                    start_dim=self.channel_dim, end_dim=self.channel_dim+1
                    ),  # DI
                inputs[2].expand(*tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1])),  # constants
                inputs[3].permute(0,2,1,3,4) # coupled inputs
            ]
            res = th.cat(result, dim=self.channel_dim)

        else:
            if not (self.n_constants > 0 or self.decoder_input_channels > 0):
                return self.fold(prognostics)

            if self.n_constants == 0:
                result = [
                    inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1),
                    inputs[1][:, :, slice(step*self.input_time_dim, (step+1)*self.input_time_dim), ...].flatten(
                        start_dim=self.channel_dim, end_dim=self.channel_dim+1
                        )  # DI
                ]
                res = th.cat(result, dim=self.channel_dim)

                # fold faces into batch dim
                res = self.fold(res)
                
                return res

            if self.decoder_input_channels == 0:
                result = [
                    inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1),
                    inputs[1].expand(*tuple([inputs[0].shape[0]] + len(inputs[1].shape)*[-1]))  # constants
                ]
                res = th.cat(result, dim=self.channel_dim)

                # fold faces into batch dim
                res = self.fold(res)
                
                return res

            result = [
                inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1),
                inputs[1][:, :, slice(step*self.input_time_dim, (step+1)*self.input_time_dim), ...].flatten(
                    start_dim=self.channel_dim, end_dim=self.channel_dim+1
                    ),  # DI
                inputs[2].expand(*tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1]))  # constants
            ]
            res = th.cat(result, dim=self.channel_dim)
        
#        print()
#        print(f'shape of res: {res.shape}')
#        print(f'shape of coupled: {coupled.shape}')
#        print()

        # fold faces into batch dim
        res = self.fold(res)
        return res

    def _reshape_outputs(self, outputs: th.Tensor) -> th.Tensor:

        # unfold:
        outputs = self.unfold(outputs)
        
        # extract shape and reshape
        shape = tuple(outputs.shape)
        res = th.reshape(outputs, shape=(shape[0], shape[1], 1 if self.is_diagnostic else self.input_time_dim, -1, *shape[3:]))
        
        return res

    def _initialize_hidden(self, inputs: Sequence, outputs: Sequence, step: int) -> None:
        self.reset()
        for prestep in range(self.presteps):
            if step < self.presteps:
                s = step + prestep
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(
                        inputs=[inputs[0][:, :, s*self.input_time_dim:(s+1)*self.input_time_dim]] +\
                                list(inputs[1:3]) + [inputs[3][prestep]],
                        step=step+prestep
                        )
                else:
                    input_tensor = self._reshape_inputs(
                        inputs=[inputs[0][:, :, s*self.input_time_dim:(s+1)*self.input_time_dim]] + \
                         list(inputs[1:]),
                        step=step+prestep
                        )
            else:
                s = step - self.presteps + prestep
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(
                        inputs=[outputs[s-1]] + \
                        list(inputs[1:3]) + [inputs[3][step-(prestep-self.presteps)]],
                        step=s+1
                        )
                else:
                    # check this with matze:
                    # consider if step=2 (input_time_dim=2) with integration_steps=8 and reset_cycle=48H and presteps=1, prestep=0
                    # then we reset hidden here.
                    # s = 1 and we're indexing ouptut[0] 
                    # wouldn't we want to initialze hidden with output[1]? 
                    input_tensor = self._reshape_inputs(
                        inputs=[outputs[s-1]] + list(inputs[1:]),
                        step=s+1
                        )
            # Forward the data through the model to initialize hidden states
            self.decoder(self.encoder(input_tensor))

    def forward(self, inputs: Sequence, output_only_last=False) -> th.Tensor:
        self.reset()
        outputs = []
        for step in range(self.integration_steps):
            
           # print()
           # print(f'==== FIGURING HIDDEN STATE LOGIC ====')
           # print(f'inputs shape [len(inputs)={len(inputs)}]:')
           # print(f'    input[0]: {inputs[0].shape}')
           # print(f'    input[1]: {inputs[1].shape}')
           # print(f'    input[2]: {inputs[2].shape}')
           # print(f'    input[3]: {inputs[3].shape}')
            #print(f'hidden state arithmatic:')
            #print(f'    presteps: {self.presteps}')
            #s = 0+step
            #print(f'    s*self.input_time_dim:(s+1)*self.input_time_dim = {s*self.input_time_dim}:{(s+1)*self.input_time_dim}')
            #print(f'skipping over presteps in foward pass:')
            #s=self.presteps
            #print(f'    s: {s}')
            #print(f'    s*self.input_time_dim:(s+1)*self.input_time_dim = {s*self.input_time_dim}:{(s+1)*self.input_time_dim}')
            #print(f'    self.delta_t = {self.delta_t}')

            # (Re-)initialize recurrent hidden states
            if (step*(self.delta_t*self.input_time_dim)) % self.reset_cycle == 0:
                self._initialize_hidden(inputs=inputs, outputs=outputs, step=step)
            
            # Construct concatenated input: [prognostics|TISR|constants]
            if step == 0:
                s = self.presteps
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(
                        inputs=[inputs[0][:, :, s*self.input_time_dim:(s+1)*self.input_time_dim]] +\
                                list(inputs[1:3]) + [inputs[3][s]],
                        step=s
                        )
                else:
                    input_tensor = self._reshape_inputs(
                        inputs=[inputs[0][:, :, s*self.input_time_dim:(s+1)*self.input_time_dim]] + list(inputs[1:]),
                        step=s
                        )
            else:
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(
                        inputs=[outputs[-1]] + list(inputs[1:3]) +\
                                [inputs[3][self.presteps+step]],
                        step=step+self.presteps
                        )
                else:
                    input_tensor = self._reshape_inputs(
                        inputs=[outputs[-1]] + list(inputs[1:]),
                        step=step+self.presteps
                        )

            # Forward through model
            encodings = self.encoder(input_tensor)
            decodings = self.decoder(encodings)
            # Absolute prediction
            #reshaped = self._reshape_outputs(decodings)
            # Residual prediction
            reshaped = self._reshape_outputs(input_tensor[:, :self.input_channels*self.input_time_dim] + decodings)
            outputs.append(reshaped)

        if output_only_last:
            return outputs[-1]
        
        return th.cat(outputs, dim=self.channel_dim)

    def reset(self):
        self.encoder.reset()
        self.decoder.reset()
