from typing import Any, Dict, Optional, Sequence, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
import torch as th

from training.dlwp.model.modules.cube_sphere import CubeSpherePadding, CubeSphereLayer


class CubeSphereUNetEncoder(th.nn.Module):
    def __init__(
            self,
            input_channels: int = 3,
            n_channels: Sequence = (16, 32, 64),
            convolutions_per_depth: int = 2,
            kernel_size: int = 3,
            dilations: list = None,
            pooling_type: str = 'torch.nn.MaxPool2d',
            pooling: int = 2,
            activation: th.nn.Module = None,
            add_polar_layer: bool = True,
            flip_north_pole: bool = True
    ):
        super().__init__()
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.pooling_type =pooling_type
        self.pooling = pooling
        self.activation = activation
        self.add_polar_layer = add_polar_layer
        self.flip_north_pole = flip_north_pole

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
                pool_config = DictConfig(dict(
                    _target_=self.pooling_type,
                    kernel_size=self.pooling
                ))
                modules.append(CubeSphereLayer(pool_config, add_polar_layer=False, flip_north_pole=False))
            # Only do one convolution at the bottom of the U-net, since the other is in the decoder
            convolution_steps = convolutions_per_depth if n < len(self.n_channels) - 1 else convolutions_per_depth // 2
            for _ in range(convolution_steps):
                conv_config = DictConfig(dict(
                    _target_='torch.nn.Conv2d',
                    in_channels=old_channels,
                    out_channels=curr_channel,
                    kernel_size=self.kernel_size,
                    dilation=dilations[n],
                    padding=0
                ))
                modules.append(CubeSpherePadding(((self.kernel_size - 1) // 2)*dilations[n]))
                modules.append(CubeSphereLayer(conv_config, add_polar_layer=self.add_polar_layer,
                                               flip_north_pole=self.flip_north_pole))
                if self.activation is not None:
                    modules.append(self.activation)
                old_channels = curr_channel
            self.encoder.append(th.nn.Sequential(*modules))

        self.encoder = th.nn.ModuleList(self.encoder)

    def forward(self, inputs: Sequence) -> Sequence:
        outputs = []
        for layer in self.encoder:
            outputs.append(layer(inputs))
            inputs = outputs[-1]
        return outputs


class UNetEncoder(th.nn.Module):
    """
    Generic UNet3Encoder that can be applied to arbitrary meshes.
    """
    def __init__(
            self,
            conv_block: DictConfig,
            down_sampling_block: DictConfig,
            recurrent_block: DictConfig = None,
            input_channels: int = 3,
            n_channels: Sequence = (16, 32, 64),
            n_layers: Sequence = (2, 2, 1),
            dilations: list = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
    ):
        super().__init__()
        self.n_channels = n_channels

        import copy
        cblock = copy.deepcopy(conv_block)

        if dilations is None:
            # Defaults to [1, 1, 1...] in accordance with the number of unet levels
            dilations = [1 for _ in range(len(n_channels))]

        # Build encoder
        old_channels = input_channels
        self.encoder = []
        for n, curr_channel in enumerate(n_channels):
            modules = list()
            if n > 0:
                modules.append(instantiate(
                    config=down_sampling_block,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad
                    ))
            else:
                down_pool_module = None

            modules.append(instantiate(
                config=conv_block,
                in_channels=old_channels,
                latent_channels=curr_channel,
                out_channels=curr_channel,
                dilation=dilations[n],
                n_layers=n_layers[n],
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                ))
            old_channels = curr_channel

            self.encoder.append(th.nn.Sequential(*modules))

        self.encoder = th.nn.ModuleList(self.encoder)

    def forward(self, inputs: Sequence) -> Sequence:
        outputs = []
        for layer in self.encoder:
            outputs.append(layer(inputs))
            inputs = outputs[-1]
        return outputs

    def reset(self):
        pass


class UNet3Encoder(th.nn.Module):
    """
    Generic UNet3Encoder that can be applied to arbitrary meshes.
    """
    def __init__(
            self,
            conv_block: DictConfig,
            down_sampling_block: DictConfig,
            recurrent_block: DictConfig = None,
            input_channels: int = 3,
            n_channels: Sequence = (16, 32, 64),
            n_layers: Sequence = (2, 2, 1),
            dilations: list = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
    ):
        super().__init__()
        self.n_channels = n_channels
        
        if dilations is None:
            # Defaults to [1, 1, 1...] in accordance with the number of unet levels
            dilations = [1 for _ in range(len(n_channels))]

        # Build encoder
        old_channels = input_channels
        self.encoder = []
        for n, curr_channel in enumerate(n_channels):
            modules = list()
            if n > 0:
                modules.append(instantiate(
                    config=down_sampling_block,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad
                    ))
            modules.append(instantiate(
                config=conv_block,
                in_channels=old_channels,
                latent_channels=curr_channel,
                out_channels=curr_channel,
                dilation=dilations[n],
                n_layers=n_layers[n],
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                ))
            old_channels = curr_channel

            #if recurrent_block is not None and n == len(n_channels) - 1:
            #    self.recurrent_block = instantiate(
            #        config=recurrent_block,
            #        in_channels=curr_channel,
            #        enable_healpixpad=enable_healpixpad
            #        )
            #    modules.append(self.recurrent_block)
            # IF USING THIS, MAKE SURE IT IS RESET PROPERLY

            self.encoder.append(th.nn.Sequential(*modules))

        self.encoder = th.nn.ModuleList(self.encoder)

    def forward(self, inputs: Sequence) -> Sequence:
        outputs = []
        for layer in self.encoder:
            outputs.append(layer(inputs))
            inputs = outputs[-1]
        return outputs

    def reset(self):
        pass
        #self.recurrent_block.reset()
