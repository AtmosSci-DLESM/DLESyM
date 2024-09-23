from typing import Any, Dict, Optional, Sequence, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
import torch as th

from training.dlwp.model.modules.cube_sphere import CubeSpherePadding, CubeSphereLayer


class CubeSphereUNetDecoder(th.nn.Module):
    def __init__(
            self,
            input_channels: Sequence = (16, 32, 64),
            n_channels: Sequence = (64, 32, 16),
            output_channels: int = 1,
            convolutions_per_depth: int = 2,
            kernel_size: int = 3,
            upsampling_type: str = 'interpolate',
            upsampling: int = 2,
            activation: th.nn.Module = None,
            add_polar_layer: bool = True,
            flip_north_pole: bool = True
    ):
        super().__init__()
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.upsampling_type = upsampling_type
        self.upsampling = upsampling
        self.activation = activation
        self.add_polar_layer = add_polar_layer
        self.flip_north_pole = flip_north_pole

        assert output_channels >= 1
        assert convolutions_per_depth >= 1
        assert len(input_channels) == len(n_channels)
        assert kernel_size >= 1 and kernel_size % 2 == 1
        assert upsampling_type in ['interpolate', 'transpose']

        input_channels = list(input_channels[::-1])
        self.decoder = []
        for n, curr_channel in enumerate(n_channels):
            modules = list()
            # Only do one convolution at the bottom of the u-net
            convolution_steps = convolutions_per_depth // 2 if n == 0 else convolutions_per_depth
            # Regular convolutions. The last convolution depth is dealt with in the next segment, because we either
            # add another conv + interpolation or a transpose conv.
            for m in range(convolution_steps - 1):
                if n == 0 and m == 0:
                    in_ch = input_channels[n]
                elif m == 0 and n > 0:
                    in_ch = input_channels[n] + curr_channel
                else:
                    in_ch = curr_channel
                conv_config = DictConfig(dict(
                    _target_='torch.nn.Conv2d',
                    in_channels=in_ch,
                    out_channels=curr_channel,
                    kernel_size=self.kernel_size,
                    padding=0
                ))
                modules.append(CubeSpherePadding((self.kernel_size - 1) // 2))
                modules.append(CubeSphereLayer(conv_config, add_polar_layer=self.add_polar_layer,
                                               flip_north_pole=self.flip_north_pole))
                modules.append(self.activation)
            # Add the conv + interpolate or transpose conv layer. If it is the last set, add the output conv.
            if n < len(n_channels) - 1:
                if self.upsampling_type == 'interpolate':
                    # Regular conv + interpolation
                    conv_config = DictConfig(dict(
                        _target_='torch.nn.Conv2d',
                        in_channels=curr_channel,
                        out_channels=n_channels[n + 1],
                        kernel_size=self.kernel_size,
                        padding=0
                    ))
                    modules.append(CubeSpherePadding((self.kernel_size - 1) // 2))
                    modules.append(CubeSphereLayer(conv_config, add_polar_layer=self.add_polar_layer,
                                                   flip_north_pole=self.flip_north_pole))
                    modules.append(self.activation)
                    upsample_config = DictConfig(dict(
                        _target_='dlwp.model.modules.utils.Interpolate',
                        scale_factor=self.upsampling,
                        mode='nearest'
                    ))
                    modules.append(CubeSphereLayer(upsample_config, add_polar_layer=False, flip_north_pole=False))
                else:
                    # Upsample transpose conv
                    upsample_config = DictConfig(dict(
                        _target_='torch.nn.ConvTranspose2d',
                        in_channels=curr_channel,
                        out_channels=n_channels[n + 1],
                        kernel_size=self.upsampling,
                        stride=self.upsampling,
                        padding=0
                    ))
                    modules.append(CubeSphereLayer(upsample_config, add_polar_layer=self.add_polar_layer,
                                                   flip_north_pole=self.flip_north_pole))
                    modules.append(self.activation)
            else:
                conv_config = DictConfig(dict(
                        _target_='torch.nn.Conv2d',
                        in_channels=curr_channel,
                        out_channels=curr_channel,
                        kernel_size=self.kernel_size,
                        padding=0
                    ))
                modules.append(CubeSpherePadding((self.kernel_size - 1) // 2))
                modules.append(CubeSphereLayer(conv_config, add_polar_layer=self.add_polar_layer,
                                               flip_north_pole=self.flip_north_pole))
                modules.append(self.activation)
                # Add the output layer
                conv_config = DictConfig(dict(
                    _target_='torch.nn.Conv2d',
                    in_channels=curr_channel,
                    out_channels=output_channels,
                    kernel_size=1,
                    padding=0
                ))
                modules.append(CubeSphereLayer(conv_config, add_polar_layer=self.add_polar_layer,
                                               flip_north_pole=self.flip_north_pole))
            self.decoder.append(th.nn.Sequential(*modules))

        self.decoder = th.nn.ModuleList(self.decoder)

    def forward(self, inputs: Sequence) -> th.Tensor:
        x = inputs[-1]
        for n, layer in enumerate(self.decoder):
            x = layer(x)
            if n < len(self.decoder) - 1:
                x = th.cat([x, inputs[-2 - n]], dim=1)
        return x


class UNetDecoder(th.nn.Module):
    """
    Generic UNetDecoder that can be applied to arbitrary meshes.
    """
    def __init__(
            self,
            conv_block: DictConfig,
            up_sampling_block: DictConfig,
            output_layer: DictConfig,
            recurrent_block: DictConfig = None,
            n_channels: Sequence = (64, 32, 16),
            n_layers: Sequence = (1, 2, 2),
            output_channels: int = 1,
            dilations: list = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
    ):
        super().__init__()
        self.channel_dim = 1  # 1 in previous layout

        if enable_nhwc and activation is not None:
            activation = activation.to(memory_format=th.channels_last)

        if dilations is None:
            # Defaults to [1, 1, 1...] in accordance with the number of unet levels
            dilations = [1 for _ in range(len(n_channels))]

        self.decoder = []
        for n, curr_channel in enumerate(n_channels):

            # Second half of the synoptic layer does not need an upsampling module
            if n == 0:
                up_sample_module = None
            else:
                up_sample_module = instantiate(
                    config=up_sampling_block,
                    in_channels=curr_channel,
                    out_channels=curr_channel,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad
                    )

            next_channel = n_channels[n+1] if n < len(n_channels) - 1 else n_channels[-1]

            conv_module = instantiate(
                config=conv_block,
                in_channels=curr_channel*2 if n > 0 else curr_channel,  # Considering skip connection
                latent_channels=curr_channel,
                out_channels=next_channel,
                dilation=dilations[n],
                n_layers=n_layers[n],
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                )

            # Recurrent module
            if recurrent_block is not None:
                rec_module = instantiate(
                    config=recurrent_block,
                    in_channels=next_channel,
                    enable_healpixpad=enable_healpixpad
                    )
            else:
                rec_module = None

            self.decoder.append(th.nn.ModuleDict(
                {"upsamp": up_sample_module,
                 "conv": conv_module,
                 "recurrent": rec_module}
                ))

        self.decoder = th.nn.ModuleList(self.decoder)

        # (Linear) Output layer
        self.output_layer = instantiate(
            config=output_layer,
            in_channels=curr_channel,
            out_channels=output_channels,
            dilation=dilations[-1],
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            )

    def forward(self, inputs: Sequence) -> th.Tensor:
        x = inputs[-1]
        for n, layer in enumerate(self.decoder):
            if layer["upsamp"] is not None:
                up = layer["upsamp"](x)
                x = th.cat([up, inputs[-1 - n]], dim=self.channel_dim)
            x = layer["conv"](x)
            if layer["recurrent"] is not None: x = layer["recurrent"](x)
        return self.output_layer(x)

    def reset(self):
        for layer in self.decoder:
            layer["recurrent"].reset()


class UNet3Decoder(th.nn.Module):
    """
    Generic UNet3Decoder that can be applied to arbitrary meshes.
    """
    def __init__(
            self,
            conv_block: DictConfig,
            down_sampling_block: DictConfig,
            up_sampling_block: DictConfig,
            skip_block: DictConfig,
            output_layer: DictConfig,
            recurrent_block: DictConfig = None,
            input_channels: Sequence = (16, 32, 64),
            n_channels: Sequence = (64, 32, 16),
            n_layers: Sequence = (1, 2, 2),
            output_channels: int = 1,
            dilations: list = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
    ):
        super().__init__()
        self.channel_dim = 1
        input_channels = n_channels[::-1]

        if dilations is None:
            # Defaults to [1, 1, 1...] according to the number of unet levels
            dilations = [1 for _ in range(len(input_channels))]

        levels = len(n_channels)  # Depth or number of hierarchies in the unet
        pow2 = [2**x for x in range(len(input_channels))][::-1]  # i.e., [..., 8, 4, 2, 1]

        input_channels = list(input_channels[::-1])
        self.decoder = []
        for n, curr_channel in enumerate(n_channels):

            # No upsamplers/downpoolers in the synoptic layer
            if n == 0:
                skip_module = None
                up_samp_modules = None
                down_pool_modules = None
            else:
                # Skipper
                skip_module = instantiate(
                    config=skip_block,
                    in_channels=curr_channel,
                    out_channels=curr_channel,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad
                    )

                # Upsamplers
                up_samp_modules = list()
                for ch_below_idx, channels_below in enumerate(input_channels[:n]):
                    up_samp_modules.append(instantiate(
                        config=up_sampling_block,
                        in_channels=input_channels[ch_below_idx+1],
                        out_channels=curr_channel,
                        upsampling=up_sampling_block.upsampling*pow2[-n:][ch_below_idx],
                        enable_nhwc=enable_nhwc,
                        enable_healpixpad=enable_healpixpad
                        ))
                up_samp_modules = th.nn.ModuleList(up_samp_modules)

                # Downpoolers
                down_pool_modules = list()
                for ch_above_idx, channels_above in enumerate(input_channels[::-1][:len(input_channels)-1-n]):
                    down_pool_modules.append(instantiate(
                        config=down_sampling_block,
                        pooling=down_sampling_block.pooling*pow2[n+1:][ch_above_idx],
                        enable_nhwc=enable_nhwc,
                        enable_healpixpad=enable_healpixpad
                        ))
                down_pool_modules = th.nn.ModuleList(down_pool_modules)

            # Convolvers
            in_ch = curr_channel
            next_channel = n_channels[n+1] if n < len(n_channels) - 1 else n_channels[-1]
            # Add channels for upsamplings coming from below and downpoolings coming from above
            if n > 0:
                for channels_below in input_channels[:n]:
                    in_ch += curr_channel  # Upsamplers convolve to curr_channel
                for channels_above in input_channels[::-1][:len(input_channels)-1-n]:
                    in_ch += channels_above  # Downpoolers keep originial number of channels
            conv_module = instantiate(
                config=conv_block,
                in_channels=in_ch,
                latent_channels=curr_channel,
                out_channels=next_channel,
                dilation=dilations[n],
                n_layers=n_layers[n],
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                )

            # Recurrent module
            if recurrent_block is not None:
                rec_module = instantiate(
                    config=recurrent_block,
                    in_channels=next_channel,
                    enable_healpixpad=enable_healpixpad
                    )
            else:
                rec_module = None

            self.decoder.append(th.nn.ModuleDict(
                {"skip": skip_module,
                 "upsamps": up_samp_modules,
                 "downpools": down_pool_modules,
                 "convs": conv_module,
                 "recurrent": rec_module}
                ))

        self.decoder = th.nn.ModuleList(self.decoder)

        # Linear Output layer
        conv_modules = list()
        conv_modules.append(instantiate(
            config=output_layer,
            in_channels=curr_channel,
            out_channels=output_channels,
            dilation=dilations[-1],
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        self.output_layer = th.nn.Sequential(*conv_modules)

    def forward(self, inputs: Sequence) -> th.Tensor:
        outputs = []
        for n, layer in enumerate(self.decoder):

            # Skip connections, upsamplings, and downpoolings do not exist in the synoptic layer
            if n > 0:
                # Skip
                skip = layer["skip"](inputs[-1 - n])

                # Upsamplings
                ups = list()
                for u_idx, upsampler in enumerate(layer["upsamps"]):
                    ups.append(upsampler(outputs[u_idx]))
                ups = th.cat(ups, dim=self.channel_dim)

                # Downpoolings
                if len(layer["downpools"]) > 0:
                    downs = list()
                    for d_idx, downpooler in enumerate(layer["downpools"]):
                        downs.append(downpooler(inputs[d_idx]))
                    downs = th.cat(downs, dim=self.channel_dim)
                    
                    # Concatenate upsamplings, (downpoolings), and skip
                    x = th.cat([ups, downs, skip], dim=self.channel_dim)
                else:
                    x = th.cat([ups, skip], dim=self.channel_dim)
            else:
                x = inputs[-1]

            # Convolutions
            x = layer["convs"](x)

            # Recurrent module
            if layer["recurrent"] is not None: x = layer["recurrent"](x)

            outputs.append(x)

        return self.output_layer(x)

    def reset(self):
        for layer in self.decoder:
            layer["recurrent"].reset()
