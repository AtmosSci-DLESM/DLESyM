from typing import Any, Dict, Optional, Sequence, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
import torch as th

from training.dlwp.model.modules.healpix import HEALPixLayer
from training.dlwp.model.modules.utils import Interpolate


#
# FOLDING/UNFOLDING BLOCKS
#

class FoldFaces(th.nn.Module):
    # perform face folding:
    # [B, F, C, H, W] -> [B*F, C, H, W]

    def __init__(self):
        super().__init__()

    def forward(self, tensor: th.Tensor) -> th.Tensor:

        N, F, C, H, W = tensor.shape
        tensor = th.reshape(tensor, shape=(N*F, C, H, W))
    
        return tensor


class UnfoldFaces(th.nn.Module):
    # perform face unfolding:
    # [B*F, C, H, W] -> [B, F, C, H, W]

    def __init__(self, num_faces=12):
        super().__init__()
        self.num_faces = num_faces

    def forward(self, tensor: th.Tensor) -> th.Tensor:
        
        NF, C, H, W = tensor.shape
        tensor = th.reshape(tensor, shape=(-1, self.num_faces, C, H, W))
    
        return tensor


#
# RECURRENT BLOCKS
#

class ConvGRUBlock(th.nn.Module):
    """
    Code modified from
    https://github.com/happyjin/ConvGRU-pytorch/blob/master/convGRU.py
    """
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            kernel_size: int = 1,
            downscale_factor: int = 4,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()

        self.channels = in_channels
        self.conv_gates = geometry_layer(
            layer="torch.nn.Conv2d",
            in_channels=in_channels + self.channels,
            out_channels=2*self.channels,  # for update_gate,reset_gate respectively
            kernel_size=kernel_size,
            padding="same",
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            )
        self.conv_can = geometry_layer(
            layer="torch.nn.Conv2d",
            in_channels=in_channels+self.channels,
            out_channels=self.channels, # for candidate neural memory
            kernel_size=kernel_size,
            padding="same",
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            )
        self.h = th.zeros(1, 1, 1, 1)

    def forward(self, inputs: Sequence) -> Sequence:
        if inputs.shape != self.h.shape:
            self.h = th.zeros_like(inputs)
        combined = th.cat([inputs, self.h], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = th.split(combined_conv, self.channels, dim=1)
        reset_gate = th.sigmoid(gamma)
        update_gate = th.sigmoid(beta)

        combined = th.cat([inputs, reset_gate*self.h], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = th.tanh(cc_cnm)

        h_next = (1 - update_gate) * self.h + update_gate * cnm
        self.h = h_next

        return inputs + h_next

    def reset(self):
        self.h = th.zeros_like(self.h)


#
# CONV BLOCKS
#

class BasicConvBlock(th.nn.Module):
    """
    Convolution block consisting of n subsequent convolutions and activations
    """
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            out_channels: int = 1,
            kernel_size: int = 3,
            dilation: int = 1,
            n_layers: int = 1,
            latent_channels: int = None,
            activation: th.nn.Module = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
        if latent_channels is None: latent_channels = max(in_channels, out_channels)
        convblock = []
        for n in range(n_layers):
            convblock.append(geometry_layer(
                layer='torch.nn.Conv2d',
                in_channels=in_channels if n == 0 else latent_channels,
                out_channels=out_channels if n == n_layers - 1 else latent_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                ))
            if activation is not None: convblock.append(activation)
        self.convblock = th.nn.Sequential(*convblock)

    def forward(self, x):
        return self.convblock(x)


class MobileNetConvBlock(th.nn.Module):
    """
    A convolution block as reported in Figure 4 (d) of https://arxiv.org/pdf/1801.04381.pdf

    Does not seem to improve performance over two simple convolutions
    """
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            out_channels: int = 1,
            kernel_size: int = 3,
            dilation: int = 1,
            activation: th.nn.Module = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
        # Instantiate 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = geometry_layer(
                layer='torch.nn.Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                )
        # Convolution block
        convblock = []
        # Map channels to output depth
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock.append(activation)
        # Depthwise convolution
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=out_channels,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock.append(activation)
        # Linear postprocessing
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        self.convblock = th.nn.Sequential(*convblock)

    def forward(self, x):
        return self.skip_module(x) + self.convblock(x)

class DoubleConvNeXtBlock(th.nn.Module):
    """
    Another modification of ConvNeXtBlock block this time putting two into a single block
    """
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            latent_channels: int = 1,
            out_channels: int = 1,
            kernel_size: int = 3,
            dilation: int = 1,
            upscale_factor: int = 4,
            n_layers: int = 1,
            activation: th.nn.Module = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
    
        if in_channels == int(latent_channels):
            self.skip_module1 = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module1 = geometry_layer(
                layer='torch.nn.Conv2d',
                in_channels=in_channels,
                out_channels=int(latent_channels),
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                )
        if out_channels == int(latent_channels):
            self.skip_module2 = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module2 = geometry_layer(
                layer='torch.nn.Conv2d',
                in_channels=int(latent_channels),
                out_channels=out_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                )

        # 1st ConvNeXt block, the output of this one remains internal 
        convblock1 = []
        # 3x3 convolution establishing latent channels channels
        convblock1.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=in_channels,
            out_channels=int(latent_channels),
            kernel_size=kernel_size,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock1.append(activation)
        # 1x1 convolution establishing increased channels
        convblock1.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=int(latent_channels),
            out_channels=int(latent_channels*upscale_factor),
            kernel_size=1,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock1.append(activation)
        # 1x1 convolution returning to latent channels
        convblock1.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=int(latent_channels*upscale_factor),
            out_channels=int(latent_channels),
            kernel_size=1,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock1.append(activation)
        self.convblock1 = th.nn.Sequential(*convblock1)

        # 2nd ConNeXt block, takes the output of the first convnext block 
        convblock2 = []
        # 3x3 convolution establishing latent channels channels
        convblock2.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=int(latent_channels),
            out_channels=int(latent_channels),
            kernel_size=kernel_size,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock2.append(activation)
        # 1x1 convolution establishing increased channels
        convblock2.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=int(latent_channels),
            out_channels=int(latent_channels*upscale_factor),
            kernel_size=1,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock2.append(activation)
        # 1x1 convolution reducing to output channels
        convblock2.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=int(latent_channels*upscale_factor),
            out_channels=out_channels,
            kernel_size=1,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock2.append(activation)
        self.convblock2 = th.nn.Sequential(*convblock2)
    


    def forward(self, x):
        # internal convnext result 
        x1 = self.skip_module1(x) + self.convblock1(x)
        # return second convnext result 
        return self.skip_module2(x1) + self.convblock2(x1)
        
class SymmetricConvNeXtBlock(th.nn.Module):
    """
    Another modification of ConvNeXtBlock block this time putting two into a single block
    """
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            latent_channels: int = 1,
            out_channels: int = 1,
            kernel_size: int = 3,
            dilation: int = 1,
            upscale_factor: int = 4,
            n_layers: int = 1,
            activation: th.nn.Module = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
    
        if in_channels == int(latent_channels):
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = geometry_layer(
                layer='torch.nn.Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,#int(latent_channels),
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                )

        # 1st ConvNeXt block, the output of this one remains internal 
        convblock = []
        # 3x3 convolution establishing latent channels channels
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=in_channels,
            out_channels=int(latent_channels),
            kernel_size=kernel_size,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock.append(activation)
        # 1x1 convolution establishing increased channels
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=int(latent_channels),
            out_channels=int(latent_channels*upscale_factor),
            kernel_size=1,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock.append(activation)
        # 1x1 convolution returning to latent channels
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=int(latent_channels*upscale_factor),
            out_channels=int(latent_channels),
            kernel_size=1,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock.append(activation)
        # 3x3 convolution from latent channels to latent channels
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=int(latent_channels),
            out_channels=out_channels,#int(latent_channels),
            kernel_size=kernel_size,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock.append(activation)
        self.convblock = th.nn.Sequential(*convblock)


    def forward(self, x):
        # residual connection with reshaped inpute and output of conv block 
        return self.skip_module(x) + self.convblock(x)

class ConvNeXtBlock(th.nn.Module):
    """
    A modification of the convolution block reported in Figure 4 of https://arxiv.org/pdf/2201.03545.pdf
    """
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            latent_channels: int = 1,
            out_channels: int = 1,
            kernel_size: int = 3,
            dilation: int = 1,
            upscale_factor: int = 4,
            n_layers: int = 1,
            activation: th.nn.Module = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()

        # Instantiate 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = geometry_layer(
                layer='torch.nn.Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                )
        # Convolution block
        convblock = []
        # 3x3 convolution increasing channels
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=in_channels,
            out_channels=int(latent_channels*upscale_factor),
            kernel_size=kernel_size,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock.append(activation)
        # 3x3 convolution maintaining increased channels
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=int(latent_channels*upscale_factor),
            out_channels=int(latent_channels*upscale_factor),
            kernel_size=kernel_size,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock.append(activation)
        # Linear postprocessing
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=int(latent_channels*upscale_factor),
            out_channels=out_channels,
            kernel_size=1,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        self.convblock = th.nn.Sequential(*convblock)

    def forward(self, x):
        return self.skip_module(x) + self.convblock(x)


#
# DOWNSAMPLING BLOCKS
#

class MaxPool(th.nn.Module):
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            pooling: int = 2,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
        self.maxpool = geometry_layer(
            layer="torch.nn.MaxPool2d",
            kernel_size=pooling,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            )
    def forward(self, x):
        return self.maxpool(x)


class AvgPool(th.nn.Module):
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            pooling: int = 2,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
        self.avgpool = geometry_layer(
            layer="torch.nn.AvgPool2d",
            kernel_size=pooling,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            )
    def forward(self, x):
        return self.avgpool(x)


class LearnedPool(th.nn.Module):
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 1,
            out_channels: int = 1,
            pooling: int = 2,
            activation: th.nn.Module = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
        # "Skip" connection
        self.skip_pool = MaxPool(
            geometry_layer=geometry_layer,
            pooling=pooling,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            )
        # Donwpooling convolution
        downpooler = []
        downpooler.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=pooling,
            stride=pooling,
            padding=0,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None:
            downpooler.append(activation)
        self.downpooler = th.nn.Sequential(*downpooler)

    def forward(self, x):
        return self.skip_pool(x) + self.downpooler(x)


#
# UPSAMPLING BLOCKS
#

class InterpolationUpsample(th.nn.Module):
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            out_channels: int = 1,
            kernel_size: int = 3,
            mode: str = "nearest",
            upsampling: int = 2,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
        self.upsampler = geometry_layer(
            layer=Interpolate,
            scale_factor=upsampling,
            mode=mode,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            )
    def forward(self, x):
        return self.upsampler(x)


class TransposedConvUpsample(th.nn.Module):
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            out_channels: int = 1,
            upsampling: int = 2,
            activation: th.nn.Module = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
        upsampler = []
        # Upsample transpose conv
        upsampler.append(geometry_layer(
            layer='torch.nn.ConvTranspose2d',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=upsampling,
            stride=upsampling,
            padding=0,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None:
            upsampler.append(activation)
        self.upsampler = th.nn.Sequential(*upsampler)

    def forward(self, x):
        return self.upsampler(x)
