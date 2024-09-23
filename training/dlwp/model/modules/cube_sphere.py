from typing import DefaultDict, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from torch.nn.utils.parametrizations import spectral_norm


class CubeSphereLayer(torch.nn.Module):
    """
    Pytorch module for applying any base torch Module on a cube-sphere tensor. Expects all input/output tensors to
    have a shape [..., 6, H, W], where 6 is the dimension of the cube face. Operations can be done either on all
    cube faces or separately for the first 4 equatorial faces and the last 2 polar faces. The final pole may also
    optionally be flipped to match the orientation of rotation around the Earth's axis.
    """
    def __init__(
            self,
            base_layer: Union[DictConfig, DefaultDict],
            add_polar_layer: bool = True,
            flip_north_pole: bool = True,
            use_spectral_norm: bool = False
    ):
        super().__init__()
        self.add_polar_layer = add_polar_layer
        self.flip_north_pole = flip_north_pole
        self.use_spectral_norm = use_spectral_norm

        if self.use_spectral_norm:
            func = spectral_norm
        else:
            func = lambda x: x

        if self.add_polar_layer:
            self.add_module('equatorial', func(instantiate(base_layer)))
            self.add_module('polar', func(instantiate(base_layer)))
        else:
            self.add_module('primary', func(instantiate(base_layer)))

    def forward(self, inputs):
        results = []
        if self.add_polar_layer:
            # Equatorial faces
            for face in range(4):
                results.append(torch.unsqueeze(self.equatorial(inputs[..., face, :, :]), -3))
            # First polar face (S)
            results.append(torch.unsqueeze(self.polar(inputs[..., 4, :, :]), -3))
            # Second polar face (N)
            if self.flip_north_pole:
                results.append(torch.unsqueeze(self.polar(inputs[..., 5, :, :].flip(-2)).flip(-2), -3))
            else:
                results.append(torch.unsqueeze(self.polar(inputs[..., 5, :, :]), -3))
        else:
            # Equatorial faces and first polar face (S)
            for face in range(5):
                results.append(torch.unsqueeze(self.primary(inputs[..., face, :, :]), -3))
            # Second polar face (N)
            if self.flip_north_pole:
                results.append(torch.unsqueeze(self.primary(inputs[..., 5, :, :].flip(-2)).flip(-2), -3))
            else:
                results.append(torch.unsqueeze(self.primary(inputs[..., 5, :, :]), -3))
        return torch.cat(results, -3)


# pylint: disable=invalid-name
class CubeSpherePadding(torch.nn.Module):
    """
    Padding layer for data on a cubed sphere. The requirements for using this layer are as follows:
    - The last three dimensions are (face=6, height, width)
    - The first four indices in the faces dimension are the equatorial faces
    - The last two faces (indices 4 and 5) are the polar faces

    Orientation and arrangement of the cube faces corresponds to that produced by the tempest-remap library.
    """
    def __init__(self, padding: int = 1):
        """
        Initialize a padding operation layer.
        :param padding: int: amount of padding to apply on each edge of each cube face
        """
        super().__init__()
        self.padding = padding
        if not isinstance(padding, int) or padding < 1:
            raise ValueError(f"invalid value for 'padding', expected int > 0 but got {padding}")

    def forward(self, inputs):
        p = self.padding
        # Face dimension prior to indexing single face
        f = len(inputs.shape) - 3
        # Height/width dimensions after removing face dimension
        h = len(inputs.shape) - 3
        w = len(inputs.shape) - 2

        # Pad the equatorial upper/lower boundaries and the polar upper/lower boundaries
        out = list()
        # Face 0
        out.append(torch.unsqueeze(
            torch.cat([
                inputs[..., 4, -p:, :],
                inputs[..., 0, :, :],
                inputs[..., 5, :p, :]
            ], dim=h), f
        ))
        # Face 1
        out.append(torch.unsqueeze(
            torch.cat([
                inputs[..., 4, :, -p:].flip(h).transpose(h, w),
                inputs[..., 1, :, :],
                inputs[..., 5, :, -p:].flip(w).transpose(h, w)
            ], dim=h), f
        ))
        # Face 2
        out.append(torch.unsqueeze(
            torch.cat([
                inputs[..., 4, :p, :].flip(h, w),
                inputs[..., 2, :, :],
                inputs[..., 5, -p:, :].flip(h, w)
            ], dim=h), f
        ))
        # Face 3
        out.append(torch.unsqueeze(
            torch.cat([
                inputs[..., 4, :, :p].flip(w).transpose(h, w),
                inputs[..., 3, :, :],
                inputs[..., 5, :, :p].flip(h).transpose(h, w)
            ], dim=h), f
        ))
        # Face 4 (south pole)
        out.append(torch.unsqueeze(
            torch.cat([
                inputs[..., 2, :p, :].flip(h, w),
                inputs[..., 4, :, :],
                inputs[..., 0, :p, :]
            ], dim=h), f
        ))
        # Face 5 (north pole)
        out.append(torch.unsqueeze(
            torch.cat([
                inputs[..., 0, -p:, :],
                inputs[..., 5, :, :],
                inputs[..., 2, -p:, :].flip(h, w)
            ], dim=h), f
        ))

        out1 = torch.cat(out, dim=f)
        del out

        # Pad the equatorial periodic lateral boundaries and the polar left/right boundaries
        out = list()
        # Face 0
        out.append(torch.unsqueeze(
            torch.cat([
                out1[..., 3, :, -p:],
                out1[..., 0, :, :],
                out1[..., 1, :, :p]
            ], dim=w), f
        ))
        # Face 1
        out.append(torch.unsqueeze(
            torch.cat([
                out1[..., 0, :, -p:],
                out1[..., 1, :, :],
                out1[..., 2, :, :p]
            ], dim=w), f
        ))
        # Face 2
        out.append(torch.unsqueeze(
            torch.cat([
                out1[..., 1, :, -p:],
                out1[..., 2, :, :],
                out1[..., 3, :, :p]
            ], dim=w), f
        ))
        # Face 3
        out.append(torch.unsqueeze(
            torch.cat([
                out1[..., 2, :, -p:],
                out1[..., 3, :, :],
                out1[..., 0, :, :p]
            ], dim=w), f
        ))
        # Face 4
        out.append(torch.unsqueeze(
            torch.cat([
                out[3][..., 0, p:2 * p, :].flip(h).transpose(h, w),
                out1[..., 4, :, :],
                out[1][..., 0, p:2 * p, :].flip(w).transpose(h, w)
            ], dim=w), f
        ))
        # Face 5
        out.append(torch.unsqueeze(
            torch.cat([
                out[3][..., 0, -2 * p:-p, :].flip(w).transpose(h, w),
                out1[..., 5, :, :],
                out[1][..., 0, -2 * p:-p, :].flip(h).transpose(h, w)
            ], dim=w), f
        ))

        del out1
        outputs = torch.cat(out, dim=f)
        del out
        return outputs
