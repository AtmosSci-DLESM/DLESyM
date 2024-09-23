import torch


class Interpolate(torch.nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.interp = torch.nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, inputs):
        return self.interp(inputs, scale_factor=self.scale_factor, mode=self.mode)
