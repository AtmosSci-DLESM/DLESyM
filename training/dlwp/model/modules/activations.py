import torch


class CappedLeakyReLU(torch.nn.Module):
    """
    Implements a ReLU with capped maximum value.
    """
    def __init__(self, cap_value=1., **kwargs):
        """
        :param cap_value: float: value at which to clip activation
        :param kwargs: passed to torch.nn.LeadyReLU
        """
        super().__init__()
        self.add_module('relu', torch.nn.LeakyReLU(**kwargs))
        #self.cap = torch.tensor(cap_value, dtype=torch.float32)
        self.register_buffer("cap", torch.tensor(cap_value, dtype=torch.float32))

    def forward(self, inputs):
        x = self.relu(inputs)
        x = torch.clamp(x, max=self.cap)
        return x


class CappedGELU(torch.nn.Module):
    """
    Implements a ReLU with capped maximum value.
    """
    def __init__(self, cap_value=1., **kwargs):
        """
        :param cap_value: float: value at which to clip activation
        :param kwargs: passed to torch.nn.LeadyReLU
        """
        super().__init__()
        self.add_module('gelu', torch.nn.GELU(**kwargs))
        #self.cap = torch.tensor(cap_value, dtype=torch.float32)
        self.register_buffer("cap", torch.tensor(cap_value, dtype=torch.float32))

    def forward(self, inputs):
        x = self.gelu(inputs)
        x = torch.clamp(x, max=self.cap)
        return x
