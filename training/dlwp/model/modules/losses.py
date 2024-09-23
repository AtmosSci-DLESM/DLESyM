import torch
from torch.nn import functional as F
import numpy as np

class LossOnStep(torch.nn.Module):
    """
    Allows computation of an existing loss function on only one step of a sequence of outputs.
    """
    def __init__(self, loss: torch.nn.Module, time_dim: int, step: int):
        super().__init__()
        self.loss = loss
        self.time_slice = slice(step * time_dim, (step + 1) * time_dim)

    def forward(self, inputs, targets, model: torch.nn.Module = None):
        
        # check weather model is required for loss calculation 
        if 'model' in self.loss.forward.__code__.co_varnames:
            return self.loss(inputs[:, self.time_slice], targets[:, self.time_slice], model)
        else: 
            return self.loss(inputs[:, self.time_slice], targets[:, self.time_slice])


class GeneratorLoss(torch.nn.Module):
    def __init__(self, loss: torch.nn.Module, disc_score_weight: float):
        super().__init__()
        self.loss = loss
        self.disc_score_weight = torch.tensor(disc_score_weight, dtype=torch.float32)

    def forward(self, inputs, targets, disc_score):
        if disc_score is not None:
            return self.loss(inputs, targets) + self.disc_score_weight * disc_score
        else:
            return self.loss(inputs, targets)

class MSE_SSIM(torch.nn.Module):
    """
    This class provides a compound loss formulation combining differential structural similarity (SSIM) and mean squared 
    error (MSE). Calling this class will compute the loss using SSIM for fields indicated by model attributes 
    (model.ssim_fields). 
    """
    def __init__(
            self,
            mse_params=None,
            ssim_params=None,
            ssim_variables = ['ttr1h', 'tcwv0'],
            weights=[0.5,0.5],
            ):
        """
        Constructor method.

        :param mse_params: (Optional) parameters to pass to MSE constructor  
        :param ssim_params: (Optional) dictionary of parameters to pass to SSIM constructor  
        :ssim variables: (Optional) list of variables over which loss will be calculated using DSSIM and MSE 
        :param weights: (Optional) variables identified as requireing SSIM-loss calculation 
            will have their loss calculated by a weighted average od the DSSIM metric and MSE.
            The weights of this weighted average are identified here. [MSE_weight, DSSIM_weight]
        """

        super(MSE_SSIM, self).__init__()
        if ssim_params is None:
            self.ssim = SSIM()  
        else: 
            self.ssim = SSIM(**ssim_params)
        if mse_params is None:
            self.mse = torch.nn.MSELoss() 
        else: 
            self.mse = torch.nn.MSELoss(**mse_params)
        if np.sum(weights) == 1:
            self.mse_dssim_weights = weights 
        else:
            raise ValueError('Weights passed to MSE_SSIM loss must sum to 1')
        self.ssim_variables = ssim_variables 

    def forward(
            self,
            outputs: torch.tensor,
            targets: torch.tensor,
            model: torch.nn.Module):

        
        # check tensor shapes to ensure proper computation of loss 
        try:
            assert outputs.shape[-1] == outputs.shape[-2]
            assert outputs.shape[3] == 12
            assert outputs.shape[2] == model.output_channels
            assert (outputs.shape[1] == model.output_time_dim) or (outputs.shape[1] == model.output_time_dim//model.input_time_dim)
        except AssertionError: 
            print(f'losses.MSE_SSIM: expected output shape [batchsize, {model.output_time_dim}, {model.output_channels}, [spatial dims]] got {outputs.shape}')
            exit()

        # store the location of output and target tensors 
        device = outputs.get_device()
        # initialize losses by var tensor that will store the variable wise loss 
        loss_by_var = torch.empty([outputs.shape[2]],device=f'cuda:{device}')
        # initialize weights tensor that will allow for a weighted average of MSE and SSIM 
        weights = torch.tensor(self.mse_dssim_weights,device=f'cuda:{device}')
        # calculate variable wise loss 
        for i,v in enumerate(model.output_variables):
            # for logging purposes calculated DSIM and MSE for each variable
            var_mse = self.mse(outputs[:,:,i:i+1,:,:,:],targets[:,:,i:i+1,:,:,:]) # the slice operation here ensures the singleton dimension is not squashed
            var_dssim = torch.min(torch.tensor([1.,float(var_mse)]))*(1-self.ssim(outputs[:,:,i:i+1,:,:,:],targets[:,:,i:i+1,:,:,:]))
            if v in self.ssim_variables: 
                # compute weighted average between mse and dssim
                loss_by_var[i] = torch.sum( weights * torch.stack([var_mse,var_dssim]) )
            else:
                loss_by_var[i] = var_mse
            model.log(f'MSEs_train/{model.output_variables[i]}', var_mse, batch_size=model.batch_size)
            model.log(f'DSIMs_train/{model.output_variables[i]}', var_dssim, batch_size=model.batch_size)
            model.log(f'losses_train/{model.output_variables[i]}', loss_by_var[i], batch_size=model.batch_size)
        loss = loss_by_var.mean()
        model.log(f'losses_train/all_vars', loss, batch_size=model.batch_size)
        return loss

class SSIM(torch.nn.Module):
    """
    This class provides a differential structural similarity (SSIM) as loss for training an artificial neural network. The
    advantage of SSIM over the conventional mean squared error is a relation to images where SSIM incorporates the local
    neighborhood when determining the quality of an individual pixel. Results are less blurry, as demonstrated here
    https://ece.uwaterloo.ca/~z70wang/research/ssim/

    Code is origininally taken from https://github.com/Po-Hsun-Su/pytorch-ssim
    Modifications include comments and an optional training phase with the mean squared error (MSE) preceding the SSIM
    loss, to bring the weights on track. Otherwise, SSIM often gets stuck early in a local minimum.
    """

    def __init__(
            self,
            window_size: int = 11,
            time_series_forecasting: bool = False,
            padding_mode: str = "constant",
            mse: bool = False,
            mse_epochs: int = 0):
        """
        Constructor method.

        :param window_size: (Optional) The patch size over which the SSIM is computed
        :param time_series_forecasting: (Optional) Boolean indicating whether time series forecasting is the task
        :param padding_mode: Padding mode used for padding input images, e.g. 'zeros', 'replicate', 'reflection'
        :param mse: Uses MSE parallel
        :param mse_epochs: (Optional) Number of MSE epochs preceding the SSIM epochs during training
        """
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.time_series_forecasting = time_series_forecasting
        self.padding_mode = padding_mode
        self.mse = torch.nn.MSELoss() if mse else None
        self.mse_epochs = mse_epochs
        self.c1, self.c2 = 0.01 ** 2, 0.03 ** 2

        self.register_buffer("window", self._create_window(window_size), persistent=False)

    def forward(
            self,
            img1: torch.Tensor,
            img2: torch.Tensor,
            mask: torch.Tensor = None,
            epoch: int = 0) -> torch.Tensor:
        """
        Forward pass of the SSIM loss

        :param img1: Predicted image of shape [B, T, C, F, H, W]
        :param img2: Ground truth image of shape [B, T, C, F, H, W]
        :param mask: (Optional) Mask for excluding pixels
        :param epoch: (Optional) The current epoch
        :return: The structural similarity loss
        """
        img1 = img1.transpose(dim0=2, dim1=3)
        img2 = img2.transpose(dim0=2, dim1=3)
        if self.time_series_forecasting:
            # Join Batch and time dimension
            img1 = torch.flatten(img1, start_dim=0, end_dim=2)
            img2 = torch.flatten(img2, start_dim=0, end_dim=2)

        window = self.window.expand(img1.shape[1], -1, -1, -1)

        if window.dtype != img1.dtype:
            window = window.to(dtype=img1.dtype)

        return self._ssim(img1, img2, window, mask, epoch)

    @staticmethod
    def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
        """
        Computes a Gaussian over the size of the window to weigh distant pixels less.

        :param window_size: The size of the patches
        :param sigma: The width of the Gaussian curve
        :return: A tensor representing the weights for each pixel in the window or patch
        """
        x = torch.arange(0, window_size) - window_size // 2
        gauss = torch.exp(-((x.div(2 * sigma)) ** 2))
        return gauss / gauss.sum()

    def _create_window(
            self,
            window_size: int,
            sigma: float = 1.5) -> torch.Tensor:
        """
        Creates the weights of the window or patches.

        :param window_size: The size of the patches
        :param sigma: The width of the Gaussian curve
        """
        _1D_window = self._gaussian(window_size, sigma).unsqueeze(-1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        return _2D_window

    def _ssim(
            self,
            img1: torch.Tensor,
            img2: torch.Tensor,
            window: torch.Tensor,
            mask: torch.Tensor = None,
            epoch: int = 0) -> torch.Tensor:
        """
        Computes the SSIM loss between two image tensors

        :param _img1: The predicted image tensor
        :param _img2: The target image tensor
        :param window: The weights for each pixel in the window over which the SSIM is computed
        :param mask: Mask for excluding pixels
        :param epoch: The current epoch
        :return: The SSIM between img1 and img2
        """
        if epoch < self.mse_epochs:
            # If specified, the MSE is used for the first self.mse_epochs epochs
            return F.mse_loss(img1, img2)

        channels = window.shape[0]
        window_size = window.shape[2]

        window = window.to(device=img1.device)

        _img1 = F.pad(
            img1,
            pad=[(window_size - 1) // 2, (window_size - 1) // 2 + (window_size - 1) % 2,
                 (window_size - 1) // 2, (window_size - 1) // 2 + (window_size - 1) % 2],
            mode=self.padding_mode
        )

        _img2 = F.pad(
            img2,
            pad=[(window_size - 1) // 2, (window_size - 1) // 2 + (window_size - 1) % 2,
                 (window_size - 1) // 2, (window_size - 1) // 2 + (window_size - 1) % 2],
            mode=self.padding_mode
        )

        mu1 = F.conv2d(_img1, window, padding=0, groups=channels)
        mu2 = F.conv2d(_img2, window, padding=0, groups=channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(_img1 * _img1, window, padding=0, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(_img2 * _img2, window, padding=0, groups=channels) - mu2_sq
        sigma12_sq = F.conv2d(_img1 * _img2, window, padding=0, groups=channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.c1) * (2 * sigma12_sq + self.c2)) /\
                   ((mu1_sq + mu2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2))

        if mask is not None:
            ssim_map = ssim_map[..., mask]
            img1 = img1[..., mask]
            img2 = img2[..., mask]

        ssim = ssim_map.mean().abs()

        if self.mse:
            ssim = ssim + self.mse(img1, img2)
    
        return ssim


def loss_hinge_disc(score_generated, score_real):
    """Discriminator hinge loss."""
    l1 = F.relu(1.0 - score_real)
    loss = torch.mean(l1)
    l2 = F.relu(1.0 + score_generated)
    loss += torch.mean(l2)
    return loss


def loss_hinge_gen(score_generated):
    """Generator hinge loss."""
    loss = -torch.mean(score_generated)
    return loss
