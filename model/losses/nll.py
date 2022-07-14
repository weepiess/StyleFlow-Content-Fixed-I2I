import torch
import torch.nn as nn
import numpy as np
from math import log, pi, exp
# class NLLLoss(nn.Module):
#     """Negative log-likelihood loss assuming isotropic gaussian with unit norm.
#     Args:
#         k (int or float): Number of discrete values in each input dimension.
#             E.g., `k` is 256 for natural images.
#     See Also:
#         Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
#     """
#     def __init__(self, k=256):
#         super(NLLLoss, self).__init__()
#         self.k = k

#     def forward(self, z, sldj):
#         prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
#         prior_ll = prior_ll.flatten(1).sum(-1) \
#             - np.log(self.k) * np.prod(z.size()[1:])
#         ll = prior_ll + sldj
#         nll = -ll.mean()

#         return nll

class NLLLoss(nn.Module):
    def __init__(self, n_bins=256):
        super(NLLLoss, self).__init__()
        self.n_bins = n_bins
        
    def forward(self, log_p, logdet, image_size=[728,128]):
        # log_p = calc_log_p([z_list])
        n_pixel = image_size[0] * image_size[1] * 3

        loss = -log(self.n_bins) * n_pixel
        loss = loss + logdet + log_p

        return (
            (-loss / (log(2) * n_pixel)).mean(),
            (log_p / (log(2) * n_pixel)).mean(),
            (logdet / (log(2) * n_pixel)).mean(),
        )