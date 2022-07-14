import torch
import torch.nn as nn


class GaussianKLLoss(nn.Module):
    r"""Compute KL loss in VAE for Gaussian distributions"""
    def __init__(self):
        super(GaussianKLLoss, self).__init__()

    def forward(self, mu, logvar=None):
        r"""Compute loss

        Args:
            mu (tensor): mean
            logvar (tensor): logarithm of variance
        """
        if logvar is None:
            logvar = torch.zeros_like(mu)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
