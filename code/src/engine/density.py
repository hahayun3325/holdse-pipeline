import torch
import torch.nn as nn


class Density(nn.Module):
    def __init__(self, params_init={}):
        super().__init__()
        for p in params_init:
            param = nn.Parameter(torch.tensor(params_init[p]))
            setattr(self, p, param)

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)


class LaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, params_init={}, beta_min=0.0001):
        super().__init__(params_init=params_init)
        self.beta_min = torch.tensor(beta_min).cuda()

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        density = alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))
        # CRITICAL FIX: Clamp to prevent unbounded values
        return torch.clamp(density, 0, 1.0)

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta


class AbsDensity(Density):  # like NeRF++
    def density_func(self, sdf, beta=None):
        # Unbounded torch.abs(sdf) can cause free_energy = dists * huge_density = Inf
        return torch.clamp(torch.abs(sdf), max=1.0)

class SimpleDensity(Density):  # like NeRF
    def __init__(self, params_init={}, noise_std=1.0):
        super().__init__(params_init=params_init)
        self.noise_std = noise_std

    def density_func(self, sdf, beta=None):
        if self.training and self.noise_std > 0.0:
            noise = torch.randn(sdf.shape).cuda() * self.noise_std
            sdf = sdf + noise
        # CRITICAL FIX: Clamp ReLU output to prevent unbounded density
        return torch.clamp(torch.relu(sdf), max=1.0)