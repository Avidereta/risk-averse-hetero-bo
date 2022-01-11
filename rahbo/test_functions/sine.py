import torch
from torch import Tensor
import math
from .benchmarks import BenchmarkBase
from botorch.utils.sampling import draw_sobol_samples


class SineBenchmark(BenchmarkBase):
    """
    One-dimensional sine function with two global optimizers with different noise level.
    Noise in the measurements is zero-mean Gaussian with heteroscedastic (i.e., input-dependent) variance induced by
    sigmoid function. This results into small noise on [0, 1] and higher noise on (1, 2].
    """

    def __init__(self, config):
        super(SineBenchmark, self).__init__()
        self.sigma = config['sigma']
        self.repeat_eval = config['repeat_eval']
        self._optimizers = [0.25, 1.25]
        self._max_value = 1
        self.seed_test = 42

    def evaluate(self, x: Tensor, seed_eval=None) -> Tensor:
        y_true = self.f(x)
        sigmas = self.get_noise_var(x)

        if seed_eval is not None:
            shape = torch.cat([y_true] * self.repeat_eval, dim=1).shape
            y = y_true + sigmas * torch.randn(shape, generator=torch.Generator().manual_seed(seed_eval))
        else:
            y_true = torch.cat([y_true] * self.repeat_eval, dim=1)
            y = y_true + sigmas.reshape((-1, 1)) * torch.randn_like(y_true)
        return y

    def evaluate_on_test(self, x: Tensor) -> Tensor:

        y_true = self.f(x)
        sigmas = self.get_noise_var(x)

        shape = y_true.shape
        noise = sigmas * torch.randn(shape, generator=torch.Generator().manual_seed(self.seed_test))
        y = y_true + noise
        return y

    def get_domain(self):

        bounds = torch.zeros(2, 1, dtype=torch.float)
        bounds[1] = 2
        return bounds

    def get_random_initial_points(self, num_points, seed) -> Tensor:

        x = draw_sobol_samples(self.get_domain(), num_points, q=1, seed=seed).reshape((-1, 1))

        return x

    @staticmethod
    def f(x):
        y_true = torch.sin(x * (2 * math.pi))
        return y_true.reshape((-1, 1))

    def get_noise_var(self, x):
        sigmas = torch.sigmoid((x - 1) * 30) * self.sigma + 0.1
        return sigmas

    def get_info_to_dump(self, x):

        dict_to_dump = {
            'f': self.f(x).squeeze(),
            'rho': self.get_noise_var(x).squeeze()
        }

        return dict_to_dump
