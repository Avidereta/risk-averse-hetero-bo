import torch
from torch import Tensor
import math
from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.utils.sampling import draw_sobol_samples
from .benchmarks import BenchmarkBase


class NegBranin(SyntheticTestFunction):
    """ Negated Branin function.
        Two-dimensional function defined over on [-5, 10] x [0, 15] with 3 optimizers.

        Definition:
            B(x) = - [ (x_2 - b x_1^2 + c x_1 - r)^2 + 10 (1-t) cos(x_1) + 10]
            Where b, c, r and t are constants defined as:
                b = 5.1 / (4 * math.pi ** 2)
                c = 5 / math.pi
                r = 6
                t = 1 / (8 * math.pi)
    """

    dim = 2
    _bounds = [(-5.0, 10.0), (0.0, 15.0)]
    _optimal_value = - 0.397887
    _optimizers = [(-math.pi, 12.275), (math.pi, 2.275), (9.42478, 2.475)]

    def evaluate_true(self, x: Tensor) -> Tensor:
        t1 = (
                x[..., 1]
                - 5.1 / (4 * math.pi ** 2) * x[..., 0] ** 2
                + 5 / math.pi * x[..., 0]
                - 6
        )
        t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(x[..., 0])
        result = t1 ** 2 + t2 + 10
        return torch.Tensor(-result)


class NegBraninBenchmark(BenchmarkBase):
    """
    Two-dimentional Negated Branin function with three global optimizers: A=(9.4, 2.5), B=(pi, 2.3), C=(-pi, 12.3) with
    different noise level. Noise in the measurements is zero-mean Gaussian with heteroscedastic (i.e., input-dependent)
    variance induced by two-dimentional sigmoid function. This results into the noisy evaluations with A being the
    noisiest input and C with the least noise.
    """

    def __init__(self, config, f: SyntheticTestFunction = NegBranin):
        super(NegBraninBenchmark, self).__init__()

        self.f = f()
        self.sigma = config['sigma']
        self.repeat_eval = config['repeat_eval']
        self._optimizers = self.f._optimizers
        self._max_value = self.f._optimal_value
        self.seed_test = 42

    def evaluate(self, x: Tensor, seed_eval=None) -> Tensor:
        y_true = self.f.evaluate_true(x).reshape((-1, 1))
        sigmas = self.get_noise_var(x[:, 0].reshape(-1, 1),
                                    x[:, 1].reshape(-1, 1))

        if seed_eval is not None:
            shape = torch.cat([y_true] * self.repeat_eval, dim=1).shape
            noise = sigmas * torch.randn(shape, generator=torch.Generator().manual_seed(seed_eval))
            y = y_true + noise
        else:
            noise = sigmas * torch.randn_like(torch.cat([y_true] * self.repeat_eval, dim=1))
            y = y_true + noise

        return y

    def evaluate_on_test(self, x: Tensor) -> Tensor:

        y_true = self.f.evaluate_true(x).reshape((-1, 1))
        sigmas = self.get_noise_var(x[:, 0].reshape(-1, 1),
                                    x[:, 1].reshape(-1, 1))
        shape = y_true.shape
        noise = sigmas * torch.randn(shape, generator=torch.Generator().manual_seed(self.seed_test))
        y = y_true + noise

        return y

    def get_domain(self):
        return torch.Tensor(self.f._bounds).T

    def get_random_initial_points(self, num_points, seed) -> Tensor:

        x = draw_sobol_samples(self.get_domain(), num_points, q=1, seed=seed).squeeze()

        return x

    def get_noise_var(self, x_0, x_1):
        var1 = self._get_noise_var_1d(x_0, sigma=self.sigma, shift=3.2)
        var2 = self._get_noise_var_1d(x_1, sigma=self.sigma, shift=0)
        return var1 * var2

    def get_info_to_dump(self, x):

        dict_to_dump = {
            'f': self.f.evaluate_true(x).squeeze(),
            'rho': self.get_noise_var(x[:, 0].reshape(-1, 1),
                                      x[:, 1].reshape(-1, 1)).squeeze()
        }

        return dict_to_dump

    @staticmethod
    def _get_noise_var_1d(x, sigma:float=2, shift:float=2):
        return torch.sigmoid((x - shift) * 2) * sigma + 0.2
