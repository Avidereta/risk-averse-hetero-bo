from .benchmarks import BenchmarkBase
import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.sampling import draw_sobol_samples
import os
import numpy as np
import h5py
import json


class FSquaredExponential(torch.nn.Module):
    def __init__(self, *layers, scale=1.):
        super().__init__()
        self.d = layers[0]

        _layers = []
        for i in range(len(layers) - 1):
            din = layers[i]
            dout = layers[i + 1]
            _layers.append(torch.nn.Linear(din, dout, bias=False))
        self.F = torch.nn.Sequential(*_layers)

        self.m = torch.nn.Parameter(torch.zeros(self.d), requires_grad=True)
        self.A = torch.nn.Parameter(torch.tensor(scale), requires_grad=True)
        self.B = torch.nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, X):
        X_torch = X - self.m
        F = self.F(X_torch.float())
        return self.A * torch.exp(-torch.sum(F * F, dim=-1)) + self.B

    @property
    def V(self):
        V = torch.eye(self.d)
        for layer in self.F:
            V = torch.matmul(layer.weight, V)
        return torch.matmul(V.T, V)


def get_data(path, run=0):
    """
    reads evaluation data from {path}
    """
    meta_data = {}
    f_eval = h5py.File(os.path.join(path, 'data/evaluations.hdf5'), 'r')
    try:
        data_0 = f_eval['0']
        data = data_0[str(run)][:]
    except KeyError:
        return None

    meta_data = {'algorithm': data_0.attrs['algorithm'],
                 'T': len(data)}

    f_eval.close()
    with open(os.path.join(path, 'sf/x0.json')) as f:
        x0 = np.array(json.load(f))
        meta_data['x0'] = x0

    return {'data': data, 'meta': meta_data}


class FELBenchmark(BenchmarkBase):

    def __init__(self, config):
        super(FELBenchmark, self).__init__()

        self.repeat_eval = config['repeat_eval']
        self.seed_test = 42

        run1 = get_data('fel_data/20200913_031909')
        run2 = get_data('fel_data/20200913_051204')

        #         x_ref = run1['meta']['x0']
        x1_bp = run1['data'][-2]['x_bp'].copy()
        x2_bp = run2['data'][-2]['x_bp'].copy()
        x_ref = (x1_bp + x2_bp) / 2

        def normalize_x(X):
            return (X - x_ref) / 0.16

        def denormalize_x(X):
            return x_ref + X * 0.16

        # normalization for the range
        def normalize_y(Y):
            return Y / 500

        def denormalize_y(Y):
            return Y * 500

        self.X_val = normalize_x(torch.Tensor(np.array(run1['data']['x']).copy()))
        self.Y_val = normalize_y(torch.Tensor(np.array(run1['data']['y']).copy()))
        self.X = normalize_x(torch.Tensor(np.array(run2['data']['x']).copy()))
        self.Y = normalize_y(torch.Tensor(np.array(run2['data']['y']).copy()))
        d = 28
        self.d = 4
        self.model = FSquaredExponential(d, 8*d, 4*d, 2*d, d, scale=1.)

        self._train()

        train_residuals = ((self.Y_val - self.model(self.X_val))**2).detach()
        train_X = self.X_val.detach()
        self.gp_model_varproxi = SingleTaskGP(train_X, train_residuals.reshape((-1, 1))).to(train_X)
        self.mll_varproxi = ExactMarginalLogLikelihood(self.gp_model_varproxi.likelihood, self.gp_model_varproxi)
        fit_gpytorch_model(self.mll_varproxi)

    def _train(self):
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)  # , weight_decay=1e-4)

        dataset = torch.utils.data.TensorDataset(self.X, self.Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)
        T = 50
        for t in range(T + 1):
            # batch
            for x, y in loader:
                # Forward pass: compute predicted y by passing x to the self.model.
                y_pred = self.model(x)
                y_tar = y  # + torch.normal(torch.zeros_like(y), 0.2)

                loss = loss_fn(y_pred,
                               y_tar) + 0.1 * self.model.B ** 2  # + 0.00001* torch.sum(model.V**2) #+ 0.01* (model.A - 1.)**2 + 0.001*torch.sum(model.m**2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def evaluate(self, x: Tensor, seed_eval=None) -> Tensor:
        n_points = x.shape[0]
        x_extended = torch.cat([self.X[-1, :5].reshape(1,-1).repeat(n_points,1),
                       x,
                       self.X[-1, 9:].reshape(1,-1).repeat(n_points,1)], dim=1)
        y_true = self.model(x_extended).detach().reshape((-1, 1))
        noise_mean = (self.gp_model_varproxi(x_extended.double()).mean**0.5).detach()

        if seed_eval is not None:
            shape = torch.cat([y_true] * self.repeat_eval, dim=1).shape
            y = y_true + noise_mean * torch.randn(shape, generator=torch.Generator().manual_seed(seed_eval))
        else:
            y_true = torch.cat([y_true] * self.repeat_eval, dim=1)
            y = y_true + noise_mean.reshape((-1, 1)) * torch.randn_like(y_true)
        return y

    def evaluate_on_test(self, x: Tensor) -> Tensor:
        n_points = x.shape[0]
        x_extended = torch.cat([self.X[-1, :5].reshape(1,-1).repeat(n_points,1),
                       x,
                       self.X[-1, 9:].reshape(1,-1).repeat(n_points,1)], dim=1)
        y_true = self.model(x_extended).detach()
        noise_mean = (self.gp_model_varproxi(x_extended.double()).mean**0.5).detach()

        shape = y_true.shape
        noise = noise_mean * torch.randn(shape, generator=torch.Generator().manual_seed(self.seed_test))
        y = y_true + noise
        return y

    def get_domain(self):
        bounds = torch.zeros(2, self.d, dtype=torch.float)
        bounds[0] = - 0.3
        bounds[1] = 0.3
        return bounds

    def get_random_initial_points(self, num_points, seed) -> Tensor:

        x = draw_sobol_samples(self.get_domain(), num_points, q=1, seed=seed).squeeze()

        return x

    def get_info_to_dump(self, x):
        n_points = x.shape[0]
        x_extended = torch.cat([self.X[-1, :5].reshape(1, -1).repeat(n_points, 1),
                                x,
                                self.X[-1, 9:].reshape(1, -1).repeat(n_points, 1)], dim=1)

        dict_to_dump = {
            'f': self.model(x_extended).detach().squeeze(),
            'rho': self.gp_model_varproxi(x_extended.double()).mean.detach().squeeze()
        }

        return dict_to_dump
