from __future__ import annotations

from typing import Optional, Union

from botorch.posteriors.posterior import Posterior
from botorch.acquisition.objective import ScalarizedObjective
from botorch.models.model import Model
from botorch.acquisition.analytic import AnalyticAcquisitionFunction, UpperConfidenceBound
from botorch.utils.transforms import convert_to_target_pre_hook, t_batch_mode_transform
import torch
from botorch.exceptions import UnsupportedError
from torch import Tensor

from botorch.acquisition.analytic import AnalyticAcquisitionFunction


class RiskAverseUpperConfidenceBound(AnalyticAcquisitionFunction):
    r"""Single-outcome Risk-averse Upper Confidence Bound (RAUCB).

    Analytic upper confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `RAUCB(x) = mu(x) + sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.

    Example:
        model = SingleTaskGP(train_X, train_Y)
        RAHBO = RiskAverseUpperConfidenceBound(model, beta=0.2)
        rahbo = RAHBO(test_X)
    """

    def __init__(
            self,
            model: Model,
            model_varproxi: Model,
            beta: Union[float, Tensor],
            beta_varproxi: Union[float, Tensor],
            gamma: Union[float, Tensor],
            objective: Optional[ScalarizedObjective] = None,
            maximize: bool = True,
    ) -> None:
        r"""Risk averse single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            model_varproxi: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta_varproxi: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            gamma: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter function optimization and varproxi of noise
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, objective=objective)
        self.maximize = maximize
        self.model_varproxi = model_varproxi
        if not torch.is_tensor(beta):
            beta = torch.tensor([beta])
        self.register_buffer("beta", beta)
        if not torch.is_tensor(beta_varproxi):
            beta_varproxi = torch.tensor([beta_varproxi])
        #         self.beta_varproxi = beta_varproxi
        self.register_buffer("beta_varproxi", beta_varproxi)
        if not torch.is_tensor(gamma):
            gamma = torch.tensor(gamma)
        #         self.gamma = gamma
        self.register_buffer("gamma", gamma)

    def _get_posterior_varproxi(self, X: Tensor, check_single_output: bool = True) -> Posterior:
        r"""Compute the posterior over varproxi at the input candidate set X.


        Args:
            X: The input candidate set
            check_single_output: If True, Raise an error if the posterior is not
                single-output.

        Returns:
            The posterior at X.
        """
        self.model_varproxi.eval()
        posterior = self.model_varproxi(X)
        if check_single_output:
            if posterior.event_shape[-1] != 1:
                raise UnsupportedError(
                    "Multi-Output posteriors are not supported for acquisition "
                    f"functions of type {self.__class__.__name__}"
                )
        return posterior

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`.
        """
        self.beta = self.beta.to(X)
        posterior = self._get_posterior(X=X)
        batch_shape = X.shape[:-2]
        mean = posterior.mean.view(batch_shape)
        variance = posterior.variance.view(batch_shape)
        delta = (self.beta.expand_as(variance) * variance).sqrt()

        self.beta_varproxi = self.beta_varproxi.to(X)
        posterior_varproxi = self._get_posterior_varproxi(X=X)
        mean_varproxi = posterior_varproxi.mean.view(batch_shape)
        variance_varproxi = posterior_varproxi.variance.view(batch_shape)
        delta_varproxi = (self.beta_varproxi.expand_as(variance_varproxi) * variance_varproxi).sqrt()

        # ucb = ucb_f - gamma*lcb_{rho}
        if self.maximize:
            return (mean + delta - self.gamma.expand_as(mean_varproxi) * (mean_varproxi - delta_varproxi))
        # lcb = lcb_f - gamma*ucb_{rho}
        else:
            return (mean - delta - self.gamma.expand_as(mean_varproxi) * (mean_varproxi + delta_varproxi))


class PosteriorVariance(AnalyticAcquisitionFunction):
    r"""Single-outcome Posterior Variance.

    Only supports the case of q=1. Requires the model's posterior to have a
    `mean` property. The model must be single-outcome.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> PM = PosteriorVariance(model)
        >>> pv = PV(test_X)
    """

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the posterior mean on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Posterior Mean values at the given design
            points `X`.
        """

        posterior = self._get_posterior(X=X)
        return posterior.variance.view(X.shape[:-2])


class LowerConfidenceBound(AnalyticAcquisitionFunction):
    r"""Single-outcome Upper Confidence Bound (UCB).

    Analytic upper confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `LCB(x) = mu(x) - sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.

    """

    def __init__(
        self,
        model: Model,
        beta: Union[float, Tensor],
        objective: Optional[ScalarizedObjective] = None,
        maximize: bool = False,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, objective=objective)
        self.maximize = maximize
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`.
        """
        self.beta = self.beta.to(X)
        posterior = self._get_posterior(X=X)
        batch_shape = X.shape[:-2]
        mean = posterior.mean.view(batch_shape)
        variance = posterior.variance.view(batch_shape)
        delta = (self.beta.expand_as(mean) * variance).sqrt()
        if self.maximize:
            return -mean - delta
        else:
            return mean - delta