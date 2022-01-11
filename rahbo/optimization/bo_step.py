from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP, FixedNoiseGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
import torch


def initialize_model(X, y, GP=None, state_dict=None, *GP_args, **GP_kwargs):
    """
    Create GP model and fit it. The function also accepts
    state_dict which is used as an initialization for the GP model.

    Parameters
    ----------
    X : torch.tensor, shape=(n_samples, dim)
        Input values

    y : torch.tensor, shape=(n_samples,)
        Output values

    GP : botorch.models.Model
        GP model class

    state_dict : dict
        GP model state dict

    Returns
    -------
    mll : gpytorch.mlls.MarginalLoglikelihood
        Marginal loglikelihood

    gp :
    """

    if GP is None:
        GP = SingleTaskGP

    model = GP(X, y, *GP_args, **GP_kwargs).to(X)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def bo_step_risk_averse(X, y, objective, bounds, GP=None, acquisition=None, q=1, state_dict=None,  *GP_args,
                        **GP_kwargs):
    """
    One iteration of Bayesian optimization:
        1. Fit GP model using (X, y)
        2. Create acquisition function
        3. Optimize acquisition function to obtain candidate point
        4. Evaluate objective at candidate point
        5. Add new point to the data set

    Parameters
    ----------
    X : torch.tensor, shape=(n_samples, dim)
        Input values

    y : torch.tensor, shape=(n_samples,)
        Objective values

    objective : callable, argument=torch.tensor
        Objective black-box function, accepting as an argument torch.tensor

    bounds : torch.tensor, shape=(2, dim)
        Box-constraints

    GP : callable
        GP model class constructor. It is a function that takes as input
        2 tensors - X, y - and returns an instance of botorch.models.Model.

    acquisition : callable
        Acquisition function construction. It is a function that receives
        one argument - GP model - and returns an instance of
        botorch.acquisition.AcquisitionFunction

    q : int
        Number of candidate points to find

    state_dict : dict
        GP model state dict

    plot : bool
        Flag indicating whether to plot the result

    Returns
    -------
    X : torch.tensor
        Tensor of input values with new point

    y : torch.tensor
        Tensor of output values with new point

    gp : botorch.models.Model
        Constructed GP model


    Example
    -------
    from botorch.models import FixedNoiseGP
    noise_var = 1e-2 * torch.ones_like(y)
    GP = lambda X, y: FixedNoiseGP(X, y, noise_var)
    acq_func = labmda gp: ExpectedImprovement(gp, y.min(), maximize=False)
    X, y = bo_step(X, y, objective, GP=GP, Acquisition=acq_func)

    """

    yvar = GP_kwargs.get('train_Yvar')

    # Create GP model
    mll, gp = initialize_model(X, y, GP=GP, state_dict=state_dict, train_Yvar=yvar)
    fit_gpytorch_model(mll)

    # Variance model
    mll_varproxi, gp_varproxi = initialize_model(X, yvar, GP=SingleTaskGP, state_dict=None)
    fit_gpytorch_model(mll_varproxi)

    # Create acquisition function
    acquisition = acquisition(gp, gp_varproxi)

    # Optimize acquisition function
    candidate, _ = optimize_acqf(
        acquisition, bounds=bounds, q=q, num_restarts=1, raw_samples=1000,
    )

    # Update data set
    X = torch.cat([X, candidate])
    y = torch.cat([y, objective(candidate).mean(dim=1).reshape((-1, 1))])
    if yvar is not None:
        yvar = torch.cat([yvar, objective(candidate).var(dim=1).reshape((-1, 1))])

    if yvar is not None:
        return X, y, gp, yvar, gp_varproxi
    else:
        return X, y, gp


def bo_step(X, y, objective, bounds, GP=None, acquisition=None, q=1, state_dict=None, *GP_args,
            **GP_kwargs):
    """
    One iteration of Bayesian optimization:
        1. Fit GP model using (X, y)
        2. Create acquisition function
        3. Optimize acquisition function to obtain candidate point
        4. Evaluate objective at candidate point
        5. Add new point to the data set

    Parameters
    ----------
    X : torch.tensor, shape=(n_samples, dim)
        Input values

    y : torch.tensor, shape=(n_samples,)
        Objective values

    objective : callable, argument=torch.tensor
        Objective black-box function, accepting as an argument torch.tensor

    bounds : torch.tensor, shape=(2, dim)
        Box-constraints

    GP : callable
        GP model class constructor. It is a function that takes as input
        2 tensors - X, y - and returns an instance of botorch.models.Model.

    acquisition : callable
        Acquisition function construction. It is a function that receives
        one argument - GP model - and returns an instance of
        botorch.acquisition.AcquisitionFunction

    q : int
        Number of candidate points to find

    state_dict : dict
        GP model state dict

    plot : bool
        Flag indicating whether to plot the result

    Returns
    -------
    X : torch.tensor
        Tensor of input values with new point

    y : torch.tensor
        Tensor of output values with new point

    gp : botorch.models.Model
        Constructed GP model


    Example
    -------
    from botorch.models import FixedNoiseGP
    noise_var = 1e-2 * torch.ones_like(y)
    GP = lambda X, y: FixedNoiseGP(X, y, noise_var)
    acq_func = labmda gp: ExpectedImprovement(gp, y.min(), maximize=False)
    X, y = bo_step(X, y, objective, GP=GP, Acquisition=acq_func)

    """

    yvar = GP_kwargs.get('train_Yvar')

    # Create GP model
    mll, gp = initialize_model(X, y, GP=GP, state_dict=state_dict, *GP_args, **GP_kwargs)
    fit_gpytorch_model(mll)

    # Create acquisition function
    acquisition = acquisition(gp)

    # Optimize acquisition function
    candidate, _ = optimize_acqf(
        acquisition, bounds=bounds, q=q, num_restarts=1, raw_samples=1000,
    )

    # Update data set
    X = torch.cat([X, candidate])
    y = torch.cat([y, objective(candidate).mean(dim=1).reshape((-1, 1))])
    if yvar is not None:
        yvar = torch.cat([GP_kwargs['train_Yvar'], objective(candidate).var(dim=1).reshape((-1, 1))])

    if yvar is not None:
        return X, y, gp, yvar
    else:
        return X, y, gp