from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.fit import fit_gpytorch_model
import torch
import tqdm

from ..acquisition.acquisition import PosteriorVariance
from .bo_step import bo_step, initialize_model


def bo_loop_learn_rho(x, yvar, ymean, cv_function, n_budget_var, bounds):
    """

    :param x:
    :param yvar:
    :param ymean:
    :param cv_function:
    :param n_budget_var:
    :param bounds:
    :param global_vars:
    :return:
    """
    inputs_var = x
    scores_var = yvar
    scores_acquired = ymean
    gp_state_dicts = []

    for iteration in range(1, n_budget_var + 1):

        GP = SingleTaskGP

        acquisition = lambda gp: PosteriorVariance(model=gp)

        # bo_step
        mll, gp_varproxi = initialize_model(inputs_var, scores_var, GP=GP)
        fit_gpytorch_model(mll)

        candidate, _ = optimize_acqf(acquisition(gp_varproxi), bounds=bounds, q=1, num_restarts=1, raw_samples=1000)
        inputs_var = torch.cat([inputs_var, candidate])

        # eval
        eval_result = cv_function(candidate)
        scores_var = torch.cat([scores_var, eval_result.std(dim=1).reshape((-1, 1))])
        scores_acquired = torch.cat([scores_acquired, eval_result.mean(dim=1).reshape((-1, 1))])

        gp_state_dicts.append(gp_varproxi.state_dict())

    return gp_varproxi, inputs_var, scores_var, scores_acquired, gp_state_dicts


def get_observation_with_learnt_rho(x, gp_varproxi, cv_function, gamma):
    """

    :param x:
    :param gp_varproxi:
    :param cv_function:
    :param gamma:
    :return:
    """
    ys = cv_function(x.detach())
    rhos = torch.cat([torch.clamp(gp_varproxi.posterior(x).mean.detach(), min=0.02)] * ys.shape[1], dim=1)

    return ys - gamma * rhos


def evaluate_rho_mean(x, gp_var, gamma, repeat_eval=1):
    """

    :param x: torch.Tensor nmb_ponts x dim, input
    :param gp_var: botorch.model, e.g., SingleTaskGP, used for modeling variance rho^2
    :param gamma: int, d coefficient of absolute risk tolerance in Mean-Variance objective
    :param repeat_eval: int, number of evaluation repeatitions
    :return: torch.Tenzor, nmb_points x 1
    """
    rhos = torch.cat([torch.clamp(gp_var.posterior(x).mean.detach(), min=0.02)] * repeat_eval, dim=1)

    return gamma * rhos


def bo_loop_rahbo_with_learnt_rho(x, y, gp_varproxi, cv_function, bounds, global_vars):
    """

    :param x:
    :param y:
    :param gp_varproxi:
    :param bounds:
    :param global_vars:
    :return:
    """
    inputs = x
    scores = y

    rhos = torch.clamp(gp_varproxi.posterior(inputs).mean.detach(), min=0.02)
    state_dict = None
    yvar = None
    gp = None

    objective = lambda x: get_observation_with_learnt_rho(x, gp_varproxi, cv_function, global_vars['gamma'])

    best_observed = []
    best_observed.append(scores.max())
    gps_state_dicts = []

    with tqdm.tqdm(total=global_vars['n_budget']) as bar:
        for iteration in range(1, global_vars['n_budget'] + 1):
            n_samples = len(scores)

            GP = FixedNoiseGP

            acquisition = lambda gp: UpperConfidenceBound(gp, beta=global_vars['beta'])

            inputs, scores, gp, yvar = bo_step(inputs, scores, objective, bounds,
                                               GP=GP, acquisition=acquisition, q=1,
                                               state_dict=state_dict,
                                               train_Yvar=rhos)
            rhos = torch.clamp(gp_varproxi.posterior(inputs).mean.detach(), min=0.02)
            state_dict = gp.state_dict()
            best_observed.append(scores.max())
            gps_state_dicts.append(gp.state_dict())
            bar.update(len(scores) - n_samples)

    return gp, inputs, scores, yvar, gps_state_dicts
