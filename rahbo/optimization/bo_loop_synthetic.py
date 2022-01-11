from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.fit import fit_gpytorch_model
import torch
import tqdm

from ..acquisition.acquisition import PosteriorVariance
from ..test_functions.sine import get_observations
from .bo_step import bo_step, initialize_model


def bo_loop_learn_rho(x, yvar, ymean, n_budget_var, bounds, global_vars):
    """

    :param x:
    :param yvar:
    :param ymean:
    :param n_budget_var:
    :param bounds:
    :param global_vars:
    :return:
    """
    inputs_var = x
    scores_var = yvar
    scores_acquired = ymean
    state_dict = None

    objective = lambda x: get_observations(x, sigma=global_vars['sigma'], repeat_each=global_vars['repeat'])

    for iteration in range(1, n_budget_var + 1):

        GP = SingleTaskGP

        acquisition = lambda gp: PosteriorVariance(model=gp)

        # bo_step
        mll, gp_varproxi = initialize_model(inputs_var, scores_var, GP=GP, state_dict=state_dict)
        fit_gpytorch_model(mll)

        candidate, _ = optimize_acqf(acquisition(gp_varproxi), bounds=bounds, q=1, num_restarts=1, raw_samples=1000)
        inputs_var = torch.cat([inputs_var, candidate])

        # eval
        eval_result = objective(candidate)
        scores_var = torch.cat([scores_var, eval_result[5].mean(dim=1).reshape((-1, 1))])
        scores_acquired = torch.cat([scores_acquired, eval_result[4].mean(dim=1).reshape((-1, 1))])

        state_dict = gp_varproxi.state_dict()

    return gp_varproxi, inputs_var, scores_var, scores_acquired


def get_observation_with_learnt_rho(x, gp_varproxi, global_vars):
    """

    :param x:
    :param gp_varproxi:
    :param global_vars:
    :return:
    """
    ys = get_observations(x.detach(), sigma=global_vars['sigma'], repeat_each=global_vars['repeat'])[4]
    rhos = torch.cat([torch.clamp(gp_varproxi.posterior(x).mean.detach(), min=0)] * global_vars['repeat'], dim=1)

    return ys - global_vars['gamma'] * rhos


def bo_loop_rahbo_with_learnt_rho(x, y, gp_varproxi, bounds, global_vars):
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

    rhos = torch.clamp(gp_varproxi.posterior(inputs).mean.detach(), min=0)
    state_dict = None

    objective = lambda x: get_observation_with_learnt_rho(x, gp_varproxi, global_vars)

    best_observed = []
    best_observed.append(scores.max())

    with tqdm.tqdm(total=global_vars['n_budget']) as bar:
        for iteration in range(1, global_vars['n_budget'] + 1):
            n_samples = len(scores)

            GP = FixedNoiseGP

            acquisition = lambda gp: UpperConfidenceBound(gp, beta=global_vars['beta'])

            inputs, scores, gp, yvar = bo_step(inputs, scores, objective, bounds,
                                               GP=GP, acquisition=acquisition, q=1,
                                               state_dict=state_dict,
                                               train_Yvar=rhos)
            rhos = torch.clamp(gp_varproxi.posterior(inputs).mean.detach(), min=0)
            state_dict = gp.state_dict()
            best_observed.append(scores.max())

            bar.update(len(scores) - n_samples)

    return gp, inputs, scores, yvar
