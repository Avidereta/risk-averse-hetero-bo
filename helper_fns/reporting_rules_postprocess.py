from botorch.models import FixedNoiseGP, SingleTaskGP
from rabo.acquisition.acquisition import RiskAverseUpperConfidenceBound, UpperConfidenceBound, LowerConfidenceBound
import torch
from torch import Tensor
from rabo.optimization.bo_loop import bo_loop_learn_rho, evaluate_rho_mean


def transform_x_to_tensor(df_values) -> Tensor:
    """
    Transforms dumped values from df to tensor format n_points x dimention
    :param df_values:
    :return:
    """
    return torch.vstack([torch.hstack([x_i for x_i in x]) for x in df_values])


def report_idx_max_ymean(resdf):
    """
    Returns list of iterations numbers where each corresponds to the best ymean observed so far
    :param resdf: pd.DataFrame with column 'ymean'
    :return: list
    """
    best_idx = [resdf['ymean'].values[:i].argmax() for i in range(1, len(resdf['ymean']) + 1)]
    return best_idx

def report_idx_max_mv(resdf, gamma):
    """
    Returns list of iterations numbers where each corresponds to the best ymean observed so far
    :param resdf: pd.DataFrame with column 'ymean'
    :return: list
    """
    mv = resdf['ymean'].values - gamma*resdf['yvar'].values

    best_idx = [mv[:i].argmax() for i in range(1, len(resdf['ymean']) + 1)]
    return best_idx


def report_idx_max_last_lcb_rahbo(resdf, t=None, dt=10):
    lcb_f = []
    idxs = []
    n_initial = resdf.iloc[0].config['n_initial']
    if resdf.iloc[0].config['bo_method']['name'] == 'rahbo_us':
        n_initial += resdf.iloc[0].config['bo_method']['n_budget_var']
    if t is None:
        for t in range(n_initial, len(resdf), dt):
            if t % 10 == 0:
                print (f'report_idx_max_last_lcb_rahbo, t={t}')
            train_y = torch.Tensor(resdf['ymean'].values[:t-1]).reshape((-1, 1))
            train_x = transform_x_to_tensor(resdf['inputs'].values[:t - 1])
            train_yvar = torch.Tensor(resdf['yvar'].values[:t - 1]).reshape((-1, 1))
            gp_var_state_dict = resdf['gps_var_state_dict'][t]
            gp_state_dict = resdf['gps'][t]

            gp = FixedNoiseGP(train_x, train_y, train_yvar)
            gp_var = SingleTaskGP(train_x, train_yvar)
            gp.load_state_dict(gp_state_dict)
            gp_var.load_state_dict(gp_var_state_dict)

            beta = resdf.iloc[0].config['bo_method']['beta']
            gamma = resdf.iloc[0].config['bo_method']['gamma']

            ralcb_function = RiskAverseUpperConfidenceBound(gp, gp_var, beta, beta, gamma, maximize=False)
            lcbs = ralcb_function.forward(train_x.reshape((len(train_x), 1, -1)))
            idxs.extend([int(lcbs.argmax())] * dt)
            lcb_f.extend([lcbs.max()] * dt)
    else:
        train_y = torch.Tensor(resdf['ymean'].values[:t - 1]).reshape((-1, 1))
        train_x = transform_x_to_tensor(resdf['inputs'].values[:t - 1])
        train_yvar = torch.Tensor(resdf['yvar'].values[:t - 1]).reshape((-1, 1))
        gp_var_state_dict = resdf['gps_var_state_dict'][t]
        gp_state_dict = resdf['gps'][t]

        gp = FixedNoiseGP(train_x, train_y, train_yvar)
        gp_var = SingleTaskGP(train_x, train_yvar)
        gp.load_state_dict(gp_state_dict)
        gp_var.load_state_dict(gp_var_state_dict)

        beta = resdf.iloc[0].config['bo_method']['beta']
        gamma = resdf.iloc[0].config['bo_method']['gamma']

        ralcb_function = RiskAverseUpperConfidenceBound(gp, gp_var, beta, beta, gamma, maximize=False)
        lcbs = ralcb_function.forward(train_x.reshape((len(train_x), 1, -1)))
        idxs.append(int(lcbs.argmax()))
        lcb_f.append(lcbs.max())

    idx_all = [i for i in range(n_initial)]
    idx_all.extend(idxs)
    return idx_all, lcb_f



def report_idx_max_last_lcb_rahbous(resdf, t=None, dt=10):
    """
    Returns list of iterations numbers where each corresponds to the max lcb for the model at this iterations computed
    for the points observed so far
    :param resdf: pd.DataFrame with column 'ymean'
    :return: list
    """

    lcb_f = []
    idxs = []
    n_initial = resdf.iloc[0].config['n_initial']
    n_initial += resdf.iloc[0].config['bo_method']['n_budget_var']
    if t is None:
        for t in range(n_initial, len(resdf), dt):
            if t % 10 == 0:
                print(f'report_idx_max_last_lcb_rahbo, t={t}')
            train_y = torch.Tensor(resdf['ymean'].values[:t - 1]).reshape((-1, 1))
            train_x = transform_x_to_tensor(resdf['inputs'].values[:t - 1])
            train_yvar = torch.Tensor(resdf['yvar'].values[:t - 1]).reshape((-1, 1))
            gp_var_state_dict = resdf['gps_var_state_dict'][t]
            gp_state_dict = resdf['gps'][t]

            gp = FixedNoiseGP(train_x, train_y, train_yvar)
            gp_var = SingleTaskGP(train_x, train_yvar)
            gp.load_state_dict(gp_state_dict)
            gp_var.load_state_dict(gp_var_state_dict)

            beta = resdf.iloc[0].config['bo_method']['beta']
            gamma = resdf.iloc[0].config['bo_method']['gamma']

            rho_mean = lambda x: evaluate_rho_mean(x, gp_var, gamma, 1)

            lcb_function = LowerConfidenceBound(gp, beta, maximize=False)
            x = train_x.reshape((len(train_x), 1, -1))
            lcbs = lcb_function.forward(x) - rho_mean(x)
            idxs.extend([int(lcbs.argmax())] * dt)
            lcb_f.extend([lcbs.max()] * dt)
    else:
        train_y = torch.Tensor(resdf['ymean'].values[:t - 1]).reshape((-1, 1))
        train_x = transform_x_to_tensor(resdf['inputs'].values[:t - 1])
        train_yvar = torch.Tensor(resdf['yvar'].values[:t - 1]).reshape((-1, 1))
        gp_var_state_dict = resdf['gps_var_state_dict'][t]
        gp_state_dict = resdf['gps'][t]

        gp = FixedNoiseGP(train_x, train_y, train_yvar)
        gp_var = SingleTaskGP(train_x, train_yvar)
        gp.load_state_dict(gp_state_dict)
        gp_var.load_state_dict(gp_var_state_dict)

        beta = resdf.iloc[0].config['bo_method']['beta']
        gamma = resdf.iloc[0].config['bo_method']['gamma']

        rho_mean = lambda x: evaluate_rho_mean(x, gp_var, gamma, 1)

        lcb_function = LowerConfidenceBound(gp, beta, maximize=False)
        x = train_x.reshape((len(train_x), 1, -1))
        lcbs = lcb_function.forward(x) - rho_mean(x)
        idxs.append(int(lcbs.argmax()))
        lcb_f.append(lcbs.max())

    idx_all = [i for i in range(n_initial)]
    idx_all.extend(idxs)
    return idx_all, lcb_f


def report_idx_max_last_lcb_ucb(resdf, t=None, dt=10):
    """
    Returns list of iterations numbers where each corresponds to the max lcb for the model at this iterations computed
    for the points observed so far
    :param resdf: pd.DataFrame with column 'ymean'
    :return: list
    """

    lcb_f = []
    idxs = []
    n_initial = resdf.iloc[0].config['n_initial']
    if t is None:
        for t in range(n_initial, len(resdf), dt):
            if t % 10 == 0:
                print(f'report_idx_max_last_lcb_ucb, t={t}')
            train_y = torch.Tensor(resdf['ymean'].values[:t - 1]).reshape((-1, 1))
            train_x = transform_x_to_tensor(resdf['inputs'].values[:t - 1])
            train_yvar = torch.Tensor(resdf['yvar'].values[:t - 1]).reshape((-1, 1))
            gp_state_dict = resdf['gps'][t]

            gp = FixedNoiseGP(train_x, train_y, train_yvar)
            gp.load_state_dict(gp_state_dict)

            beta = resdf.iloc[0].config['bo_method']['beta']

            lcb_function = LowerConfidenceBound(gp, beta, maximize=False)
            lcbs = lcb_function.forward(train_x.reshape((len(train_x), 1, -1)))
            idxs.extend([int(lcbs.argmax())] * dt)
            lcb_f.extend([lcbs.max()] * dt)
    else:
        train_y = torch.Tensor(resdf['ymean'].values[:t - 1]).reshape((-1, 1))
        train_x = transform_x_to_tensor(resdf['inputs'].values[:t - 1])
        train_yvar = torch.Tensor(resdf['yvar'].values[:t - 1]).reshape((-1, 1))
        gp_state_dict = resdf['gps'][t]

        gp = FixedNoiseGP(train_x, train_y, train_yvar)
        gp.load_state_dict(gp_state_dict)

        beta = resdf.iloc[0].config['bo_method']['beta']

        lcb_function = LowerConfidenceBound(gp, beta, maximize=False)
        lcbs = lcb_function.forward(train_x.reshape((len(train_x), 1, -1)))
        idxs.append(int(lcbs.argmax()))
        lcb_f.append(lcbs.max())

    idx_all = [i for i in range(n_initial)]
    idx_all.extend(idxs)
    return idx_all, lcb_f