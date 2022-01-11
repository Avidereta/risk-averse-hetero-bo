from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.acquisition import UpperConfidenceBound
import time
import torch
import datetime
from multiprocessing import Process
from rahbo.acquisition.acquisition import RiskAverseUpperConfidenceBound
from rahbo.optimization.bo_step import bo_step_risk_averse
from rahbo.optimization.bo_step import bo_step
from rahbo.optimization.bo_loop import bo_loop_learn_rho, evaluate_rho_mean
from rahbo.test_functions.benchmark_factory import build_benchmark
from runner_utils.preprocess_config import parse_config
from runner_utils.postprocess_results import dump_exp_results
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from copy import deepcopy

import argparse

import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='Start ucb experiment')
parser.add_argument('--config_path', type=str,
                    help='Defines path to config file')
parser.add_argument('--retries_num', type=int, default=4,
                    help='Number of retries on fail')
parser.add_argument('--max_processes', type=int, default=10,
                    help='Maximum number of processes')
parser.add_argument('--dry_run', type=bool, default=False,
                    help='Defines path to config file')


def dump_progress(config, progress, new_file=False):
    with open(config['fname_progress'], 'w' if new_file else 'a') as progress_file:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d, %H:%M:%S')
        progress_file.write(timestamp + ' ::: ' + progress + '\n')


def run_experiment(config, new_progress_file=True):
    dump_progress(config, 'STARTED', new_file=new_progress_file)
    try:
        tracked_test_y = None
        benchmark = build_benchmark(config['benchmark'])
        bounds = benchmark.get_domain()
        objective = benchmark.evaluate
        objective_test = benchmark.evaluate_on_test

        ## initial data generation
        train_x = benchmark.get_random_initial_points(num_points=config['n_initial'],
                                                      seed=config['seed'] + config['restart'])
        train_y_cv = objective(train_x)
        train_yvar = train_y_cv.var(dim=1).reshape((-1, 1))
        train_ymean = train_y_cv.mean(dim=1).reshape((-1, 1))

        ## BO initialization
        inputs = train_x
        ymean = train_ymean
        yvar = train_yvar
        state_dict = None

        # initialize dictionaries used for logging
        extra_benchmark_info = [benchmark.get_info_to_dump(inputs)]*len(inputs)
        gps_state_dict = [None]*len(inputs)
        extra_method_info = None
        if config['bo_method']['name'] in ['rahbo', 'rahbo_us']:
            extra_method_info = dict(gps_var_state_dict=[None]*len(inputs))

        # rahbo_us specific: learn variance (rho) via uncertainty sampling prior to BO for objective
        if config['bo_method']['name'] == 'rahbo_us':
            # learn rho
            gp_var, inputs, yvar, ymean, gp_var_state_dict = bo_loop_learn_rho(inputs,
                                                                               yvar,
                                                                               ymean,
                                                                               objective,
                                                                               config['bo_method']['n_budget_var'],
                                                                               bounds)
            tracked_test_y = objective_test(inputs)
            k = config['bo_method']['n_budget_var']
            extra_method_info['gps_var_state_dict'].extend([gp_var.state_dict()] * k)
            extra_benchmark_info.extend([benchmark.get_info_to_dump(inputs)] * k)
            gps_state_dict.extend([None] * k)
            dump_exp_results(config, inputs, ymean, yvar, tracked_test_y, gps_state_dict,
                             extra_benchmark_info, extra_method_info)

            mv_objective = lambda x: benchmark.evaluate(x) - evaluate_rho_mean(x, gp_var, config['bo_method']['gamma'],
                                                                               config['benchmark']['repeat_eval'])

        ## BO loop
#        with tqdm.tqdm(total=config['n_budget']) as bar:
        for iteration in range(1, config['n_budget'] + 1):
            n_samples = len(ymean)

            GP = FixedNoiseGP

            if config['bo_method']['name'] == 'rahbo':
                acquisition = lambda gp, gp_var: RiskAverseUpperConfidenceBound(gp, gp_var,
                                                                                beta=config['bo_method']['beta'],
                                                                                beta_varproxi=config['bo_method']['beta'],
                                                                                gamma=config['bo_method']['gamma'])

                inputs, ymean, gp, yvar, gp_var = bo_step_risk_averse(inputs, ymean, objective, bounds,
                                                                      GP=GP, acquisition=acquisition, q=1,
                                                                      state_dict=state_dict,
                                                                      train_Yvar=yvar)
                extra_method_info['gps_var_state_dict'].append(gp_var.state_dict())

            elif config['bo_method']['name'] == 'ucb':
                acquisition = lambda gp: UpperConfidenceBound(gp, beta=config['bo_method']['beta'])

                inputs, ymean, gp, yvar = bo_step(inputs, ymean, objective, bounds,
                                                  GP=GP, acquisition=acquisition, q=1,
                                                  state_dict=state_dict,
                                                  train_Yvar=yvar)

            elif config['bo_method']['name'] == 'rahbo_naive':
                acquisition = lambda gp: UpperConfidenceBound(gp, beta=config['bo_method']['beta'])

                GP = SingleTaskGP

                inputs, ymean, gp = bo_step(inputs, ymean, objective, bounds,
                                                  GP=GP, acquisition=acquisition, q=1,
                                                  state_dict=state_dict)

            elif config['bo_method']['name'] == 'rahbo_us':

                # update rho model
                if iteration % 10 == 0:
                    gp_model = SingleTaskGP(inputs, yvar).to(inputs)
                    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
                    gp_model.load_state_dict(gp_var.state_dict())
                    fit_gpytorch_model(mll)
                    gp_var = deepcopy(gp_model)

                    mv_objective = lambda x: benchmark.evaluate(x) - evaluate_rho_mean(x, gp_var,
                                                                                       config['bo_method']['gamma'],
                                                                                       config['benchmark'][
                                                                                           'repeat_eval'])

                acquisition = lambda gp: UpperConfidenceBound(gp, beta=config['bo_method']['beta'])

                inputs, ymean, gp, yvar = bo_step(inputs, ymean, mv_objective, bounds,
                                                  GP=GP, acquisition=acquisition, q=1,
                                                  state_dict=state_dict,
                                                  train_Yvar=yvar)

                extra_method_info['gps_var_state_dict'].append(gp_var.state_dict())

            if config['bo_method']['name'] != 'random_search':
                state_dict = gp.state_dict()
                gps_state_dict.append(state_dict)
                dump_progress(config, str(config['n_budget'] + config['n_initial'] - n_samples))
                extra_benchmark_info.append(benchmark.get_info_to_dump(inputs))

            if iteration % config['iter_save'] == 0:
                if tracked_test_y is None:
                    tracked_test_y = objective_test(inputs)
                    # print('start: tracked_test_y', tracked_test_y.shape)
                else:
                    test_y_tmp = objective_test(inputs[-config['iter_save']:])
                    # print ('start', tracked_test_y.shape, test_y_tmp.shape)
                    tracked_test_y = torch.cat([tracked_test_y, test_y_tmp], dim=0)
                    # print('start: tracked_test_y', tracked_test_y.shape)
                # print (f'start: extra_method_info {extra_method_info}')
                dump_exp_results(config, inputs, ymean, yvar, tracked_test_y, gps_state_dict,
                                 extra_benchmark_info, extra_method_info)

    except Exception as e:
        dump_progress(config, 'FAILED')
        raise e
    else:
        dump_progress(config, 'FINISHED')


def start_processes(processes, step, num_steps):
    print('Starting processes')
    for p, _, _, _, _ in processes:
        p.start()
    alive = True
    while alive:
        alive = False
        time.sleep(10)
        print(f'RUNNING STEP {step} OUT OF {num_steps}')
        for i, (p, dname, fname_progress, c, attempt) in enumerate(processes):
            with open(fname_progress, 'r') as progress:
                status = progress.readlines()[-1]
                print(f'[ {attempt} ] {dname} : {status}')
            if p.is_alive():
                alive = True
            elif 'FAILED' in status:
                if attempt < args.retries_num:
                    p.join()
                    p = Process(target=run_experiment, args=(c, False))
                    attempt += 1
                    processes[i] = p, c['dname_expanded'], c['fname_progress'], c, attempt
                    p.start()

    print('Waiting for all processes to stop')
    for p, _, _, _, _ in processes:
        p.join()


if __name__ == "__main__":
    args = parser.parse_args()

    configs = parse_config(args.config_path)

    # create list of processes based on config 'expand' to launch in queue
    all_processes = []
    for i, c in enumerate(configs):
        if not args.dry_run:
            print(f'Creating process {i}: ', c['dname_expanded'])
            p = Process(target=run_experiment, args=(c,))
            all_processes.append((p, c['dname_expanded'], c['fname_progress'], c, 0))
        else:
            print('Would have started for dname:', c['dname_expanded'])
    # in loop choose max_processes (e.g., 5) processes from list and launch
    step = 0
    while step < len(all_processes):
        begin = int(step * args.max_processes)
        end = min(begin+args.max_processes, len(all_processes))
        start_processes(all_processes[begin:end], step, int((len(all_processes) + args.max_processes - 1) / args.max_processes))
        step += 1









