import pandas as pd


def dump_exp_results(config, inputs, ymean, yvar, test_y, gps_state_dict,
                     extra_benchmark_info: dict = None, extra_method_info: dict = None):

    results = pd.DataFrame()

    results['config'] = [config] * len(inputs)
    results['inputs'] = inputs
    results['restart'] = [config['restart']] * len(inputs)
    results['ymean'] = ymean.reshape((-1,))
    results['yvar'] = yvar.reshape((-1,))
    results['test'] = test_y
    results['gps'] = gps_state_dict
    results['iteration'] = [i for i in range(len(inputs))]
    if extra_benchmark_info is not None:
        results['extra_benchmark_info'] = extra_benchmark_info
    if extra_method_info is not None:
        for key in extra_method_info.keys():
            results[key] = extra_method_info[key]

    results.to_csv(config['fname_csv'])
    results.to_pickle(config['fname_pickle'])