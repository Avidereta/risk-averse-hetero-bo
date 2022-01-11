import yaml
import os
from copy import deepcopy


def expand_config(config):
    result = [config]
    expandable_fields = config.get('expand', [])
    for field_path in expandable_fields:
        new_result = []
        for current_config in result:
            new_config = deepcopy(current_config)
            path = field_path[0:-1]
            field_name = field_path[-1]
            subconfig = new_config
            for part in path:
                subconfig = new_config[part]
            expansion_list = deepcopy(subconfig[field_name])
            for expansion in expansion_list:
                subconfig[field_name] = expansion
                new_result.append(deepcopy(new_config))
        result = new_result
    return result


def parse_config(config_path):
    with open(config_path, 'r') as yamlfile:
        config = yaml.safe_load(yamlfile)

    configs = expand_config(config)
    return [preprocess_single_task_config(c) for c in configs]


# TODO: add config check for some vars
def preprocess_single_task_config(config):
    dname = config['dname']
    dname = dname.format(c=config)

    if not os.path.exists(dname):
        os.makedirs(dname)

    fname_csv = os.path.join(dname, "results.csv")
    fname_pickle = os.path.join(dname, "results.p")
    fname_progress = os.path.join(dname, "progress")

    config_extension = {
        'dname_expanded': dname,
        'fname_csv': fname_csv,
        'fname_pickle': fname_pickle,
        'fname_progress': fname_progress,
    }

    config.update(config_extension)

    new_config_path = os.path.join(dname, 'config.yaml')
    with open(new_config_path, 'w') as yamlfile:
        yaml.safe_dump(config, yamlfile)

    return config


# TODO: add config check for some vars
def old_preprocess_config(config_path):

    with open(config_path, 'r') as yamlfile:
        config = yaml.safe_load(yamlfile)

    mname = config['bo_method']['name']
    dname = config['dname']
    if mname == 'ucb':
        dname = os.path.join(dname, mname)
    else:
        dname = os.path.join(dname, mname, f"gamma{config['bo_method']['gamma']}")
    dname_csv = os.path.join(dname, 'csv')
    dname_pickle = os.path.join(dname, 'pickle')
    for d in [dname, dname_csv, dname_pickle]:
        if not os.path.exists(d):
            os.makedirs(d)

    fname_csv = os.path.join(dname_csv, f"{mname}_restart{config['restart']}.csv")
    fname_pickle = os.path.join(dname_pickle, f"{mname}_restart{config['restart']}.p")

    config_extension = {
        'fname_csv': fname_csv,
        'fname_pickle': fname_pickle,
        'dname_csv': dname_csv,
        'dname_pickle': dname_pickle
    }

    config.update(config_extension)

    new_config_path = os.path.join(dname, 'config.yaml')
    with open(new_config_path, 'w') as yamlfile:
        yaml.safe_dump(config, yamlfile)

    return config
