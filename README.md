# Risk-averse-Heteroscedastic-BO
Risk-averse heteroscedastic Bayesian optimization algorithm (RAHBO) aims to identify a solution with high return and low noise variance, while learning the noise distribution on the fly. To this end, we model both expectation and variance as (unknown) RKHS functions, and propose a novel risk-aware acquisition function.

#### This branch was prepared for anonymous submission for NeurIPS 2021.


### Installion in depelop mode:
```console
$ pip install -r requirements.txt
$ pip install -e .
```

### Illustrative ipynb
Check ipynb.rahbo_illustrative.ipynb to see an illustrative example of Risk-averse-Heteroscedastic-BO (RAHBO) applied to sine function that has two global optima with different noise level.

### Running experiments
To run the experiments, configure yaml file (see 'Risk-averse-BO/runner_files/configs/EXAMPLE_config.yaml' as an example), then run 
```console
python runner_files/start_experiment.py --config_path="path_to_yaml" 
```
When running experiments under number of processes constraints (e.g., AutoML tuning), use --max_processes
```console
python runner_files/start_experiment.py --config_path="path_to_yaml" --max_processes=5
```
