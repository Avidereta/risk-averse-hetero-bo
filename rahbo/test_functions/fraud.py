from .benchmarks import BenchmarkBase
import torch
from torch import Tensor
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,accuracy_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize


def _generate_validation_splits(ds, num_splits):
    splits = []
    for i in range(num_splits):
        time_lower_bound = np.percentile(ds['Time'].values, i * 100 / num_splits)
        time_upper_bound = np.percentile(ds['Time'].values, (i+1) * 100 / num_splits)
        splits.append(ds[(time_lower_bound <= ds.Time) & (ds.Time < time_upper_bound)])
    return splits


def _get_X_y_from_ds(ds):
    feature_columns = [f'V{i+1}' for i in range(28)] + ['Amount']
    X = ds[feature_columns].values
    y = ds['Class'].values
    return X, y


def _get_split_results(clf, splits, scoring):
    eval_results = []
    for val_ds in splits:
        X_val, y_val = _get_X_y_from_ds(val_ds)
        y_pred = clf.predict(X_val)
        eval_results.append(scoring(y_val, y_pred))
    return eval_results


class FraudBenchmark(BenchmarkBase):
    def __init__(self, config):
        super(FraudBenchmark, self).__init__()

        self.cv_splits = config['repeat_eval']
        self.space = config['space']

        fraud_ds = pd.read_csv('data/creditcard.csv')
        val_time_threshold = np.percentile(fraud_ds['Time'].values, 50)
        test_time_threshold = np.percentile(fraud_ds['Time'].values, 75)

        self.train_ds = fraud_ds[fraud_ds.Time < val_time_threshold]
        self.val_ds = fraud_ds[(val_time_threshold <= fraud_ds.Time) &
                               (fraud_ds.Time <= test_time_threshold)]
        self.test_ds = fraud_ds[fraud_ds.Time > test_time_threshold]
        self.train_val_ds = fraud_ds[fraud_ds.Time < test_time_threshold]

        self.validation_splits = _generate_validation_splits(self.val_ds, self.cv_splits)
        self.test_splits = _generate_validation_splits(self.test_ds, self.cv_splits)

    def get_domain(self):
        bounds_01 = torch.zeros(2, len(self.space), dtype=torch.float64)
        bounds_01[1] = 1
        return bounds_01

    def get_random_initial_points(self, num_points, seed) -> Tensor:
        x = draw_sobol_samples(self.get_domain(), num_points, q=1, seed=seed).squeeze()

        return x

    def evaluate(self, x=None):
        # print('@@Evaluate, x=', x)
        return self._get_evaluations(self.train_ds, self.validation_splits, x, self.space)

    def evaluate_on_test(self, x=None):
        return self._get_evaluations(self.train_val_ds, self.test_splits, x, self.space)

    def get_info_to_dump(self, x):

        bounds = torch.Tensor([var['domain'] for var in self.space]).to(x).t()
        parameters = self._wrap_x(unnormalize(x, bounds), self.space)
        dict_to_dump = {
            'inputs_param': parameters
        }

        return dict_to_dump

    @staticmethod
    def _get_evaluations(train_ds, eval_splits, parameters, space, ml_model=RandomForestClassifier):
        # Rescale to original bounds
        dtype = parameters.dtype
        device = parameters.device
        bounds = torch.tensor([var['domain'] for var in space]).to(parameters).t()
        parameters = unnormalize(parameters, bounds)

        # Convert tensor to a list of dictionaries
        parameters = FraudBenchmark._wrap_x(parameters, space)

        def _get_score(params_dict):
            model = ml_model(n_jobs=20, **params_dict)
            X, y = _get_X_y_from_ds(train_ds)
            model.fit(X, y)
            scores = _get_split_results(model, eval_splits, balanced_accuracy_score)
            return np.array(scores) - 1

        #     score_list = Parallel(-1)(delayed(_get_score)(p) for p in parameters)
        score_list = [_get_score(p) for p in parameters]

        return torch.tensor(score_list, dtype=dtype, device=device)

    @staticmethod
    def _wrap_x(x, space):
        """
        Wrap tensor to a list of dictionaries
        """

        def _wrap_row(row):
            wrapped_row = {}
            for i, x_ in enumerate(row):
                wrapped_row[space[i]['name']] = x_.item()

                if space[i]['type'] == 'discrete':
                    wrapped_row[space[i]['name']] = int(np.round(x_.item()))
            return wrapped_row

        wrapped_x = []
        for i in range(x.shape[0]):
            wrapped_x.append(_wrap_row(x[i]))

        return wrapped_x

    @staticmethod
    def _unwrap_x(parameters, space):
        """
        Unwrap list of dictionaries to torch.tensor
        """
        x = torch.zeros(len(parameters), len(space),
                        dtype=torch.float64)
        for i, p in enumerate(parameters):
            x_ = [p[var['name']] for var in space]
            x[i] = torch.tensor(x_, dtype=torch.float64)

        return x

