from torch import Tensor


class BenchmarkBase:
    """
    Base class for benchmark.
    """

    def __init__(self):
        self.repeat_eval = None
        self.seed = None
        self._max_value = None   # if known: max value achievable
        self._optimizers = None  # if known: input at which max is achieved

    def get_domain(self) -> Tensor:
        """
        Function to be implemented by actual benchmark.
        Provides with the dim of optimization domain as well as bounds

        :return: torch.Tensor dim_domain x 2
        """
        raise NotImplementedError

    def evaluate(self, x: Tensor) -> Tensor:
        """
        Function to be implemented by actual benchmark.
        Evaluates objective function for input x.

        :param x: torch.Tensor nmb_points x dim_domain
        :return: torch.Tensor nmb_points x 1
        """
        raise NotImplementedError

    def evaluate_on_test(self, x: Tensor) -> Tensor:
        """
        Function to be implemented by actual benchmark.
        Evaluates test objective function for input x: useful to evaluate BO results in test mode,
        e.g., new dataset or new simulator settings. If training and test modes are the same,
        self.evaluate_on_test should replicate self.evaluate

        :param x: torch.Tensor nmb_points x dim_domain
        :return: torch.Tensor nmb_points x 1
        """
        raise NotImplementedError

    def get_random_initial_points(self,  num_points: int, seed: int) -> Tensor:
        """
        Function to be implemented by actual benchmark.
        Provides with initial points from the domain.
        The points are used for GP model training and strategy for sampling is benchmark specific.

        Example:
        x = draw_sobol_samples(self.get_domain(), num_points, q=1, seed=seed).reshape((-1, 1))

        :param num_points: int, number of points to be generated
        :param seed: int
        :return: torch.Tensor
        """
        raise NotImplementedError

    def get_info_to_dump(self, x: Tensor):
        """
        Function to be implemented by actual benchmark.
        Help function to dump (log) extra information about function evaluation during the optimization.
        For example, true noiseless function values available for synthetic functions.
        This is not used in optimization procedure, but can be useful for postprocessing.

        :param x: torch.Tensor (if info is input dependent)
        :return: dictionary, e.g., 'f': 0.0
        """

        dict_to_dump = {}

        return dict_to_dump




