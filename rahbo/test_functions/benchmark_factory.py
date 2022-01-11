from .benchmarks import BenchmarkBase


def build_benchmark(config) -> BenchmarkBase:
    if config['type'] == 'branin':
        from .branin import NegBraninBenchmark
        return NegBraninBenchmark(config)
    elif config['type'] == 'sine':
        from .sine import SineBenchmark
        return SineBenchmark(config)
    elif config['type'] == 'fel':
        from .fel import FELBenchmark
        return FELBenchmark(config)
    elif config['type'] == 'fraud':
        from .fraud import FraudBenchmark
        return FraudBenchmark(config)
    else:
        raise NotImplementedError

