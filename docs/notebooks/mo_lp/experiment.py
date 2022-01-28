from typing import Callable
import tensorflow as tf
import gpflow
import numpy as np
import time
import pathlib
from dataclasses import dataclass, field


from trieste.acquisition.interface import AcquisitionFunction

from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.interfaces import TrainablePredictJointReparamModelStack
from trieste.objectives.utils import mk_observer
from trieste.observer import OBJECTIVE

from trieste.acquisition import BatchMonteCarloExpectedHypervolumeImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.bayesian_optimizer import BayesianOptimizer

from trieste.acquisition.multi_objective.pareto import Pareto, get_reference_point


from test_functions import TestFunction, get_test_function
from generate_true_pareto_fronts import read_true_pf
from mo_penalization import MOLocalPenalizationAcquisitionFunction


def get_acquisition_function(name):
    if name == "BatchMC":
        return BatchMonteCarloExpectedHypervolumeImprovement(sample_size=100).using(OBJECTIVE)
    elif name == "DistanceBased":
        return MOLocalPenalizationAcquisitionFunction().using(OBJECTIVE)
    else:
        raise ValueError(f"Unknown method {name}")


@dataclass
class Config:
    acquisition_method_name: str
    test_function_name: str
    test_function: Callable = field(init=False)
    n_initial_points: int = 3
    n_query_points: int = 4
    n_optimization_steps: int = 3
    n_repeats: int = 5
    seed: int = None

    def __post_init__(self):
        # it's ok to create it once as re-use
        # because test functions are supposed to be stateless
        self.test_function = get_test_function(self.test_function_name)

    def create_acquisition_function(self):
        # acquisition functions cn be stateful
        # so we need to re-create it each time
        return get_acquisition_function(self.acquisition_method_name)

    def get_filename(self):
        return f"{self.acquisition_method_name}_" \
               f"{self.test_function_name}_" \
               f"n_initial_points_{self.n_initial_points}_" \
               f"n_query_points_{self.n_query_points}_" \
               f"n_optimization_steps_{self.n_optimization_steps}_" \
               f"n_repeats_{self.n_repeats}_" \
               f"seed_{self.seed}"

    @classmethod
    def from_dict(cls, args):
        config = Config(**args)
        return config


def build_stacked_independent_objectives_model(data, n_obj):
    gprs = []
    for idx in range(n_obj):
        single_obj_data = Dataset(
            data.query_points, tf.gather(data.observations, [idx], axis=1)
        )
        variance = tf.math.reduce_variance(single_obj_data.observations)
        kernel = gpflow.kernels.Matern52(variance, tf.constant(0.2, tf.float64))
        gpr = gpflow.models.GPR(single_obj_data.astuple(), kernel, noise_variance=1e-5)
        gpflow.utilities.set_trainable(gpr.likelihood, False)
        gprs.append((GaussianProcessRegression(gpr), 1))

    return TrainablePredictJointReparamModelStack(*gprs)


def get_hv_regret(true_points, observed_points, num_initial_points):
    ref_point = get_reference_point(observed_points)
    ideal_hv = Pareto(true_points).hypervolume_indicator(ref_point)

    hv_regret = []
    for i in range(num_initial_points, len(observed_points)+1):
        observations = observed_points[:i, :]
        observed_hv = Pareto(observations).hypervolume_indicator(ref_point)

        hv_regret.append((ideal_hv - observed_hv).numpy())
    
    return hv_regret


def single_run(config: Config, save_to_file=False):
    print(f"Running {config.acquisition_method_name} on {config.test_function_name}")

    if config.seed is not None:
        np.random.seed(config.seed)
        tf.random.set_seed(config.seed)

    test_function: TestFunction = config.test_function
    true_pf = read_true_pf(test_function.true_pf_filename)
    observer = mk_observer(test_function, OBJECTIVE)

    hv_regret = []
    for i in range(config.n_repeats):
        print(f"Repeat #{i}")
        initial_query_points = test_function.search_space.sample(config.n_initial_points)
        initial_data = observer(initial_query_points)

        model = build_stacked_independent_objectives_model(initial_data[OBJECTIVE], test_function.n_objectives)
        acq_fn = config.create_acquisition_function()
        acq_rule = EfficientGlobalOptimization(acq_fn, num_query_points=config.n_query_points)

        start = time.time()
        result = BayesianOptimizer(observer, test_function.search_space).optimize(config.n_optimization_steps,
                                                                                initial_data,
                                                                                {OBJECTIVE: model},
                                                                                acq_rule)
        stop = time.time()
        print(f"Finished in {stop - start}s")

        dataset = result.try_get_final_datasets()[OBJECTIVE]
        hv_regret.append(get_hv_regret(true_pf, dataset.observations, config.n_initial_points))

    if save_to_file:
        current_dir = pathlib.Path(__file__).parent
        file_path = current_dir.joinpath("results", config.get_filename()).resolve()
        np.savetxt(str(file_path), hv_regret, delimiter=",")
        print(f"Saved results to {file_path}")

    return hv_regret

