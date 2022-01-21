from typing import Callable
import tensorflow as tf
import gpflow
import numpy as np
import time
from dataclasses import dataclass, field


from trieste.acquisition.interface import AcquisitionFunction

from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.interfaces import TrainableModelStack
from trieste.objectives.utils import mk_observer
from trieste.observer import OBJECTIVE

from trieste.acquisition import BatchMonteCarloExpectedHypervolumeImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.bayesian_optimizer import BayesianOptimizer

from trieste.acquisition.multi_objective.pareto import Pareto, get_reference_point


from mo_lp.test_functions import TestFunction, get_test_function
from mo_lp.generate_true_pareto_fronts import read_true_pf
from mo_lp.mo_penalization import MOLocalPenalizationAcquisitionFunction


def get_acquisition_function(name):
    if name == "BatchMC":
        return BatchMonteCarloExpectedHypervolumeImprovement(sample_size=250).using(OBJECTIVE)
    elif name == "DistanceBased":
        return MOLocalPenalizationAcquisitionFunction().using(OBJECTIVE)
    else:
        raise ValueError(f"Unknown method {name}")


@dataclass
class Config:
    acquisition_method_name: str
    test_function_name: str
    acquisition_function: AcquisitionFunction = field(init=False)
    test_function: Callable = field(init=False)
    n_initial_points: int = 3
    n_query_points: int = 4
    n_optimization_steps: int = 3
    n_repeats: int = 5

    def __post_init__(self):
        self.test_function = get_test_function(self.test_function_name)
        self.acquisition_function = get_acquisition_function(self.acquisition_method_name)

    @classmethod
    def from_dict(args):
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

    return TrainableModelStack(*gprs)


def get_hv_regret(true_points, observed_points, num_initial_points):
    ref_point = get_reference_point(observed_points)
    ideal_hv = Pareto(true_points).hypervolume_indicator(ref_point)

    hv_regret = []
    for i in range(num_initial_points, len(observed_points)+1):
        observations = observed_points[:i, :]
        observed_hv = Pareto(observations).hypervolume_indicator(ref_point)

        hv_regret.append((ideal_hv - observed_hv).numpy())
    
    return hv_regret


def single_run(config: Config):
    test_function: TestFunction = config.test_function
    observer = mk_observer(test_function, OBJECTIVE)
    true_pf = read_true_pf(test_function.true_pf_filename)

    hv_regret = []
    for _ in range(config.n_repeats):
        initial_query_points = test_function.search_space.sample(config.n_initial_points)
        initial_data = observer(initial_query_points)

        model = build_stacked_independent_objectives_model(initial_data[OBJECTIVE], test_function.n_objectives)
        acq_rule = EfficientGlobalOptimization(config.acquisition_function, num_query_points=config.n_query_points)

        print(f"Running {config.method_name} with batch size {config.n_query_points} for {config.n_optimization_steps} iterations")
        start = time.time()
        result = BayesianOptimizer(observer, test_function.search_space).optimize(config.n_optimization_steps,
                                                                                initial_data,
                                                                                {OBJECTIVE: model},
                                                                                acq_rule)
        stop = time.time()
        print(f"Finished in {stop - start}s")

        dataset = result.try_get_final_datasets()[OBJECTIVE]
        hv_regret.append(get_hv_regret(true_pf, dataset.observations, config.n_initial_points))
    
    return hv_regret


# def run_experiment(f, n_obj, search_space, num_initial_points, num_steps, num_query_points):
#     observer = mk_observer(f, OBJECTIVE)

#     initial_query_points = search_space.sample(num_initial_points)
#     initial_data = observer(initial_query_points)

#     print(f"Running Batch MC with batch size {num_query_points} for {num_steps} iterations")
#     model = build_stacked_independent_objectives_model(initial_data[OBJECTIVE], n_obj)
#     acq_function = BatchMonteCarloExpectedHypervolumeImprovement(sample_size=250).using(OBJECTIVE)
#     acq_rule = EfficientGlobalOptimization(acq_function, num_query_points=num_query_points)

#     start = time.time()
#     result = BayesianOptimizer(observer, search_space).optimize(num_steps, initial_data, {OBJECTIVE: model}, acq_rule)
#     stop = time.time()

#     batch_mc_dataset = result.try_get_final_datasets()[OBJECTIVE]
#     print(f"Batch MC finished in {stop - start}s")

#     print(f"Running MO LP with batch size {num_query_points} for {num_steps} iterations")
#     model = build_stacked_independent_objectives_model(initial_data[OBJECTIVE], n_obj)
#     acq_function = MOLocalPenalizationAcquisitionFunction().using(OBJECTIVE)
#     acq_rule = EfficientGlobalOptimization(acq_function, num_query_points=num_query_points)
    
#     start = time.time()
#     result = BayesianOptimizer(observer, search_space).optimize(num_steps, initial_data, {OBJECTIVE: model}, acq_rule)
#     stop = time.time()

#     mo_lp_dataset = result.try_get_final_datasets()[OBJECTIVE]
#     print(f"MO LP finished in {stop - start}s")


#     return batch_mc_dataset.observations, mo_lp_dataset.observations









