import tensorflow as tf
import gpflow
import numpy as np
import time

from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.interfaces import ModelStack
from trieste.objectives.utils import mk_observer
from trieste.observer import OBJECTIVE

from trieste.acquisition import BatchMonteCarloExpectedHypervolumeImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.bayesian_optimizer import BayesianOptimizer

from trieste.acquisition.multi_objective.pareto import Pareto, get_reference_point


from mo_lp.mo_penalization import MOLocalPenalizationAcquisitionFunction

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

    return ModelStack(*gprs)


def run_experiment(f, n_obj, search_space, num_initial_points, num_steps, num_query_points):
    observer = mk_observer(f, OBJECTIVE)

    initial_query_points = search_space.sample(num_initial_points)
    initial_data = observer(initial_query_points)

    print(f"Running Batch MC with batch size {num_query_points} for {num_steps} iterations")
    model = build_stacked_independent_objectives_model(initial_data[OBJECTIVE], n_obj)
    acq_function = BatchMonteCarloExpectedHypervolumeImprovement(sample_size=250).using(OBJECTIVE)
    acq_rule = EfficientGlobalOptimization(acq_function, num_query_points=num_query_points)

    start = time.time()
    result = BayesianOptimizer(observer, search_space).optimize(num_steps, initial_data, {OBJECTIVE: model}, acq_rule)
    stop = time.time()

    batch_mc_dataset = result.try_get_final_datasets()[OBJECTIVE]
    print(f"Batch MC finished in {stop - start}s")

    print(f"Running MO LP with batch size {num_query_points} for {num_steps} iterations")
    model = build_stacked_independent_objectives_model(initial_data[OBJECTIVE], n_obj)
    acq_function = MOLocalPenalizationAcquisitionFunction().using(OBJECTIVE)
    acq_rule = EfficientGlobalOptimization(acq_function, num_query_points=num_query_points)
    
    start = time.time()
    result = BayesianOptimizer(observer, search_space).optimize(num_steps, initial_data, {OBJECTIVE: model}, acq_rule)
    stop = time.time()

    mo_lp_dataset = result.try_get_final_datasets()[OBJECTIVE]
    print(f"MO LP finished in {stop - start}s")


    return batch_mc_dataset.observations, mo_lp_dataset.observations


def get_hv_regret(true_points, observed_points, num_initial_points):
    ref_point = get_reference_point(observed_points)
    ideal_hv = Pareto(true_points).hypervolume_indicator(ref_point)

    hv_regret = []
    for i in range(num_initial_points, len(observed_points)+1):
        observations = observed_points[:i, :]
        observed_hv = Pareto(observations).hypervolume_indicator(ref_point)

        hv_regret.append((ideal_hv - observed_hv).numpy())
    
    return hv_regret
