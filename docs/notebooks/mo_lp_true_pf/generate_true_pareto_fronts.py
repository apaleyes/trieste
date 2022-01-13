import tensorflow as tf
import numpy as np
import trieste
import math

from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2


class PymooProblem(Problem):
    def __init__(self, search_space, f, n_obj):
        n_var = int(search_space.dimension)
        xl = search_space.lower.numpy()
        xu = search_space.upper.numpy()
        
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)
        
        self._f = f
        
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self._f(x)


def get_true_pf(problem, n_gen=1000):
    algorithm = NSGA2(pop_size=100)
    res = minimize(problem, algorithm, ('n_gen', n_gen), seed=1, verbose=False)

    return res.F


def save_true_pf(true_pf, filename):
    np.savetxt(filename, true_pf, delimiter=",")


def read_true_pf(filename):
    return np.genfromtxt(filename, delimiter=',')


############################
# Simple single input space case
############################
SIMPLE_1D_INPUT_FILENAME = "simple_1d_input.csv"

def save_simple_1d_input():
    search_space = trieste.space.Box([0], [2*math.pi])

    def f1(x):
        return tf.cos(2 * x) + tf.sin(x)

    def f2(x):
        return 0.2 * (tf.cos(x) - tf.sin(x)) + 0.3

    def f(x):
        return tf.concat([f1(x), f2(x)], axis=-1)

    print("Generating true Pareto front for simple 1d case")
    problem = PymooProblem(search_space, f, n_obj=2)
    true_pf = get_true_pf(problem)
    save_true_pf(true_pf, SIMPLE_1D_INPUT_FILENAME)
    print("Saved to " + SIMPLE_1D_INPUT_FILENAME)


############################
# 2d input case from Gardner et al. 2014
############################
GARDNER_2D_INPUT_FILENAME = "gardner_2d_input.csv"

def save_gardner_2d_input():
    search_space = trieste.space.Box([0, 0], [2*math.pi, 2*math.pi])

    def f1(input_data):
        x, y = input_data[..., -2], input_data[..., -1]
        z = tf.cos(2.0 * x) * tf.cos(y) + tf.sin(x)
        return z[:, None]

    def f2(input_data):
        x, y = input_data[:, -2], input_data[:, -1]
        # changes are made so that the function is between 0 and 1
        z = 1.0 - (tf.cos(x) * tf.cos(y) - tf.sin(x) * tf.sin(y) + 1.0) / 2.0
        return z[:, None]

    def f(x):
        return tf.concat([f1(x), f2(x)], axis=-1)

    print("Generating true Pareto front for functions taken from Gardner et al. 2014")
    problem = PymooProblem(search_space, f, n_obj=2)
    true_pf = get_true_pf(problem)
    save_true_pf(true_pf, GARDNER_2D_INPUT_FILENAME)
    print("Saved to " + GARDNER_2D_INPUT_FILENAME)


############################
# ZDT3
############################
ZDT3_FILENAME = "zdt3_2d_input.csv"

def save_zdt3():
    search_space = trieste.space.Box([0, 0], [1, 1])

    def f1(x):
        return tf.reshape(x[:, 0], (-1, 1))

    def f2(x):
        x1 = x[:, 0]
        n = tf.cast(tf.shape(x)[-1], tf.float64)
        g = 1.0 + tf.reduce_sum(x[:, 1:], axis=1) * 9.0 / (n - 1.0)
        h = g*(1 - tf.sqrt(x1 / g) - x1 / g * tf.sin(10.0 * math.pi * x1))
        return h[:, None]

    def f(x):
        return tf.concat([f1(x), f2(x)], axis=-1)

    print("Generating true Pareto front for ZDT 3")
    problem = PymooProblem(search_space, f, n_obj=2)
    true_pf = get_true_pf(problem)
    save_true_pf(true_pf, ZDT3_FILENAME)
    print("Saved to " + ZDT3_FILENAME)


############################
# Hartmann-Ackley in 6d input
############################
HARTMANN_ACKLEY_FILENAME = "hartmann_ackley_6d_input.csv"

def save_hartmann_ackley_6d_input():
    from trieste.objectives.single_objectives import hartmann_6, ackley_5
    from trieste.types import TensorType

    # Ackley funciton in Trieste is defined over 5d domain, and we want 6d
    def ackley_6(x: TensorType) -> TensorType:
        tf.debugging.assert_shapes([(x, (..., 6))])
        
        x_5d = x[..., :-1]

        return ackley_5(x_5d)

    search_space = trieste.space.Box([0]*6, [1]*6)

    def f(x):
        return tf.concat([hartmann_6(x), ackley_6(x)], axis=-1)

    print("Generating true Pareto front for Hartmann-Ackley")
    problem = PymooProblem(search_space, f, n_obj=2)
    true_pf = get_true_pf(problem)
    save_true_pf(true_pf, HARTMANN_ACKLEY_FILENAME)
    print("Saved to " + HARTMANN_ACKLEY_FILENAME)


############################
# DTLZ 2 - 3 objectives, 4d input
############################
DTLZ2_FILENAME = "dtlz2_4d_input.csv"

def save_dtlz2_4d_input():
    search_space = trieste.space.Box([0, 0, 0, 0], [1, 1, 1, 1])

    def g(x): 
        return tf.pow(x[:, 2] - 0.5, 2) + tf.pow(x[:, 3] - 0.5, 2)

    def f1(x):
        value = (1.0 + g(x)) * tf.cos(x[:, 0] * math.pi * 0.5) * tf.cos(x[:, 1] * math.pi * 0.5)
        return value[:, None]

    def f2(x):
        value = (1.0 + g(x)) * tf.cos(x[:, 0] * math.pi * 0.5) * tf.sin(x[:, 1] * math.pi * 0.5)
        return value[:, None]

    def f3(x):
        value = (1.0 + g(x)) * tf.sin(x[:, 0] * math.pi * 0.5)
        return value[:, None]

    def f(x):
        return tf.concat([f1(x), f2(x), f3(x)], axis=-1)

    print("Generating true Pareto front for DTLZ 2")
    problem = PymooProblem(search_space, f, n_obj=3)
    true_pf = get_true_pf(problem)
    save_true_pf(true_pf, DTLZ2_FILENAME)
    print("Saved to " + DTLZ2_FILENAME)


############

if __name__ == "__main__":
    # save_simple_1d_input()
    # save_gardner_2d_input()
    # save_hartmann_ackley_6d_input()
    # save_zdt3()
    save_dtlz2_4d_input()
