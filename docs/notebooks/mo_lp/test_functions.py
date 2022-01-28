import math
import tensorflow as tf
import trieste

from abc import ABC, abstractclassmethod, abstractmethod

def get_test_function(name: str):
    if name == Simple1D.name:
        return Simple1D()
    elif name == Gardner2D.name:
        return Gardner2D()
    elif name == ZDT3.name:
        return ZDT3()
    elif name == HartmannAckley6D.name:
        return HartmannAckley6D()
    elif name == DTLZ2.name:
        return DTLZ2()
    else:
        raise ValueError(f"Unknown test function {name}")


# from https://stackoverflow.com/questions/128573/using-property-on-classmethods
class classproperty(object):
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class TestFunction(ABC):
    def __init__(self, search_space, true_pf_filename, n_objectives=2):
        self._search_space = search_space
        self._n_objectives = n_objectives
        self._true_pf_filename = true_pf_filename

    @abstractclassmethod
    def name(self):
        ...

    @property
    def true_pf_filename(self):
        return self._true_pf_filename

    @property
    def search_space(self):
        return self._search_space

    @property
    def n_objectives(self):
        return self._n_objectives

    @abstractmethod
    def f(self, x):
        ...

    def __call__(self, x):
        return self.f(x)


class Simple1D(TestFunction):
    """Simple 1d input space case
    """
    def __init__(self):
        super().__init__(trieste.space.Box([0], [2*math.pi]), "simple_1d_input.csv")

    @classproperty
    def name(self):
        return "Simple1D"

    def f1(self, x):
        return tf.cos(2 * x) + tf.sin(x)

    def f2(self, x):
        return 0.2 * (tf.cos(x) - tf.sin(x)) + 0.3

    def f(self, x):
        return tf.concat([self.f1(x), self.f2(x)], axis=-1)



class Gardner2D(TestFunction):
    """2d input case from Gardner et al. 2014
    """
    def __init__(self):
        super().__init__(trieste.space.Box([0, 0], [2*math.pi, 2*math.pi]), "gardner_2d_input.csv")

    @classproperty
    def name(self):
        return "Gardner2D"

    def f1(self, input_data):
        x, y = input_data[..., -2], input_data[..., -1]
        z = tf.cos(2.0 * x) * tf.cos(y) + tf.sin(x)
        return z[:, None]

    def f2(self, input_data):
        x, y = input_data[:, -2], input_data[:, -1]
        # changes are made so that the function is between 0 and 1
        z = 1.0 - (tf.cos(x) * tf.cos(y) - tf.sin(x) * tf.sin(y) + 1.0) / 2.0
        return z[:, None]

    def f(self, x):
        return tf.concat([self.f1(x), self.f2(x)], axis=-1)



class ZDT3(TestFunction):
    """ZDT3 - 6d input, discontinuous front
    """
    def __init__(self):
        super().__init__(trieste.space.Box([0, 0], [1, 1]), "zdt3_2d_input.csv")

    @classproperty
    def name(self):
        return "ZDT3"

    def f1(self, x):
        return tf.reshape(x[:, 0], (-1, 1))

    def f2(self, x):
        x1 = x[:, 0]
        n = tf.cast(tf.shape(x)[-1], tf.float64)
        g = 1.0 + tf.reduce_sum(x[:, 1:], axis=1) * 9.0 / (n - 1.0)
        h = g*(1 - tf.sqrt(x1 / g) - x1 / g * tf.sin(10.0 * math.pi * x1))
        return h[:, None]

    def f(self, x):
        return tf.concat([self.f1(x), self.f2(x)], axis=-1)



from trieste.objectives.single_objectives import hartmann_6, ackley_5

# Ackley function in Trieste is defined over 5d domain, and we want 6d
def ackley_6(x):
    tf.debugging.assert_shapes([(x, (..., 6))])
    x_5d = x[..., :-1]
    return ackley_5(x_5d)

class HartmannAckley6D(TestFunction):
    """Hartmann-Ackley in 6d input
    """
    def __init__(self):
        super().__init__(trieste.space.Box([0]*6, [1]*6), "hartmann_ackley_6d_input.csv")

    @classproperty
    def name(self):
        return "Hartmann-Ackley"

    def f(self, x):
        return tf.concat([hartmann_6(x), ackley_6(x)], axis=-1)



class DTLZ2(TestFunction):
    """DTLZ 2 - 3 objectives, 4d input
    """
    def __init__(self):
        super().__init__(trieste.space.Box([0]*4, [1]*4), "dtlz2_4d_input.csv", n_objectives=3)

    @classproperty
    def name(self):
        return "DTLZ2"

    def g(self, x): 
        return tf.pow(x[:, 2] - 0.5, 2) + tf.pow(x[:, 3] - 0.5, 2)

    def f1(self, x):
        value = (1.0 + self.g(x)) * tf.cos(x[:, 0] * math.pi * 0.5) * tf.cos(x[:, 1] * math.pi * 0.5)
        return value[:, None]

    def f2(self, x):
        value = (1.0 + self.g(x)) * tf.cos(x[:, 0] * math.pi * 0.5) * tf.sin(x[:, 1] * math.pi * 0.5)
        return value[:, None]

    def f3(self, x):
        value = (1.0 + self.g(x)) * tf.sin(x[:, 0] * math.pi * 0.5)
        return value[:, None]

    def f(self, x):
        return tf.concat([self.f1(x), self.f2(x), self.f3(x)], axis=-1)
