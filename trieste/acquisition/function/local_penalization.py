# Copyright 2021 The Trieste Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains local penalization-based acquisition function builders.
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional, Union, cast

import tensorflow as tf
import tensorflow_probability as tfp

from ...data import Dataset
from ...models import ProbabilisticModel
from ...space import SearchSpace
from ...types import TensorType
from ..interface import (
    AcquisitionFunction,
    PenalizationFunction,
    SingleModelAcquisitionBuilder,
    SingleModelGreedyAcquisitionBuilder,
    UpdatablePenalizationFunction,
)
from .entropy import MinValueEntropySearch
from .function import ExpectedImprovement, expected_improvement


class LocalPenalizationAcquisitionFunction(SingleModelGreedyAcquisitionBuilder):
    r"""
    Builder of the acquisition function maker for greedily collecting batches by local
    penalization.  The resulting :const:`AcquisitionFunctionMaker` takes in a set of pending
    points and returns a base acquisition function penalized around those points.
    An estimate of the objective function's Lipschitz constant is used to control the size
    of penalization.

    Local penalization allows us to perform batch Bayesian optimization with a standard (non-batch)
    acquisition function. All that we require is that the acquisition function takes strictly
    positive values. By iteratively building a batch of points though sequentially maximizing
    this acquisition function but down-weighted around locations close to the already
    chosen (pending) points, local penalization provides diverse batches of candidate points.

    Local penalization is applied to the acquisition function multiplicatively. However, to
    improve numerical stability, we perform additive penalization in a log space.

    The Lipschitz constant and additional penalization parameters are estimated once
    when first preparing the acquisition function with no pending points. These estimates
    are reused for all subsequent function calls.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        num_samples: int = 500,
        penalizer: Callable[
            [ProbabilisticModel, TensorType, TensorType, TensorType],
            Union[PenalizationFunction, UpdatablePenalizationFunction],
        ] = None,
        base_acquisition_function_builder: ExpectedImprovement
        | MinValueEntropySearch
        | None = None,
    ):
        """
        :param search_space: The global search space over which the optimisation is defined.
        :param num_samples: Size of the random sample over which the Lipschitz constant
            is estimated. We recommend scaling this with search space dimension.
        :param penalizer: The chosen penalization method (defaults to soft penalization). This
            should be a function that accepts a model, pending points, lipschitz constant and eta
            and returns a PenalizationFunction.
        :param base_acquisition_function_builder: Base acquisition function to be
            penalized (defaults to expected improvement). Local penalization only supports
            strictly positive acquisition functions.
        :raise tf.errors.InvalidArgumentError: If ``num_samples`` is not positive.
        """
        tf.debugging.assert_positive(num_samples)

        self._search_space = search_space
        self._num_samples = num_samples

        self._lipschitz_penalizer = soft_local_penalizer if penalizer is None else penalizer

        if base_acquisition_function_builder is None:
            self._base_builder: SingleModelAcquisitionBuilder = ExpectedImprovement()
        else:
            self._base_builder = base_acquisition_function_builder

        self._lipschitz_constant = None
        self._eta = None
        self._base_acquisition_function: Optional[AcquisitionFunction] = None
        self._penalization: Optional[PenalizationFunction | UpdatablePenalizationFunction] = None
        self._penalized_acquisition: Optional[AcquisitionFunction] = None

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :param pending_points: The points we penalize with respect to.
        :return: The (log) expected improvement penalized with respect to the pending points.
        :raise tf.errors.InvalidArgumentError: If the ``dataset`` is empty.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")

        acq = self._update_base_acquisition_function(dataset, model)
        if pending_points is not None and len(pending_points) != 0:
            acq = self._update_penalization(acq, dataset, model, pending_points)

        return acq

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
        new_optimization_step: bool = True,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :param pending_points: Points already chosen to be in the current batch (of shape [M,D]),
            where M is the number of pending points and D is the search space dimension.
        :param new_optimization_step: Indicates whether this call to update_acquisition_function
            is to start of a new optimization step, of to continue collecting batch of points
            for the current step. Defaults to ``True``.
        :return: The updated acquisition function.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(self._base_acquisition_function is not None, [])

        if new_optimization_step:
            self._update_base_acquisition_function(dataset, model)

        if pending_points is None or len(pending_points) == 0:
            # no penalization required if no pending_points
            return cast(AcquisitionFunction, self._base_acquisition_function)

        return self._update_penalization(function, dataset, model, pending_points)

    def _update_penalization(
        self,
        function: Optional[AcquisitionFunction],
        dataset: Dataset,
        model: ProbabilisticModel,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        tf.debugging.assert_rank(pending_points, 2)

        if self._penalized_acquisition is not None and isinstance(
            self._penalization, UpdatablePenalizationFunction
        ):
            # if possible, just update the penalization function variables
            self._penalization.update(pending_points, self._lipschitz_constant, self._eta)
            return self._penalized_acquisition
        else:
            # otherwise construct a new penalized acquisition function
            self._penalization = self._lipschitz_penalizer(
                model, pending_points, self._lipschitz_constant, self._eta
            )

            @tf.function
            def penalized_acquisition(x: TensorType) -> TensorType:
                log_acq = tf.math.log(
                    cast(AcquisitionFunction, self._base_acquisition_function)(x)
                ) + tf.math.log(cast(PenalizationFunction, self._penalization)(x))
                return tf.math.exp(log_acq)

            self._penalized_acquisition = penalized_acquisition
            return penalized_acquisition

    @tf.function(experimental_relax_shapes=True)
    def _get_lipschitz_estimate(
        self, model: ProbabilisticModel, sampled_points: TensorType
    ) -> tuple[TensorType, TensorType]:
        with tf.GradientTape() as g:
            g.watch(sampled_points)
            mean, _ = model.predict(sampled_points)
        grads = g.gradient(mean, sampled_points)
        grads_norm = tf.norm(grads, axis=1)
        max_grads_norm = tf.reduce_max(grads_norm)
        eta = tf.reduce_min(mean, axis=0)
        return max_grads_norm, eta

    def _update_base_acquisition_function(
        self, dataset: Dataset, model: ProbabilisticModel
    ) -> AcquisitionFunction:
        samples = self._search_space.sample(num_samples=self._num_samples)
        samples = tf.concat([dataset.query_points, samples], 0)

        lipschitz_constant, eta = self._get_lipschitz_estimate(model, samples)
        if lipschitz_constant < 1e-5:  # threshold to improve numerical stability for 'flat' models
            lipschitz_constant = 10

        self._lipschitz_constant = lipschitz_constant
        self._eta = eta

        if self._base_acquisition_function is not None:
            self._base_acquisition_function = self._base_builder.update_acquisition_function(
                self._base_acquisition_function,
                model,
                dataset=dataset,
            )
        elif isinstance(self._base_builder, ExpectedImprovement):  # reuse eta estimate
            self._base_acquisition_function = cast(
                AcquisitionFunction, expected_improvement(model, self._eta)
            )
        else:
            self._base_acquisition_function = self._base_builder.prepare_acquisition_function(
                model,
                dataset=dataset,
            )
        return self._base_acquisition_function


class local_penalizer(UpdatablePenalizationFunction):
    def __init__(
        self,
        model: ProbabilisticModel,
        pending_points: TensorType,
        lipschitz_constant: TensorType,
        eta: TensorType,
    ):
        """Initialize the local penalizer.

        :param model: The model over the specified ``dataset``.
        :param pending_points: The points we penalize with respect to.
        :param lipschitz_constant: The estimated Lipschitz constant of the objective function.
        :param eta: The estimated global minima.
        :return: The local penalization function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one."""
        self._model = model

        mean_pending, variance_pending = model.predict(pending_points)
        self._pending_points = tf.Variable(pending_points, shape=[None, *pending_points.shape[1:]])
        self._radius = tf.Variable(
            tf.transpose((mean_pending - eta) / lipschitz_constant),
            shape=[1, None],
        )
        self._scale = tf.Variable(
            tf.transpose(tf.sqrt(variance_pending) / lipschitz_constant),
            shape=[1, None],
        )

    def update(
        self,
        pending_points: TensorType,
        lipschitz_constant: TensorType,
        eta: TensorType,
    ) -> None:
        """Update the local penalizer with new variable values."""
        mean_pending, variance_pending = self._model.predict(pending_points)
        self._pending_points.assign(pending_points)
        self._radius.assign(tf.transpose((mean_pending - eta) / lipschitz_constant))
        self._scale.assign(tf.transpose(tf.sqrt(variance_pending) / lipschitz_constant))


class soft_local_penalizer(local_penalizer):

    r"""
    Return the soft local penalization function used for single-objective greedy batch Bayesian
    optimization in :cite:`Gonzalez:2016`.

    Soft penalization returns the probability that a candidate point does not belong
    in the exclusion zones of the pending points. For model posterior mean :math:`\mu`, model
    posterior variance :math:`\sigma^2`, current "best" function value :math:`\eta`, and an
    estimated Lipschitz constant :math:`L`,the penalization from a set of pending point
    :math:`x'` on a candidate point :math:`x` is given by
    .. math:: \phi(x, x') = \frac{1}{2}\textrm{erfc}(-z)
    where :math:`z = \frac{1}{\sqrt{2\sigma^2(x')}}(L||x'-x|| + \eta - \mu(x'))`.

    The penalization from a set of pending points is just product of the individual
    penalizations. See :cite:`Gonzalez:2016` for a full derivation.

    :param model: The model over the specified ``dataset``.
    :param pending_points: The points we penalize with respect to.
    :param lipschitz_constant: The estimated Lipschitz constant of the objective function.
    :param eta: The estimated global minima.
    :return: The local penalization function. This function will raise
        :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
        greater than one.
    """

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This penalization function cannot be calculated for batches of points.",
        )

        pairwise_distances = tf.norm(
            tf.expand_dims(x, 1) - tf.expand_dims(self._pending_points, 0), axis=-1
        )
        standardised_distances = (pairwise_distances - self._radius) / self._scale

        normal = tfp.distributions.Normal(tf.cast(0, x.dtype), tf.cast(1, x.dtype))
        penalization = normal.cdf(standardised_distances)
        return tf.reduce_prod(penalization, axis=-1)


class hard_local_penalizer(local_penalizer):
    r"""
    Return the hard local penalization function used for single-objective greedy batch Bayesian
    optimization in :cite:`Alvi:2019`.

    Hard penalization is a stronger penalizer than soft penalization and is sometimes more effective
    See :cite:`Alvi:2019` for details. Our implementation follows theirs, with the penalization from
    a set of pending points being the product of the individual penalizations.

    :param model: The model over the specified ``dataset``.
    :param pending_points: The points we penalize with respect to.
    :param lipschitz_constant: The estimated Lipschitz constant of the objective function.
    :param eta: The estimated global minima.
    :return: The local penalization function. This function will raise
        :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
        greater than one.
    """

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This penalization function cannot be calculated for batches of points.",
        )

        pairwise_distances = tf.norm(
            tf.expand_dims(x, 1) - tf.expand_dims(self._pending_points, 0), axis=-1
        )

        p = -5  # following experiments of :cite:`Alvi:2019`.
        penalization = ((pairwise_distances / (self._radius + self._scale)) ** p + 1) ** (1 / p)
        return tf.reduce_prod(penalization, axis=-1)






from .multi_objective import ExpectedHypervolumeImprovement

class MOLocalPenalizationAcquisitionFunction(SingleModelGreedyAcquisitionBuilder):
    def __init__(
        self
    ):
        self._base_builder: SingleModelAcquisitionBuilder = ExpectedHypervolumeImprovement()
        self._base_acquisition_function: Optional[AcquisitionFunction] = None
        self._penalization: Optional[mo_penalizer] = None
        self._penalized_acquisition: Optional[AcquisitionFunction] = None

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :param pending_points: The points we penalize with respect to.
        :return: 
        :raise tf.errors.InvalidArgumentError: If the ``dataset`` is empty.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")

        acq = self._update_base_acquisition_function(model, dataset)
        if pending_points is not None and len(pending_points) != 0:
            acq = self._update_penalization(acq, model, pending_points)

        return acq

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
        new_optimization_step: bool = True,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :param pending_points: Points already chosen to be in the current batch (of shape [M,D]),
            where M is the number of pending points and D is the search space dimension.
        :param new_optimization_step: Indicates whether this call to update_acquisition_function
            is to start of a new optimization step, of to continue collecting batch of points
            for the current step. Defaults to ``True``.
        :return: The updated acquisition function.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(self._base_acquisition_function is not None, [])

        if new_optimization_step:
            self._update_base_acquisition_function(model, dataset)

        if pending_points is None or len(pending_points) == 0:
            # no penalization required if no pending_points
            return cast(AcquisitionFunction, self._base_acquisition_function)

        return self._update_penalization(function, model, pending_points)

    def _update_penalization(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        tf.debugging.assert_rank(pending_points, 2)

        if self._penalized_acquisition is not None and isinstance(
            self._penalization, mo_penalizer
        ):
            # if possible, just update the penalization function variables
            self._penalization.update(pending_points)
            return self._penalized_acquisition
        else:
            # otherwise construct a new penalized acquisition function
            self._penalization = mo_penalizer(model, pending_points)

        # # handy plotting for acquisition function
        # import matplotlib.pyplot as plt

        # def plot_fn(fn, label, c):
        #     import numpy as np
        #     import math 
            
        #     x = np.linspace(0, 2*math.pi, 1000)
        #     y = fn(x.reshape(-1, 1, 1))

        #     plt.plot(x, y, label=label, c=c)

        @tf.function
        def penalized_acquisition(x: TensorType) -> TensorType:
            log_acq = tf.math.log(
                cast(AcquisitionFunction, self._base_acquisition_function)(x)
            ) + tf.math.log(self._penalization(x))
            return tf.math.exp(log_acq)

        # # plot acquisition function and batch points
        # plt.figure()
        # plt.vlines(tf.squeeze(pending_points), ymin=0, ymax=1, label="batch points so far", colors="green")
        # plot_fn(self._base_acquisition_function, "base function", c="blue")
        # plot_fn(self._penalization, "penalization", c="red")
        # plot_fn(penalized_acquisition, "penalized acquisition", c="purple")
        # # tf.print("------------------------------------")
        # # tf.print(pending_points)
        # # tf.print(self._base_acquisition_function(tf.expand_dims(pending_points, axis=1)))
        # # tf.print(self._penalization(tf.expand_dims(pending_points, axis=1)))
        # # tf.print(penalized_acquisition(tf.expand_dims(pending_points, axis=1)))
        # # tf.print("------------------------------------")
        # plt.legend()
        # plt.show()

        self._penalized_acquisition = penalized_acquisition
        return penalized_acquisition

    def _update_base_acquisition_function(
        self, model: ProbabilisticModel, dataset: Dataset
    ) -> AcquisitionFunction:
        if self._base_acquisition_function is None:
            self._base_acquisition_function = self._base_builder.prepare_acquisition_function(
                model,
                dataset=dataset,
            )
        else:
            self._base_acquisition_function = self._base_builder.update_acquisition_function(
                self._base_acquisition_function,
                model,
                dataset=dataset,
            )
        return self._base_acquisition_function

class mo_penalizer():
    def __init__(self, model: ProbabilisticModel, pending_points: TensorType):
        tf.debugging.Assert(pending_points is not None and len(pending_points) != 0, [])

        self._model = model
        self._pending_points = tf.Variable(pending_points, shape=[None, *pending_points.shape[1:]])
        pending_means, pending_vars = self._model.predict(self._pending_points)
        self._pending_means = tf.Variable(pending_means, shape=[None, *pending_means.shape[1:]])
        self._pending_vars = tf.Variable(pending_vars, shape=[None, *pending_vars.shape[1:]])

    def update(
        self,
        pending_points: TensorType
    ) -> None:
        """Update the local penalizer with new pending points."""
        tf.debugging.Assert(pending_points is not None and len(pending_points) != 0, [])

        self._pending_points.assign(pending_points)
        pending_means, pending_vars = self._model.predict(self._pending_points)
        self._pending_means.assign(pending_means)
        self._pending_vars.assign(pending_vars)

    # @tf.function
    # def __call__(self, x: TensorType) -> TensorType:
    #     # x is [N, 1, D]
    #     x = tf.squeeze(x, axis=1) # x is now [N, D]

    #     # pending_points is [B, D] where B is the size of the batch collected so far
    #     cov_with_pending_points = self._model.covariance_between_points(x, self._pending_points) # [N, B, K], K is the number of models in the stack
    #     pending_means, pending_covs = self._model.predict(self._pending_points) # pending_means is [B, K], pending_covs is [B, K]
    #     x_means, x_covs = self._model.predict(x) # x_mean is [N, K], x_cov is [N, K]

    #     tf.debugging.assert_shapes(
    #         [
    #             (x, ["N", "D"]),
    #             (self._pending_points, ["B", "D"]),
    #             (cov_with_pending_points, ["N", "B", "K"]),
    #             (pending_means, ["B", "K"]),
    #             (pending_covs, ["B", "K"]),
    #             (x_means, ["N", "K"]),
    #             (x_covs, ["N", "K"])
    #         ],
    #         message="uh-oh"
    #     )

    #     # N = tf.shape(x)[0]
    #     # B = tf.shape(self._pending_points)[0]
    #     # K = tf.shape(cov_with_pending_points)[-1]

    #     x_means_expanded = x_means[:, None, :]
    #     x_covs_expanded = x_covs[:, None, :]
    #     pending_means_expanded = pending_means[None, :, :]
    #     pending_covs_expanded = pending_covs[None, :, :]

    #     # tf.print(x_covs_expanded)
    #     # tf.print(pending_covs_expanded)
    #     # tf.print(cov_with_pending_points)
    #     # tf.print(x_covs_expanded + pending_covs_expanded - 2.0 * cov_with_pending_points)

    #     CLAMP_LB = 1e-12
    #     variance = x_covs_expanded + pending_covs_expanded - 2.0 * cov_with_pending_points
    #     variance = tf.clip_by_value(variance, CLAMP_LB, variance.dtype.max)

    #     # mean = tf.clip_by_value(pending_means_expanded - x_means_expanded, CLAMP_LB, x_means_expanded.dtype.max)
    #     # stddev = tf.clip_by_value(tf.math.sqrt(variance), CLAMP_LB, variance.dtype.max)
    #     mean = pending_means_expanded - x_means_expanded
    #     stddev = tf.math.sqrt(variance)

    #     # print(variance)
    #     # print(stddev)

    #     f_diff_normal = tfp.distributions.Normal(loc=mean, scale=stddev)
    #     cdf = f_diff_normal.cdf(0.0)

    #     # print(cdf)

    #     tf.debugging.assert_shapes(
    #         [
    #             (x, ["N", "D"]),
    #             (self._pending_points, ["B", "D"]),
    #             (mean, ["N", "B", "K"]),
    #             (stddev, ["N", "B", "K"]),
    #             (cdf, ["N", "B", "K"])
    #         ],
    #         message="uh-oh-oh"
    #     )

    #     # tf.print(mean)
    #     # tf.print(stddev)
    #     # tf.print(cdf)
    #     # penalty = tf.reduce_prod((1.0 - tf.reduce_prod(1 - cdf, axis=-1)), axis=-1)
    #     penalty = tf.reduce_prod(1.0 - tf.reduce_prod(cdf, axis=-1), axis=-1)

    #     return tf.reshape(penalty, (-1, 1))

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        # x is [N, 1, D]
        x = tf.squeeze(x, axis=1) # x is now [N, D]
        x_means, x_vars = self._model.predict(x) # x_means is [N, K], x_vars is [N, K], where K is the number of models/objectives

        # self._pending_points is [B, D] where B is the size of the batch collected so far
        tf.debugging.assert_shapes(
            [
                (x, ["N", "D"]),
                (self._pending_points, ["B", "D"]),
                (self._pending_means, ["B", "K"]),
                (self._pending_vars, ["B", "K"]),
                (x_means, ["N", "K"]),
                (x_vars, ["N", "K"])
            ],
            message="Encountered unexpected shapes while calculating mean and variance of given point x and pending points"
        )

        x_means_expanded = x_means[:, None, :]
        pending_means_expanded = self._pending_means[None, :, :]
        pending_vars_expanded = self._pending_vars[None, :, :]
        pending_stddevs_expanded = tf.sqrt(pending_vars_expanded)
        standardize_mean_diff = (x_means_expanded - pending_means_expanded) / pending_stddevs_expanded # [N, B, K]

        d = tf.norm(standardize_mean_diff, axis=-1) # [N, B]

        # warp the distance so that resulting value is from 0 to nearly 1
        # 2 * (sigmoid(d) - 0.5)
        warped_d = 2 * (1.0 / (1.0 + tf.exp(-d)) - 0.5) # [N, B]
        # 1 - 1/(1+d)
        # warped_d = 1.0 - 1.0 / (1.0 + d) # [N, B]

        penalty = tf.reduce_prod(warped_d, axis=-1) # [N,]

        return tf.reshape(penalty, (-1, 1))
