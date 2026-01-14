# Code adapted from https://github.com/ray-project/ray/blob/master/rllib/models/tf/tf_action_dist.py
# Copyright 2021 Ray Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import gymnasium as gym
# import gym
from math import log
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch, try_import_torch_distributions
from ray.rllib.utils.typing import TensorType, List, Union, ModelConfigDict
from ray.rllib.utils import SMALL_NUMBER

torch, nn = try_import_torch()
th = try_import_torch_distributions()


class TruncatedNormal(TorchActionDistribution):
    """Normal distribution that is truncated such that all values are within [low, high].
    The distribution is defined by the mean and the std deviation of a normal distribution that is not truncated.
    (loc=mean -> first half of input, scale=exp(log_std) -> log_std = second half of input)
    KL corresponds to the KL of the underlying normal distributions
    """

    def __init__(self, inputs: List[TensorType], model: ModelV2, low: float = -1.0,
                 high: float = 1.0):
        mean_normal, log_std_normal = torch.split(inputs, inputs.shape[1] // 2, dim=1)
        self.mean_normal = mean_normal
        self.log_std_normal = log_std_normal
        self.std_normal = torch.exp(log_std_normal)
        self.low = low
        self.high = high
        self.zeros = torch.sum(torch.zeros_like(self.mean_normal), dim=1)
        # Use torch.distributions.TransformedDistribution for truncated normal
        base_dist = th.Normal(loc=self.mean_normal, scale=self.std_normal)
        # Create truncation transforms
        transforms = [th.affine_transform(torch.tensor(1.0), torch.tensor(0.0)),
                      th.ExpTransform().inv,
                      th.AffineTransform(torch.tensor(0.0), torch.tensor(1.0))]
        # For simplicity, use Normal with clamping (approximation)
        self.dist = base_dist
        super().__init__(inputs, model)

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        mean = self.dist.mean
        return torch.clamp(mean, self.low, self.high)

    @override(ActionDistribution)
    def logp(self, x: TensorType) -> TensorType:
        # Clamp x to valid range
        x_clamped = torch.clamp(x, self.low, self.high)
        # Use normal distribution log_prob (approximation for truncated)
        log_prob = self.dist.log_prob(x_clamped)
        return torch.sum(log_prob, dim=-1)

    @override(ActionDistribution)
    def kl(self, other: ActionDistribution) -> TensorType:
        assert isinstance(other, TruncatedNormal)
        # Return the kl_divergence of the underlying normal distributions
        return torch.sum(
            other.log_std_normal - self.log_std_normal +
            (torch.square(self.std_normal) + torch.square(self.mean_normal - other.mean_normal))
            / (2.0 * torch.square(other.std_normal)) - 0.5,
            dim=1)

    @override(ActionDistribution)
    def entropy(self) -> TensorType:
        return torch.sum(self.dist.entropy(), dim=1)

    @override(TorchActionDistribution)
    def _build_sample_op(self) -> TensorType:
        sample = self.dist.sample()
        return torch.clamp(sample, self.low, self.high)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(
            action_space: gym.Space,
            model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        return np.prod(action_space.shape) * 2


class TruncatedNormalZeroKL(TruncatedNormal):
    """Normal distribution that is truncated such that all values are within [low, high].
    The distribution is defined by the mean and the std deviation of a normal distribution that is not truncated.
    (loc=mean -> first half of input, scale=exp(log_std) -> log_std = second half of input).
    KL is always set to zero.
    """

    def __init__(self, inputs: List[TensorType], model: ModelV2, low: float = -1.0,
                 high: float = 1.0):
        super().__init__(inputs, model, low, high)
        self.zeros = torch.sum(torch.zeros_like(self.mean_normal), dim=1)

    @override(TruncatedNormal)
    def kl(self, other: ActionDistribution) -> TensorType:
        assert isinstance(other, TruncatedNormal)
        return self.zeros


class BetaBase(TorchActionDistribution):
    """
    A Beta distribution is defined on the interval [0, 1] and parameterized by
    shape parameters alpha and beta (also called concentration parameters).
    PDF(x; alpha, beta) = x**(alpha - 1) (1 - x)**(beta - 1) / Z
        with Z = Gamma(alpha) Gamma(beta) / Gamma(alpha + beta)
        and Gamma(n) = (n - 1)!
    """

    def __init__(self,
                 inputs: List[TensorType],
                 model: ModelV2,
                 low: float = -1.0,
                 high: float = 1.0):

        self.dist = None
        self.low = low
        self.high = high

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        mean = self.dist.mean
        return self._squash(mean)

    @override(TorchActionDistribution)
    def _build_sample_op(self) -> TensorType:
        return self._squash(self.dist.sample())

    @override(ActionDistribution)
    def entropy(self) -> TensorType:
        return torch.sum(self.dist.entropy(), dim=1)

    @override(ActionDistribution)
    def kl(self, other: ActionDistribution) -> TensorType:
        assert isinstance(other, BetaBase)
        return torch.sum(th.kl_divergence(self.dist, other.dist), dim=1)

    @override(ActionDistribution)
    def logp(self, x: TensorType) -> TensorType:
        unsquashed_values = self._unsquash(x)
        return torch.sum(
            self.dist.log_prob(unsquashed_values), dim=-1)

    def _squash(self, raw_values: TensorType) -> TensorType:
        return raw_values * (self.high - self.low) + self.low

    def _unsquash(self, values: TensorType) -> TensorType:
        return (values - self.low) / (self.high - self.low)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(
            action_space: gym.Space,
            model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        return np.prod(action_space.shape) * 2


class BetaMeanTotal(BetaBase):
    """
    A Beta distribution defined by the mean (squashed) and the log total concentration
    """

    def __init__(self,
                 inputs: List[TensorType],
                 model: ModelV2,
                 low: float = -1.0,
                 high: float = 1.0):
        super().__init__(inputs, model, low, high)
        mean_squashed, log_total_concentration = torch.split(self.inputs, self.inputs.shape[1] // 2, dim=-1)
        log_total_concentration = torch.clamp(log_total_concentration, log(SMALL_NUMBER),
                                              -log(SMALL_NUMBER))
        # total_concentration > 0
        mean = self._unsquash(mean_squashed)
        total_concentration = torch.exp(log_total_concentration)
        alpha = mean * total_concentration
        beta = (1.0 - mean) * total_concentration
        self.dist = th.Beta(
            concentration1=alpha, concentration0=beta)
        super(BetaBase, self).__init__(inputs, model)


class BetaAlphaBeta(BetaBase):
    """
    A Beta distribution defined by alpha and beta with alpha, beta > 1
    """

    def __init__(self,
                 inputs: List[TensorType],
                 model: ModelV2,
                 low: float = -1.0,
                 high: float = 1.0):
        inputs = torch.clamp(inputs, log(SMALL_NUMBER),
                            -log(SMALL_NUMBER))
        inputs = torch.log(torch.exp(inputs) + 1.0) + 1.0  # ensures alpha > 1, beta > 1
        super().__init__(inputs, model, low, high)
        alpha, beta = torch.split(inputs, inputs.shape[1] // 2, dim=-1)

        self.dist = th.Beta(
            concentration1=alpha, concentration0=beta)
        super(BetaBase, self).__init__(inputs, model)
