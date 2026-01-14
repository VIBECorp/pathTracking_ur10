# Code adapted from https://github.com/ray-project/ray/blob/master/rllib/models/tf/fcnet.py
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
from typing import Dict, Optional, Sequence, List

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import normc_initializer, SlimFC
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType, ModelConfigDict

torch, nn = try_import_torch()


def get_torch_activation_fn(activation_name: str):
    """Get PyTorch activation function by name."""
    if activation_name is None or activation_name == "linear":
        return None
    
    activation_name_lower = activation_name.lower()
    
    if activation_name_lower == "relu":
        return nn.ReLU
    elif activation_name_lower == "tanh":
        return nn.Tanh
    elif activation_name_lower == "sigmoid":
        return nn.Sigmoid
    elif activation_name_lower == "elu":
        return nn.ELU
    elif activation_name_lower == "gelu":
        return nn.GELU
    elif activation_name_lower == "selu":
        return nn.SELU
    elif activation_name_lower == "leaky_relu":
        return nn.LeakyReLU
    elif activation_name_lower == "swish":
        # Swish/SiLU: x * sigmoid(x)
        # PyTorch 1.7+ has nn.SiLU, but for compatibility we can use a lambda
        try:
            return nn.SiLU  # Available in PyTorch 1.7+
        except AttributeError:
            # Fallback for older PyTorch versions
            def swish(x):
                return x * torch.sigmoid(x)
            return swish
    else:
        raise ValueError(f"Unknown activation function: {activation_name}")


class FullyConnectedNetworkLastLayerActivation(TorchModelV2, nn.Module):
    """Generic fully connected network implemented in PyTorch with last layer activation support."""

    def __init__(
            self,
            obs_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: Optional[int],
            model_config: ModelConfigDict,
            name: str,
            **kwargs
    ):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # Get configuration
        custom_model_config = model_config.get("custom_model_config", {})
        fcnet_hiddens = model_config.get("fcnet_hiddens", [256, 128])
        post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        fcnet_activation = model_config.get("fcnet_activation", "tanh")
        post_fcnet_activation = model_config.get("post_fcnet_activation")
        no_final_linear = model_config.get("no_final_linear", False)
        vf_share_layers = model_config.get("vf_share_layers", False)
        free_log_std = model_config.get("free_log_std", False)
        last_layer_activation = custom_model_config.get("last_layer_activation")
        no_log_std_activation = custom_model_config.get("no_log_std_activation", False)
        log_std_range = custom_model_config.get("log_std_range")
        output_intermediate_layers = custom_model_config.get("output_intermediate_layers", False)

        hiddens = list(fcnet_hiddens or []) + list(post_fcnet_hiddens or [])
        activation = get_torch_activation_fn(fcnet_activation)
        if not fcnet_hiddens:
            activation = get_torch_activation_fn(post_fcnet_activation)

        if last_layer_activation is not None:
            last_layer_activation_fn = get_torch_activation_fn(last_layer_activation)
        else:
            last_layer_activation_fn = None

        # Generate free-floating bias variables for the second half of the outputs
        if free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two", num_outputs)
            num_outputs = num_outputs // 2
            self.log_std_var = nn.Parameter(torch.zeros(num_outputs))

        self._log_std_range = log_std_range
        self._output_intermediate_layers = output_intermediate_layers
        self._intermediate_layer_names = []
        self._vf_share_layers = vf_share_layers

        # Input size
        input_size = int(np.prod(obs_space.shape))

        # Build action network
        self._action_layers = nn.ModuleList()
        self._action_intermediate_outputs = []
        i = 1

        # Create layers 0 to second-last
        current_size = input_size
        for size in hiddens[:-1] if len(hiddens) > 1 else []:
            layer = SlimFC(
                in_size=current_size,
                out_size=size,
                activation_fn=activation,
                initializer=normc_initializer(1.0)
            )
            self._action_layers.append(layer)
            self._intermediate_layer_names.append(f"fc_{i}")
            i += 1
            current_size = size
        
        # The last layer
        if no_final_linear and num_outputs:
            layer = SlimFC(
                in_size=current_size,
                out_size=num_outputs,
                activation_fn=activation,
                initializer=normc_initializer(1.0)
            )
            self._action_layers.append(layer)
            self._logits_out = None
            last_hidden_size = current_size
        else:
            if len(hiddens) > 0:
                layer = SlimFC(
                    in_size=current_size,
                    out_size=hiddens[-1],
                    activation_fn=activation,
                    initializer=normc_initializer(1.0)
                )
                self._action_layers.append(layer)
                self._intermediate_layer_names.append(f"fc_{i}")
                last_hidden_size = hiddens[-1]
            else:
                last_hidden_size = current_size

            if num_outputs:
                if no_log_std_activation and not vf_share_layers:
                    # Separate outputs for actions and log_std
                    self._actions_out = SlimFC(
                        in_size=last_hidden_size,
                        out_size=num_outputs // 2,
                        activation_fn=last_layer_activation_fn,
                        initializer=normc_initializer(0.01)
                    )
                    self._log_std_out = SlimFC(
                        in_size=last_hidden_size,
                        out_size=num_outputs // 2,
                        activation_fn=None,
                        initializer=normc_initializer(0.01)
                    )
                    self._logits_out = None
                else:
                    self._logits_out = SlimFC(
                        in_size=last_hidden_size,
                        out_size=num_outputs,
                        activation_fn=last_layer_activation_fn,
                        initializer=normc_initializer(0.01)
                    )
                    self._actions_out = None
                    self._log_std_out = None
            else:
                self._logits_out = None
                self._actions_out = None
                self._log_std_out = None

        # Build value network
        if not vf_share_layers:
            self._value_layers = nn.ModuleList()
            vf_input_size = int(np.prod(obs_space.shape))
            for size in hiddens:
                layer = SlimFC(
                    in_size=vf_input_size,
                    out_size=size,
                    activation_fn=activation,
                    initializer=normc_initializer(1.0)
                )
                self._value_layers.append(layer)
                vf_input_size = size
        else:
            self._value_layers = None

        # Value output
        vf_input_size = last_hidden_size if vf_share_layers else (hiddens[-1] if hiddens else int(np.prod(obs_space.shape)))
        self._value_out = SlimFC(
            in_size=vf_input_size,
            out_size=1,
            activation_fn=None,
            initializer=normc_initializer(0.01)
        )

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType):
        obs = input_dict[SampleBatch.OBS]
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()

        # Action network forward pass
        x = obs
        intermediate_outputs = []
        
        for i, layer in enumerate(self._action_layers):
            x = layer(x)
            if self._output_intermediate_layers and i < len(self._action_layers) - 1:
                intermediate_outputs.append(x)

        # Get logits output
        if self._logits_out is not None:
            logits_out = self._logits_out(x)
        elif self._actions_out is not None and self._log_std_out is not None:
            actions_out = self._actions_out(x)
            log_std_out = self._log_std_out(x)
            logits_out = torch.cat([actions_out, log_std_out], dim=1)
        else:
            logits_out = x

        # Handle free_log_std
        if hasattr(self, 'log_std_var'):
            batch_size = logits_out.shape[0]
            log_std_tiled = self.log_std_var.unsqueeze(0).expand(batch_size, -1)
            logits_out = torch.cat([logits_out, log_std_tiled], dim=1)

        # Handle log_std_range
        if self._log_std_range:
            mean, log_std = torch.split(logits_out, logits_out.shape[1] // 2, dim=1)
            log_std = self._log_std_range[0] + 0.5 * (log_std + 1) * (self._log_std_range[1] - self._log_std_range[0])
            logits_out = torch.cat([mean, log_std], dim=1)

        # Value network forward pass
        if self._vf_share_layers:
            vf_x = x
        else:
            vf_x = obs
            if self._value_layers is not None:
                for layer in self._value_layers:
                    vf_x = layer(vf_x)

        value_out = self._value_out(vf_x)
        self._value_out_flat = value_out.squeeze(-1)

        # Store intermediate outputs if needed
        if self._output_intermediate_layers:
            self._intermediate_outputs = {}
            for i, output in enumerate(intermediate_outputs):
                self._intermediate_outputs[self._intermediate_layer_names[i]] = output
            self._intermediate_outputs['logits'] = logits_out

        return logits_out, state

    def value_function(self):
        return self._value_out_flat

    def get_extra_outs(self, input_dict: Dict[str, TensorType]) -> Dict[str, TensorType]:
        """Get extra outputs for intermediate layers if enabled."""
        if self._output_intermediate_layers:
            return self._intermediate_outputs
        return {}
