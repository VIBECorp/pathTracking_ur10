#!/usr/bin/env python

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import json
import os
import sys
import inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(current_dir))
import numpy as np
import ray
from ray import tune
from ray.tune.logger import TBXLoggerCallback
from ray.tune import Callback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.policy import Policy
from typing import Dict
import multiprocessing
from collections import defaultdict
import logging
import klimits


METRIC_OPS = ['sum', 'average', 'max', 'min']


class CustomTrainCallbacks(DefaultCallbacks):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("="*80)
        print("CustomTrainCallbacks.__init__ CALLED")
        print("="*80)
        sys.stdout.flush()

    def _get_episode_data(self, episode):
        """Get user_data or custom_data depending on episode type."""
        # Ray 2.x: SingleAgentEpisode uses custom_data, MultiAgentEpisode uses user_data
        if isinstance(episode, SingleAgentEpisode):
            return episode.custom_data
        elif isinstance(episode, MultiAgentEpisode):
            return episode.user_data
        else:
            # Fallback: try custom_data first, then user_data
            if hasattr(episode, 'custom_data'):
                return episode.custom_data
            elif hasattr(episode, 'user_data'):
                return episode.user_data
            else:
                # Create a dict if neither exists
                episode.custom_data = {}
                return episode.custom_data

    def _set_custom_metric(self, episode, key, value):
        """Set custom metric, supporting both SingleAgentEpisode and MultiAgentEpisode."""
        # Ray 2.x: Store custom metrics in multiple places for compatibility
        # 1. Store in custom_data/user_data (for our own tracking)
        episode_data = self._get_episode_data(episode)
        if 'custom_metrics' not in episode_data:
            episode_data['custom_metrics'] = {}
        episode_data['custom_metrics'][key] = value
        
        # 2. Try to set as episode attribute for Ray RLlib auto-aggregation
        # Ray RLlib may check for episode.custom_metrics attribute
        try:
            if not hasattr(episode, 'custom_metrics'):
                episode.custom_metrics = {}
            episode.custom_metrics[key] = value
        except (AttributeError, TypeError):
            # If we can't set custom_metrics attribute, that's okay
            # We'll collect from custom_data in on_train_result
            pass
        
        # Debug: Print when setting spline_deviation_termination_rate (only first few episodes)
        if 'spline_deviation_termination_rate' in key:
            # Use a simple counter to limit prints
            if not hasattr(self, '_episode_count'):
                self._episode_count = 0
            self._episode_count += 1
            if self._episode_count <= 5:
                print(f"[_set_custom_metric] Setting {key} = {value}")
                sys.stdout.flush()

    def _get_last_info(self, episode):
        """Get last info from episode, supporting both SingleAgentEpisode and MultiAgentEpisode."""
        # Ray 2.x: Use get_infos() instead of last_info_for()
        if hasattr(episode, 'get_infos'):
            infos = episode.get_infos()
            if infos and len(infos) > 0:
                return infos[-1]
        # Fallback for MultiAgentEpisode (if it has last_info_for)
        if hasattr(episode, 'last_info_for'):
            return episode.last_info_for()
        return None

    def on_episode_start(self, *, episode, env_index: int,
                         env_runner=None, worker=None, base_env: BaseEnv = None,
                         policies: Dict[str, Policy] = None, **kwargs):
        # Ray 2.x: episode and env_index are required, others are optional
        # Support both SingleAgentEpisode and MultiAgentEpisode
        if not hasattr(self, '_episode_start_count'):
            self._episode_start_count = 0
        self._episode_start_count += 1
        if self._episode_start_count <= 3:
            print(f"[on_episode_start] Episode {self._episode_start_count} started")
            sys.stdout.flush()
        episode_data = self._get_episode_data(episode)
        episode_data['op'] = {}
        for op in METRIC_OPS:
            episode_data['op'][op] = defaultdict(list)

    def on_episode_step(self, *, episode, env_index: int,
                        env_runner=None, worker=None, base_env: BaseEnv = None, **kwargs):
        # Ray 2.x: episode and env_index are required
        episode_data = self._get_episode_data(episode)
        episode_info = self._get_last_info(episode)
        if episode_info:
            for op in list(episode_info.keys() & METRIC_OPS):
                for k, v in episode_info[op].items():
                    episode_data['op'][op][k].append(v)

    def on_episode_end(self, *, episode, env_index: int,
                       env_runner=None, worker=None, base_env: BaseEnv = None,
                       policies: Dict[str, Policy] = None, **kwargs):
        # Ray 2.x: episode and env_index are required
        # Debug: Print when callback is called
        if not hasattr(self, '_episode_end_count'):
            self._episode_end_count = 0
        self._episode_end_count += 1
        if self._episode_end_count <= 5:
            print(f"[on_episode_end] Episode {self._episode_end_count} ended, episode_id={getattr(episode, 'id_', 'unknown')}")
            sys.stdout.flush()
        
        # Get base_env from env_runner if not provided
        if base_env is None and env_runner is not None:
            if hasattr(env_runner, 'env'):
                base_env = env_runner.env
            elif hasattr(env_runner, 'base_env'):
                base_env = env_runner.base_env
        
        # Try to get the actual environment from env_runner
        env = None
        if env_runner is not None:
            # Try to get env from env_runner's envs list
            if hasattr(env_runner, 'envs') and env_runner.envs:
                if isinstance(env_runner.envs, list) and len(env_runner.envs) > env_index:
                    env = env_runner.envs[env_index]
                elif not isinstance(env_runner.envs, list):
                    env = env_runner.envs
        
        if base_env is None and env is None:
            return
        def __apply_op_on_list(operator, data_list):
            if operator == 'sum':
                return sum(data_list)
            elif operator == 'average':
                return sum(data_list) / len(data_list)
            elif operator == 'max':
                return max(data_list)
            elif operator == 'min':
                return min(data_list)

        episode_data = self._get_episode_data(episode)
        episode_info = self._get_last_info(episode)
        if episode_info is None:
            return
        episode_length = episode_info.get('episode_length', 0)
        trajectory_length = episode_info.get('trajectory_length', 0)
        
        # Ray 2.x: Try to get unwrapped environment from base_env or env_runner
        # If env is not already set from env_runner, try base_env
        if env is None and base_env is not None:
            if hasattr(base_env, 'get_unwrapped'):
                try:
                    unwrapped = base_env.get_unwrapped()
                    if unwrapped and len(unwrapped) > 0:
                        env = unwrapped[-1]
                except (AttributeError, TypeError):
                    pass
            # If get_unwrapped() doesn't work, try accessing env directly
            if env is None:
                if hasattr(base_env, 'env'):
                    env = base_env.env
                    # If env is a list (vectorized), get the first one
                    if isinstance(env, list) and len(env) > 0:
                        env = env[0]
                elif hasattr(base_env, 'unwrapped'):
                    env = base_env.unwrapped
        
        # Unwrap SyncVectorEnv or other gymnasium wrappers to get the actual environment
        if env is not None:
            # Unwrap gymnasium wrappers (SyncVectorEnv, DictInfoToList, etc.)
            while hasattr(env, 'env') or hasattr(env, 'unwrapped'):
                if hasattr(env, 'envs') and isinstance(env.envs, list) and len(env.envs) > 0:
                    # SyncVectorEnv: get the first environment
                    env = env.envs[0]
                elif hasattr(env, 'env'):
                    env = env.env
                    # If it's still a list, get the first element
                    if isinstance(env, list) and len(env) > 0:
                        env = env[0]
                elif hasattr(env, 'unwrapped'):
                    env = env.unwrapped
                else:
                    break
                # Stop if we've reached the actual TrackingBase environment
                if hasattr(env, 'TERMINATION_JOINT_LIMITS'):
                    break
        
        # Set custom metrics using helper method
        for op in METRIC_OPS:
            for k, v in episode_data['op'][op].items():
                self._set_custom_metric(episode, k + '_' + op, __apply_op_on_list(op, episode_data['op'][op][k]))

        for k, v in episode_info.items():
            if k.startswith('obstacles'):
                self._set_custom_metric(episode, k, v)
            if "spline" in k and not np.isnan(v):
                self._set_custom_metric(episode, k, v)

        self._set_custom_metric(episode, 'episode_length', float(episode_length))
        self._set_custom_metric(episode, 'trajectory_length', trajectory_length)

        if 'trajectory_fraction' in episode_info:
            self._set_custom_metric(episode, 'trajectory_fraction', episode_info['trajectory_fraction'])

        # if 'reference_spline_length' in episode_info:  # should be covered by spline in k
        #    self._set_custom_metric(episode, 'reference_spline_length', episode_info['reference_spline_length'])

        if 'trajectory_successful' in episode_info:
            self._set_custom_metric(episode, 'success_rate', episode_info['trajectory_successful'])
        else:
            self._set_custom_metric(episode, 'success_rate', 0.0)

        # Environment-specific metrics (only if env is available)
        if env is not None:
            termination_reason = episode_info.get('termination_reason', -1)
            if termination_reason == env.TERMINATION_JOINT_LIMITS:
                self._set_custom_metric(episode, 'joint_limit_violation_termination_rate', 1.0)
            else:
                self._set_custom_metric(episode, 'joint_limit_violation_termination_rate', 0.0)

            if env.use_splines:
                termination_reasons_spline_dict = {env.TERMINATION_TRAJECTORY_LENGTH: 'trajectory_length_termination_rate',
                                                   env.TERMINATION_SPLINE_LENGTH: 'spline_length_termination_rate',
                                                   env.TERMINATION_SPLINE_DEVIATION: 'spline_deviation_termination_rate',
                                                   env.TERMINATION_ROBOT_STOPPED: 'robot_stopped_termination_rate'}

                for k, v in termination_reasons_spline_dict.items():
                    value = 1.0 if termination_reason == k else 0.0
                    self._set_custom_metric(episode, v, value)
                    # Debug: Print when setting spline_deviation_termination_rate
                    if v == 'spline_deviation_termination_rate':
                        if not hasattr(self, '_spline_debug_count'):
                            self._spline_debug_count = 0
                        self._spline_debug_count += 1
                        if self._spline_debug_count <= 3:
                            print(f"[on_episode_end] termination_reason={termination_reason}, TERMINATION_SPLINE_DEVIATION={env.TERMINATION_SPLINE_DEVIATION}, setting {v}={value}")
                            episode_data = self._get_episode_data(episode)
                            if 'custom_metrics' in episode_data:
                                print(f"  episode.custom_data['custom_metrics'] keys: {list(episode_data['custom_metrics'].keys())[:10]}")
                            sys.stdout.flush()

                if env.sphere_balancing_mode:
                    if termination_reason == env.TERMINATION_BALANCING:
                        self._set_custom_metric(episode, 'balancing_termination_rate', 1.0)
                    else:
                        self._set_custom_metric(episode, 'balancing_termination_rate', 0.0)

            if env.floating_robot_base:
                termination_reasons_base_dict = \
                    {env.TERMINATION_BASE_POS_DEVIATION: 'robot_base_pos_deviation_termination_rate',
                     env.TERMINATION_BASE_ORN_DEVIATION: 'robot_base_orn_deviation_termination_rate',
                     env.TERMINATION_BASE_Z_ANGLE_DEVIATION: 'robot_base_z_angle_deviation_termination_rate'}

                for k, v in termination_reasons_base_dict.items():
                    self._set_custom_metric(episode, v, 1.0 if termination_reason == k else 0.0)

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # Ray 2.x: 'trainer' parameter renamed to 'algorithm'
        result['callback_ok'] = True
        
        iteration = result.get('training_iteration', 0)
        print(f"\n{'='*80}")
        print(f"on_train_result CALLED for iteration {iteration}")
        print(f"{'='*80}")
        sys.stdout.flush()
        
        # Aggregate custom metrics from episodes and add to result dict for TensorBoard logging
        # Ray 2.x: Custom metrics need to be explicitly added to result dict
        if 'custom_metrics' not in result:
            result['custom_metrics'] = {}
        
        # Collect custom metrics from all workers
        # Ray 2.x: Check various places where custom metrics might be stored
        
        # Debug: Print hist_stats keys to see what's available
        print(f"Checking hist_stats...")
        sys.stdout.flush()
        if 'hist_stats' in result:
            hist_stats = result['hist_stats']
            if isinstance(hist_stats, dict):
                all_keys = list(hist_stats.keys())
                print(f"hist_stats has {len(all_keys)} keys")
                print(f"Sample keys: {all_keys[:20]}")
                # Check specifically for spline_deviation_termination_rate
                spline_keys = [k for k in all_keys if 'spline_deviation' in k.lower() or 'termination_rate' in k.lower()]
                if spline_keys:
                    print(f"✓ Found spline/termination_rate keys: {spline_keys[:10]}")
                else:
                    print(f"✗ spline_deviation_termination_rate NOT found in hist_stats!")
                sys.stdout.flush()
        else:
            print("✗ hist_stats not in result!")
            sys.stdout.flush()
        
        # 1. Check hist_stats - Ray RLlib stores episode metrics here
        # Custom metrics from episode.custom_data['custom_metrics'] should appear here
        if 'hist_stats' in result:
            hist_stats = result['hist_stats']
            
            # Check for episode_custom_metrics dict in hist_stats
            if isinstance(hist_stats, dict):
                # Look for keys that match our custom metric names
                # Ray RLlib may store them with 'episode_' prefix or directly
                custom_metric_keys = [
                    'trajectory_fraction',
                    'spline_deviation_termination_rate',
                    'trajectory_length_termination_rate',
                    'spline_length_termination_rate',
                    'robot_stopped_termination_rate',
                    'joint_limit_violation_termination_rate',
                    'balancing_termination_rate',
                    'robot_base_pos_deviation_termination_rate',
                    'robot_base_orn_deviation_termination_rate',
                    'robot_base_z_angle_deviation_termination_rate',
                    'success_rate',
                    'episode_length',
                    'trajectory_length',
                ]
                
                # Process all metrics in hist_stats and add mean, min, max
                for key, values in hist_stats.items():
                    if isinstance(values, list) and len(values) > 0:
                        # Remove 'episode_' prefix if present
                        metric_name = key.replace('episode_', '')
                        
                        # Check if this is a custom metric we care about
                        is_custom_metric = (
                            metric_name in custom_metric_keys or
                            any(cmk in metric_name for cmk in custom_metric_keys) or
                            metric_name.endswith('_termination_rate') or
                            metric_name.endswith('_rate') or
                            metric_name == 'trajectory_fraction'
                        )
                        
                        if is_custom_metric:
                            # Calculate statistics
                            mean_value = float(np.mean(values))
                            min_value = float(np.min(values))
                            max_value = float(np.max(values))
                            
                            # Add all three statistics to custom_metrics
                            # Ray Tune will automatically log these to TensorBoard under ray/tune/custom_metrics/
                            result['custom_metrics'][metric_name] = mean_value
                            result['custom_metrics'][metric_name + '_mean'] = mean_value
                            result['custom_metrics'][metric_name + '_min'] = min_value
                            result['custom_metrics'][metric_name + '_max'] = max_value
                            
                            # Debug: Print when we find spline_deviation_termination_rate
                            if 'spline_deviation_termination_rate' in metric_name:
                                print(f"✓ Found {metric_name} in hist_stats[{key}]")
                                print(f"  Values count: {len(values)}, sample: {values[:5] if len(values) > 5 else values}")
                                print(f"  mean={mean_value:.6f}, min={min_value:.6f}, max={max_value:.6f}")
                                print(f"  Added to custom_metrics: {metric_name}, {metric_name}_mean, {metric_name}_min, {metric_name}_max")
                                sys.stdout.flush()
        
        # 4. If hist_stats doesn't have the metric, try to collect from workers directly
        # This is a fallback in case Ray RLlib 2.x doesn't auto-aggregate custom_data['custom_metrics']
        if 'spline_deviation_termination_rate' not in result.get('custom_metrics', {}) and hasattr(algorithm, 'workers'):
            try:
                # Try to get custom metrics from workers' episodes
                # This is a workaround for Ray 2.x if auto-aggregation doesn't work
                all_custom_metrics = {}
                if hasattr(algorithm.workers, 'local_worker'):
                    local_worker = algorithm.workers.local_worker()
                    if hasattr(local_worker, 'sampler') and hasattr(local_worker.sampler, 'episodes'):
                        episodes = local_worker.sampler.episodes
                        for episode in episodes:
                            episode_data = self._get_episode_data(episode)
                            if 'custom_metrics' in episode_data:
                                for key, value in episode_data['custom_metrics'].items():
                                    if key not in all_custom_metrics:
                                        all_custom_metrics[key] = []
                                    all_custom_metrics[key].append(value)
                
                # Aggregate collected metrics
                for key, values in all_custom_metrics.items():
                    if len(values) > 0:
                        mean_value = float(np.mean(values))
                        min_value = float(np.min(values))
                        max_value = float(np.max(values))
                        result['custom_metrics'][key] = mean_value
                        result['custom_metrics'][key + '_mean'] = mean_value
                        result['custom_metrics'][key + '_min'] = min_value
                        result['custom_metrics'][key + '_max'] = max_value
                        
                        if 'spline_deviation_termination_rate' in key:
                            print(f"✓ Collected {key} from workers")
                            print(f"  Values count: {len(values)}, sample: {values[:5] if len(values) > 5 else values}")
                            print(f"  mean={mean_value:.6f}, min={min_value:.6f}, max={max_value:.6f}")
                            print(f"  Added to custom_metrics: {key}, {key}_mean, {key}_min, {key}_max")
                            sys.stdout.flush()
            except Exception as e:
                # Silently fail if we can't access workers
                if result.get('training_iteration', 0) <= 3:
                    logging.debug("Could not collect custom metrics from workers: %s", str(e))
        
        # 2. Check episode_custom_metrics (direct aggregation)
        if 'episode_custom_metrics' in result:
            for key, value in result['episode_custom_metrics'].items():
                result['custom_metrics'][key] = value
        
        # 3. Check info dict for custom metrics
        if 'info' in result:
            info = result['info']
            if 'learner' in info:
                learner_info = info['learner']
                if 'default_policy' in learner_info:
                    policy_info = learner_info['default_policy']
                    if 'custom_metrics' in policy_info:
                        for key, value in policy_info['custom_metrics'].items():
                            result['custom_metrics'][key] = value
        
        # Debug: Print final custom_metrics to verify they're being added
        print(f"Final custom_metrics check...")
        sys.stdout.flush()
        custom_metrics_keys = list(result.get('custom_metrics', {}).keys())
        spline_custom_keys = [k for k in custom_metrics_keys if 'spline_deviation_termination_rate' in k]
        if spline_custom_keys:
            print(f"✓ Found spline_deviation_termination_rate in custom_metrics: {spline_custom_keys}")
            for key in spline_custom_keys:
                print(f"  {key} = {result['custom_metrics'][key]}")
        else:
            print(f"✗ spline_deviation_termination_rate NOT in custom_metrics!")
            print(f"  Available custom_metrics keys: {custom_metrics_keys[:30]}")
        
        print(f"{'='*80}")
        print(f"End Iteration {iteration}")
        print(f"{'='*80}\n")
        sys.stdout.flush()


class CustomTuneCallback(Callback):
    """Ray Tune callback to collect and log custom metrics from RLlib results."""
    
    def on_trial_result(self, iteration, trials, trial, result, **info):
        """Called when a trial reports a result."""
        # This is called by Ray Tune, not RLlib
        # result contains the metrics from RLlib's training iteration
        
        iteration_num = result.get('training_iteration', 0)
        if iteration_num <= 3:
            print(f"\n{'='*80}")
            print(f"[CustomTuneCallback] on_trial_result for iteration {iteration_num}")
            print(f"Result keys: {list(result.keys())[:30]}")
            sys.stdout.flush()
        
        # Check if custom_metrics are already in result (from RLlib callback)
        if 'custom_metrics' not in result:
            result['custom_metrics'] = {}
        
        # Debug: Print all result keys to understand structure
        if iteration_num <= 3:
            print(f"  Checking result structure...")
            if 'hist_stats' in result:
                print(f"  ✓ hist_stats found at top level: {list(result['hist_stats'].keys())[:20]}")
            else:
                print(f"  ✗ hist_stats NOT found at top level")
            if 'env_runners' in result:
                env_runners_keys = list(result['env_runners'].keys())
                print(f"  ✓ env_runners found: {env_runners_keys[:20]}")
                if 'hist_stats' in result['env_runners']:
                    hist_stats_keys = list(result['env_runners']['hist_stats'].keys())
                    print(f"  ✓ hist_stats found in env_runners: {hist_stats_keys[:30]}")
                    # Check for spline_deviation_termination_rate
                    spline_keys = [k for k in hist_stats_keys if 'spline_deviation' in k.lower() or 'termination_rate' in k.lower()]
                    if spline_keys:
                        print(f"  ✓ Found spline/termination_rate keys: {spline_keys[:10]}")
                    # Debug: Print sample values from hist_stats
                    for key in hist_stats_keys[:5]:
                        value = result['env_runners']['hist_stats'][key]
                        if isinstance(value, list):
                            print(f"    hist_stats['{key}']: list with {len(value)} items, sample: {value[:3] if len(value) > 3 else value}")
                        else:
                            print(f"    hist_stats['{key}']: {type(value).__name__}, value: {value}")
                if 'custom_metrics' in result['env_runners']:
                    custom_metrics_keys = list(result['env_runners']['custom_metrics'].keys())
                    print(f"  ✓ custom_metrics found in env_runners: {custom_metrics_keys[:20]}")
            sys.stdout.flush()
        
        # Check hist_stats for custom metrics - it's in env_runners!
        hist_stats = None
        if 'env_runners' in result and 'hist_stats' in result['env_runners']:
            hist_stats = result['env_runners']['hist_stats']
        elif 'hist_stats' in result:
            hist_stats = result['hist_stats']
        
        if hist_stats:
            if isinstance(hist_stats, dict):
                custom_metric_keys = [
                    'trajectory_fraction',
                    'spline_deviation_termination_rate',
                    'trajectory_length_termination_rate',
                    'spline_length_termination_rate',
                    'robot_stopped_termination_rate',
                    'joint_limit_violation_termination_rate',
                    'balancing_termination_rate',
                    'robot_base_pos_deviation_termination_rate',
                    'robot_base_orn_deviation_termination_rate',
                    'robot_base_z_angle_deviation_termination_rate',
                    'success_rate',
                    'episode_length',
                    'trajectory_length',
                ]
                
                for key, values in hist_stats.items():
                    if isinstance(values, list) and len(values) > 0:
                        metric_name = key.replace('episode_', '')
                        
                        is_custom_metric = (
                            metric_name in custom_metric_keys or
                            any(cmk in metric_name for cmk in custom_metric_keys) or
                            metric_name.endswith('_termination_rate') or
                            metric_name.endswith('_rate') or
                            metric_name == 'trajectory_fraction'
                        )
                        
                        if is_custom_metric:
                            mean_value = float(np.mean(values))
                            min_value = float(np.min(values))
                            max_value = float(np.max(values))
                            
                            result['custom_metrics'][metric_name] = mean_value
                            result['custom_metrics'][metric_name + '_mean'] = mean_value
                            result['custom_metrics'][metric_name + '_min'] = min_value
                            result['custom_metrics'][metric_name + '_max'] = max_value
                            
                            if 'spline_deviation_termination_rate' in metric_name:
                                print(f"✓ Found {metric_name} in hist_stats[{key}]")
                                print(f"  Values count: {len(values)}, mean={mean_value:.6f}, min={min_value:.6f}, max={max_value:.6f}")
                                sys.stdout.flush()
        
        # Also check env_runners/custom_metrics (Ray 2.x may store metrics here)
        if 'env_runners' in result and 'custom_metrics' in result['env_runners']:
            env_runners_custom_metrics = result['env_runners']['custom_metrics']
            if isinstance(env_runners_custom_metrics, dict):
                for key, value in env_runners_custom_metrics.items():
                    result['custom_metrics'][key] = float(value) if isinstance(value, (int, float)) else value
                    if iteration_num <= 3 and ('spline_deviation' in key.lower() or 'termination_rate' in key.lower()):
                        print(f"✓ Found {key} in env_runners/custom_metrics: {value}")
                        sys.stdout.flush()
        
        # Also check env_runners dict directly for scalar metrics
        if 'env_runners' in result:
            env_runners_metrics = result['env_runners']
            if isinstance(env_runners_metrics, dict):
                for key, value in env_runners_metrics.items():
                    if isinstance(value, (int, float)) and ('spline_deviation' in key.lower() or 'termination_rate' in key.lower()):
                        result['custom_metrics'][key] = float(value)
                        if iteration_num <= 3:
                            print(f"✓ Found {key} in env_runners: {value}")
                            sys.stdout.flush()
        
        # Debug: Print final custom_metrics
        if iteration_num <= 3:
            custom_metrics_keys = list(result.get('custom_metrics', {}).keys())
            spline_custom_keys = [k for k in custom_metrics_keys if 'spline_deviation_termination_rate' in k]
            if spline_custom_keys:
                print(f"✓ spline_deviation_termination_rate in custom_metrics: {spline_custom_keys}")
                for key in spline_custom_keys:
                    print(f"  {key} = {result['custom_metrics'][key]}")
            else:
                print(f"✗ spline_deviation_termination_rate NOT in custom_metrics!")
                print(f"  Available custom_metrics keys: {custom_metrics_keys[:20]}")
            print(f"{'='*80}\n")
            sys.stdout.flush()


def _make_env_config():

    env_config = {
        'experiment_name': args.name,
        'm_prev': args.m_prev,
        'pos_limit_factor': args.pos_limit_factor,
        'vel_limit_factor': args.vel_limit_factor,
        'acc_limit_factor': args.acc_limit_factor,
        'jerk_limit_factor': args.jerk_limit_factor,
        'torque_limit_factor': args.torque_limit_factor,
        'action_mapping_factor': args.action_mapping_factor,
        'set_highest_action_to_one': args.set_highest_action_to_one,
        'action_preprocessing_function': args.action_preprocessing_function,
        'normalize_reward_to_frequency': args.normalize_reward_to_frequency,
        'online_trajectory_duration': args.online_trajectory_duration,
        'online_trajectory_time_step': args.online_trajectory_time_step,
        'obs_add_target_point_pos': args.obs_add_target_point_pos,
        'obs_add_target_point_relative_pos': args.obs_add_target_point_relative_pos,
        'punish_action': args.punish_action,
        'action_punishment_min_threshold': args.action_punishment_min_threshold,
        'action_max_punishment': args.action_max_punishment,
        'reward_action': args.reward_action,
        'action_reward_min_threshold': args.action_reward_min_threshold,
        'action_reward_peak': args.action_reward_peak,
        'action_max_reward': args.action_max_reward,
        'punish_adaptation': args.punish_adaptation,
        'adaptation_max_punishment': args.adaptation_max_punishment,
        'punish_end_min_distance': args.punish_end_min_distance,
        'end_min_distance_max_threshold': args.end_min_distance_max_threshold,
        'end_min_distance_max_punishment': args.end_min_distance_max_punishment,
        'punish_end_max_torque': args.punish_end_max_torque,
        'end_max_torque_min_threshold': args.end_max_torque_min_threshold,
        'end_max_torque_max_punishment': args.end_max_torque_max_punishment,
        'obstacle_scene': args.obstacle_scene,
        'activate_obstacle_collisions': args.activate_obstacle_collisions,
        'log_obstacle_data': False,
        'check_braking_trajectory_collisions': args.check_braking_trajectory_collisions,
        'check_braking_trajectory_torque_limits': args.check_braking_trajectory_torque_limits,
        'collision_check_time': args.collision_check_time,
        'closest_point_safety_distance': args.closest_point_safety_distance,
        'use_target_points': args.use_target_points,
        'acc_limit_factor_braking': args.acc_limit_factor_braking,
        'jerk_limit_factor_braking': args.jerk_limit_factor_braking,
        'punish_braking_trajectory_min_distance': args.punish_braking_trajectory_min_distance,
        'braking_trajectory_min_distance_max_threshold': args.braking_trajectory_min_distance_max_threshold,
        'braking_trajectory_max_punishment': args.braking_trajectory_max_punishment,
        'punish_braking_trajectory_max_torque': args.punish_braking_trajectory_max_torque,
        'braking_trajectory_max_torque_min_threshold': args.braking_trajectory_max_torque_min_threshold,
        'robot_scene': args.robot_scene,
        'num_virtual_motors': args.num_virtual_motors,
        'no_self_collision': args.no_self_collision,
        'terminate_on_robot_stop': args.terminate_on_robot_stop,
        'use_controller_target_velocities': args.use_controller_target_velocities,
        'target_point_cartesian_range_scene': args.target_point_cartesian_range_scene,
        'target_point_relative_pos_scene': args.target_point_relative_pos_scene,
        'target_point_radius': args.target_point_radius,
        'target_point_sequence': args.target_point_sequence,
        'target_point_reached_reward_bonus': args.target_point_reached_reward_bonus,
        'target_point_reward_factor': args.target_point_reward_factor,
        'target_point_use_actual_position': args.target_point_use_actual_position,
        'normalize_reward_to_initial_target_point_distance': args.normalize_reward_to_initial_target_point_distance,
        'use_splines': args.spline_dir is not None,
        'spline_dir': args.spline_dir,
        'spline_config_path': args.spline_config_path,
        'spline_u_arc_start_range': args.spline_u_arc_start_range,
        'spline_u_arc_diff_min': args.spline_u_arc_diff_min,
        'spline_deviation_weighting_factors': args.spline_deviation_weighting_factors,
        'spline_speed_range': args.spline_speed_range,
        'spline_random_speed_per_time_step': args.spline_random_speed_per_time_step,
        'spline_normalize_duration': args.spline_normalize_duration,
        'spline_use_reflection_vectors': args.spline_use_reflection_vectors,
        'spline_final_overshoot_factor': args.spline_final_overshoot_factor,
        'spline_final_distance_reward': args.spline_final_distance_reward,
        'spline_termination_max_deviation': args.spline_termination_max_deviation,
        'spline_termination_extra_time_steps': args.spline_termination_extra_time_steps,
        'spline_max_final_deviation': args.spline_max_final_deviation,
        'spline_braking_extra_time_steps': args.spline_braking_extra_time_steps,
        'obs_spline_n_next': args.obs_spline_n_next,
        'obs_spline_add_length': args.obs_spline_add_length,
        'obs_spline_add_distance_per_knot': args.obs_spline_add_distance_per_knot,
        'obs_spline_use_distance_between_knots': args.obs_spline_use_distance_between_knots,
        'spline_distance_max_reward': args.spline_distance_max_reward,
        'spline_deviation_max_threshold': args.spline_deviation_max_threshold,
        'punish_spline_max_deviation': args.punish_spline_max_deviation,
        'spline_max_deviation_max_punishment': args.spline_max_deviation_max_punishment,
        'punish_spline_mean_deviation': args.punish_spline_mean_deviation,
        'spline_mean_deviation_max_punishment': args.spline_mean_deviation_max_punishment,
        'spline_cartesian_deviation_max_threshold': args.spline_cartesian_deviation_max_threshold,
        'punish_spline_max_cartesian_deviation': args.punish_spline_max_cartesian_deviation,
        'spline_max_cartesian_deviation_max_punishment': args.spline_max_cartesian_deviation_max_punishment,
        'punish_spline_mean_cartesian_deviation': args.punish_spline_mean_cartesian_deviation,
        'spline_mean_cartesian_deviation_max_punishment': args.spline_mean_cartesian_deviation_max_punishment,
        'sphere_balancing_mode': args.sphere_balancing_mode,
        'balancing_sphere_dev_min_max': args.balancing_sphere_dev_min_max,
        'terminate_on_balancing_sphere_deviation': args.terminate_on_balancing_sphere_deviation,
        'terminate_balancing_sphere_not_on_board': args.terminate_balancing_sphere_not_on_board,
        'balancing_sphere_max_reward': args.balancing_sphere_max_reward,
        'floating_robot_base': args.floating_robot_base,
        'robot_base_balancing_mode': args.robot_base_balancing_mode,
        'balancing_robot_base_max_pos_deviation': args.balancing_robot_base_max_pos_deviation,
        'balancing_robot_base_pos_max_reward': args.balancing_robot_base_pos_max_reward,
        'balancing_robot_base_max_orn_deviation': args.balancing_robot_base_max_orn_deviation,
        'balancing_robot_base_orn_max_reward': args.balancing_robot_base_orn_max_reward,
        'balancing_robot_base_max_z_angle_deviation_rad': args.balancing_robot_base_max_z_angle_deviation_rad,
        'balancing_robot_base_z_angle_max_reward': args.balancing_robot_base_z_angle_max_reward,
        'balancing_robot_base_punish_last_cartesian_action_point':
            args.balancing_robot_base_punish_last_cartesian_action_point,
        'balancing_robot_base_spline_last_cartesian_action_point_deviation_max_threshold':
            args.balancing_robot_base_spline_last_cartesian_action_point_deviation_max_threshold,
        'balancing_robot_base_spline_last_cartesian_action_point_deviation_max_punishment':
            args.balancing_robot_base_spline_last_cartesian_action_point_deviation_max_punishment,
        'terminate_on_balancing_robot_base_pos_deviation': args.terminate_on_balancing_robot_base_pos_deviation,
        'terminate_on_balancing_robot_base_orn_deviation': args.terminate_on_balancing_robot_base_orn_deviation,
        'terminate_on_balancing_robot_base_z_angle_deviation': args.terminate_on_balancing_robot_base_z_angle_deviation,
        'target_link_offset': args.target_link_offset,
        'obstacle_use_computed_actual_values': args.obstacle_use_computed_actual_values,
        'solver_iterations': args.solver_iterations,
        'logging_level': args.logging_level,
    }

    if hasattr(klimits, '__version__'):
        env_config['klimits_version'] = klimits.__version__

    if hasattr(ray, '__version__'):
        env_config['ray_version'] = ray.__version__

    return env_config


if __name__ == '__main__':
    config = {
        # Ray 2.x: TensorFlow support removed from RLlib, using PyTorch
        'framework': 'torch',
        # Ray 2.x: disable new API stack to use legacy custom_model API
        '_enable_rl_module_and_learner': False,
        # Force legacy API: disable execution plan API and use simple optimizer
        '_disable_execution_plan_api': True,
        'enable_env_runner_and_connector_v2': False,
        'simple_optimizer': True,
        'num_learners': 0,
        'model': {
            'conv_filters': None,
            'fcnet_hiddens': [256, 128],
            'fcnet_activation': None,  # set at a later point
            'use_lstm': False,
        },
        'gamma': 0.99,
        'use_gae': True,
        'lambda': 1.0,
        'kl_coeff': 0.2,
        'rollout_fragment_length': None,  # set at a later point
        'train_batch_size': 49920,
        'sgd_minibatch_size': 1024,
        'num_sgd_iter': 16,
        'lr': 5e-5,
        'lr_schedule': None,
        'vf_loss_coeff': 1.0,
        'entropy_coeff': None,
        'clip_param': 0.3,
        'vf_clip_param': None,
        'kl_target': None,
        'batch_mode': 'complete_episodes',
        'normalize_actions': True,
        'evaluation_interval': None,
        # Ray 2.x: evaluation_num_episodes -> evaluation_duration + evaluation_duration_unit
        'evaluation_duration': 624,
        'evaluation_duration_unit': 'episodes',
        'evaluation_parallel_to_training': False,
        'evaluation_config': {
            "explore": False,
            "rollout_fragment_length": 1},
        # sample a single episode per self.evaluation_workers.local_worker().sample() in trainer.py
        # Ray 2.x: evaluation_num_workers is deprecated, use evaluation_num_env_runners
        'evaluation_num_env_runners': 0,
    }

    algorithm = 'PPO'
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='default_name')
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a checkpoint if training should be continued.')
    parser.add_argument('--time', type=int, required=True,
                        help='Total time of the training in hours.')
    parser.add_argument('--iterations_per_checkpoint', type=int, default=500,
                        help='The number of training iterations per checkpoint')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--num_threads_per_worker', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=None)
    parser.add_argument('--m_prev', type=int, default=0)
    parser.add_argument('--pos_limit_factor', type=float, default=1.0)
    parser.add_argument('--vel_limit_factor', type=float, default=1.0)
    parser.add_argument('--acc_limit_factor', type=float, default=1.0)
    parser.add_argument('--jerk_limit_factor', type=float, default=1.0)
    parser.add_argument('--torque_limit_factor', type=float, default=1.0)
    parser.add_argument('--action_mapping_factor', type=float, default=1.0)
    parser.add_argument('--set_highest_action_to_one', action='store_true', default=False)
    parser.add_argument('--action_preprocessing_function', default=None, choices=['tanh'])
    parser.add_argument('--normalize_reward_to_frequency', dest='normalize_reward_to_frequency', action='store_true',
                        default=False)
    parser.add_argument('--batch_size_factor', type=float, default=1.0)
    parser.add_argument('--online_trajectory_duration', type=float, default=8.0)
    parser.add_argument('--online_trajectory_time_step', type=float, default=0.1)
    parser.add_argument('--obs_add_target_point_pos', action='store_true', default=False)
    parser.add_argument('--obs_add_target_point_relative_pos', action='store_true', default=False)
    parser.add_argument('--punish_action', action='store_true', default=False)
    parser.add_argument('--action_punishment_min_threshold', type=float, default=0.9)
    parser.add_argument('--action_max_punishment', type=float, default=0.5)
    parser.add_argument('--reward_action', action='store_true', default=False)
    parser.add_argument('--action_reward_min_threshold', type=float, default=0.9)
    parser.add_argument('--action_reward_peak', type=float, default=0.95)
    parser.add_argument('--action_max_reward', type=float, default=1.0)
    parser.add_argument('--punish_adaptation', action='store_true', default=False)
    parser.add_argument('--adaptation_max_punishment', type=float, default=1.0)
    parser.add_argument('--punish_end_min_distance', action='store_true', default=False)
    parser.add_argument('--end_min_distance_max_threshold', type=float, default=0.05)
    parser.add_argument('--end_min_distance_max_punishment', type=float, default=1.0)
    parser.add_argument('--punish_end_max_torque', action='store_true', default=False)
    parser.add_argument('--end_max_torque_min_threshold', type=float, default=0.9)
    parser.add_argument('--end_max_torque_max_punishment', type=float, default=1.0)
    parser.add_argument('--punish_braking_trajectory_min_distance', action='store_true', default=False)
    parser.add_argument('--braking_trajectory_min_distance_max_threshold', type=float, default=0.05)
    parser.add_argument('--braking_trajectory_max_punishment', type=float, default=1.0)
    parser.add_argument('--punish_braking_trajectory_max_torque', action='store_true', default=False)
    parser.add_argument('--braking_trajectory_max_torque_min_threshold', type=float, default=0.8)
    parser.add_argument('--obstacle_scene', type=int, default=0)
    parser.add_argument('--activate_obstacle_collisions', action='store_true', default=False)
    parser.add_argument('--check_braking_trajectory_collisions', action='store_true', default=False)
    parser.add_argument('--check_braking_trajectory_torque_limits', action='store_true', default=False)
    parser.add_argument('--collision_check_time', type=float, default=None)
    parser.add_argument('--closest_point_safety_distance', type=float, default=0.1)
    parser.add_argument('--use_target_points', action='store_true', default=False)
    parser.add_argument('--acc_limit_factor_braking', type=float, default=1.0)
    parser.add_argument('--jerk_limit_factor_braking', type=float, default=1.0)
    parser.add_argument('--robot_scene', type=int, default=0)
    parser.add_argument('--num_virtual_motors', type=int, default=0)
    parser.add_argument('--no_self_collision', action='store_true', default=False)
    parser.add_argument('--terminate_on_robot_stop', action='store_true', default=False)
    parser.add_argument('--use_controller_target_velocities', action='store_true', default=False)
    parser.add_argument('--target_point_cartesian_range_scene', type=int, default=0)
    parser.add_argument('--target_point_relative_pos_scene', type=int, default=0)
    parser.add_argument('--target_point_radius', type=float, default=0.065)
    parser.add_argument('--target_point_sequence', type=int, default=0)
    parser.add_argument('--target_point_reached_reward_bonus', type=float, default=0.00)
    parser.add_argument('--target_point_use_actual_position', action='store_true', default=False)
    parser.add_argument('--target_link_offset', type=json.loads, default='[0, 0, 0.126]')
    parser.add_argument('--target_point_reward_factor', type=float, default=1.0)
    parser.add_argument('--normalize_reward_to_initial_target_point_distance', action='store_true', default=False)
    # spline settings
    parser.add_argument('--spline_dir', type=str, default=None)
    parser.add_argument('--spline_config_path', type=str, default=None)
    parser.add_argument('--spline_u_arc_start_range', type=json.loads, default=(0, 0))
    parser.add_argument('--spline_u_arc_diff_min', type=float, default=1.0)
    parser.add_argument('--spline_deviation_weighting_factors', type=json.loads, default=None)
    parser.add_argument('--spline_normalize_duration', action='store_true', default=False)
    parser.add_argument('--spline_use_reflection_vectors', action='store_true', default=False)
    parser.add_argument('--spline_speed_range', type=json.loads, default=None)
    parser.add_argument('--spline_random_speed_per_time_step', action='store_true', default=False)
    parser.add_argument('--spline_final_overshoot_factor', type=float, default=1.0)
    parser.add_argument('--spline_final_distance_reward', type=float, default=None)
    parser.add_argument('--spline_termination_max_deviation', type=float, default=None)
    parser.add_argument('--spline_termination_extra_time_steps', type=int, default=None)
    parser.add_argument('--spline_max_final_deviation', type=float, default=0.05)
    parser.add_argument('--spline_braking_extra_time_steps', type=float, default=None)
    parser.add_argument('--obs_spline_n_next', type=int, default=5)
    parser.add_argument('--obs_spline_add_length', action='store_true', default=False)
    parser.add_argument('--obs_spline_add_distance_per_knot', action='store_true', default=False)
    parser.add_argument('--obs_spline_use_distance_between_knots', action='store_true', default=False)
    parser.add_argument('--spline_distance_max_reward', type=float, default=1.0)
    parser.add_argument('--spline_deviation_max_threshold', type=float, default=0.25)
    parser.add_argument('--punish_spline_max_deviation', action='store_true', default=False)
    parser.add_argument('--spline_max_deviation_max_punishment', type=float, default=0.1)
    parser.add_argument('--punish_spline_mean_deviation', action='store_true', default=False)
    parser.add_argument('--spline_mean_deviation_max_punishment', type=float, default=0.1)
    parser.add_argument('--spline_cartesian_deviation_max_threshold', type=float, default=0.1)
    parser.add_argument('--punish_spline_max_cartesian_deviation', action='store_true', default=False)
    parser.add_argument('--spline_max_cartesian_deviation_max_punishment', type=float, default=0.1)
    parser.add_argument('--punish_spline_mean_cartesian_deviation', action='store_true', default=False)
    parser.add_argument('--spline_mean_cartesian_deviation_max_punishment', type=float, default=0.1)
    # end of spline settings
    # sphere balancing settings
    parser.add_argument('--sphere_balancing_mode', action='store_true', default=False)
    parser.add_argument('--balancing_sphere_dev_min_max', type=json.loads, default=None)
    parser.add_argument('--terminate_on_balancing_sphere_deviation', action='store_true', default=False)
    parser.add_argument('--terminate_balancing_sphere_not_on_board', action='store_true', default=False)
    parser.add_argument('--balancing_sphere_max_reward', type=float, default=1.0)
    # end of sphere balancing settings
    # robot base balancing settings
    parser.add_argument('--floating_robot_base', action='store_true', default=False)
    parser.add_argument('--robot_base_balancing_mode', action='store_true', default=False)
    parser.add_argument('--balancing_robot_base_max_pos_deviation', type=float, default=0.2)
    parser.add_argument('--balancing_robot_base_pos_max_reward', type=float, default=0.0)
    parser.add_argument('--balancing_robot_base_max_orn_deviation', type=float, default=1.0)
    parser.add_argument('--balancing_robot_base_orn_max_reward', type=float, default=0.0)
    parser.add_argument('--balancing_robot_base_max_z_angle_deviation_rad', type=float, default=0.52)
    parser.add_argument('--balancing_robot_base_z_angle_max_reward', type=float, default=0.0)
    parser.add_argument('--balancing_robot_base_punish_last_cartesian_action_point', action='store_true', default=False)
    parser.add_argument('--balancing_robot_base_spline_last_cartesian_action_point_deviation_max_threshold',
                        type=float, default=0.4)
    parser.add_argument('--balancing_robot_base_spline_last_cartesian_action_point_deviation_max_punishment',
                        type=float, default=1.0)
    parser.add_argument('--terminate_on_balancing_robot_base_pos_deviation', action='store_true', default=False)
    parser.add_argument('--terminate_on_balancing_robot_base_orn_deviation', action='store_true', default=False)
    parser.add_argument('--terminate_on_balancing_robot_base_z_angle_deviation', action='store_true', default=False)
    # end of robot base balancing settings
    parser.add_argument('--obstacle_use_computed_actual_values', action='store_true', default=False)
    parser.add_argument('--logging_level', default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--hidden_layer_activation', default='selu', choices=['relu', 'selu', 'tanh', 'sigmoid', 'elu',
                                                                              'gelu', 'swish', 'leaky_relu'])
    parser.add_argument('--last_layer_activation', default=None, choices=['linear', 'tanh'])
    parser.add_argument('--no_log_std_activation', action='store_true', default=False)
    parser.add_argument('--solver_iterations', type=int, default=None)
    parser.add_argument('--use_dashboard', action='store_true', default=False)
    parser.add_argument('--evaluation_interval', type=int, default=None)
    parser.add_argument('--vf_clip_param', type=float, default=10.0)
    parser.add_argument('--entropy_coeff', type=float, default=0.0)
    parser.add_argument('--kl_target', type=float, default=0.01)
    parser.add_argument('--fcnet_hiddens', type=json.loads, default=None)
    parser.add_argument('--log_std_range', type=json.loads, default=None)
    parser.add_argument('--action_distribution', default=None,
                        choices=['truncated_normal', 'truncated_normal_zero_kl', 'beta_alpha_beta'])
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger().setLevel(args.logging_level)

    config['model']['fcnet_activation'] = args.hidden_layer_activation
    if args.fcnet_hiddens is not None:
        config['model']['fcnet_hiddens'] = args.fcnet_hiddens
    config['evaluation_interval'] = args.evaluation_interval
    config['vf_clip_param'] = args.vf_clip_param
    config['entropy_coeff'] = args.entropy_coeff
    config['kl_target'] = args.kl_target

    # PyTorch custom model with last_layer_activation support
    if args.last_layer_activation is not None and args.last_layer_activation != 'linear':
        from tracking.model.torch_fcnet_last_layer_activation import FullyConnectedNetworkLastLayerActivation
        ModelCatalog.register_custom_model('torch_fcnet_last_layer_activation',
                                           FullyConnectedNetworkLastLayerActivation)
        config['model']['custom_model'] = 'torch_fcnet_last_layer_activation'
        config['model']['custom_model_config'] = {
            'last_layer_activation': args.last_layer_activation,
            'no_log_std_activation': args.no_log_std_activation,
            'log_std_range': args.log_std_range,
        }
        # Copy model config parameters
        for key in ['fcnet_hiddens', 'fcnet_activation', 'post_fcnet_hiddens', 'post_fcnet_activation',
                    'no_final_layer', 'vf_share_layers', 'free_log_std']:
            if key in config['model']:
                config['model']['custom_model_config'][key] = config['model'][key]

    if args.action_distribution is not None:
        if args.action_distribution == 'truncated_normal':
            from tracking.model.custom_action_dist import TruncatedNormal
            ModelCatalog.register_custom_action_dist("truncated_normal", TruncatedNormal)
            config['model']['custom_action_dist'] = 'truncated_normal'
        if args.action_distribution == 'truncated_normal_zero_kl':
            from tracking.model.custom_action_dist import TruncatedNormalZeroKL
            ModelCatalog.register_custom_action_dist("truncated_normal_zero_kl", TruncatedNormalZeroKL)
            config['model']['custom_action_dist'] = 'truncated_normal_zero_kl'
        if args.action_distribution == 'beta_alpha_beta':
            from tracking.model.custom_action_dist import BetaAlphaBeta
            ModelCatalog.register_custom_action_dist("beta_alpha_beta", BetaAlphaBeta)
            config['model']['custom_action_dist'] = 'beta_alpha_beta'

    if args.action_preprocessing_function == "tanh":
        config['normalize_actions'] = False
        config['clip_actions'] = False

    if args.checkpoint is not None:
        if not os.path.isdir(args.checkpoint) and not os.path.isfile(args.checkpoint):
            checkpoint_path = os.path.join(current_dir, 'trained_networks', args.checkpoint)
        else:
            checkpoint_path = args.checkpoint

        if os.path.isdir(checkpoint_path):
            if os.path.basename(checkpoint_path) == 'checkpoint':
                checkpoint_path = os.path.join(checkpoint_path, 'checkpoint')
            else:
                checkpoint_path = os.path.join(checkpoint_path, 'checkpoint', 'checkpoint')

        if not os.path.isfile(checkpoint_path):
            raise ValueError('Could not find checkpoint {}'.format(checkpoint_path))

        params_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        params_path = os.path.join(params_dir, 'params.json')

        with open(params_path) as params_file:
            checkpoint_config = json.load(params_file)
        config['env_config'] = checkpoint_config['env_config']
        config['train_batch_size'] = checkpoint_config['train_batch_size']
        config['sgd_minibatch_size'] = checkpoint_config['sgd_minibatch_size']

    else:
        checkpoint_path = None
        config['env_config'] = _make_env_config()
        config['train_batch_size'] = int(config['train_batch_size'] * args.batch_size_factor)
        config['sgd_minibatch_size'] = int(config['sgd_minibatch_size'] * args.batch_size_factor)

    if args.logdir is None:
        experiment_path = config['env_config']['experiment_name']
    else:
        experiment_path = os.path.join(args.logdir, config['env_config']['experiment_name'])

    if args.seed is not None:
        config['seed'] = args.seed
        config['env_config']['seed'] = args.seed
        np.random.seed(args.seed)

    # Ray 2.x: num_workers is removed, use num_env_runners instead
    # Set default to 5 workers if not specified
    if args.num_workers is None:
        num_env_runners = 0 
    else:
        num_env_runners = args.num_workers
    
    # Ensure num_env_runners is at least 0 (0 means local worker only)
    num_env_runners = max(num_env_runners, 0)
    
    # Set num_env_runners (Ray 2.x uses this instead of num_workers)
    config['num_env_runners'] = num_env_runners

    # For rollout_fragment_length, use num_env_runners but ensure at least 1 for division
    # When num_env_runners=0, use 1 for calculation
    config['rollout_fragment_length'] = int(config['train_batch_size'] / max(num_env_runners, 1))

    # define number of threads per worker for parallel execution based on OpenMP
    os.environ['OMP_NUM_THREADS'] = str(args.num_threads_per_worker)

    if config['env_config']['use_splines']:
        if not os.path.isdir(config['env_config']['spline_dir']):
            config['env_config']['spline_dir'] = os.path.join(current_dir, "dataset",
                                                              config['env_config']['spline_dir'])
        if not os.path.isdir(config['env_config']['spline_dir']):
            raise FileNotFoundError("Could not find spline_dir {}".format(config['env_config']['spline_dir']))
        from tracking.envs.tracking_env import TrackingEnvSpline as Env
    else:
        from tracking.envs.tracking_env import TrackingEnv as Env

    env_name = Env.__name__
    config.update(env=env_name)
    
    # Register environment for Ray - ensure it's registered before ray.init()
    def env_creator(env_config):
        """Environment creator function for Ray."""
        return Env(**env_config)
    
    tune.register_env(env_name, env_creator)
    
    # Verify environment can be created
    try:
        test_env = env_creator(config['env_config'])
        logging.info(f"Successfully created test environment: {env_name}")
        obs_space = test_env.observation_space
        action_space = test_env.action_space
        logging.info(f"Observation space: {obs_space}, Action space: {action_space}")
        test_env.close()
    except Exception as e:
        logging.error(f"Failed to create test environment: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Ray 2.x: dashboard configuration
    ray_init_kwargs = {
        'ignore_reinit_error': True,
        'logging_level': args.logging_level
    }
    if args.use_dashboard:
        ray_init_kwargs['dashboard_host'] = '0.0.0.0'
        ray_init_kwargs['include_dashboard'] = True
    ray.init(**ray_init_kwargs)
    # Note: callback will be set via algo_config.callbacks() after AlgorithmConfig is created
    # Don't set it here in config dict, as it will be overridden

    if args.num_gpus is not None:
        config['num_gpus'] = args.num_gpus

    stop = {'time_total_s': args.time * 3600}

    # Ray 2.x: use tune.run() instead of tune.run_experiments()
    # In Ray 2.x, algorithm name can be passed as string or class
    from ray.rllib.algorithms import ppo
    
    # Create algorithm class from string
    if algorithm == 'PPO':
        algo_class = ppo.PPO
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Ray 2.x: disable new API stack to use legacy custom_model API
    # Convert to AlgorithmConfig to properly set API stack flags
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    algo_config = algo_class.get_default_config()
    algo_config = algo_config.update_from_dict(config)
    
    # CRITICAL: Set callbacks as class (callable), not instance
    # Ray 2.x's callbacks() method expects a callable that returns a DefaultCallbacks subclass
    print("="*80)
    print(f"Setting callback class: {CustomTrainCallbacks}")
    print("="*80)
    sys.stdout.flush()
    algo_config.callbacks(CustomTrainCallbacks)
    
    # Disable new API stack to use legacy ModelV2 API with custom_model
    algo_config.api_stack(enable_rl_module_and_learner=False)
    # Disable validation to allow legacy API usage
    algo_config.experimental(_validate_config=False)
    # CRITICAL: Disable env runner and connector v2 to force legacy API
    algo_config.enable_env_runner_and_connector_v2 = False
    
    # Ensure num_env_runners is set explicitly (Ray 2.x requirement)
    # Ray 2.x: num_workers is removed, only num_env_runners exists
    # num_env_runners=0 means local worker only (no remote workers)
    num_env_runners = max(config.get('num_env_runners', config.get('num_workers', 1)), 0)
    # num_env_runners = 0
    
    # Set num_env_runners using the config method (preferred way)
    algo_config.num_env_runners = num_env_runners
    
    # CRITICAL: For legacy API, ensure num_learners is 0 to prevent learner_group initialization
    # When using legacy API with num_env_runners > 0, num_learners must be 0
    algo_config.num_learners = 0
    
    # Also ensure simple_optimizer is used for legacy API
    algo_config.simple_optimizer = True
    
    # Disable execution plan API (required for legacy API)
    algo_config._disable_execution_plan_api = True
    
    logging.info(f"Algorithm config - num_env_runners: {algo_config.num_env_runners}, num_learners: {algo_config.num_learners}, simple_optimizer: {algo_config.simple_optimizer}, env: {config.get('env')}")
    
    # Debug: Print config to verify settings
    logging.info(f"Config before conversion - env: {config.get('env')}, num_env_runners: {num_env_runners}, env_config present: {'env_config' in config}")
    
    # Convert back to dict for tune.run()
    config = algo_config.to_dict()
    
    # CRITICAL: Ensure callback class is preserved in dict
    # Ray 2.x may lose callback during to_dict(), so set it explicitly
    if 'callbacks' not in config or config['callbacks'] != CustomTrainCallbacks:
        print("="*80)
        print(f"Re-setting callback in config dict: {CustomTrainCallbacks}")
        print(f"Previous callback in config: {config.get('callbacks')}")
        print("="*80)
        sys.stdout.flush()
        config['callbacks'] = CustomTrainCallbacks
    
    # Also set in dict to ensure it's applied (CRITICAL for legacy API)
    config['_enable_rl_module_and_learner'] = False
    config['_enable_learner_api'] = False
    config['_enable_rl_module_api'] = False
    config['_disable_execution_plan_api'] = True
    config['enable_env_runner_and_connector_v2'] = False
    config['num_learners'] = 0
    config['simple_optimizer'] = True
    # Ensure these are not overridden
    if '_enable_new_api_stack' in config:
        config['_enable_new_api_stack'] = False
    
    # CRITICAL: Ensure num_env_runners is explicitly set in final config
    # This is required for EnvRunnerGroup to initialize properly
    # Ray 2.x: num_workers is removed, only num_env_runners exists
    config['num_env_runners'] = num_env_runners
    # Remove num_workers from config (Ray 2.x doesn't use it)
    if 'num_workers' in config:
        del config['num_workers']
    
    # Ray 2.x: num_envs_per_worker is deprecated, use num_envs_per_env_runner
    # Remove deprecated num_envs_per_worker if present
    if 'num_envs_per_worker' in config:
        del config['num_envs_per_worker']
    # Set num_envs_per_env_runner (default is 1)
    if 'num_envs_per_env_runner' not in config:
        config['num_envs_per_env_runner'] = 1
    
    # CRITICAL: Ensure environment is properly set for EnvRunnerGroup
    # The environment must be registered and accessible
    if 'env' not in config or config['env'] is None:
        raise ValueError("Environment not set in config!")
    
    # Additional check: ensure env_config is present
    if 'env_config' not in config:
        logging.warning("env_config not found in config, this may cause issues")
    
    logging.info(f"Final config - num_env_runners: {config.get('num_env_runners')}, num_envs_per_env_runner: {config.get('num_envs_per_env_runner')}, env: {config.get('env')}")
    
    # Use tune.run() for Ray 2.x (still supported, but Tuner is preferred)
    # For backward compatibility, we use tune.run() with algorithm class
    run_kwargs = {
        'name': experiment_path,
        'config': config,
        'stop': stop,
        'checkpoint_freq': args.iterations_per_checkpoint,
        'checkpoint_at_end': True,
        # 'keep_checkpoints_num': 10,
        'max_failures': 0,
        'restore': checkpoint_path,
        'callbacks': [TBXLoggerCallback(), CustomTuneCallback()],
    }
    # Ray 2.x: use storage_path instead of deprecated local_dir
    # storage_path must be an absolute path (not relative)
    if args.logdir:
        run_kwargs['storage_path'] = os.path.abspath(args.logdir)
    
    analysis = tune.run(algo_class, **run_kwargs)
