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
import ray
import klimits
import datetime
import time
import errno
import logging
import numpy as np
from ray import tune
from pathlib import Path
from ray.rllib.env import BaseEnv
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID
from typing import Dict
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(current_dir))

# Termination reason
TERMINATION_UNSET = -1
TERMINATION_SUCCESS = 0  # unused
TERMINATION_JOINT_LIMITS = 1
TERMINATION_TRAJECTORY_LENGTH = 2
TERMINATION_SPLINE_LENGTH = 3
TERMINATION_SPLINE_DEVIATION = 4
TERMINATION_ROBOT_STOPPED = 5
TERMINATION_BALANCING = 6
TERMINATION_BASE_POS_DEVIATION = 7
TERMINATION_BASE_ORN_DEVIATION = 8
TERMINATION_BASE_Z_ANGLE_DEVIATION = 9

termination_reasons_dict = {TERMINATION_JOINT_LIMITS: 'joint_limit_violation_termination_rate',
                            TERMINATION_TRAJECTORY_LENGTH: 'trajectory_length_termination_rate',
                            TERMINATION_SPLINE_LENGTH: 'spline_length_termination_rate',
                            TERMINATION_SPLINE_DEVIATION: 'spline_deviation_termination_rate',
                            TERMINATION_ROBOT_STOPPED: 'robot_stopped_termination_rate',
                            TERMINATION_BALANCING: 'balancing_termination_rate',
                            TERMINATION_BASE_POS_DEVIATION: 'robot_base_pos_deviation_termination_rate',
                            TERMINATION_BASE_ORN_DEVIATION: 'robot_base_orn_deviation_termination_rate',
                            TERMINATION_BASE_Z_ANGLE_DEVIATION: 'robot_base_z_angle_deviation_termination_rate'}

RENDERER = {'opengl': 0,
            'egl': 1,
            'cpu': 2,
            'imagegrab': 3}

METRIC_OPS = ['sum', 'average', 'max', 'min']


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


def get_metrics_dir(base_dir, real_robot):
    if real_robot:
        metrics_dir = os.path.join(base_dir, "trajectory_logs_real")
    else:
        metrics_dir = os.path.join(base_dir, "trajectory_logs_sim")

    return metrics_dir


def make_metrics_dir(base_dir, real_robot):
    metrics_dir = get_metrics_dir(base_dir, real_robot)
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), default=np_encoder))
        f.flush()


def get_network_data_dir(base_dir, real_robot):
    if real_robot:
        metrics_dir = os.path.join(base_dir, "network_data_real")
    else:
        metrics_dir = os.path.join(base_dir, "network_data_sim")

    return metrics_dir


def make_network_data_dir(base_dir, real_robot):
    network_data_dir = get_network_data_dir(base_dir, real_robot)
    os.makedirs(network_data_dir, exist_ok=True)
    with open(os.path.join(network_data_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), default=np_encoder))
        f.flush()


def store_network_data(base_dir, real_robot, pid, episode_counter, reward_total, network_data_list):
    network_data_file = "episode_{}_{}_{:.3f}.json".format(episode_counter, pid, reward_total)
    network_data_dir = get_network_data_dir(base_dir, real_robot)

    for i in range(len(network_data_list)):
        for key, value in network_data_list[i].items():
            if isinstance(value, np.ndarray):
                network_data_list[i][key] = list(value)

    with open(os.path.join(network_data_dir, network_data_file), 'w') as f:
        f.write(json.dumps(network_data_list, default=np_encoder, sort_keys=True))
        f.flush()


def store_env_config(eval_dir, env_config):
    if not os.path.exists(eval_dir):
        try:
            os.makedirs(eval_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    with open(os.path.join(eval_dir, "env_config.json"), 'w') as f:
        f.write(json.dumps(env_config, sort_keys=True))
        f.flush()


def store_metrics(base_dir, real_robot, pid, episode_counter, reward_total, last_info, episode_info):
    metric_file = "episode_{}_{}_{:.3f}.json".format(episode_counter, pid, reward_total)
    episode_info['reward'] = float(reward_total)
    episode_info['episode_length'] = int(last_info['episode_length'])
    episode_info['trajectory_length'] = int(last_info['trajectory_length'])
    episode_info['success_rate'] = last_info['trajectory_successful'] if 'trajectory_successful' in last_info else 0.0

    for key, value in last_info.items():
        if key.startswith("obstacles") or "spline" in key:
            episode_info[key] = value

    for key, value in episode_info['sum'].items():
        episode_info['sum'][key] = float(np.sum(np.array(value)))
    for key, value in episode_info['max'].items():
        episode_info['max'][key] = float(np.max(np.array(value)))
    for key, value in episode_info['average'].items():
        episode_info['average'][key] = float(np.mean(np.array(value)))
    for key, value in episode_info['min'].items():
        episode_info['min'][key] = float(np.min(np.array(value)))

    for key, value in termination_reasons_dict.items():
        episode_info[value] = 1.0 if last_info['termination_reason'] == key else 0.0

    metrics_dir = get_metrics_dir(base_dir, real_robot)
    with open(os.path.join(metrics_dir, metric_file), 'w') as f:
        f.write(json.dumps(episode_info, default=np_encoder))
        f.flush()


def rollout_multiple_workers():
    # Ray 2.x: Try to access env_runner_group directly first
    remote_workers = None
    
    # Try 1: env_runner_group (Ray 2.x primary way)
    if hasattr(agent, 'env_runner_group'):
        try:
            env_runner_group = agent.env_runner_group
            print(f"env_runner_group type: {type(env_runner_group)}")
            print(f"env_runner_group attributes: {[attr for attr in dir(env_runner_group) if not attr.startswith('__')][:30]}")
            
            # Try _worker_manager first (Ray 2.x uses this)
            if hasattr(env_runner_group, '_worker_manager'):
                worker_manager = env_runner_group._worker_manager
                print(f"_worker_manager type: {type(worker_manager)}")
                print(f"_worker_manager attributes: {[attr for attr in dir(worker_manager) if not attr.startswith('__')][:20]}")
                
                # Try to get remote workers from worker_manager
                # FaultTolerantActorManager uses 'actors' property
                if hasattr(worker_manager, 'actors'):
                    actors_attr = getattr(worker_manager, 'actors')
                    if callable(actors_attr):
                        actors = actors_attr()
                    else:
                        actors = actors_attr
                    # actors is a dict mapping actor_id to actor handle
                    if isinstance(actors, dict):
                        remote_workers = list(actors.values())
                        print(f"Found remote_workers via _worker_manager.actors: {type(remote_workers)}, count: {len(remote_workers)}")
                    elif isinstance(actors, list):
                        remote_workers = actors
                        print(f"Found remote_workers via _worker_manager.actors (list): {type(remote_workers)}, count: {len(remote_workers)}")
                elif hasattr(worker_manager, 'actor_ids'):
                    # Try to get actors by IDs
                    actor_ids = worker_manager.actor_ids
                    if isinstance(actor_ids, list) and len(actor_ids) > 0:
                        # Get actors from IDs
                        remote_workers = [worker_manager._actors.get(actor_id) for actor_id in actor_ids if actor_id in worker_manager._actors]
                        remote_workers = [w for w in remote_workers if w is not None]
                        print(f"Found remote_workers via _worker_manager.actor_ids: {type(remote_workers)}, count: {len(remote_workers)}")
                elif hasattr(worker_manager, '_actors'):
                    actors_dict = worker_manager._actors
                    if isinstance(actors_dict, dict):
                        remote_workers = list(actors_dict.values())
                        print(f"Found remote_workers via _worker_manager._actors: {type(remote_workers)}, count: {len(remote_workers)}")
                elif hasattr(worker_manager, 'remote_workers'):
                    remote_workers_attr = getattr(worker_manager, 'remote_workers')
                    if callable(remote_workers_attr):
                        remote_workers = remote_workers_attr()
                    else:
                        remote_workers = remote_workers_attr
                    print(f"Found remote_workers via _worker_manager.remote_workers: {type(remote_workers)}, count: {len(remote_workers) if remote_workers else 0}")
                elif hasattr(worker_manager, '_remote_workers'):
                    remote_workers = worker_manager._remote_workers
                    print(f"Found remote_workers via _worker_manager._remote_workers: {type(remote_workers)}, count: {len(remote_workers) if remote_workers else 0}")
            
            # Fallback: try direct attributes on env_runner_group
            if remote_workers is None:
                if hasattr(env_runner_group, 'remote_workers'):
                    remote_workers_attr = getattr(env_runner_group, 'remote_workers')
                    if callable(remote_workers_attr):
                        remote_workers = remote_workers_attr()
                    else:
                        remote_workers = remote_workers_attr
                    print(f"Found remote_workers via env_runner_group.remote_workers: {type(remote_workers)}")
                elif hasattr(env_runner_group, '_remote_env_runners'):
                    remote_workers = env_runner_group._remote_env_runners
                    print(f"Found remote_workers via env_runner_group._remote_env_runners: {type(remote_workers)}")
                elif hasattr(env_runner_group, 'remote_env_runners'):
                    remote_workers_attr = getattr(env_runner_group, 'remote_env_runners')
                    if callable(remote_workers_attr):
                        remote_workers = remote_workers_attr()
                    else:
                        remote_workers = remote_workers_attr
                    print(f"Found remote_workers via env_runner_group.remote_env_runners: {type(remote_workers)}")
        except Exception as e:
            print(f"Failed to get remote_workers from env_runner_group: {e}")
            import traceback
            traceback.print_exc()
    
    # Try 2: workers (legacy API, but might work)
    if remote_workers is None:
        try:
            # Try to access workers as property (not method)
            workers_obj = agent.workers
            print(f"workers_obj type: {type(workers_obj)}")
            
            # Only try to access if it's not a method
            if not isinstance(workers_obj, type(lambda: None)):
                print(f"workers_obj attributes: {[attr for attr in dir(workers_obj) if not attr.startswith('__')][:30]}")
                
                if hasattr(workers_obj, 'remote_workers'):
                    remote_workers_attr = getattr(workers_obj, 'remote_workers')
                    if callable(remote_workers_attr):
                        remote_workers = remote_workers_attr()
                    else:
                        remote_workers = remote_workers_attr
                    print(f"Found remote_workers via workers.remote_workers: {type(remote_workers)}")
                elif hasattr(workers_obj, '_remote_workers'):
                    remote_workers = workers_obj._remote_workers
                    print(f"Found remote_workers via workers._remote_workers: {type(remote_workers)}")
        except Exception as e:
            print(f"Failed to get remote_workers from workers: {e}")
            import traceback
            traceback.print_exc()
    
    if remote_workers is None:
        # Last resort: print all attributes for debugging
        print(f"agent type: {type(agent)}")
        print(f"agent attributes: {[attr for attr in dir(agent) if not attr.startswith('__') and 'worker' in attr.lower() or 'env' in attr.lower()][:30]}")
        raise AttributeError(f"Cannot find remote_workers in agent")
    
    # Check if remote_workers is empty (num_env_runners=0 means no remote workers)
    if not remote_workers or len(remote_workers) == 0:
        print(f"Warning: remote_workers is empty (num_env_runners=0). Falling back to single worker mode.")
        # Fall back to single worker mode
        # Use env_runner_group to get local worker (same as in main code)
        if hasattr(agent, 'env_runner_group'):
            env_runner_group = agent.env_runner_group
            if hasattr(env_runner_group, '_local_env_runner'):
                local_worker = env_runner_group._local_env_runner
            else:
                raise ValueError("Cannot find _local_env_runner in env_runner_group")
        else:
            # Fallback: try agent.workers
            workers_obj = agent.workers
            if hasattr(workers_obj, '_local_env_runner'):
                local_worker = workers_obj._local_env_runner
            elif hasattr(workers_obj, 'local_worker'):
                if callable(workers_obj.local_worker):
                    local_worker = workers_obj.local_worker()
                else:
                    local_worker = workers_obj.local_worker
            else:
                raise ValueError("Cannot find local worker for single worker mode")
        
        # Get env from local_worker
        # local_worker might be an EnvRunner, so get env from it
        if hasattr(local_worker, 'env'):
            local_env = local_worker.env
            # If env is a list (vectorized), get the first one
            if isinstance(local_env, list) and len(local_env) > 0:
                local_env = local_env[0]
        elif hasattr(local_worker, 'envs') and isinstance(local_worker.envs, list) and len(local_worker.envs) > 0:
            local_env = local_worker.envs[0]
        else:
            raise ValueError("Cannot find env in local_worker")
        
        # Unwrap gymnasium wrappers if needed
        while hasattr(local_env, 'env') or hasattr(local_env, 'unwrapped'):
            if hasattr(local_env, 'envs') and isinstance(local_env.envs, list) and len(local_env.envs) > 0:
                local_env = local_env.envs[0]
            elif hasattr(local_env, 'env'):
                local_env = local_env.env
                if isinstance(local_env, list) and len(local_env) > 0:
                    local_env = local_env[0]
            elif hasattr(local_env, 'unwrapped'):
                local_env = local_env.unwrapped
            else:
                break
            # Stop if we've reached the actual TrackingBase environment
            if hasattr(local_env, 'TERMINATION_JOINT_LIMITS'):
                break
        
        # Set global env variable for rollout_single_worker_manually()
        global env
        env = local_env
        
        if args.store_metrics:
            make_metrics_dir(env.evaluation_dir, args.use_real_robot)
        if args.store_network_data:
            make_network_data_dir(env.evaluation_dir, args.use_real_robot)
        if args.store_trajectory:
            store_env_config(env.evaluation_dir, args.config["env_config"])
        # Use single worker rollout instead
        rollout_single_worker_manually()
        return
    
    if args.store_metrics or args.store_network_data or args.store_trajectory:
        evaluation_dir = ray.get(remote_workers[0].foreach_env.remote(lambda env: env.evaluation_dir))[0]
        if args.store_metrics:
            make_metrics_dir(evaluation_dir, args.use_real_robot)
        if args.store_network_data:
            make_network_data_dir(evaluation_dir, args.use_real_robot)
        if args.store_trajectory:
            store_env_config(evaluation_dir, args.config["env_config"])
    if args.seed is not None:  # increment the seed for each worker by one
        for i in range(0, len(remote_workers)):
            remote_workers[i].foreach_env.remote(lambda env: env.set_seed(args.seed + i))
    for _ in range(args.episodes):  # episodes per worker
        ray.get([worker.sample.remote() for worker in remote_workers])

    ray.get([worker.stop.remote() for worker in remote_workers])


def rollout_single_worker_manually():
    episodes_sampled = 0
    episode_computation_time_list = []
    episode_control_phase_list = []
    episode_trajectory_duration_list = []
    start = time.time()

    if args.seed is not None:
        env.set_seed(args.seed)

    while True:
        if args.episodes:
            if episodes_sampled >= args.episodes:
                break

        network_data_list = []
        if not args.spline_name_list:
            obs = env.reset()
        else:
            obs = env.reset(spline_name=args.spline_name_list[episodes_sampled % len(args.spline_name_list)])
        done = False
        reward_total = 0.0
        episode_info = {}
        steps = -1
        start_episode_timer = time.time()
        while not done:
            steps = steps + 1
            if args.store_network_data:
                # Ray 2.x: For single agent, use compute_single_action
                # compute_single_action returns (action, state_out, info) tuple
                # Check if full_fetch is supported
                try:
                    result = agent.compute_single_action(obs, full_fetch=True)
                    if isinstance(result, tuple) and len(result) >= 3:
                        action, state_out, info = result
                    else:
                        # If full_fetch doesn't work, try without it
                        action = result
                        state_out = None
                        info = {}
                except TypeError:
                    # full_fetch might not be supported, try without it
                    action = agent.compute_single_action(obs)
                    state_out = None
                    info = {}
                
                network_data = (action, state_out, info)
                network_data[2]['action'] = action  # add action to extra_outs
                network_data[2]['observation'] = obs
                network_data_list.append(network_data[2])
            else:
                # Ray 2.x: For single agent, use compute_single_action
                action = agent.compute_single_action(obs)
            if args.store_metrics:
                if not episode_info:
                    for op in METRIC_OPS:
                        episode_info[op] = defaultdict(list)

                # Gymnasium API: step() returns (obs, reward, terminated, truncated, info) or (obs, reward, done, info)
                step_result = env.step(action)
                if len(step_result) == 5:
                    # Gymnasium new API: (obs, reward, terminated, truncated, info)
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                elif len(step_result) == 4:
                    # Old API: (obs, reward, done, info)
                    next_obs, reward, done, info = step_result
                else:
                    raise ValueError(f"Unexpected step() return value: {step_result}")

                for op in list(episode_info.keys() & METRIC_OPS):
                    for k, v in info[op].items():
                        episode_info[op][k].append(v)

            else:
                # Gymnasium API: step() returns (obs, reward, terminated, truncated, info) or (obs, reward, done, info)
                step_result = env.step(action)
                if len(step_result) == 5:
                    # Gymnasium new API: (obs, reward, terminated, truncated, info)
                    next_obs, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                elif len(step_result) == 4:
                    # Old API: (obs, reward, done, info)
                    next_obs, reward, done, _ = step_result
                else:
                    raise ValueError(f"Unexpected step() return value: {step_result}")

            reward_total += reward
            obs = next_obs

        end_episode_timer = time.time()
        episode_computation_time = end_episode_timer - start_episode_timer
        logging.info("Computing episode %s took %s seconds", episodes_sampled + 1, episode_computation_time)
        episode_computation_time_list.append(episode_computation_time)
        trajectory_duration = (steps + 1) * env.trajectory_time_step
        episode_trajectory_duration_list.append(trajectory_duration)
        if env.precomputation_time is not None:
            control_phase_duration = episode_computation_time - env.precomputation_time
            logging.info("Trajectory duration: %s seconds. Control phase: %s seconds.",
                         trajectory_duration, control_phase_duration)
            episode_control_phase_list.append(control_phase_duration)
        else:
            logging.info("Trajectory duration: %s seconds", (steps + 1) * env.trajectory_time_step)
        logging.info("Episode reward: %s", reward_total)
        episodes_sampled += 1

        if args.store_metrics:
            store_metrics(env.evaluation_dir, env.use_real_robot, env.pid, episodes_sampled, reward_total,
                          info, episode_info)
        if args.store_network_data:
            store_network_data(env.evaluation_dir, env.use_real_robot, env.pid, episodes_sampled, reward_total,
                               network_data_list)

    end = time.time()
    env.close()
    logging.info("Computed %s episode(s) in %s seconds.", len(episode_computation_time_list), end - start)
    logging.info("Mean computation time: %s seconds, Max computation time: %s seconds.",
                 np.mean(episode_computation_time_list),
                 np.max(episode_computation_time_list))
    computation_time_fraction = np.array(episode_computation_time_list) * 100 / \
        np.array(episode_trajectory_duration_list)
    logging.info("Mean computation time fraction: %s %%, Max computation time fraction: %s %%.",
                 np.mean(computation_time_fraction),
                 np.max(computation_time_fraction))
    if episode_control_phase_list:
        max_relative_delay = (np.max(np.array(episode_control_phase_list) /
                                     np.array(episode_trajectory_duration_list)) - 1) * 100
        logging.info("Mean control phase: %s seconds, Max control phase: %s seconds. Max delay: %s percent.",
                     np.mean(episode_control_phase_list),
                     np.max(episode_control_phase_list),
                     max_relative_delay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None,
                        help="The name of the evaluation.")
    parser.add_argument('--evaluation_dir', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to the checkpoint for evaluation.")
    parser.add_argument('--episodes', type=int, default=20,
                        help="The number of episodes for evaluation per worker.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_threads_per_worker', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=None)
    parser.add_argument('--use_real_robot', action='store_true', default=None)
    parser.add_argument('--real_robot_debug_mode', dest='real_robot_debug_mode', action='store_true', default=False)
    parser.add_argument('--use_gui', action='store_true', default=False)
    parser.add_argument('--switch_gui', action='store_true', default=False)
    parser.add_argument('--online_trajectory_duration', type=float, default=None)
    parser.add_argument('--check_braking_trajectory_collisions', action='store_true', default=False)
    parser.add_argument('--check_braking_trajectory_torque_limits', action='store_true', default=False)
    parser.add_argument('--collision_check_time', type=float, default=None)
    parser.add_argument('--store_metrics', action='store_true', default=False)
    parser.add_argument('--plot_trajectory', action='store_true', default=False)
    parser.add_argument('--save_trajectory_plot', action='store_true', default=False)
    parser.add_argument('--plot_acc_limits', action='store_true', default=False)
    parser.add_argument('--plot_actual_values', action='store_true', default=False)
    parser.add_argument('--plot_actual_torques', action='store_true', default=False)
    parser.add_argument('--plot_computed_actual_values', action='store_true', default=False)
    parser.add_argument('--plot_joint', type=json.loads, default=None)
    # spline settings
    parser.add_argument('--plot_spline', action='store_true', default=False)
    parser.add_argument('--visualize_action_spline', action='store_true', default=False)
    parser.add_argument('--spline_speed', type=float, default=None)
    parser.add_argument('--spline_braking_extra_time_steps', type=float, default=None)
    parser.add_argument('--terminate_on_robot_stop', action='store_true', default=False)
    parser.add_argument('--spline_dir', type=str, default=None)
    parser.add_argument('--spline_config_path', type=str, default=None)
    parser.add_argument('--spline_use_reflection_vectors', action='store_true', default=False)
    parser.add_argument('--spline_u_arc_start_range', type=json.loads, default=None)
    parser.add_argument('--spline_u_arc_diff_min', type=float, default=None)
    parser.add_argument('--spline_u_arc_diff_max', type=float, default=None)
    parser.add_argument('--spline_name_list', type=json.loads, default=None)
    parser.add_argument('--spline_termination_max_deviation', type=float, default=None)
    parser.add_argument('--spline_termination_extra_time_steps', type=int, default=None)
    parser.add_argument('--spline_compute_total_spline_metrics', action='store_true', default=False)
    # end of spline settings
    # sphere balancing settings
    parser.add_argument('--sphere_balancing_mode', action='store_true', default=False)
    parser.add_argument('--terminate_balancing_sphere_not_on_board', action='store_true', default=False)
    # end of sphere balancing settings
    # robot base balancing settings
    parser.add_argument('--floating_robot_base', action='store_true', default=False)
    parser.add_argument('--no_terminate_on_balancing_robot_base_z_angle_deviation', action='store_true', default=False)
    parser.add_argument('--no_terminate_on_balancing_robot_base_pos_deviation', action='store_true', default=False)
    parser.add_argument('--no_terminate_on_balancing_robot_base_orn_deviation', action='store_true', default=False)
    # end of robot base balancing settings
    parser.add_argument('--torque_limit_factor', type=float, default=None)
    parser.add_argument('--store_actions', action='store_true', default=False)
    parser.add_argument('--store_trajectory', action='store_true', default=False)
    parser.add_argument('--store_network_data', action='store_true', default=False)
    parser.add_argument('--no_exploration', action='store_true', default=False)
    parser.add_argument('--log_obstacle_data', action='store_true', default=False)
    parser.add_argument('--obstacle_scene', type=int, default=None)
    parser.add_argument('--obstacle_use_computed_actual_values', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--solver_iterations', type=int, default=None)
    parser.add_argument('--random_agent', action='store_true', default=False)
    parser.add_argument('--no_self_collision', action='store_true', default=False)
    parser.add_argument('--use_thread_for_movement', action='store_true', default=False)
    parser.add_argument('--use_process_for_movement', action='store_true', default=False)
    parser.add_argument('--no_use_control_rate_sleep', action='store_true', default=False)
    parser.add_argument('--control_time_step', type=float, default=None)
    parser.add_argument('--time_step_fraction_sleep_observation', type=float, default=None)
    parser.add_argument("--logging_level", default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--render', action='store_true', default=False,
                        help="If set, videos of the generated episodes are recorded.")
    parser.add_argument("--renderer", default='opengl', choices=['opengl', 'egl', 'cpu', 'imagegrab'])
    parser.add_argument('--render_no_shadows', action='store_true', default=False)
    parser.add_argument('--camera_angle', type=int, default=0)
    parser.add_argument('--video_frame_rate', type=float, default=None)
    parser.add_argument('--video_height', type=int, default=None)
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--video_add_text', action='store_true', default=False)
    parser.add_argument('--fixed_video_filename', action='store_true', default=False)
    parser.add_argument('--use_dashboard', action='store_true', default=False)

    args = parser.parse_args()

    if args.render and args.renderer == 'egl':
        os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
        os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'

    if args.evaluation_dir is None:
        evaluation_dir = os.path.join(Path.home(), "safe_motions_evaluation")
    else:
        evaluation_dir = os.path.join(args.evaluation_dir, "safe_motions_evaluation")

    if not os.path.isdir(args.checkpoint) and not os.path.isfile(args.checkpoint):
        checkpoint_path = os.path.join(current_dir, "trained_networks", args.checkpoint)
    else:
        checkpoint_path = args.checkpoint

    # Check if this is Ray RLlib 2.x format (PyTorch) or old format (TensorFlow)
    is_ray2x_format = False
    if os.path.isdir(checkpoint_path):
        # Ray RLlib 2.x format: checkpoint directory contains rllib_checkpoint.json
        rllib_checkpoint_json = os.path.join(checkpoint_path, "rllib_checkpoint.json")
        if os.path.isfile(rllib_checkpoint_json):
            is_ray2x_format = True
            # For Ray 2.x, use the directory path directly
            # params.json is in the parent directory
            params_dir = os.path.dirname(checkpoint_path)
        else:
            # Old format: look for checkpoint file inside directory
            if os.path.basename(checkpoint_path) == "checkpoint":
                checkpoint_path = os.path.join(checkpoint_path, "checkpoint")
            else:
                checkpoint_path = os.path.join(checkpoint_path, "checkpoint", "checkpoint")
            params_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    else:
        # checkpoint_path is a file (old format)
        params_dir = os.path.dirname(os.path.dirname(checkpoint_path))

    if not is_ray2x_format and not os.path.isfile(checkpoint_path):
        raise ValueError("Could not find checkpoint {}".format(checkpoint_path))

    params_path = os.path.join(params_dir, "params.json")

    with open(params_path) as params_file:
        checkpoint_config = json.load(params_file)
    checkpoint_config['evaluation_interval'] = None
    env_config = checkpoint_config['env_config']

    if args.name is not None:
        env_config['experiment_name'] = args.name

    logging.basicConfig()
    logging.getLogger().setLevel(args.logging_level)
    env_config['logging_level'] = args.logging_level

    if args.render:
        env_config.update(render_video=True)
        env_config['camera_angle'] = args.camera_angle
        env_config['renderer'] = RENDERER[args.renderer]
        env_config['render_no_shadows'] = args.render_no_shadows
        env_config['video_frame_rate'] = args.video_frame_rate
        env_config['video_height'] = args.video_height
        env_config['video_dir'] = args.video_dir
        env_config['fixed_video_filename'] = args.fixed_video_filename
        env_config['video_add_text'] = args.video_add_text
        if args.fixed_video_filename and args.num_workers is not None and args.num_workers >= 2:
            raise ValueError("fixed_video_filename requires num_workers < 2")

    else:
        env_config.update(render_video=False)

    if args.use_gui:
        env_config.update(use_gui=True)
    else:
        env_config.update(use_gui=False)

    env_config.update(use_real_robot=args.use_real_robot)
    env_config['time_stamp'] = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

    if args.store_actions:
        env_config['store_actions'] = True

    if args.store_trajectory:
        env_config['store_trajectory'] = True

    if args.log_obstacle_data:
        env_config['log_obstacle_data'] = True

    if args.save_trajectory_plot:
        env_config['save_trajectory_plot'] = True

    if args.switch_gui:
        env_config['switch_gui'] = True

    if args.plot_spline:
        env_config['plot_spline'] = True

    if args.visualize_action_spline:
        env_config['visualize_action_spline'] = True

    if args.plot_actual_torques:
        env_config['plot_actual_torques'] = True

    if args.use_thread_for_movement:
        env_config['use_thread_for_movement'] = True

    if args.use_process_for_movement:
        env_config['use_process_for_movement'] = True

    if args.no_use_control_rate_sleep:
        env_config['use_control_rate_sleep'] = False

    if args.obstacle_use_computed_actual_values:
        env_config['obstacle_use_computed_actual_values'] = True

    if args.obstacle_scene is not None:
        env_config['obstacle_scene'] = args.obstacle_scene

    if args.online_trajectory_duration is not None:
        env_config['online_trajectory_duration'] = args.online_trajectory_duration

    if args.plot_trajectory:

        env_config['plot_trajectory'] = True

        if args.plot_acc_limits:
            env_config['plot_acc_limits'] = True

        if args.plot_actual_values:
            env_config['plot_actual_values'] = True

        if args.plot_computed_actual_values:
            env_config['plot_computed_actual_values'] = True

        if args.plot_joint is not None:
            env_config['plot_joint'] = args.plot_joint

    if args.random_agent:
        env_config['random_agent'] = True

    if args.collision_check_time is not None:
        env_config['collision_check_time'] = args.collision_check_time

    if args.torque_limit_factor is not None:
        env_config['torque_limit_factor'] = args.torque_limit_factor

    if args.solver_iterations:
        env_config['solver_iterations'] = args.solver_iterations

    if args.real_robot_debug_mode:
        env_config['real_robot_debug_mode'] = True

    if args.no_self_collision:
        env_config['no_self_collision'] = True

    if args.floating_robot_base:
        env_config['floating_robot_base'] = True

    if args.no_terminate_on_balancing_robot_base_z_angle_deviation:
        env_config['terminate_on_balancing_robot_base_z_angle_deviation'] = False

    if args.no_terminate_on_balancing_robot_base_pos_deviation:
        env_config['terminate_on_balancing_robot_base_pos_deviation'] = False

    if args.no_terminate_on_balancing_robot_base_orn_deviation:
        env_config['terminate_on_balancing_robot_base_orn_deviation'] = False

    if args.time_step_fraction_sleep_observation is not None:
        env_config['time_step_fraction_sleep_observation'] = args.time_step_fraction_sleep_observation

    if args.control_time_step is not None:
        env_config['control_time_step'] = args.control_time_step

    if args.spline_braking_extra_time_steps is not None:
        env_config['spline_braking_extra_time_steps'] = args.spline_braking_extra_time_steps

    if args.terminate_on_robot_stop:
        env_config['terminate_on_robot_stop'] = True

    if args.spline_use_reflection_vectors:
        env_config['spline_use_reflection_vectors'] = True

    if args.spline_u_arc_start_range is not None:
        env_config['spline_u_arc_start_range'] = args.spline_u_arc_start_range

    if args.spline_u_arc_diff_min is not None:
        env_config['spline_u_arc_diff_min'] = args.spline_u_arc_diff_min

    if args.spline_u_arc_diff_max is not None:
        env_config['spline_u_arc_diff_max'] = args.spline_u_arc_diff_max

    if args.spline_termination_max_deviation is not None:
        if args.spline_termination_max_deviation == -1:
            env_config['spline_termination_max_deviation'] = None  # deactivate spline termination
        else:
            env_config['spline_termination_max_deviation'] = args.spline_termination_max_deviation

    if args.spline_termination_extra_time_steps is not None:
        env_config['spline_termination_extra_time_steps'] = args.spline_termination_extra_time_steps

    if args.spline_compute_total_spline_metrics:
        env_config['spline_compute_total_spline_metrics'] = True

    if args.sphere_balancing_mode:
        if 'sphere_balancing_mode' not in env_config or not env_config['sphere_balancing_mode']:
            env_config['sphere_balancing_mode'] = True
            env_config['obs_no_balancing_sphere'] = True

    if args.terminate_balancing_sphere_not_on_board:
        env_config['terminate_balancing_sphere_not_on_board'] = True

    env_config['spline_speed'] = args.spline_speed

    if 'use_splines' in env_config and env_config['use_splines']:
        if not os.path.isdir(env_config['spline_dir']):
            env_config['spline_dir'] = os.path.join(current_dir, "dataset", env_config['spline_dir'])
        if args.spline_dir is not None:
            if not os.path.isdir(args.spline_dir):
                env_config['spline_dir'] = os.path.join(os.path.dirname(env_config['spline_dir']), args.spline_dir)
                if not os.path.isdir(env_config['spline_dir']):
                    env_config['spline_dir'] = os.path.join(current_dir, "dataset", args.spline_dir)
            else:
                env_config['spline_dir'] = args.spline_dir
        if not os.path.isdir(env_config['spline_dir']):
            raise FileNotFoundError("Could not find spline_dir {}".format(env_config['spline_dir']))
        if args.spline_config_path is not None:
            env_config['spline_config_path'] = args.spline_config_path

    checkpoint_config['num_workers'] = args.num_workers

    if args.num_gpus is not None:
        checkpoint_config['num_gpus'] = args.num_gpus
        if args.num_gpus == 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

    if args.seed is not None:
        checkpoint_config['seed'] = args.seed
        env_config['seed'] = args.seed

    if 'ray_version' in env_config and hasattr(ray, '__version__'):
        if env_config['ray_version'] != ray.__version__:
            logging.warning('This network was trained with ray=={} but you are using ray=={}'.format(
                env_config['ray_version'], ray.__version__))

    if 'klimits_version' in env_config and hasattr(klimits, '__version__'):
        if env_config['klimits_version'] != klimits.__version__:
            logging.warning('This network was trained with klimits=={} but you are using klimits=={}'.format(
                env_config['klimits_version'], klimits.__version__))

    if 'use_braking_trajectory_method' in env_config:  # for compatibility with older versions
        env_config['check_braking_trajectory_collisions'] = env_config['use_braking_trajectory_method']
        del env_config['use_braking_trajectory_method']

    if args.check_braking_trajectory_collisions:
        env_config['check_braking_trajectory_collisions'] = True

    if args.check_braking_trajectory_torque_limits:
        env_config['check_braking_trajectory_torque_limits'] = True

    checkpoint_config['env_config'] = env_config

    if 'sample_batch_size' in checkpoint_config:
        del checkpoint_config['sample_batch_size']
    checkpoint_config['rollout_fragment_length'] = 1  # stop sampling of remote workers after a single episode

    if args.no_exploration:
        checkpoint_config['explore'] = False

    # Register custom models BEFORE algorithm initialization (Ray 2.x requirement)
    if 'custom_model' in checkpoint_config['model']:
        from ray.rllib.models.catalog import ModelCatalog
        custom_model_name = checkpoint_config['model']['custom_model']
        
        if custom_model_name == 'fcnet_last_layer_activation':
            from tracking.model.fcnet_v2_last_layer_activation import FullyConnectedNetworkLastLayerActivation
            ModelCatalog.register_custom_model('fcnet_last_layer_activation', FullyConnectedNetworkLastLayerActivation)
        elif custom_model_name == 'torch_fcnet_last_layer_activation':
            from tracking.model.torch_fcnet_last_layer_activation import FullyConnectedNetworkLastLayerActivation
            ModelCatalog.register_custom_model('torch_fcnet_last_layer_activation', FullyConnectedNetworkLastLayerActivation)
        elif custom_model_name == 'keras_fcnet_last_layer_activation':
            from tracking.model.keras_fcnet_last_layer_activation import FullyConnectedNetworkLastLayerActivation
            ModelCatalog.register_custom_model('keras_fcnet_last_layer_activation',
                                               FullyConnectedNetworkLastLayerActivation)
            for key in ['fcnet_hiddens', 'fcnet_activation', 'post_fcnet_hiddens', 'post_fcnet_activation',
                        'no_final_layer', 'vf_share_layers', 'free_log_std']:
                if key in checkpoint_config['model'] and key not in checkpoint_config['model']['custom_model_config']:
                    checkpoint_config['model']['custom_model_config'][key] = checkpoint_config['model'][key]
        
        if 'custom_options' in checkpoint_config['model']:
            checkpoint_config['model']['custom_model_config'] = checkpoint_config['model']['custom_options']
            del checkpoint_config['model']['custom_options']

        if args.store_network_data:
            checkpoint_config['model']['custom_model_config']['output_intermediate_layers'] = True

    args.config = checkpoint_config
    args.run = "PPO"
    args.env = checkpoint_config['env']
    args.out = None
    if args.seed is not None:
        np.random.seed(args.seed)

    # define number of threads per worker for parallel execution based on OpenMP
    os.environ['OMP_NUM_THREADS'] = str(args.num_threads_per_worker)

    if 'use_splines' in env_config and env_config['use_splines']:
        from tracking.envs.tracking_env import TrackingEnvSpline as Env
    else:
        from tracking.envs.tracking_env import TrackingEnv as Env
    from tracking.train import CustomTrainCallbacks

    class CustomEvaluationCallbacks(CustomTrainCallbacks):

        def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                             policies: Dict[str, Policy],
                             episode: MultiAgentEpisode, env_index: int, **kwargs):

            episode.user_data['start_time'] = time.time()
            if args.store_metrics:
                super().on_episode_start(worker=worker, base_env=base_env,
                                         policies=policies,
                                         episode=episode, env_index=env_index, **kwargs)

        def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                            episode: MultiAgentEpisode, env_index: int, **kwargs):
            if args.store_metrics:
                super().on_episode_step(worker=worker, base_env=base_env,
                                        episode=episode, env_index=env_index, **kwargs)

        def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                           policies: Dict[str, Policy], episode: MultiAgentEpisode,
                           env_index: int, **kwargs):
            episode_computation_time = time.time() - episode.user_data['start_time']
            env = base_env.get_unwrapped()[-1]
            print("Computing episode {} took {} seconds".format(env.episode_counter, episode_computation_time))
            last_info = episode.last_info_for()
            episode_length = last_info['episode_length']
            print("Trajectory duration: {} seconds".format(episode_length * env.trajectory_time_step))
            reward_total = episode.agent_rewards[('agent0', 'default_policy')]
            print("Episode reward: {}".format(reward_total))
            if args.store_metrics:
                store_metrics(env.evaluation_dir, env.use_real_robot, env.pid,
                              env.episode_counter, reward_total, last_info, episode.user_data['op'])
            if 'network_data_list' in episode.user_data:
                store_network_data(env.evaluation_dir, env.use_real_robot, env.pid, env.episode_counter, reward_total,
                                   episode.user_data['network_data_list'])

        def on_train_result(self, *, algorithm, result: dict, **kwargs):
            # Ray 2.x: 'trainer' parameter renamed to 'algorithm'
            pass

        def on_postprocess_trajectory(
                self, *, worker: "RolloutWorker", episode: MultiAgentEpisode,
                agent_id: AgentID, policy_id: PolicyID,
                policies: Dict[PolicyID, Policy], postprocessed_batch: SampleBatch,
                original_batches: Dict[AgentID, SampleBatch], **kwargs) -> None:
            if args.store_network_data:
                network_data_list = []
                for i in range(len(postprocessed_batch['obs'])):
                    network_data_step = {}
                    for key in postprocessed_batch:
                        network_key = None
                        if key == 'actions':
                            network_key = 'action'
                        elif key == 'obs':
                            network_key = 'observation'
                        elif key in ['action_prob', 'action_logp', 'action_dist_inputs', 'vf_preds', 'fc_1', 'fc_2',
                                     'fc_value_1', 'fc_value_2', 'logits']:
                            network_key = key
                        if network_key is not None:
                            network_data_step[network_key] = postprocessed_batch[key][i]
                    network_data_list.append(network_data_step)
                episode.user_data['network_data_list'] = network_data_list

    tune.register_env(Env.__name__,
                      lambda config_args: Env(**config_args))
    args.config['callbacks'] = CustomEvaluationCallbacks

    ray.init(dashboard_host="127.0.0.1", include_dashboard=args.use_dashboard, ignore_reinit_error=True,
             num_gpus=args.num_gpus)
    
    # Re-register custom models AFTER ray.init() so workers can access them
    if 'custom_model' in checkpoint_config['model']:
        from ray.rllib.models.catalog import ModelCatalog
        custom_model_name = checkpoint_config['model']['custom_model']
        
        if custom_model_name == 'fcnet_last_layer_activation':
            from tracking.model.fcnet_v2_last_layer_activation import FullyConnectedNetworkLastLayerActivation
            ModelCatalog.register_custom_model('fcnet_last_layer_activation', FullyConnectedNetworkLastLayerActivation)
        elif custom_model_name == 'torch_fcnet_last_layer_activation':
            from tracking.model.torch_fcnet_last_layer_activation import FullyConnectedNetworkLastLayerActivation
            ModelCatalog.register_custom_model('torch_fcnet_last_layer_activation', FullyConnectedNetworkLastLayerActivation)
        elif custom_model_name == 'keras_fcnet_last_layer_activation':
            from tracking.model.keras_fcnet_last_layer_activation import FullyConnectedNetworkLastLayerActivation
            ModelCatalog.register_custom_model('keras_fcnet_last_layer_activation',
                                               FullyConnectedNetworkLastLayerActivation)
    
    # Ray 2.x: Import algorithm class directly instead of using rollout.get_trainable_cls
    if args.run == "PPO":
        from ray.rllib.algorithms import ppo
        algo_class = ppo.PPO
    else:
        raise ValueError(f"Unsupported algorithm: {args.run}")
    
    # Ray 2.x: Use AlgorithmConfig to properly configure the algorithm
    # Instead of using update_from_dict (which may reintroduce problematic keys),
    # manually set only the necessary config values directly
    algo_config = algo_class.get_default_config()
    
    # Set environment first
    algo_config.environment(args.env)
    
    # Set env_config if present
    if 'env_config' in args.config:
        algo_config.env_config = args.config['env_config']
    
    # Set model config if present
    if 'model' in args.config:
        model_config = args.config['model']
        if isinstance(model_config, dict):
            # Set model parameters individually to avoid problematic keys
            if 'custom_model' in model_config:
                algo_config.model['custom_model'] = model_config['custom_model']
            if 'custom_model_config' in model_config:
                algo_config.model['custom_model_config'] = model_config['custom_model_config']
            if 'fcnet_hiddens' in model_config:
                algo_config.model['fcnet_hiddens'] = model_config['fcnet_hiddens']
            if 'fcnet_activation' in model_config:
                algo_config.model['fcnet_activation'] = model_config['fcnet_activation']
            if 'use_lstm' in model_config:
                algo_config.model['use_lstm'] = model_config['use_lstm']
            if 'conv_filters' in model_config:
                algo_config.model['conv_filters'] = model_config['conv_filters']
    
    # Set PPO-specific parameters if present
    if args.run == "PPO":
        if 'gamma' in args.config:
            algo_config.gamma = args.config['gamma']
        if 'lambda' in args.config:
            algo_config.lambda_ = args.config['lambda']
        if 'kl_coeff' in args.config:
            algo_config.kl_coeff = args.config['kl_coeff']
        if 'kl_target' in args.config:
            algo_config.kl_target = args.config['kl_target']
        if 'lr' in args.config:
            algo_config.lr = args.config['lr']
        if 'lr_schedule' in args.config:
            algo_config.lr_schedule = args.config['lr_schedule']
        if 'num_sgd_iter' in args.config:
            algo_config.num_sgd_iter = args.config['num_sgd_iter']
        if 'sgd_minibatch_size' in args.config:
            algo_config.sgd_minibatch_size = args.config['sgd_minibatch_size']
        if 'train_batch_size' in args.config:
            algo_config.train_batch_size = args.config['train_batch_size']
        if 'rollout_fragment_length' in args.config:
            algo_config.rollout_fragment_length = args.config['rollout_fragment_length']
        if 'vf_clip_param' in args.config:
            algo_config.vf_clip_param = args.config['vf_clip_param']
        if 'vf_loss_coeff' in args.config:
            algo_config.vf_loss_coeff = args.config['vf_loss_coeff']
        if 'use_gae' in args.config:
            algo_config.use_gae = args.config['use_gae']
    
    # Set general parameters
    if 'normalize_actions' in args.config:
        algo_config.normalize_actions = args.config['normalize_actions']
    if 'num_gpus' in args.config:
        algo_config.num_gpus = args.config['num_gpus']
    
    # Set num_env_runners: Use args.num_workers for evaluation (override checkpoint config)
    # Evaluation typically uses num_workers=0 (local worker only)
    num_env_runners = args.num_workers if args.num_workers is not None else 0
    algo_config.num_env_runners = num_env_runners
    algo_config.num_learners = 0
    algo_config.simple_optimizer = True
    algo_config._disable_execution_plan_api = True
    
    # Ray 2.x: Disable new API stack to use legacy custom_model API
    algo_config.api_stack(enable_rl_module_and_learner=False)
    algo_config.experimental(_validate_config=False)
    algo_config.enable_env_runner_and_connector_v2 = False
    
    # Fix policy_mapping_fn: must be None or callable (Ray 2.x requirement)
    if 'policy_mapping_fn' in args.config:
        policy_mapping_fn = args.config['policy_mapping_fn']
        if policy_mapping_fn is not None and not callable(policy_mapping_fn):
            algo_config.multi_agent(policy_mapping_fn=None)
        elif policy_mapping_fn is None:
            algo_config.multi_agent(policy_mapping_fn=None)
    else:
        algo_config.multi_agent(policy_mapping_fn=None)
    
    # Set callbacks
    algo_config.callbacks(CustomEvaluationCallbacks)
    
    # Ray 2.x: Use build_algo() method (build() is deprecated)
    agent = algo_config.build_algo()
    agent.restore(checkpoint_path)

    # Ray 2.x: Check num_env_runners or num_workers for compatibility
    num_workers_value = checkpoint_config.get('num_env_runners', checkpoint_config.get('num_workers', 0))
    if num_workers_value == 0:
        # Ray 2.x: workers is a property, not a method
        workers_obj = agent.workers
        
        # Get local worker
        if hasattr(workers_obj, 'local_worker'):
            # local_worker might be a method or property
            if callable(workers_obj.local_worker):
                local_worker = workers_obj.local_worker()
            else:
                local_worker = workers_obj.local_worker
        elif hasattr(workers_obj, '_local_env_runner'):
            local_worker = workers_obj._local_env_runner
        else:
            raise AttributeError("Cannot find local_worker in agent.workers")
        
        env = local_worker.env
        if args.store_metrics:
            make_metrics_dir(env.evaluation_dir, args.use_real_robot)
        if args.store_network_data:
            make_network_data_dir(env.evaluation_dir, args.use_real_robot)
        if args.store_trajectory:
            store_env_config(env.evaluation_dir, args.config["env_config"])
        rollout_single_worker_manually()
    else:
        if args.use_real_robot:
            raise ValueError("--use_real_robot requires --num_workers=0")
        rollout_multiple_workers()




