# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

import copy
import datetime
import pathlib
import time
from typing import Any, Callable, Dict, Tuple

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from flax.training import orbax_utils
from jax import tree
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from mava.experiments.arena import calculate_winrate_vs_heuristic
from mava.experiments.evaluator import get_eval_fn, make_ff_eval_act_fn
from mava.experiments.smax import map_name_to_scenario
from mava.experiments.smax.heuristic_enemy_smax_env import (
    HeuristicEnemySMAX,
)
from mava.experiments.smax.league_smax import LeagueManager, LeagueSMAX, LeagueState
from mava.experiments.wrappers import RecordEpisodeMetrics, SmaxWrapper
from mava.networks import FeedForwardActor as Actor
from mava.networks import FeedForwardValueNet as Critic
from mava.systems.ppo.types import LearnerState, OptStates, Params, PPOTransition
from mava.types import (
    ActorApply,
    CriticApply,
    ExperimentOutput,
    LearnerFn,
    Metrics,
)
from mava.utils.checkpointing import Checkpointer
from mava.utils.jax_utils import merge_leading_dims, unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.multistep import calculate_gae
from mava.utils.network_utils import get_action_head
from mava.utils.training import make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics


def make_envs(config, network: Actor):
    kwargs = dict(config.env.kwargs)
    kwargs["scenario"] = map_name_to_scenario(config.env.scenario.task_name)

    train = RecordEpisodeMetrics(SmaxWrapper(LeagueSMAX(network, **kwargs), False))
    eval = RecordEpisodeMetrics(SmaxWrapper(LeagueSMAX(network, **kwargs), False))

    return train, eval


def get_learner_fn(
    env: LeagueSMAX,
    apply_fns: Tuple[ActorApply, CriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[LearnerState]:
    """Get the learner function."""
    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(learner_state: LearnerState, _: Any) -> Tuple[LearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The current model parameters.
                - opt_states (OptStates): The current optimizer states.
                - key (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
            _ (Any): The current metrics info.

        """

        def _env_step(
            learner_state: LearnerState, _: Any
        ) -> Tuple[LearnerState, Tuple[PPOTransition, Metrics]]:
            """Step the environment."""
            params, opt_states, key, env_state, last_timestep, last_done = learner_state

            # Select action
            key, policy_key = jax.random.split(key)
            actor_policy = actor_apply_fn(params.actor_params, last_timestep.observation)
            value = critic_apply_fn(params.critic_params, last_timestep.observation)

            action = actor_policy.sample(seed=policy_key)
            log_prob = actor_policy.log_prob(action)

            # Step environment
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            done = timestep.last().repeat(env.num_agents).reshape(config.arch.num_envs, -1)
            transition = PPOTransition(
                last_done, action, value, timestep.reward, log_prob, last_timestep.observation
            )
            learner_state = LearnerState(params, opt_states, key, env_state, timestep, done)
            return learner_state, (transition, timestep.extras["episode_metrics"])

        # Step environment for rollout length
        learner_state, (traj_batch, episode_metrics) = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # Calculate advantage
        params, opt_states, key, env_state, last_timestep, last_done = learner_state
        last_val = critic_apply_fn(params.critic_params, last_timestep.observation)

        advantages, targets = calculate_gae(
            traj_batch, last_val, last_done, config.system.gamma, config.system.gae_lambda
        )

        # Update winrates
        jax.debug.print("Winrates: {}", jnp.round(jnp.mean(env_state.env_state.state.winrates, axis=0),2))

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""
                params, opt_states, key = train_state
                traj_batch, advantages, targets = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    traj_batch: PPOTransition,
                    gae: chex.Array,
                    key: chex.PRNGKey,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # Rerun network
                    actor_policy = actor_apply_fn(actor_params, traj_batch.obs)
                    log_prob = actor_policy.log_prob(traj_batch.action)

                    # Calculate actor loss
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    # Nomalise advantage at minibatch level
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    actor_loss1 = ratio * gae
                    actor_loss2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config.system.clip_eps,
                            1.0 + config.system.clip_eps,
                        )
                        * gae
                    )
                    actor_loss = -jnp.minimum(actor_loss1, actor_loss2)
                    actor_loss = actor_loss.mean()
                    # The seed will be used in the TanhTransformedDistribution:
                    entropy = actor_policy.entropy(seed=key).mean()

                    total_actor_loss = actor_loss - config.system.ent_coef * entropy
                    return total_actor_loss, (actor_loss, entropy)

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    traj_batch: PPOTransition,
                    targets: chex.Array,
                ) -> Tuple:
                    """Calculate the critic loss."""
                    # Rerun network
                    value = critic_apply_fn(critic_params, traj_batch.obs)

                    # Clipped MSE loss
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -config.system.clip_eps, config.system.clip_eps
                    )
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    total_value_loss = config.system.vf_coef * value_loss
                    return total_value_loss, value_loss

                # Calculate actor loss
                key, entropy_key = jax.random.split(key)
                actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                actor_loss_info, actor_grads = actor_grad_fn(
                    params.actor_params, traj_batch, advantages, entropy_key
                )

                # Calculate critic loss
                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                value_loss_info, critic_grads = critic_grad_fn(
                    params.critic_params, traj_batch, targets
                )

                # Compute the parallel mean (pmean) over the batch.
                # This pmean could be a regular mean as the batch axis is on the same device.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="batch"
                )
                # pmean over devices.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="device"
                )

                critic_grads, value_loss_info = jax.lax.pmean(
                    (critic_grads, value_loss_info), axis_name="batch"
                )
                # pmean over devices.
                critic_grads, value_loss_info = jax.lax.pmean(
                    (critic_grads, value_loss_info), axis_name="device"
                )

                # Update params and optimiser state
                actor_updates, actor_new_opt_state = actor_update_fn(
                    actor_grads, opt_states.actor_opt_state
                )
                actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

                critic_updates, critic_new_opt_state = critic_update_fn(
                    critic_grads, opt_states.critic_opt_state
                )
                critic_new_params = optax.apply_updates(params.critic_params, critic_updates)

                new_params = Params(actor_new_params, critic_new_params)
                new_opt_state = OptStates(actor_new_opt_state, critic_new_opt_state)

                actor_loss, (_, entropy) = actor_loss_info
                value_loss, unscaled_value_loss = value_loss_info

                total_loss = actor_loss + value_loss
                loss_info = {
                    "total_loss": total_loss,
                    "value_loss": unscaled_value_loss,
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                }
                return (new_params, new_opt_state, entropy_key), loss_info

            params, opt_states, traj_batch, advantages, targets, key = update_state
            key, shuffle_key, entropy_key = jax.random.split(key, 3)

            # Shuffle data and create minibatches
            batch_size = config.system.rollout_length * config.arch.num_envs
            permutation = jax.random.permutation(shuffle_key, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = tree.map(lambda x: merge_leading_dims(x, 2), batch)
            shuffled_batch = tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = tree.map(
                lambda x: jnp.reshape(x, (config.system.num_minibatches, -1, *x.shape[1:])),
                shuffled_batch,
            )

            # Update minibatches
            (params, opt_states, entropy_key), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states, entropy_key), minibatches
            )

            update_state = (params, opt_states, traj_batch, advantages, targets, key)
            return update_state, loss_info

        update_state = (params, opt_states, traj_batch, advantages, targets, key)

        # Update epochs
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs
        )

        params, opt_states, traj_batch, advantages, targets, key = update_state
        learner_state = LearnerState(params, opt_states, key, env_state, last_timestep, last_done)
        return learner_state, (episode_metrics, loss_info)

    def learner_fn(learner_state: LearnerState) -> ExperimentOutput[LearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The initial model parameters.
                - opt_states (OptStates): The initial optimizer state.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.

        """
        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.system.num_updates_per_eval
        )
        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def enemy_setup(key, config) -> Tuple[Actor, FrozenDict]:
    kwargs = dict(config.env.kwargs)
    kwargs["scenario"] = map_name_to_scenario(config.env.scenario.task_name)
    mock_env = SmaxWrapper(HeuristicEnemySMAX(**kwargs))

    # Define network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    action_head, _ = get_action_head(mock_env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=mock_env.action_dim)

    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)

    obs = mock_env.observation_spec.generate_value()
    actor_params = actor_network.init(key, obs)

    return (actor_network, actor_params)


def get_optim(config):
    # Define network and optimiser.
    actor_lr = make_learning_rate(config.system.actor_lr, config)
    critic_lr = make_learning_rate(config.system.critic_lr, config)

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )

    return actor_optim, critic_optim

def learner_setup(
    env: LeagueSMAX, key: jax.random.PRNGKey, league_state: LeagueState, config: DictConfig, actor_params, critic_params = None,
) -> Tuple[Callable, Actor, LearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of agents.
    config.system.num_agents = env.num_agents

    # PRNG keys.
    key, critic_net_key = jax.random.split(key)

    # Initialise observation with obs of all agents.
    obs = env.observation_spec.generate_value()
    init_x = tree.map(lambda x: x[jnp.newaxis, ...], obs)

    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    action_head, _ = get_action_head(env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=env.action_dim)
    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)

    critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    critic_network = Critic(torso=critic_torso)
    if critic_params is None:
        critic_params = critic_network.init(critic_net_key, init_x)

    actor_optim, critic_optim = get_optim(config)

    # Initialise optimiser state.
    actor_opt_state = actor_optim.init(actor_params)
    critic_opt_state = critic_optim.init(critic_params)

    # Pack params.
    params = Params(actor_params, critic_params)

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.system.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0, None))(
        jnp.stack(env_keys),
        league_state,
    )
    reshape_states = lambda x: x.reshape(
        (n_devices, config.system.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    # (devices, update batch size, num_envs, ...)
    env_states = tree.map(reshape_states, env_states)
    timesteps = tree.map(reshape_states, timesteps)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        # Update the params
        params = restored_params

    # Define params to be replicated across devices and batches.
    dones = jnp.zeros(
        (config.arch.num_envs, config.system.num_agents),
        dtype=bool,
    )
    key, step_keys = jax.random.split(key)
    opt_states = OptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (params, opt_states, step_keys, dones)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape))
    replicate_learner = tree.map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states, step_keys, dones = replicate_learner
    init_learner_state = LearnerState(params, opt_states, step_keys, env_states, timesteps, dones)

    return (actor_network, critic_network), init_learner_state

def learn_func_setup(env, networks, config):
    actor_optim, critic_optim = get_optim(config)

    # Pack apply and update functions.
    actor_network, critic_network = networks
    apply_fns = (actor_network.apply, critic_network.apply)
    update_fns = (actor_optim.update, critic_optim.update)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    return learn

def run_league_experiment(_config: DictConfig):
    """Runs experiment."""
    _config.logger.system_name = "sp_ff_ippo"
    config = copy.deepcopy(_config)

    # Logger setup
    logger = MavaLogger(config)
    cfg: Dict = OmegaConf.to_container(config, resolve=True)
    cfg["arch"]["devices"] = jax.devices()
    pprint(cfg)

    # Setup checkpointing
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_dir = (pathlib.Path().absolute() / f"checkpoints/league/{timestamp}").absolute()
    save_dir.mkdir(exist_ok=True, parents=True)

    # Initialize league
    league = LeagueManager()

    key = jax.random.PRNGKey(config.system.seed)

    # Add initial policy to league
    key, init_actor_key = jax.random.split(key)
    network, actor_params = enemy_setup(init_actor_key, config)
    league_state = league.reset(actor_params, config.league.max_members)
    critic_params = None

    # Save initial checkpoint
    save_args = orbax_utils.save_args_from_target(actor_params)
    orbax_checkpointer.save(save_dir / "random", actor_params, save_args=save_args)

    key, setup_key = jax.random.split(key,2)

    env, eval_env = make_envs(config, network)

    # Setup learner with provided parameters or initialize new ones
    networks, learner_state = learner_setup(
        env, setup_key, league_state, config, actor_params, critic_params
    )

    learn = learn_func_setup(env, networks, config)

    # Setup evaluator.
    # One key per device for evaluation.
    eval_act_fn = make_ff_eval_act_fn(networks[0].apply, config)
    evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    # Self-play training loop
    t=0
    for sp_iter in range(config.league.num_league_steps):
        print(f"{Fore.GREEN}Starting self-play iteration {sp_iter+1}/{config.league.num_league_steps}{Style.RESET_ALL}")

        # Train against the league
        key, sp_key = jax.random.split(key)
        learner_state, league_state, t = self_play_step(sp_key,learn, learner_state, evaluator, logger, config, t)

        actor_params = unreplicate_n_dims(learner_state.params.actor_params)

        if sp_iter % config.league.steps_per_heuristic_eval == 0:
            print(f"{Fore.GREEN} Calculating WR vs Heuristic Enemy {sp_iter+1}/{config.league.num_league_steps}{Style.RESET_ALL}")
            wr = calculate_winrate_vs_heuristic(actor_params, config)
            logger.log({"winrate_vs_heuristic": wr}, sp_iter, sp_iter/config.league.steps_per_heuristic_eval, LogEvent.ABSOLUTE)

        league_state = league.add_policy(actor_params, league_state)
        key, setup_key = jax.random.split(key,2)
        _, learner_state = learner_setup(
            env, setup_key, league_state, config, actor_params, critic_params
        )

        # Save checkpoint for this iteration
        save_args = orbax_utils.save_args_from_target(actor_params)
        orbax_checkpointer.save(save_dir / str(sp_iter), actor_params, save_args=save_args)


def self_play_step(key, learn, learner_state, evaluate, logger: MavaLogger, config, t):
    # Calculate number of updates per evaluation.
    n_devices = len(jax.devices())
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.system.num_updates_per_eval
        * config.system.rollout_length
        * config.system.update_batch_size
        * config.arch.num_envs
    )

    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t += steps_per_rollout
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actor metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

        # Prepare for evaluation.
        trained_params = unreplicate_batch_dim(learner_state.params.actor_params)
        key, *eval_keys = jax.random.split(key, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)
        league_state = unreplicate_n_dims(learner_state.env_state.env_state.state, 3)
        # Evaluate.
        eval_metrics = evaluate(
            trained_params,
            eval_keys,
            {},
            flax.jax_utils.replicate(
                league_state,
                devices=jax.devices()
            )
        )
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)

        if jnp.mean(eval_metrics["win_rate"]) > config.league.eval_cutoff:
            return learner_state, league_state, t

    return learner_state, league_state, t


@hydra.main(
    config_path="../configs/default",
    config_name="ff_ippo_sp.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Overrides
    cfg.system.seed = 2025

    # Run experiment.
    eval_performance = run_league_experiment(cfg)
    # eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}IPPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
