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
import time
from typing import Any, Dict, Tuple, Callable
from functools import partial
import dataclasses
import pathlib
import datetime
import pickle

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint
from flax.training import orbax_utils
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from jax import tree
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint


from mava.experiments.smax.heuristic_enemy_smax_env import (
    HeuristicEnemySMAX,
    LearnedPolicyEnemySMAX,
)
from mava.experiments.smax.smax_env import State as SMAXState
from mava.experiments.smax import map_name_to_scenario
from mava.evaluator import get_eval_fn, make_ff_eval_act_fn
from mava.networks import FeedForwardActor as Actor
from mava.networks import FeedForwardValueNet as Critic
from mava.systems.ppo.types import LearnerState, OptStates, Params, PPOTransition
from mava.types import (
    ActorApply,
    CriticApply,
    ExperimentOutput,
    LearnerFn,
    MarlEnv,
    Metrics,
    Observation,
)
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import merge_leading_dims, unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.multistep import calculate_gae
from mava.utils.network_utils import get_action_head
from mava.utils.training import make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics
from mava.wrappers import SmaxWrapper
from mava.experiments.wrappers import batchify


# class EnemySMAX(MultiAgentEnv):
#     """Class that presents the SMAX environment as a single-player
#     (but still multi-agent) environment. Functions like a wrapper, but
#     not linked with any of the wrapper code because that is used differently."""

#     def __init__(self, **env_kwargs):
#         self._env = SMAX(**env_kwargs)
#         # only one team
#         self.num_agents = self._env.num_allies
#         self.num_enemies = self._env.num_enemies
#         # want to provide a consistent API between this and SMAX
#         self.num_allies = self._env.num_allies
#         self.agents = [f"ally_{i}" for i in range(self.num_agents)]
#         self.enemy_agents = [f"enemy_{i}" for i in range(self.num_enemies)]
#         self.all_agents = self.agents + self.enemy_agents
#         self.observation_spaces = {i: self._env.observation_spaces[i] for i in self.agents}
#         self.action_spaces = {i: self._env.action_spaces[i] for i in self.agents}

#     def __getattr__(self, name: str):
#         return getattr(self._env, name)

#     @partial(jax.jit, static_argnums=(0,))
#     def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
#         key, reset_key = jax.random.split(key)
#         obs, state = self._env.reset(reset_key)
#         enemy_policy_state = self.get_enemy_policy_initial_state(key)
#         new_obs = {agent: obs[agent] for agent in self.agents}
#         new_obs["world_state"] = obs["world_state"]
#         return new_obs, State(state=state, enemy_policy_state=enemy_policy_state)

#     def get_enemy_actions(self, key, enemy_policy_state, enemy_obs, state):
#         raise NotImplementedError

#     def get_enemy_policy_initial_state(self, key):
#         raise NotImplementedError

#     @partial(jax.jit, static_argnums=(0, 4))
#     def step_env(
#         self,
#         key: chex.PRNGKey,
#         state: State,
#         actions: Dict[str, chex.Array],
#         get_state_sequence=False,
#     ):
#         jaxmarl_state = state.state
#         obs = self._env.get_obs(jaxmarl_state)
#         enemy_obs = self._env.get_obs_unit_list(jaxmarl_state)
#         enemy_obs = jnp.array([enemy_obs[agent] for agent in self.enemy_agents])
#         key, action_key = jax.random.split(key)
#         enemy_actions, enemy_policy_state = self.get_enemy_actions(
#             action_key, state.enemy_policy_state, enemy_obs, state
#         )
#         enemy_actions = jnp.array([enemy_actions[i] for i in self.enemy_agents])
#         actions = jnp.array([actions[i] for i in self.agents])
#         enemy_movement_actions, enemy_attack_actions = self._env._decode_discrete_actions(
#             enemy_actions
#         )
#         if self._env.action_type == "continuous":
#             cont_actions = jnp.zeros((len(self.all_agents), 4))
#             cont_actions = cont_actions.at[: self.num_allies].set(actions)
#             key, action_key = jax.random.split(key)
#             ally_movement_actions, ally_attack_actions = self._env._decode_continuous_actions(
#                 action_key, jaxmarl_state, cont_actions
#             )
#             ally_movement_actions = ally_movement_actions[: self.num_allies]
#             ally_attack_actions = ally_attack_actions[: self.num_allies]
#         else:
#             ally_movement_actions, ally_attack_actions = self._env._decode_discrete_actions(actions)

#         movement_actions = jnp.concatenate([ally_movement_actions, enemy_movement_actions], axis=0)
#         attack_actions = jnp.concatenate([ally_attack_actions, enemy_attack_actions], axis=0)

#         if not get_state_sequence:
#             obs, jaxmarl_state, rewards, dones, infos = self._env.step_env_no_decode(
#                 key,
#                 jaxmarl_state,
#                 (movement_actions, attack_actions),
#                 get_state_sequence=get_state_sequence,
#             )
#             new_obs = {agent: obs[agent] for agent in self.agents}
#             new_obs["world_state"] = obs["world_state"]
#             rewards = {agent: rewards[agent] for agent in self.agents}
#             all_done = dones["__all__"]
#             dones = {agent: dones[agent] for agent in self.agents}
#             dones["__all__"] = all_done

#             state = state.replace(enemy_policy_state=enemy_policy_state, state=jaxmarl_state)
#             return new_obs, state, rewards, dones, infos
#         else:
#             states = self._env.step_env_no_decode(
#                 key,
#                 jaxmarl_state,
#                 (movement_actions, attack_actions),
#                 get_state_sequence=get_state_sequence,
#             )
#             return states

#     @partial(jax.jit, static_argnums=(0,))
#     def get_avail_actions(self, state: State):
#         avail_actions = self._env.get_avail_actions(state.state)
#         return {agent: avail_actions[agent] for agent in self.agents}

#     def get_all_unit_obs(self, state: State):
#         return self._env.get_obs(state.state)

#     def get_obs(self, state: State) -> Dict[str, chex.Array]:
#         obs = self.get_all_unit_obs(state)
#         return {agent: obs[agent] for agent in self.agents}

#     def get_world_state(self, state: State):
#         return self._env.get_world_state(state.state)

#     def is_terminal(self, state: State):
#         return self._env.is_terminal(state.state)

#     def expand_state_seq(self, state_seq):
#         # TODO jit/scan this
#         expanded_state_seq = []

#         # TODO this actually can't take a key because recording this key is really hard
#         # it's not exposed to the user so we can't ask them to store it. Not a problem
#         # for now but will have to get creative in the future potentially.
#         for key, state, actions in state_seq:
#             # There is a split in the step function of MultiAgentEnv
#             # We call split here so that the action key is the same.
#             key, _ = jax.random.split(key)
#             states = self.step_env(key, state, actions, get_state_sequence=True)
#             states = list(map(SMAXState, *dataclasses.astuple(states)))
#             viz_actions = {
#                 agent: states[-1].prev_attack_actions[i] for i, agent in enumerate(self.all_agents)
#             }

#             expanded_state_seq.append((key, state.state, viz_actions))
#             expanded_state_seq.extend(zip([key] * len(states), states, [viz_actions] * len(states)))
#             state = state.replace(state=state.state.replace(terminal=self.is_terminal(state)))
#         return expanded_state_seq


# class LearnedPolicyEnemySMAX(EnemySMAX):
#     def __init__(self, policy, params, **env_kwargs):
#         super().__init__(**env_kwargs)
#         self.policy = policy
#         self.params = params

#     def get_enemy_policy_initial_state(self, key):
#         return self.params

#     def get_enemy_actions(self, key, policy_state, enemy_obs, state):
#         enemy_obs = Observation(
#             agents_view=enemy_obs,
#             action_mask=batchify(self._env.get_avail_actions(state.state), self.enemy_agents),
#         )
#         pi = self.policy.apply(policy_state, enemy_obs)
#         enemy_actions = pi.sample(seed=key)
#         enemy_actions = {
#             agent: enemy_actions[self._env.agent_ids[agent] - self.num_agents]
#             for agent in self.enemy_agents
#         }
#         enemy_actions = {k: v.squeeze() for k, v in enemy_actions.items()}
#         return enemy_actions, policy_state


def make_envs(config, enemy_net: Actor, enemy_params: FrozenDict):
    kwargs = dict(config.env.kwargs)
    kwargs["scenario"] = map_name_to_scenario(config.env.scenario.task_name)

    train = SmaxWrapper(LearnedPolicyEnemySMAX(enemy_net, enemy_params, **kwargs), False)

    eval = SmaxWrapper(LearnedPolicyEnemySMAX(enemy_net, enemy_params, **kwargs), False)

    return environments.add_extra_wrappers(train, eval, config)


def get_learner_fn(
    env: MarlEnv,
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


def learner_setup(
    env: MarlEnv, key: jax.random.PRNGKey, config: DictConfig, actor_params, critic_params = None
) -> Tuple[Callable, Actor, LearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of agents.
    config.system.num_agents = env.num_agents

    # PRNG keys.
    key, critic_net_key = jax.random.split(key)

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

    # Initialise optimiser state.
    actor_opt_state = actor_optim.init(actor_params)
    critic_opt_state = critic_optim.init(critic_params)

    # Pack params.
    params = Params(actor_params, critic_params)

    # Pack apply and update functions.
    apply_fns = (actor_network.apply, critic_network.apply)
    update_fns = (actor_optim.update, critic_optim.update)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.system.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
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

    return learn, actor_network, init_learner_state


def run_self_play_experiment(_config: DictConfig):
    """Runs experiment."""
    _config.logger.system_name = "sp_ff_ippo"
    config = copy.deepcopy(_config)

    # Logger setup
    logger = MavaLogger(config)
    cfg: Dict = OmegaConf.to_container(config, resolve=True)
    cfg["arch"]["devices"] = jax.devices()
    pprint(cfg)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_dir = (pathlib.Path().parent.parent / f"checkpoints/self_play/{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}").absolute()


    key = jax.random.PRNGKey(config.system.seed)

    actor_params = None
    critic_params = None
    for sp_iter in range(config.system.num_self_play_steps):
        key, sp_key = jax.random.split(key)
        actor_params, critic_params = self_play_step(sp_key, actor_params, critic_params, logger, config)

        #Save Checkpoint
        save_args = orbax_utils.save_args_from_target(actor_params)
        orbax_checkpointer.save(save_dir/str(sp_iter), actor_params, save_args=save_args)


def self_play_step(key, actor_params, critic_params, logger, config):
    key, setup_key, eval_key, rand_actor_key = jax.random.split(key,4)

    if actor_params is None:
        actor_network, actor_params = enemy_setup(rand_actor_key, config)
    else:
        actor_network, _ = enemy_setup(rand_actor_key, config)

    env, eval_env = make_envs(config, actor_network, actor_params)

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(
        env, setup_key, config, actor_params, critic_params
    )

    # Setup evaluator.
    # One key per device for evaluation.
    n_devices = len(jax.devices())
    eval_keys = jax.random.split(eval_key, n_devices)
    eval_act_fn = make_ff_eval_act_fn(actor_network.apply, config)
    evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    # Calculate number of updates per evaluation.
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
        t = int(steps_per_rollout * (eval_step + 1))
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
        # Evaluate.
        eval_metrics = evaluator(trained_params, eval_keys, {})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)

        if jnp.mean(eval_metrics["win_rate"]) > 90:
            return (unreplicate_n_dims(learner_state.params.actor_params), unreplicate_n_dims(learner_state.params.critic_params))


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    _config.logger.system_name = "ff_ippo"
    config = copy.deepcopy(_config)

    n_devices = len(jax.devices())

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key, enemy_actor_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=5
    )

    enemy = enemy_setup(enemy_actor_net_key, config)

    # Create the enviroments for train and eval.
    env, eval_env = make_envs(config, *enemy)

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )

    # Setup evaluator.
    # One key per device for evaluation.
    eval_keys = jax.random.split(key_e, n_devices)
    eval_act_fn = make_ff_eval_act_fn(actor_network.apply, config)
    evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    # Calculate total timesteps.
    config = check_total_timesteps(config)
    assert config.system.num_updates > config.arch.num_evaluation, (
        "Number of updates per evaluation must be less than total number of updates."
    )

    assert config.arch.num_envs % config.system.num_minibatches == 0, (
        "Number of envs must be divisibile by number of minibatches."
    )

    # Calculate number of updates per evaluation.
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.system.num_updates_per_eval
        * config.system.rollout_length
        * config.system.update_batch_size
        * config.arch.num_envs
    )

    # Logger setup
    logger = MavaLogger(config)
    cfg: Dict = OmegaConf.to_container(config, resolve=True)
    cfg["arch"]["devices"] = jax.devices()
    pprint(cfg)

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.logger.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_params = None
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actor metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        trained_params = unreplicate_batch_dim(learner_state.params.actor_params)
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)
        # Evaluate.
        eval_metrics = evaluator(trained_params, eval_keys, {})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)
        episode_return = jnp.mean(eval_metrics["episode_return"])

        if save_checkpoint:
            # Save checkpoint of learner state
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Record the performance for the final evaluation run.
    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    # Measure absolute metric.
    if config.arch.absolute_metric:
        abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
        eval_keys = jax.random.split(key, n_devices)

        eval_metrics = abs_metric_evaluator(best_params, eval_keys, {})

        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()

    save_dir = pathlib.Path(f"./checkpoints/{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_params(save_dir, (actor_network, learner_state.params.actor_params), enemy, config)

    return eval_performance

def save_params(dir, ally, enemy, config):
    with open(dir / "ally.pkl", "wb") as file:
        pickle.dump(ally, file)

    with open(dir / "enemy.pkl", "wb") as file:
        pickle.dump(enemy, file)

    with open(dir / "conf.pkl", "wb") as file:
        pickle.dump(config, file)

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
    eval_performance = run_self_play_experiment(cfg)
    # eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}IPPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()