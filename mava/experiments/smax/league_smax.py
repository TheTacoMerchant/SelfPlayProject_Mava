from functools import partial
from typing import Dict, Optional, Tuple
from math import floor

import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from flax.linen.initializers import lecun_normal

from mava.experiments.smax.heuristic_enemy_smax_env import State
from mava.experiments.smax.smax_env import SMAX
from mava.experiments.wrappers import batchify
from mava.types import Observation


def tree_stack(trees):
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)


def tree_concat(trees):
    return jax.tree.map(lambda *v: jnp.concat(v), *trees)


def tree_unstack(tree):
    leaves, treedef = jax.tree.flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]


@dataclass
class LeagueState:
    env_state: State
    max_league_members: int
    n_league_members: int
    current_step:int

    agent_idxes: chex.Array

    selected_opponent: int
    selected_params: chex.Array
    winrates: chex.Array
    member_params: chex.ArrayTree


class LeagueManager:
    """
    A league of policies for self-play training.
    LS:
    0 - Main Agent
    1 - Main Exploiter
    2 - League Exploiter
    """
    def __init__(self, total_agents:int, pattern: list):
        n_reps = floor((total_agents-1) / len(pattern))
        rem = total_agents - len(pattern) * n_reps

        ls = pattern * n_reps + [0]*rem
        self.learner_schedule_types = jnp.array(ls)

        id_ls = list(range(len(pattern)))
        id_ls = id_ls * n_reps + [0]*rem
        self.learner_schedule_ids = jnp.array(id_ls)


    def reset(self, current_learner, max_members, num_persistent=None) -> LeagueState:
        broadcast = lambda x: jnp.broadcast_to(x, (max_members, *x.shape))

        return LeagueState(
            env_state=None,
            max_league_members=max_members,
            current_step=0,
            agent_idxes=jnp.zeros(num_persistent, dtype=jnp.uint16),
            selected_opponent=0,
            selected_params=None,
            n_league_members=1,
            winrates=jnp.where(jnp.arange(max_members) < 1, 1e-5, 1.0),
            member_params=jax.tree.map(broadcast, current_learner),
        )

    def add_policy(self, actor_params, old_state: LeagueState) -> LeagueState:
        """Add a new policy to the league."""
        if old_state.n_league_members < old_state.max_league_members:
            n_league_members = old_state.n_league_members+1
            member_params = jax.tree.map(lambda x,y : x.at[n_league_members-1].set(y), old_state.member_params, actor_params)
            last_idx = n_league_members-1
        else:
            n_league_members = old_state.max_league_members
            weakest_idx = jnp.argmax(old_state.winrates)
            member_params = jax.tree.map(lambda x,y : x.at[weakest_idx].set(y), old_state.member_params, actor_params)
            last_idx = weakest_idx

        agent_idxes = old_state.agent_idxes.at[self.learner_schedule_ids[old_state.current_step]].set(last_idx)

        main_mask = jnp.where(jnp.arange(old_state.max_league_members) < n_league_members, 1e-5, 1.0)
        me_mask = jnp.where(jnp.arange(old_state.max_league_members) == agent_idxes[0], 1e-5, 1.0)

        mask = jnp.where(self.learner_schedule_types[old_state.current_step+1] == 1, me_mask, main_mask)

        return old_state.replace(n_league_members=n_league_members,
                                 winrates=mask,
                                 agent_idxes=agent_idxes,
                                 current_step=old_state.current_step+1,
                                 member_params=member_params)

    def get_init_params(self, key, state: LeagueState) -> Optional[Dict]:
        if self.learner_schedule_types[state.current_step] == 0:
            return jax.tree.map(lambda x: x[state.agent_idxes[0]], state.member_params)
        elif self.learner_schedule_types[state.current_step] == 2:
            current_id = self.learner_schedule_ids[state.current_step]
            latest_params = jax.tree.map(lambda x: x[state.agent_idxes[current_id]], state.member_params)
            return reset_action_head(key, latest_params)
        else:
            return None # For now, we always reset exploiters

def reset_action_head(key, params):
    init_fn = lecun_normal()
    params["params"]["action_head"]["Dense_0"]["kernel"] = init_fn(key, params["params"]["action_head"]["Dense_0"]["kernel"].shape)
    params["params"]["action_head"]["Dense_0"]["bias"] = jnp.zeros_like(params["params"]["action_head"]["Dense_0"]["bias"])

    return params


class LeagueSMAX:
    def __init__(self, network, pfsp_factor, **env_kwargs):
        self._env = SMAX(**env_kwargs)
        # only one team
        self.num_agents = self._env.num_allies
        self.num_enemies = self._env.num_enemies
        # want to provide a consistent API between this and SMAX
        self.num_allies = self._env.num_allies
        self.agents = [f"ally_{i}" for i in range(self.num_agents)]
        self.enemy_agents = [f"enemy_{i}" for i in range(self.num_enemies)]
        self.all_agents = self.agents + self.enemy_agents
        self.observation_spaces = {i: self._env.observation_spaces[i] for i in self.agents}
        self.action_spaces = {i: self._env.action_spaces[i] for i in self.agents}
        self.network= network

        self.pfsp_factor = pfsp_factor

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LeagueState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[State] = None,
    ) -> Tuple[Dict[str, chex.Array], LeagueState, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment. Resets the environment if done.
        To control the reset state, pass `reset_state`. Otherwise, the environment will reset randomly."""

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions)
        win = (rewards['ally_0'] >= 1.0)

        updated_wr = state.winrates.at[state.selected_opponent].set(state.winrates[state.selected_opponent]*0.9 + 0.1*win)

        if reset_state is None:
            obs_re, states_re = self.reset(key_reset, state)
        else:
            states_re = reset_state
            obs_re = self.get_obs(states_re)

        # Auto-reset environment based on termination
        states = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        states = states.replace(winrates = jax.lax.select(dones["__all__"], updated_wr, state.winrates))
        return obs, states, rewards, dones, infos

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, league_state: LeagueState) -> Tuple[Dict[str, chex.Array], State]:
        key, reset_key = jax.random.split(key)
        obs, state = self._env.reset(reset_key)
        league_state = self.get_enemy_policy_initial_state(key, league_state)
        new_obs = {agent: obs[agent] for agent in self.agents}
        new_obs["world_state"] = obs["world_state"]
        league_state = league_state.replace(env_state=state)
        return new_obs, league_state

    def _select_opponent(self, key: chex.PRNGKey, league_state: LeagueState) -> LeagueState:
        key, subkey = jax.random.split(key)
        # index = jax.random.randint(subkey, shape=(), minval=0, maxval=league_state.n_league_members)
        hard_probs = (1-league_state.winrates)**self.pfsp_factor/jnp.sum((1-league_state.winrates)**self.pfsp_factor)
        var_probs = league_state.winrates*(1-league_state.winrates)
        struggling = (jnp.mean(league_state.winrates, where=(league_state.winrates != 1.0)) < 0.0) #TODO: Make this configurable
        probs = jnp.where(struggling, var_probs, hard_probs)
        index = jax.random.categorical(subkey, logits=jnp.log(probs))

        opp_params = jax.tree.map(lambda x: x[index], league_state.member_params)

        league_state = league_state.replace(selected_opponent=index, selected_params=opp_params)

        return league_state

    def get_enemy_policy_initial_state(self, key, league_state: LeagueState):
        return self._select_opponent(key, league_state)

    def get_enemy_actions(self, key, policy_state, enemy_obs, state):
        enemy_obs = Observation(
            agents_view=enemy_obs,
            action_mask=batchify(self._env.get_avail_actions(state), self.enemy_agents),
        )
        pi = self.network.apply(policy_state, enemy_obs)
        enemy_actions = pi.sample(seed=key)
        enemy_actions = {
            agent: enemy_actions[self._env.agent_ids[agent] - self.num_agents]
            for agent in self.enemy_agents
        }
        enemy_actions = {k: v.squeeze() for k, v in enemy_actions.items()}
        return enemy_actions, policy_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: LeagueState,
        actions: Dict[str, chex.Array],
        get_state_sequence=False,
    ):
        jaxmarl_state = state.env_state
        obs = self._env.get_obs(jaxmarl_state)
        enemy_obs = self._env.get_obs_unit_list(jaxmarl_state)
        enemy_obs = jnp.array([enemy_obs[agent] for agent in self.enemy_agents])
        key, action_key = jax.random.split(key)
        enemy_actions, _ = self.get_enemy_actions(
            action_key, state.selected_params, enemy_obs, jaxmarl_state
        )
        enemy_actions = jnp.array([enemy_actions[i] for i in self.enemy_agents])
        actions = jnp.array([actions[i] for i in self.agents])
        enemy_movement_actions, enemy_attack_actions = self._env._decode_discrete_actions(
            enemy_actions
        )
        if self._env.action_type == "continuous":
            cont_actions = jnp.zeros((len(self.all_agents), 4))
            cont_actions = cont_actions.at[: self.num_allies].set(actions)
            key, action_key = jax.random.split(key)
            ally_movement_actions, ally_attack_actions = self._env._decode_continuous_actions(
                action_key, jaxmarl_state, cont_actions
            )
            ally_movement_actions = ally_movement_actions[: self.num_allies]
            ally_attack_actions = ally_attack_actions[: self.num_allies]
        else:
            ally_movement_actions, ally_attack_actions = self._env._decode_discrete_actions(actions)

        movement_actions = jnp.concatenate([ally_movement_actions, enemy_movement_actions], axis=0)
        attack_actions = jnp.concatenate([ally_attack_actions, enemy_attack_actions], axis=0)

        if not get_state_sequence:
            obs, jaxmarl_state, rewards, dones, infos = self._env.step_env_no_decode(
                key,
                jaxmarl_state,
                (movement_actions, attack_actions),
                get_state_sequence=get_state_sequence,
            )
            new_obs = {agent: obs[agent] for agent in self.agents}
            new_obs["world_state"] = obs["world_state"]
            rewards = {agent: rewards[agent] for agent in self.agents}
            all_done = dones["__all__"]
            dones = {agent: dones[agent] for agent in self.agents}
            dones["__all__"] = all_done

            state = state.replace(env_state=jaxmarl_state)
            return new_obs, state, rewards, dones, infos
        else:
            states = self._env.step_env_no_decode(
                key,
                jaxmarl_state,
                (movement_actions, attack_actions),
                get_state_sequence=get_state_sequence,
            )
            return states

    def get_avail_actions(self, state: LeagueState) -> Dict[str, chex.Array]:
        return self._env.get_avail_actions(state.env_state)
