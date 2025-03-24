from typing import Dict

from flax.struct import dataclass
import chex
import jax
import jax.numpy as jnp

from mava.experiments.utils import calculate_winrate
from mava.experiments.smax.heuristic_enemy_smax_env import EnemySMAX
from mava.types import Observation
from mava.wrappers.jaxmarl import batchify


@dataclass
class LeagueState:
    n_league_members: int
    winrates: chex.Array
    member_params: chex.Array

def tree_stack(trees):
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)


def tree_concat(trees):
    return jax.tree.map(lambda *v: jnp.concat(v), *trees)


def tree_unstack(tree):
    leaves, treedef = jax.tree.flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]


class LeagueManager:
    """A league of policies for self-play training."""

    def reset(self, current_learner) -> LeagueState:
        return LeagueState(
            n_league_members=1,
            winrates=jnp.array([0.5]),
            member_params=jax.tree.map(lambda x: jnp.expand_dims(x, 0), current_learner),
        )
    
    def add_policy(self, actor_params, old_state: LeagueState, cfg: Dict) -> LeagueState:
        """Add a new policy to the league."""
        n_league_members = old_state.n_league_members+1
        winrates = jnp.array([calculate_winrate(actor_params, opponent, cfg) for opponent in tree_unstack(old_state.member_params)] + [0.5])
        print(f"Winrates: {list(winrates)}")
        member_params = tree_concat([old_state.member_params, jax.tree.map(lambda x: jnp.expand_dims(x, 0),actor_params)])

        return LeagueState(n_league_members=n_league_members, winrates=winrates, member_params=member_params)


class LeagueSMAX(EnemySMAX):
    def __init__(self, network, league_state: LeagueState, **env_kwargs):
        super().__init__(**env_kwargs)
        self.league_state: LeagueState = league_state
        self.network= network

    def _select_opponent(self, key: chex.PRNGKey, league_state: LeagueState):
        key, subkey = jax.random.split(key)
        # index = jax.random.randint(subkey, shape=(), minval=0, maxval=league_state.n_league_members)
        probs = (1-league_state.winrates)**2/jnp.sum((1-league_state.winrates)**2)
        index = jax.random.categorical(subkey, logits=jnp.log(probs))

        opp_params = jax.tree.map(lambda x: x[index], league_state.member_params)

        return opp_params

    def get_enemy_policy_initial_state(self, key):
        return self._select_opponent(key, self.league_state)

    def get_enemy_actions(self, key, policy_state, enemy_obs, state):
        enemy_obs = Observation(
            agents_view=enemy_obs,
            action_mask=batchify(self._env.get_avail_actions(state), self.enemy_agents),
        )
        # jax.debug.print("Sample of policy params: {x}", x=policy_state['params']['torso']['Dense_0']['kernel'][0][:10])
        pi = self.network.apply(policy_state, enemy_obs)
        enemy_actions = pi.sample(seed=key)
        enemy_actions = {
            agent: enemy_actions[self._env.agent_ids[agent] - self.num_agents]
            for agent in self.enemy_agents
        }
        enemy_actions = {k: v.squeeze() for k, v in enemy_actions.items()}
        return enemy_actions, policy_state
        
    def step_env(
        self,
        key: chex.PRNGKey,
        state,
        actions: Dict[str, chex.Array],
        get_state_sequence=False,
    ):
        new_obs, state, rewards, dones, infos = super().step_env(key, state, actions, get_state_sequence)
        return new_obs, state, rewards, dones, infos