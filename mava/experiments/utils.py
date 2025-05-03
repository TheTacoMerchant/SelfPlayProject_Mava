from mava.networks import FeedForwardActor as Actor
from mava.types import Observation
from mava.experiments.wrappers import batchify
from mava.experiments.smax.smax_env import SMAX, map_name_to_scenario

import hydra
import jax
import jax.numpy as jnp


def load_params_from_checkpoint(checkpointer, filepath):
    params = checkpointer.restore(filepath)
    return params


def simulate_traj(key, env: SMAX, ally_net, ally_params, enemy_net, enemy_params):
    key, reset_key = jax.random.split(key)
    _, state = env.reset(reset_key)

    @jax.jit
    def _step(carry, _):
        state, key = carry

        ally_agents = [f"ally_{i}" for i in range(env.num_allies)]
        enemy_agents = [f"enemy_{i}" for i in range(env.num_enemies)]

        obs = env.get_obs_unit_list(state)
        ally_obs = jnp.array([obs[agent] for agent in ally_agents])
        enemy_obs = jnp.array([obs[agent] for agent in enemy_agents])

        ally_obs = Observation(
            agents_view=ally_obs,
            action_mask=batchify(env.get_avail_actions(state), ally_agents),
        )
        enemy_obs = Observation(
            agents_view=enemy_obs,
            action_mask=batchify(env.get_avail_actions(state), enemy_agents),
        )

        actions = {}

        key, ally_key, enemy_key, step_key = jax.random.split(key, 4)
        ally_policy = ally_net.apply(ally_params, ally_obs)
        ally_actions = ally_policy.sample(seed=ally_key)

        enemy_policy = enemy_net.apply(enemy_params,  enemy_obs)
        enemy_actions = enemy_policy.sample(seed=enemy_key)

        action_cat = jnp.concatenate([ally_actions, enemy_actions], -1)
        actions = {x:action_cat[i] for i, x in enumerate(ally_agents+enemy_agents)}

        obs, new_state, rewards, dones, infos = env.step(step_key, state, actions)

        return ((new_state, key), (step_key, state, actions, rewards, dones, infos))

    _, traj = jax.lax.scan(_step, (state, key), None, length=env.max_steps)

    return traj


def calculate_winrate(ally, enemy, config, num_traj = 100):
    key = jax.random.PRNGKey(2025)
    key, *traj_keys = jax.random.split(key, num_traj+1)

    kwargs = dict(config.env.kwargs)
    kwargs["scenario"] = map_name_to_scenario(config.env.scenario.task_name)

    # Initialize environment
    env = SMAX(**kwargs)

    # Define network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    action_head = {"_target_": "mava.networks.heads.DiscreteActionHead"}
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=env.action_spaces["ally_0"].n)

    network = Actor(torso=actor_torso, action_head=actor_action_head)    

    traj = jax.vmap(simulate_traj, in_axes=[0,None,None,None,None,None])(jnp.stack(traj_keys), env, network, ally, network, enemy)
    done_idxes = jnp.argmax(traj[4]["__all__"], axis=1)
    done_idxes = jnp.where(done_idxes == 0, env.max_steps, done_idxes)
    print(f"{done_idxes=}")

    won_episodes = 0
    enemy_won = 0
    for i in range(num_traj):
        won_episodes += 1 if jnp.any(traj[3]["ally_0"][i][:done_idxes[i]+1] >= 1) else 0
        enemy_won += 1 if jnp.any(traj[3]["enemy_0"][i][:done_idxes[i]+1] >= 1) else 0

    return won_episodes / num_traj


def simulate_traj_vs_heuristic(key, env, network, params):
    key, reset_key = jax.random.split(key)
    _, state = env.reset(reset_key)

    @jax.jit
    def _step(carry, _):
        state, key = carry

        agents = [f"ally_{i}" for i in range(env.num_allies)]

        obs = env.get_obs_unit_list(state.state)
        obs = jnp.array([obs[agent] for agent in agents])

        obs = Observation(
            agents_view=obs,
            action_mask=batchify(env.get_avail_actions(state), agents),
        )

        key, ally_key, step_key = jax.random.split(key, 3)
        policy = network.apply(params, obs)
        actions = policy.sample(seed=ally_key)

        actions = {x:actions[i] for i, x in enumerate(agents)}

        obs, new_state, rewards, dones, infos = env.step(step_key, state, actions)

        return ((new_state, key), (step_key, state, actions, rewards, dones, infos))

    _, traj = jax.lax.scan(_step, (state, key), None, length=200)

    return traj