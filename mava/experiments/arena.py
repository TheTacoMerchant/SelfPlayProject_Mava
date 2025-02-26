import pickle
import pathlib

import hydra
import orbax.checkpoint
import jax
import jax.numpy as jnp
from mava.networks import FeedForwardActor as Actor
from mava.experiments.smax.smax_env import SMAX
# from mava.experiments.smax.heuristic_enemy_smax_env import HeuristicEnemySMAX
from jaxmarl.environments.smax.heuristic_enemy_smax_env import HeuristicEnemySMAX
from mava.experiments.smax import map_name_to_scenario
from mava.experiments.utils import simulate_traj, load_params_from_checkpoint
from mava.wrappers.jaxmarl import batchify
from mava.types import Observation

def load(ally_model_path, enemy_model_path, config_path):
        # Load models
    ally_net, ally_params = pickle.load(open(ally_model_path, 'rb'))
    if jax.tree.leaves(ally_params)[0].ndim > 2:
        ally_params = jax.tree.map(lambda x : x[0][0], ally_params)
    enemy_net, enemy_params = pickle.load(open(enemy_model_path, 'rb'))
    if jax.tree.leaves(enemy_params)[0].ndim > 2:
        enemy_params = jax.tree.map(lambda x : x[0][0], enemy_params)
    config = pickle.load(open(config_path, "rb"))

    return ((ally_net, ally_params), (enemy_net, enemy_params), config)

def simulate_traj_vs_heuristic(key, env, network, params):
    key, reset_key = jax.random.split(key)
    _, state = env.reset(reset_key)

    @jax.jit
    def _step(carry, _):
        state, key = carry

        agents = [f"ally_{i}" for i in range(env.num_allies)]

        # jax.debug.breakpoint()
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

def calculate_winrate(ally, enemy, config, num_traj = 100):
    key = jax.random.PRNGKey(2025)
    key, *traj_keys = jax.random.split(key, num_traj+1)

    kwargs = dict(config.env.kwargs)
    kwargs["scenario"] = map_name_to_scenario(config.env.scenario.task_name)

    # Define network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    action_head = {"_target_": "mava.networks.heads.DiscreteActionHead"}
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=10)

    network = Actor(torso=actor_torso, action_head=actor_action_head)

    # Initialize environment
    env = SMAX(**kwargs)

    traj = jax.vmap(simulate_traj, in_axes=[0,None,None,None,None,None])(jnp.stack(traj_keys), env, network, ally, network, enemy)
    done_idxes = jnp.argmax(traj[4]["__all__"], axis=1)
    done_idxes = jnp.where(done_idxes == 0, 200, done_idxes)

    won_episodes = 0
    enemy_won = 0
    for i in range(num_traj):
        won_episodes += 1 if jnp.any(traj[3]["ally_0"][i][:done_idxes[i]+1] >= 1) else 0
        enemy_won += 1 if jnp.any(traj[3]["enemy_0"][i][:done_idxes[i]+1] >= 1) else 0

    return won_episodes / num_traj

def calculate_winrate_vs_heuristic(params, config, num_traj = 100):
    key = jax.random.PRNGKey(2025)
    key, *traj_keys = jax.random.split(key, num_traj+1)

    kwargs = dict(config.env.kwargs)
    kwargs["scenario"] = map_name_to_scenario(config.env.scenario.task_name)

    # Define network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    action_head = {"_target_": "mava.networks.heads.DiscreteActionHead"}
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=10)

    network = Actor(torso=actor_torso, action_head=actor_action_head)

    # Initialize environment
    env = HeuristicEnemySMAX(**kwargs)

    traj = jax.vmap(simulate_traj_vs_heuristic, in_axes=[0,None,None,None])(jnp.stack(traj_keys), env, network, params)
    done_idxes = jnp.argmax(traj[4]["__all__"], axis=1)
    done_idxes = jnp.where(done_idxes == 0, 200, done_idxes)

    won_episodes = 0
    enemy_won = 0
    for i in range(num_traj):
        won_episodes += 1 if jnp.any(traj[3]["ally_0"][i][:done_idxes[i]+1] >= 1) else 0
        # enemy_won += 1 if jnp.any(traj[3]["enemy_0"][i][:done_idxes[i]+1] >= 1) else 0

    return won_episodes / num_traj

@hydra.main(
    config_path="../configs/default",
    config_name="ff_ippo_sp.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg):
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    # base_dir = (pathlib.Path().parent.parent / "checkpoints").absolute()
    base_dir = (pathlib.Path().parent.parent / "checkpoints/self_play").absolute()
    # ally_str = "2025_02_25_11_11_21/9"
    # enemy_str = "2025_02_25_11_11_21/8"
    # ally_params = load_params_from_checkpoint(checkpointer, base_dir / ally_str)
    # enemy_params = load_params_from_checkpoint(checkpointer, base_dir / enemy_str)

    # wr = calculate_winrate(ally_params, enemy_params, cfg, num_traj=200)
    for i in range(10):
        ally_str = f"2025_02_25_11_11_21/{i}"
        ally_params = load_params_from_checkpoint(checkpointer, base_dir / ally_str)
        wr = calculate_winrate_vs_heuristic(ally_params, cfg, num_traj=200)
        print(f"Winrate of model {ally_str} vs Heuristic is {wr*100}")

if __name__ == "__main__":
    hydra_entry_point()
