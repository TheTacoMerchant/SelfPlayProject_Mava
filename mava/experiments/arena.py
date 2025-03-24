import pickle
import pathlib

import hydra
import orbax.checkpoint
import jax
import jax.numpy as jnp
from mava.networks import FeedForwardActor as Actor
# from mava.experiments.smax.heuristic_enemy_smax_env import HeuristicEnemySMAX
from jaxmarl.environments.smax.heuristic_enemy_smax_env import HeuristicEnemySMAX
from mava.experiments.smax import map_name_to_scenario
from mava.experiments.utils import load_params_from_checkpoint, calculate_winrate, simulate_traj_vs_heuristic


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
    # enemy_won = 0
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
    base_dir = (pathlib.Path().parent.parent / "checkpoints/league").absolute()
    # ally_str = "2025_02_28_13_43_10/4"
    # enemy_str = "2025_02_28_13_43_10/5"
    # ally_params = load_params_from_checkpoint(checkpointer, base_dir / ally_str)
    # enemy_params = load_params_from_checkpoint(checkpointer, base_dir / enemy_str)

    # wr = calculate_winrate(ally_params, enemy_params, cfg, num_traj=200)
    # print(f"Winrate of model {ally_str} vs {enemy_str} is {wr*100}")

    for i in range(15):
        ally_str = f"2025_03_06_13_00_41/{i}"
        ally_params = load_params_from_checkpoint(checkpointer, base_dir / ally_str)
        wr = calculate_winrate_vs_heuristic(ally_params, cfg, num_traj=400)
        print(f"Winrate of model {ally_str} vs Heuristic is {wr*100}")

if __name__ == "__main__":
    hydra_entry_point()
