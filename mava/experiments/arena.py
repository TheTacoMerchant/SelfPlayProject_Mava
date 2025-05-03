import pickle
import pathlib
import json

import matplotlib.pyplot as plt
import hydra
import orbax.checkpoint
import jax
import jax.numpy as jnp
from mava.networks import FeedForwardActor as Actor
# from mava.experiments.smax.heuristic_enemy_smax_env import HeuristicEnemySMAX
from jaxmarl.environments.smax.heuristic_enemy_smax_env import HeuristicEnemySMAX
from mava.experiments.smax import map_name_to_scenario
from mava.experiments.utils import load_params_from_checkpoint, calculate_winrate, simulate_traj_vs_heuristic


def calculate_winrate_vs_heuristic(params, config, num_traj = 100, max_steps = None):
    key = jax.random.PRNGKey(2025)
    key, *traj_keys = jax.random.split(key, num_traj+1)

    kwargs = dict(config.env.kwargs)
    kwargs["scenario"] = map_name_to_scenario(config.env.scenario.task_name)
    if max_steps is not None:
        kwargs["max_steps"] = max_steps

    # Initialize environment
    env = HeuristicEnemySMAX(**kwargs)

    # Define network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    action_head = {"_target_": "mava.networks.heads.DiscreteActionHead"}
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=env.action_spaces["ally_0"].n)

    network = Actor(torso=actor_torso, action_head=actor_action_head)

    traj = jax.vmap(simulate_traj_vs_heuristic, in_axes=[0,None,None,None])(jnp.stack(traj_keys), env, network, params)
    done_idxes = jnp.argmax(traj[4]["__all__"], axis=1)
    done_idxes = jnp.where(done_idxes == 0, 200, done_idxes)
    print(f"{done_idxes=}")

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
    base_dir = (pathlib.Path().parent.parent / "checkpoints").absolute()
    date_str = "2025_04_24_12_14_16"

    out_dir = (base_dir.parent / "arena_outputs")
    if not (out_dir / date_str).exists():
        wrs = []
        for i in range(50):
            ally_str = f"league/{date_str}/{i}"
            ally_params = load_params_from_checkpoint(checkpointer, base_dir / ally_str)
            wr = calculate_winrate_vs_heuristic(ally_params, cfg, num_traj=400, max_steps=200)
            print(f"Winrate of model {ally_str} vs Heuristic is {wr*100}")
            wrs.append(wr)

        (out_dir/date_str).mkdir(exist_ok=True)
        with open(out_dir/date_str/"wr.json", "w") as f:
            json.dump({"winrates": wrs}, f)
    else:
        with open(out_dir/date_str/"wr.json", "r") as f:
            wrs = json.load(f)["winrates"]

    plt.plot(wrs)
    plt.xlabel("Iterations")
    plt.ylabel("Win-rate")
    plt.show()
    plt.savefig(out_dir/date_str/"wr_plot.pdf", transparent=True, format="pdf")
    

    # ally_str = f"ippo/2025_04_17_12_05_16"
    # ally_params = load_params_from_checkpoint(checkpointer, base_dir / ally_str)
    # ally_params = jax.tree.map(lambda x : jnp.squeeze(x), ally_params)
    # wr = calculate_winrate_vs_heuristic(ally_params, cfg, num_traj=400, max_steps=2000)
    # print(f"Winrate of model {ally_str} vs Heuristic is {wr*100}")

if __name__ == "__main__":
    hydra_entry_point()
