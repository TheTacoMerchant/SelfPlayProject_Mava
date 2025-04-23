import pathlib

import hydra
import jax
import jax.numpy as jnp
import orbax.checkpoint
from mava.experiments.utils import calculate_winrate, load_params_from_checkpoint

@hydra.main(
    config_path="../configs/default",
    config_name="ff_ippo_sp.yaml",
    version_base="1.2",
)
def main(cfg):
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    base_dir = (pathlib.Path().parent.parent / "checkpoints").absolute()

    for iter in range(10):
        ally_params = load_params_from_checkpoint(checkpointer, base_dir / f"league/2025_04_16_14_56_43/{iter}")

        enemy_params = load_params_from_checkpoint(checkpointer, base_dir / "ippo/2025_04_17_09_30_55")
        enemy_params = jax.tree.map(lambda x : jnp.squeeze(x), enemy_params)

        print(f"Iter {iter} vs IPPO-trained: {100 * calculate_winrate(ally_params, enemy_params, cfg)}")

main()