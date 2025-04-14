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

    for iter in range(20):
        ally_params = load_params_from_checkpoint(checkpointer, base_dir / f"league/2025_03_30_17_01_04/{iter}")

        enemy_params = load_params_from_checkpoint(checkpointer, base_dir / "ippo/2025_04_14_11_32_00")
        enemy_params = jax.tree.map(lambda x : jnp.squeeze(x), enemy_params)

        print(f"Iter {iter} vs IPPO-trained: {100 * calculate_winrate(ally_params, enemy_params, cfg)}")

main()