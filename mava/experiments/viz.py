import pickle
import pathlib
from typing import Tuple, Optional

import hydra
import orbax.checkpoint
from mava.experiments.utils import load_params_from_checkpoint, simulate_traj
from mava.networks import FeedForwardActor as Actor
from jax import tree
import jax
import jax.numpy as jnp
from mava.experiments.smax.smax_env import SMAX
from mava.experiments.smax.heuristic_enemy_smax_env import EnemySMAX
from mava.experiments.smax import map_name_to_scenario
import matplotlib.pyplot as plt
from matplotlib import animation


class FasterFFMpegWriter(animation.FFMpegWriter):
    '''FFMpeg-pipe writer bypassing figure.savefig.'''
    def __init__(self, **kwargs):
        '''Initialize the Writer object and sets the default frame_format.'''
        super().__init__(**kwargs)
        self.frame_format = 'argb'

    def grab_frame(self, **savefig_kwargs):
        '''Grab the image information from the figure and save as a movie frame.

        Doesn't use savefig to be faster: savefig_kwargs will be ignored.
        '''
        try:
            # re-adjust the figure size and dpi in case it has been changed by the
            # user.  We must ensure that every frame is the same size or
            # the movie will not save correctly.
            self.fig.set_size_inches(self._w, self._h)
            self.fig.set_dpi(self.dpi)
            # Draw and save the frame as an argb string to the pipe sink
            self.fig.canvas.draw()
            self._proc.stdin.write(self.fig.canvas.tostring_argb())
        except (RuntimeError, IOError) as e:
            out, err = self._proc.communicate()
            raise IOError('Error saving animation to file (cause: {0}) '
                      'Stdout: {1} StdError: {2}. It may help to re-run '
                      'with --verbose-debug.'.format(e, out, err)) 


class Visualizer(object):
    def __init__(
        self,
        env,
        state_seq,
        reward_seq=None,
    ):
        self.env = env

        self.interval = 64
        self.state_seq = state_seq
        self.reward_seq = reward_seq
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 5))

    def animate(
        self,
        save_fname: Optional[str] = None,
        view: bool = False,
    ):
        """Anim for 2D fct - x (#steps, #pop, 2) & fitness (#steps, #pop)"""
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.state_seq),
            init_func=self.init,
            blit=False,
            interval=self.interval,
        )
        # Save the animation to a gif
        if save_fname is not None:
            ani.save(save_fname, writer=FasterFFMpegWriter(fps=20))
        # Simply view it 3 times
        if view:
            plt.show(block=True)
            # plt.pause(30)
            # plt.close()

    def init(self):
        self.im = self.env.init_render(self.ax, self.state_seq[0])

    def update(self, frame):
        self.im = self.env.update_render(
            self.im, self.state_seq[frame]
        )


class SMAXVisualizer(Visualizer):
    """Visualiser especially for the SMAX environments. Needed because they have an internal model that ticks much faster
    than the learner's 'step' calls. This  means that we need to expand the state_sequence
    """

    def __init__(
        self,
        env,
        state_seq,
        reward_seq=None,
    ):
        super().__init__(env, state_seq, reward_seq)
        self.heuristic_enemy = isinstance(env, EnemySMAX)
        self.have_expanded = False

    def expand_state_seq(self):
        """Because the smax environment ticks faster than the states received
        we need to expand the states to visualise them"""
        self.state_seq = self.env.expand_state_seq(self.state_seq)
        self.have_expanded = True

    def animate(self, save_fname: Optional[str] = None, view: bool = True):
        if not self.have_expanded:
            self.expand_state_seq()
        return super().animate(save_fname, view)

    def init(self):
        self.im = self.env.init_render(
            self.ax, self.state_seq[0], 0, 0
        )

    def update(self, frame):
        self.im = self.env.update_render(
            self.im,
            self.state_seq[frame],
            frame % self.env.world_steps_per_env_step,
            frame // self.env.world_steps_per_env_step,
        )


def pytree_to_list(pytree):
    """Converts a JAX pytree with leading dimension to a list of pytrees.

    Args:
        pytree: A JAX pytree where all leaves have the same leading dimension.

    Returns:
        A Python list, where each element is a pytree with the same 
        structure as the input, but without the leading dimension.
        Returns an empty list if the input pytree is empty or None.
    """

    if pytree is None:
      return []

    leaves, treedef = tree.flatten(pytree)

    if not leaves:
      return []

    leading_dim = leaves[0].shape[0]  # Get the leading dimension

    result = []
    for i in range(leading_dim):
        # Efficiently slice all leaves at the same index
        sliced_leaves = [leaf[i] for leaf in leaves]
        sliced_pytree = treedef.unflatten(sliced_leaves)
        result.append(sliced_pytree)

    return result

def visualize_episode(env, traj, filename):
    state_seq = pytree_to_list(traj)

    viz = SMAXVisualizer(env, state_seq)

    viz.animate(filename, view=False)

def load(ally_model_path, enemy_model_path, config_path):
        # Load models
    ally_net, ally_params = pickle.load(open(ally_model_path, 'rb'))
    ally_params = jax.tree.map(lambda x : x[0][0], ally_params)
    enemy_net, enemy_params = pickle.load(open(enemy_model_path, 'rb'))
    config = pickle.load(open(config_path, "rb"))

    return ((ally_net, ally_params), (enemy_net, enemy_params), config)

def viz_specific(ally_params, enemy_params, config, filename, max_search=100, ally_won_cond=False):
    kwargs = dict(config.env.kwargs)
    kwargs["scenario"] = map_name_to_scenario(config.env.scenario.task_name)

    # Define network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    action_head = {"_target_": "mava.networks.heads.DiscreteActionHead"}
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=10)

    network = Actor(torso=actor_torso, action_head=actor_action_head)

    # Initialize environment
    env = SMAX(**kwargs)

    key = jax.random.PRNGKey(2025)
    for i in range(max_search):
        key, traj_key = jax.random.split(key)

        traj = simulate_traj(traj_key, env, network, ally_params, network, enemy_params)
        done_idx = jnp.argmax(traj[4]["__all__"])
        done_idx = jnp.where(done_idx == 0, 200, done_idx).item()
        ally_won = jnp.any(traj[3]["ally_0"][:done_idx+1] >= 1)

        if bool(ally_won) == ally_won_cond:
            traj= jax.tree.map(lambda x : x[:done_idx+1], traj)
            visualize_episode(env, traj[:3], filename)
            break

def viz_random(ally_model_path, enemy_model_path, config_path, filename="random.gif"):
        # Load models
    ally_net, ally_params = pickle.load(open(ally_model_path, 'rb'))
    if jax.tree.leaves(ally_params)[0].ndim > 2:
        ally_params = jax.tree.map(lambda x : x[0][0], ally_params)
    enemy_net, enemy_params = pickle.load(open(enemy_model_path, 'rb'))
    if jax.tree.leaves(enemy_params)[0].ndim > 2:
        enemy_params = jax.tree.map(lambda x : x[0][0], enemy_params)


    config = pickle.load(open(config_path, "rb"))

    kwargs = dict(config.env.kwargs)
    kwargs["scenario"] = map_name_to_scenario(config.env.scenario.task_name)

    # Initialize environment
    env = SMAX(**kwargs)

    key = jax.random.PRNGKey(2025)

    traj = simulate_traj(key, env, ally_net, ally_params, enemy_net, enemy_params)

    visualize_episode(env, traj[:3], filename)

@hydra.main(
    config_path="../configs/default",
    config_name="ff_ippo_sp.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg):
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    base_dir = (pathlib.Path().parent.parent / "checkpoints/self_play").absolute()
    ally_params = load_params_from_checkpoint(checkpointer, base_dir / "2025_02_25_11_11_21/9")
    enemy_params = load_params_from_checkpoint(checkpointer, base_dir / "2025_02_25_11_11_21/8")

    viz_specific(ally_params, enemy_params, cfg, "trained_sp_1_a9_e8.mp4", ally_won_cond=True)
    print('done!')

if __name__ == "__main__":
    hydra_entry_point()