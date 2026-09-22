import ale_py
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dqn import DQN

gym.register_envs(ale_py)


def moving_average(values, window):
    return np.convolve(values, np.ones(window) / window, mode="valid")


def save_metric_plots(model, output_dir="outputs/plots"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    updates = model.metrics["updates"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    update_plots = (
        ("loss", "Huber loss (smoothed)"),
        ("target_max", "Average maximum target Q-value (smoothed)"),
        ("grad_norm", "Gradient norm (smoothed)"),
        ("eps", "Epsilon"),
    )
    window = 100
    for axis, (metric_name, title) in zip(axes.flat, update_plots):
        if metric_name == "eps":
            axis.plot(updates["timestep"], updates[metric_name])
        else:
            axis.plot(
                updates["timestep"][window - 1 :],
                moving_average(updates[metric_name], window),
            )
        axis.set_title(title)
        axis.set_xlabel("Environment timestep")
        axis.grid(alpha=0.3)
    fig.savefig(output_dir / "training_metrics.png", dpi=150)
    plt.close(fig)

    episodes = model.metrics["episodes"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    episode_plots = (
        ("ret", "Episode return"),
        ("len", "Episode length"),
    )
    for axis, (metric_name, title) in zip(axes, episode_plots):
        axis.plot(episodes["end_timestep"], episodes[metric_name])
        axis.set_title(title)
        axis.grid(alpha=0.3)
    fig.savefig(output_dir / "episode_metrics.png", dpi=150)
    plt.close(fig)


def main():
    # environment options:
    # 1. Breakout: "BreakoutNoFrameskip-v4"
    # 2. Pong: "PongNoFrameskip-v4"
    # 3. Space Invaders: "SpaceInvadersNoFrameskip-v4"
    # 4. Seaquest: "SeaquestNoFrameskip-v4"

    env = gym.make("SpaceInvadersNoFrameskip-v4")
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
    env = gym.wrappers.FrameStackObservation(env, 4)

    model = DQN(env=env)
    model.train()
    run_dir = Path("outputs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_metric_plots(model, run_dir / "plots")
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.network.state_dict(), models_dir / "dqn_final.pt")
    env.close()


if __name__ == "__main__":
    main()
