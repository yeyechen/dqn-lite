import ale_py
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from pathlib import Path

from src.dqn import DQN

gym.register_envs(ale_py)


def save_metric_plots(model, output_dir="outputs/plots"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    updates = model.metrics["updates"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    update_plots = (
        ("loss", "Huber loss"),
        ("avg_q", "Average Q-value"),
        ("grad_norm", "Gradient norm"),
        ("eps", "Epsilon"),
    )
    for axis, (metric_name, title) in zip(axes.flat, update_plots):
        axis.plot(updates["timestep"], updates[metric_name])
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
    # environment
    env = gym.make("PongNoFrameskip-v4")
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
