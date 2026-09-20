import ale_py
import gymnasium as gym
import torch

from src.dqn import DQN, QLearningConfig

gym.register_envs(ale_py)


def test_smoke():
    # environment
    env = gym.make("PongNoFrameskip-v4")
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
    env = gym.wrappers.FrameStackObservation(env, 4)

    model = DQN(
        env=env,
        config=QLearningConfig(
            total_timesteps=10,
            learning_starts=4,
            train_frequency=1,
            target_update_freq=5,
            buffer_size=64,
            batch_size=8,
        ),
    )
    model.train()

    assert model.buffer.pos > 0  # experience stored
    assert all(torch.isfinite(p).all() for p in model.network.parameters())  # no divergence
    assert all(model.metrics["updates"][name] for name in model.metrics["updates"])  # metrics were recorded
