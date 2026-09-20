import ale_py
import gymnasium as gym
import torch

from src.dqn import DQN, QLearningConfig

gym.register_envs(ale_py)


def make_env():
    env = gym.make("PongNoFrameskip-v4")
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
    )
    return gym.wrappers.FrameStackObservation(env, 4)


def train_once(config):
    env = make_env()
    model = DQN(env=env, config=config)
    model.train()
    env.close()
    return model


def test_smoke():
    config = QLearningConfig(
        total_timesteps=10,
        learning_starts=4,
        train_frequency=1,
        target_update_freq=5,
        buffer_size=64,
        batch_size=8,
    )

    model = train_once(config)

    assert model.buffer.pos > 0
    assert all(torch.isfinite(parameter).all() for parameter in model.network.parameters())
    assert all(model.metrics["updates"][name] for name in model.metrics["updates"])


def test_same_seed_is_reproducible():
    config = QLearningConfig(
        seed=123,
        total_timesteps=1000,
        learning_starts=80,
        train_frequency=4,
        target_update_freq=100,
        buffer_size=10_000,
        batch_size=32,
    )

    model1 = train_once(config)
    model2 = train_once(config)

    for parameter1, parameter2 in zip(
        model1.network.parameters(),
        model2.network.parameters(),
    ):
        assert torch.equal(parameter1, parameter2)

    assert model1.metrics == model2.metrics
