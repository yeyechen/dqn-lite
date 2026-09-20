import numpy as np

from src.buffer import ReplayBuffer


def test_overflow_overwrites():
    size = 10
    shape = (4, 84, 84)
    buffer = ReplayBuffer(size=size, obs_shape=shape)
    n = size + 1
    for i in range(n):
        obs = np.full(shape, i, np.uint8)
        next_obs = np.full(shape, i + 1, np.uint8)
        buffer.add(obs, next_obs, i, 0, False)  # obs, next_obs, action, reward, done

    assert buffer.obs[0][0, 0, 0] == size
    assert buffer.full
