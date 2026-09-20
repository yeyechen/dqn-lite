import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, size, obs_shape):
        self.obs = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.pos, self.full, self.capacity = 0, False, size

    def add(self, obs, next_obs, action, reward, done):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def sample(self, batch_size):
        idx = np.random.randint(0, self.capacity if self.full else self.pos, batch_size)
        return (
            torch.tensor(self.obs[idx]),
            torch.tensor(self.actions[idx]),
            torch.tensor(self.rewards[idx]),
            torch.tensor(self.next_obs[idx]),
            torch.tensor(self.dones[idx]),
        )
