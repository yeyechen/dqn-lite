import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict

from src.buffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),  # (4, 84, 84) -> (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),  # (32, 20, 20) -> (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),  # (64, 9, 9) -> (64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),  # 64 * 7 * 7 = 3136
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.network(x / 255.0)


@dataclass
class QLearningConfig:
    learning_rate: float = 1e-4
    buffer_size: int = 200_000
    batch_size: int = 32
    start_eps: int = 1
    end_eps: float = 0.01
    exploration_fraction: float = 0.10
    total_timesteps: int = 3_000_000
    learning_starts: int = 80_000
    train_frequency: int = 4
    target_update_freq: int = 1000
    gamma: float = 0.99
    seed: int = 67


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DQN:
    def __init__(self, env, config: QLearningConfig | None = None):
        self.cfg = config or QLearningConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.cfg.seed)

        self.env = env
        self.buffer = ReplayBuffer(self.cfg.buffer_size, obs_shape=(4, 84, 84))
        self.network = QNetwork(env.action_space.n).to(self.device)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.cfg.learning_rate)

        self.metrics = defaultdict(lambda: defaultdict(list))

    def record_metrics(self, group, **values):
        for name, value in values.items():
            self.metrics[group][name].append(value)

    def train(self):
        # "initialise sequence"
        obs, _ = self.env.reset(seed=self.cfg.seed)
        episode_ret, episode_len = 0, 0
        for timestep in tqdm(range(self.cfg.total_timesteps)):
            # "with probability episilon select a random action a_t"
            # "otherwise select a = argmax Q(phi(s_t), a; theta)"
            eps = linear_schedule(
                self.cfg.start_eps,
                self.cfg.end_eps,
                self.cfg.exploration_fraction * self.cfg.total_timesteps,
                timestep,
            )
            if random.random() < eps:
                action = self.env.action_space.sample()
            else:
                q_values = self.network(torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0).to(self.device))
                action = q_values.argmax().item()

            # "execute an action a_t in emulator and observe reward r_t and image x_t+1"
            # "set s_t+1 = s_t, a_t, x_t+1 and preprocess phi_t+1 = phi(s_t+1)" - fused in the env
            # "store transition in D"
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            real_next_obs = info["final_observation"] if truncated else next_obs
            reward = np.sign(reward)  # clip reward to {-1, 0, +1}

            episode_ret += reward
            episode_len += 1
            self.buffer.add(obs, real_next_obs, action, reward, terminated or truncated)
            obs = next_obs

            # end of episode
            if terminated or truncated:
                self.record_metrics("episodes", ret=episode_ret, len=episode_len, end_timestep=timestep)
                episode_ret, episode_len = 0, 0
                obs, _ = self.env.reset()

            # "sample random minibatch of transitions from D"
            # "set y_j = r_j if terminal; else y_j = r_j + gamma * max Q(phi_j+1, a'; theta)"
            # "perform a gradient descent step"
            if timestep > self.cfg.learning_starts and timestep % self.cfg.train_frequency == 0:
                obs_b, acts_b, rews_b, next_obs_b, dones_b = [t.to(self.device) for t in self.buffer.sample(self.cfg.batch_size)]
                with torch.no_grad():
                    target_max = self.target_network(next_obs_b).max(dim=1).values
                    td_target = rews_b + self.cfg.gamma * target_max * (1 - dones_b)
                current_q = self.network(obs_b).gather(1, acts_b.unsqueeze(1)).squeeze()
                loss = nn.HuberLoss()(td_target, current_q)
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=float("inf"))
                self.optimizer.step()

                # record
                self.record_metrics(
                    "updates", timestep=timestep, loss=loss.item(), avg_q=current_q.mean().item(), grad_norm=grad_norm.item(), eps=eps
                )
            if timestep % self.cfg.target_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())
