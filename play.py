from pathlib import Path

import ale_py
import gymnasium as gym
import torch

from src.dqn import QNetwork

gym.register_envs(ale_py)

ENV_ID = "SpaceInvadersNoFrameskip-v4"
WEIGHTS = Path("outputs/20260921_151144/models/dqn_final.pt")


def make_env(env_id, video_dir):
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=video_dir, episode_trigger=lambda episode: True)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
    return gym.wrappers.FrameStackObservation(env, 4)


def load_model(env, weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QNetwork(env.action_space.n).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    return model, device


def main():
    env = make_env(ENV_ID, WEIGHTS.parent.parent / "gameplay")
    model, device = load_model(env, WEIGHTS)

    obs, _ = env.reset()
    score = 0
    terminated = truncated = False

    while not (terminated or truncated):
        state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = model(state).argmax(dim=1).item()
        obs, reward, terminated, truncated, _ = env.step(action)
        score += reward

    print(f"Episode score: {score}")
    env.close()


if __name__ == "__main__":
    main()
