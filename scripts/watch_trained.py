# scripts/watch_trained.py

import os
import sys

# Add the project root to sys.path so 'pong_env' is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import gymnasium as gym
from stable_baselines3 import PPO
import pong_env  # Registers CustomPong-v0

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "ppo_pong_agent.zip")

def main():
    env = gym.make("CustomPong-v0", render_mode="human", sound_enabled=True)
    model = PPO.load(MODEL_PATH, device="cpu")

    obs, _ = env.reset()

    try:
        while True:
            # Always use obs[None, :] to avoid batch/shape errors
            obs_batch = obs[None, :] if obs.ndim == 1 else obs
            action, _ = model.predict(obs_batch, deterministic=True)
            action = int(action) if not hasattr(action, "__len__") else int(action[0])
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("Execution interrupted by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
