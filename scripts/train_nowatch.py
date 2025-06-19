# scripts/train_nowatch.py

import os
import sys

# Add the project root to sys.path so 'pong_env' is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import pong_env  # Registers CustomPong-v0

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
TENSORBOARD_DIR = os.path.join(PROJECT_ROOT, "ppo_pong_tensorboard")
MODEL_PATH = os.path.join(MODELS_DIR, "ppo_pong_agent.zip")

for d in [MODELS_DIR, CHECKPOINTS_DIR, TENSORBOARD_DIR]:
    os.makedirs(d, exist_ok=True)

# Training settings
N_ENVS = 2048
TOTAL_TIMESTEPS = 50_000_000

def make_env():
    return gym.make(
        "CustomPong-v0",
        render_mode=None,
        sound_enabled=False
    )

if __name__ == "__main__":
    # Create vectorized environments
    env = DummyVecEnv([make_env for _ in range(N_ENVS)])

    # Load existing model or create a new one
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        model = PPO.load(MODEL_PATH, env=env, device="cpu")
    else:
        print("Creating new model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=TENSORBOARD_DIR,
            device="cpu",
            n_steps=256,
            batch_size=256,
            gae_lambda=0.95,
            gamma=0.99,
            n_epochs=4,
            clip_range=0.15,
            ent_coef=0.05,
            learning_rate=1e-4,
        )

    # Set up checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000,
        save_path=CHECKPOINTS_DIR,
        name_prefix="ppo_pong"
    )

    # Train
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback
        )
        model.save(MODEL_PATH)
        print("Training completed. Final model saved.")
    except KeyboardInterrupt:
        print("Training interrupted! Saving current model...")
        model.save(MODEL_PATH)
    finally:
        env.close()
