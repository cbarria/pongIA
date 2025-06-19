# scripts/train_watch.py

import os
import sys
import threading
import time
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from stable_baselines3 import PPO
from pong_env import PongEnv

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "ppo_pong_agent")
TRAINING = True  # Global flag

def train_loop(env, model):
    global TRAINING
    try:
        while TRAINING:
            model.learn(total_timesteps=2048, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("Training interrupted (background).")
    finally:
        TRAINING = False
        try:
            model.save(MODEL_PATH)
        except Exception as e:
            print(f"Error saving model: {e}")
        try:
            env.close()
        except Exception:
            pass

def main():
    global TRAINING
    env = PongEnv(render_mode="human", sound_enabled=True)
    if os.path.exists(MODEL_PATH + ".zip"):
        print("Loading existing model...")
        model = PPO.load(MODEL_PATH, env=env, device="cpu")
    else:
        print("Creating new model...")
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")

    # Start training in a background thread
    t_train = threading.Thread(target=train_loop, args=(env, model), daemon=True)
    t_train.start()

    obs, _ = env.reset()
    try:
        while TRAINING:
            # --- FIX: Always batch the observation! ---
            obs_batch = obs.reshape(1, -1) if obs.ndim == 1 else obs
            action, _ = model.predict(obs_batch, deterministic=False)
            action = int(np.asarray(action).flatten()[0])  # ensure int
            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()
            time.sleep(1 / env.metadata["render_fps"])
            if terminated or truncated:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Stopping...")
        TRAINING = False
        try:
            model.save(MODEL_PATH)
        except Exception as e:
            print(f"Error saving model: {e}")
        try:
            env.close()
        except Exception:
            pass
    finally:
        TRAINING = False
        t_train.join(timeout=2)
        env.close()

if __name__ == "__main__":
    main()
