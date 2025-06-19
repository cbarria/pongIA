# scripts/train_watch.py

import os
import sys
import threading
import time
import numpy as np

# Add the project root to sys.path so 'pong_env' is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from stable_baselines3 import PPO
from pong_env import PongEnv

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "ppo_pong_agent.zip")
STOP_THREADS = False

def train_loop(env, model):
    global STOP_THREADS
    try:
        while not STOP_THREADS:
            model.learn(total_timesteps=2048, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
    except Exception as e:
        print(f"Train thread error: {e}")
    finally:
        try:
            model.save(MODEL_PATH)
        except Exception as e:
            print(f"Error saving model: {e}")
        STOP_THREADS = True
        try:
            env.close()
        except Exception:
            pass

def render_loop(env, model):
    global STOP_THREADS
    obs, _ = env.reset()
    try:
        while not STOP_THREADS:
            # Observation validation
            if not isinstance(obs, np.ndarray):
                print(f"[DEBUG] Observation is not np.ndarray: type={type(obs)}, value={obs}")
                break
            if obs.shape != (5,):
                print(f"[DEBUG] Unexpected observation shape: shape={obs.shape}, value={obs}")
                break

            obs_batch = obs[None, :] if obs.ndim == 1 else obs
            try:
                action, _ = model.predict(obs_batch, deterministic=False)
                action = int(np.asarray(action).flatten()[0])
            except Exception as e:
                print(f"[DEBUG] Error in model.predict: {e}")
                print(f"[DEBUG] obs_batch={obs_batch}, type={type(obs_batch)}, shape={getattr(obs_batch, 'shape', None)}")
                break

            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()
            time.sleep(1 / env.metadata["render_fps"])
            if terminated or truncated:
                obs, _ = env.reset()
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as e:
        print(f"Render thread error: {e}")
    finally:
        STOP_THREADS = True
        try:
            env.close()
        except Exception:
            pass

def main():
    global STOP_THREADS
    env = PongEnv(render_mode="human", sound_enabled=True)
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        model = PPO.load(MODEL_PATH, env=env, device="cpu")
    else:
        print("Creating new model...")
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")

    t_train = threading.Thread(target=train_loop, args=(env, model), daemon=True)
    t_render = threading.Thread(target=render_loop, args=(env, model), daemon=True)

    t_train.start()
    t_render.start()

    try:
        while not STOP_THREADS:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Stopping...")
        STOP_THREADS = True
        try:
            model.save(MODEL_PATH)
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()