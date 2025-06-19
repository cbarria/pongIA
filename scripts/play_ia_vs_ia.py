import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
import numpy as np
from pong_env import PongEnv
import time

MODEL_PATH_RIGHT = "models/ppo_pong_agent"
MODEL_PATH_LEFT  = "models/ppo_pong_agent_left"  # Usa el mismo para ambos si no tienes dos

def main():
    print("Current working dir:", os.getcwd())
    print("Absolute path for right model:", os.path.abspath(MODEL_PATH_RIGHT + ".zip"))
    print("File exists?", os.path.exists(MODEL_PATH_RIGHT + ".zip"))

    if not os.path.exists(MODEL_PATH_RIGHT + ".zip"):
        print("\nERROR: Model file not found!")
        print(f"Train a model first with: python scripts/train_nowatch.py")
        return

    model_right = PPO.load(MODEL_PATH_RIGHT, device="cpu")
    try:
        model_left = PPO.load(MODEL_PATH_LEFT, device="cpu")
    except Exception:
        print("No left model found, using the same model for both sides.")
        model_left = model_right

    env = PongEnv(render_mode="human", sound_enabled=True)
    obs, _ = env.reset()

    try:
        while True:
            obs_right = obs.copy()
            obs_left = obs.copy()
            obs_left[0] = env.WIDTH - obs_left[0]
            obs_left[2] = -obs_left[2]
            obs_left[4] = env.HEIGHT - obs_left[4]

            action_right, _ = model_right.predict(obs_right, deterministic=True)
            action_left,  _ = model_left.predict(obs_left, deterministic=True)

            # --------- IA controla la paleta izquierda (opponent) ----------
            # Este hack pisa el movimiento automático del bot
            if action_left == 1 and env.opponent.top > 0:
                env.opponent.y -= env.PADDLE_SPEED
            elif action_left == 2 and env.opponent.bottom < env.HEIGHT:
                env.opponent.y += env.PADDLE_SPEED

            # El step mueve la paleta derecha (el agente clásico)
            obs, reward, terminated, truncated, _ = env.step(action_right)
            env.render()
            time.sleep(1 / env.metadata["render_fps"])

            if terminated or truncated:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("Execution interrupted by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()

