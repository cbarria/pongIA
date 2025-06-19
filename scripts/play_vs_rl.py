# scripts/play_vs_rl.py

import os
import sys
import numpy as np
import pygame

# Add the project root to sys.path so pong_env is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from stable_baselines3 import PPO
from pong_env import PongEnv

ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "ppo_pong_agent.zip")

pygame.init()
pygame.mixer.init()

bounce_sound = pygame.mixer.Sound(os.path.join(ASSETS_DIR, "bounce.wav"))
score_sound = pygame.mixer.Sound(os.path.join(ASSETS_DIR, "score.wav"))

WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100
BALL_SIZE = 15
PADDLE_SPEED = 6
BALL_SPEED = 5

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong: Human vs RL Agent")
font = pygame.font.SysFont("Arial", 36)
clock = pygame.time.Clock()

# Rects: left (human), right (RL)
player = pygame.Rect(10, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
rl_agent = pygame.Rect(WIDTH - 20, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
ball = pygame.Rect(WIDTH // 2, HEIGHT // 2, BALL_SIZE, BALL_SIZE)

player_score = 0
rl_score = 0

# Load RL agent
model = PPO.load(MODEL_PATH, device="cpu")

def reset_ball():
    ball.x = WIDTH // 2
    ball.y = HEIGHT // 2
    ball_speed_x = BALL_SPEED * np.random.choice([1, -1])
    ball_speed_y = BALL_SPEED * np.random.choice([1, -1])
    return ball_speed_x, ball_speed_y

ball_speed_x, ball_speed_y = reset_ball()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Human controls (arrows)
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP] and player.top > 0:
        player.y -= PADDLE_SPEED
    if keys[pygame.K_DOWN] and player.bottom < HEIGHT:
        player.y += PADDLE_SPEED

    # RL agent: build observation for its side (right paddle)
    obs = np.array([
        ball.x,
        ball.y,
        ball_speed_x,
        ball_speed_y,
        rl_agent.y
    ], dtype=np.float32)
    obs_batch = obs[None, :] if obs.ndim == 1 else obs
    action, _ = model.predict(obs_batch, deterministic=True)
    action = int(action) if not hasattr(action, "__len__") else int(action[0])
    # RL actions: 0 = stay, 1 = up, 2 = down
    if action == 1 and rl_agent.top > 0:
        rl_agent.y -= PADDLE_SPEED
    elif action == 2 and rl_agent.bottom < HEIGHT:
        rl_agent.y += PADDLE_SPEED

    player.clamp_ip(screen.get_rect())
    rl_agent.clamp_ip(screen.get_rect())

    # Ball movement
    ball.x += ball_speed_x
    ball.y += ball_speed_y

    # Wall bounce
    if ball.top <= 0 or ball.bottom >= HEIGHT:
        ball_speed_y *= -1
        bounce_sound.play()

    # Paddle collision
    if ball.colliderect(player):
        ball.left = player.right
        ball_speed_x *= -1
        bounce_sound.play()
    elif ball.colliderect(rl_agent):
        ball.right = rl_agent.left
        ball_speed_x *= -1
        bounce_sound.play()

    # Score logic
    if ball.left <= 0:
        rl_score += 1
        score_sound.play()
        ball_speed_x, ball_speed_y = reset_ball()
    elif ball.right >= WIDTH:
        player_score += 1
        score_sound.play()
        ball_speed_x, ball_speed_y = reset_ball()

    # Draw everything
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, player)
    pygame.draw.rect(screen, WHITE, rl_agent)
    pygame.draw.ellipse(screen, WHITE, ball)
    pygame.draw.aaline(screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT))

    score_text = font.render(f"You: {player_score}   RL: {rl_score}", True, WHITE)
    screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 20))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()