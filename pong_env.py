# pong_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

class PongEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, sound_enabled=False):
        super().__init__()

        self.WIDTH, self.HEIGHT = 800, 600
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 10, 100
        self.BALL_SIZE = 15
        self.PADDLE_SPEED = 6
        self.BALL_SPEED = 5

        high = np.array([
            self.WIDTH, self.HEIGHT,
            10.0, 10.0,
            self.HEIGHT
        ], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.render_mode = render_mode
        self.sound_enabled = sound_enabled

        self.player_score = 0   # right paddle (RL agent)
        self.opponent_score = 0 # left paddle (auto opponent)

        if self.sound_enabled:
            pygame.mixer.init()
            self.bounce_sound = pygame.mixer.Sound(os.path.join(ASSETS_DIR, "bounce.wav"))
            self.score_sound = pygame.mixer.Sound(os.path.join(ASSETS_DIR, "score.wav"))
        else:
            self.bounce_sound = None
            self.score_sound = None

        self.font = None

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Pong RL")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 30)

        self._setup()

    def _setup(self):
        self.player = pygame.Rect(self.WIDTH - 20, self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2,
                                  self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        self.opponent = pygame.Rect(10, self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2,
                                    self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        self.ball = pygame.Rect(self.WIDTH // 2, self.HEIGHT // 2,
                                self.BALL_SIZE, self.BALL_SIZE)
        self.ball_speed_x = self.BALL_SPEED * random.choice([1, -1])
        self.ball_speed_y = self.BALL_SPEED * random.choice([1, -1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup()
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.ball.x,
            self.ball.y,
            self.ball_speed_x,
            self.ball_speed_y,
            self.player.y
        ], dtype=np.float32)

    def step(self, action):
        reward = 0.0
        terminated = False

        # Agent paddle
        if action == 1 and self.player.top > 0:
            self.player.y -= self.PADDLE_SPEED
        elif action == 2 and self.player.bottom < self.HEIGHT:
            self.player.y += self.PADDLE_SPEED

        # Simple auto opponent
        if self.opponent.centery < self.ball.centery:
            self.opponent.y += self.PADDLE_SPEED
        elif self.opponent.centery > self.ball.centery:
            self.opponent.y -= self.PADDLE_SPEED

        # Ball movement
        self.ball.x += self.ball_speed_x
        self.ball.y += self.ball_speed_y

        # Wall bounce
        if self.ball.top <= 0 or self.ball.bottom >= self.HEIGHT:
            self.ball_speed_y *= -1
            if self.bounce_sound:
                self.bounce_sound.play()

        # Paddle collisions
        if self.ball.colliderect(self.player):
            self.ball.right = self.player.left
            self.ball_speed_x *= -1
            reward += 0.1
            if self.bounce_sound:
                self.bounce_sound.play()
        elif self.ball.colliderect(self.opponent):
            self.ball.left = self.opponent.right
            self.ball_speed_x *= -1
            if self.bounce_sound:
                self.bounce_sound.play()

        # Scoring
        if self.ball.left <= 0:
            reward = 1.0
            self.player_score += 1
            terminated = True
            self._reset_ball()
            if self.score_sound:
                self.score_sound.play()
        elif self.ball.right >= self.WIDTH:
            reward = -1.0
            self.opponent_score += 1
            terminated = True
            self._reset_ball()
            if self.score_sound:
                self.score_sound.play()
        else:
            reward += 0.001
            distance = abs(self.player.centery - self.ball.centery)
            reward -= 0.01 * (distance / self.HEIGHT)
            if action != 0:
                reward -= 0.005

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        if self.render_mode != "human":
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), self.player)
        pygame.draw.rect(self.screen, (255, 255, 255), self.opponent)
        pygame.draw.ellipse(self.screen, (255, 255, 255), self.ball)
        pygame.draw.aaline(self.screen, (255, 255, 255), (self.WIDTH // 2, 0), (self.WIDTH // 2, self.HEIGHT))

        if self.font is None:
            self.font = pygame.font.SysFont("Arial", 30)
        score_text = self.font.render(
            f"IA Train: {self.player_score}   Auto: {self.opponent_score}",
            True, (255, 255, 255)
        )
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 20))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _reset_ball(self):
        self.ball.x = self.WIDTH // 2
        self.ball.y = self.HEIGHT // 2
        self.ball_speed_x = self.BALL_SPEED * random.choice([1, -1])
        self.ball_speed_y = self.BALL_SPEED * random.choice([1, -1])

    def close(self):
        if self.render_mode == "human":
            pygame.quit()

from gymnasium.envs.registration import register
register(
    id="CustomPong-v0",
    entry_point="pong_env:PongEnv",
    kwargs={
        "render_mode": None,
        "sound_enabled": False
    }
)
