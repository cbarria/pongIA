"""Custom Gymnasium Pong Environment."""

import os
import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


class PongEnv(gym.Env):
    """Custom Pong Environment for RL agents."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, sound_enabled=False):
        super().__init__()

        self.width = 800
        self.height = 600
        self.paddle_width = 10
        self.paddle_height = 100
        self.ball_size = 15
        self.paddle_speed = 6
        self.ball_speed = 5

        high = np.array([
            self.width, self.height,
            10.0, 10.0,
            self.height
        ], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.render_mode = render_mode
        self.sound_enabled = sound_enabled

        self.player_score = 0   # Right paddle (RL agent)
        self.opponent_score = 0 # Left paddle (auto or RL)

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
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Pong RL")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 30)

        self._setup()

    def _setup(self):
        """Initialize game objects."""
        self.player = pygame.Rect(
            self.width - 20, self.height // 2 - self.paddle_height // 2,
            self.paddle_width, self.paddle_height
        )
        self.opponent = pygame.Rect(
            10, self.height // 2 - self.paddle_height // 2,
            self.paddle_width, self.paddle_height
        )
        self.ball = pygame.Rect(
            self.width // 2, self.height // 2,
            self.ball_size, self.ball_size
        )
        self.ball_speed_x = self.ball_speed * random.choice([1, -1])
        self.ball_speed_y = self.ball_speed * random.choice([1, -1])

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self._setup()
        return self._get_obs(), {}

    def _get_obs(self):
        """Return current observation."""
        return np.array([
            self.ball.x,
            self.ball.y,
            self.ball_speed_x,
            self.ball_speed_y,
            self.player.y
        ], dtype=np.float32)

    def step(self, action):
        """Apply agent action and update environment."""
        reward = 0.0
        terminated = False

        # Agent paddle
        if action == 1 and self.player.top > 0:
            self.player.y -= self.paddle_speed
        elif action == 2 and self.player.bottom < self.height:
            self.player.y += self.paddle_speed

        # Simple auto opponent
        if self.opponent.centery < self.ball.centery:
            self.opponent.y += self.paddle_speed
        elif self.opponent.centery > self.ball.centery:
            self.opponent.y -= self.paddle_speed

        # Ball movement
        self.ball.x += self.ball_speed_x
        self.ball.y += self.ball_speed_y

        # Wall bounce
        if self.ball.top <= 0 or self.ball.bottom >= self.height:
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
        elif self.ball.right >= self.width:
            reward = -1.0
            self.opponent_score += 1
            terminated = True
            self._reset_ball()
            if self.score_sound:
                self.score_sound.play()
        else:
            reward += 0.001
            distance = abs(self.player.centery - self.ball.centery)
            reward -= 0.01 * (distance / self.height)
            if action != 0:
                reward -= 0.005

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        """Render the game window."""
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
        pygame.draw.aaline(
            self.screen, (255, 255, 255),
            (self.width // 2, 0), (self.width // 2, self.height)
        )

        if self.font is None:
            self.font = pygame.font.SysFont("Arial", 30)
        score_text = self.font.render(
            f"IA Train: {self.player_score}   Auto: {self.opponent_score}",
            True, (255, 255, 255)
        )
        self.screen.blit(
            score_text,
            (self.width // 2 - score_text.get_width() // 2, 20)
        )

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _reset_ball(self):
        """Reset the ball to the center with random direction."""
        self.ball.x = self.width // 2
        self.ball.y = self.height // 2
        self.ball_speed_x = self.ball_speed * random.choice([1, -1])
        self.ball_speed_y = self.ball_speed * random.choice([1, -1])

    def close(self):
        """Close the game window."""
        if self.render_mode == "human":
            pygame.quit()


register(
    id="CustomPong-v0",
    entry_point="pong_env:PongEnv",
    kwargs={
        "render_mode": None,
        "sound_enabled": False
    }
)
