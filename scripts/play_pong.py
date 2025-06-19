# scripts/play_pong.py

import pygame
import random
import os

# Use the assets folder relative to project root
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")

pygame.init()
pygame.mixer.init()

# Load sounds from assets directory
bounce_sound = pygame.mixer.Sound(os.path.join(ASSETS_DIR, "bounce.wav"))
score_sound = pygame.mixer.Sound(os.path.join(ASSETS_DIR, "score.wav"))

WIDTH, HEIGHT = 800, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100
BALL_SIZE = 15

PADDLE_SPEED = 6
BALL_SPEED_X = 5 * random.choice((1, -1))
BALL_SPEED_Y = 5 * random.choice((1, -1))

opponent = pygame.Rect(10, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
player = pygame.Rect(WIDTH - 20, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
ball = pygame.Rect(WIDTH // 2, HEIGHT // 2, BALL_SIZE, BALL_SIZE)

player_score = 0
opponent_score = 0
font = pygame.font.Font(None, 74)
clock = pygame.time.Clock()

def reset_ball():
    global BALL_SPEED_X, BALL_SPEED_Y
    ball.center = (WIDTH // 2, HEIGHT // 2)
    BALL_SPEED_X = 5 * random.choice((1, -1))
    BALL_SPEED_Y = 5 * random.choice((1, -1))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Human player movement (arrow keys)
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP] and player.top > 0:
        player.y -= PADDLE_SPEED
    if keys[pygame.K_DOWN] and player.bottom < HEIGHT:
        player.y += PADDLE_SPEED

    # Simple AI for opponent
    if opponent.centery < ball.centery:
        opponent.y += PADDLE_SPEED
    elif opponent.centery > ball.centery:
        opponent.y -= PADDLE_SPEED

    player.clamp_ip(SCREEN.get_rect())
    opponent.clamp_ip(SCREEN.get_rect())

    # Ball movement
    ball.x += BALL_SPEED_X
    ball.y += BALL_SPEED_Y

    # Bounce logic
    if ball.top <= 0 or ball.bottom >= HEIGHT:
        BALL_SPEED_Y *= -1
        bounce_sound.play()

    if ball.colliderect(player) or ball.colliderect(opponent):
        BALL_SPEED_X *= -1
        bounce_sound.play()

    # Score logic
    if ball.left <= 0:
        player_score += 1
        score_sound.play()
        reset_ball()
    elif ball.right >= WIDTH:
        opponent_score += 1
        score_sound.play()
        reset_ball()

    SCREEN.fill(BLACK)
    pygame.draw.rect(SCREEN, WHITE, player)
    pygame.draw.rect(SCREEN, WHITE, opponent)
    pygame.draw.ellipse(SCREEN, WHITE, ball)
    pygame.draw.aaline(SCREEN, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT))

    player_text = font.render(str(player_score), True, WHITE)
    opponent_text = font.render(str(opponent_score), True, WHITE)
    SCREEN.blit(opponent_text, (WIDTH // 2 - 60, 20))
    SCREEN.blit(player_text, (WIDTH // 2 + 20, 20))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
