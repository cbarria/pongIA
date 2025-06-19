PongIA

PongIA is a minimalist deep reinforcement learning (RL) project for the classic Pong game, using Stable Baselines3, Gymnasium, and Pygame.
You can train, watch, and play against an AI agent — or just play classic Pong.

Features:

    Custom Pong environment (pong_env.py)

    Train your RL agent with PPO using many environments in parallel (scripts/train_nowatch.py)

    Watch the training process live (scripts/train_watch.py)

    Watch a pre-trained agent (scripts/watch_trained.py)

    Play as a human vs. the RL agent (scripts/play_vs_rl.py)

    Classic Pong (human vs. simple bot) (scripts/play_pong.py)

Installation:

    Clone the repository:
    git clone https://github.com/cbarria/PongIA.git
    cd PongIA

    Install dependencies (Python 3.9+ recommended):
    pip install -r requirements.txt

    Place bounce.wav and score.wav into the assets/ folder.
    (You can use any short .wav sounds you like.)

Usage:

    Train the RL Agent (Required First!)

Important: You must train a model before using watch_trained.py or play_vs_rl.py.

Start training with vectorized environments and checkpoints:
python scripts/train_nowatch.py

The first run will create models/ppo_pong_agent.zip and store checkpoints in checkpoints/.
You can interrupt (Ctrl+C) and resume later.

Watch Training Live (optional, slower):
python scripts/train_watch.py

This will show a live PongIA window while the agent is training.

    Visualize a Trained Agent

After training at least once, you can watch your RL agent play PongIA:
python scripts/watch_trained.py

    Play Against the RL Agent

Once you have a trained model:
python scripts/play_vs_rl.py

You are the left paddle (use ↑/↓ arrow keys).
The RL agent is the right paddle.

    Play Classic Pong (Human vs. Bot)

For classic PongIA against a simple bot:
python scripts/play_pong.py

File Structure:

assets/ - Sound files (bounce.wav, score.wav)
models/ - Saved RL models (.zip)
checkpoints/ - Training checkpoints
scripts/ - Training, playing, and visualization scripts
pong_env.py - Custom Pong Gymnasium environment
requirements.txt - Python dependencies

Notes & Limitations:

    Sound: Requires working audio with Pygame.

    Performance: Training is much faster with a modern CPU and >8GB RAM.

    Cross-platform: Tested on Linux, Windows, and macOS. All scripts use relative paths.

    Gymnasium: Only supports new Gymnasium (not legacy gym).

    Stable-Baselines3: RL agent uses PPO (you can tweak hyperparameters).

TODO & Ideas:

    Add support for agent vs agent.

    Save and plot training stats (TensorBoard supported).

    Add difficulty levels for the bot.

License:

MIT License (see LICENSE file)