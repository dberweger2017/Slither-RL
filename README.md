# Slither.io Reinforcement Learning AI

A complete Gymnasium environment, Python physics engine, and Stable Baselines 3 PPO training pipeline designed to learn how to play Slither.io from scratch using visual CNN inputs.

---

## 1. Training the AI (`train.py`)

### Basic Training
```bash
python3 train.py
```

### All Flags
| Flag | Default | Description |
|---|---|---|
| `--stage <1\|2\|3>` | `3` | Curriculum stage (see below) |
| `--resume <path>` | — | Resume from `"latest"` or a specific checkpoint path |
| `--timesteps <N>` | `10000000` | Total training steps before auto-stop |
| `--num-envs <N>` | `4` | Parallel environment subprocesses |
| `--render` | off | Opens a live Pygame training window (slow) |

### Curriculum Learning Stages
| Stage | Command | Arena | Goal |
|---|---|---|---|
| **1** | `python3 train.py --stage 1` | Agent + Food only | Learn movement, eating, wall avoidance |
| **2** | `python3 train.py --stage 2` | Agent + 9 scripted bots | Learn combat, dodging, hunting |
| **3** | `python3 train.py --stage 3` | Agent + 9 bots + 6 self-play clones | Develop robust, generalised strategies |

### Resuming Training
```bash
python3 train.py --resume latest --stage 2
```
Always picks `policy_final.zip` first (saved on Ctrl+C), then falls back to the highest numbered checkpoint.

### TensorBoard
```bash
tensorboard --logdir logs/
```
Custom metrics logged under `slither/`: `reward_mean`, `episode_length`, `peak_mass_mean`, `kills_mean`, `food_eaten_mean`, `death_wall_pct`, `death_collision_pct`, `survival_rate`, `boost_pct`, `wall_close_pct`.

---

## 2. Testing Locally (`test_rl.py`)

Play against your trained models with your mouse.

```bash
python3 test_rl.py
```

| Flag | Default | Description |
|---|---|---|
| `--rl <N>` | `6` | Number of golden RL agents from checkpoints |
| `--bots <N>` | `9` | Number of algorithmic scripted bots |

**Example — pure RL arena:**
```bash
python3 test_rl.py --rl 5 --bots 0
```

**Controls:** Mouse to steer, Left Click to boost, Escape to quit.

---

## 3. Web Deployment

### Export
```bash
python3 export_onnx.py          # produces slither_policy.onnx
python3 -m http.server 8080     # host it locally
```

### Browser
1. Install `slither_ai.user.js` into **Tampermonkey**.
2. Open [slither.io](http://slither.io) and spawn in.
3. Press **Q** to toggle AI control on/off.

---

## 4. Google Colab

```python
!unzip -o slither_rl.zip
!pip install stable-baselines3[extra] gymnasium pygame numpy torch
!python3 train.py --timesteps 5000000 --stage 1
```

Download the `checkpoints/` folder back to your machine to test locally.

---

## Architecture

- **CNN:** 5-channel 84×84 ego-centric mini-map (Self, Enemies, Food, Boundary, Velocity Streaks)
- **MLP:** 8-float proprioception vector (mass, turn rate, speed, boost state, wall distance, etc.)
- **Policy:** PPO with `[256, 128]` hidden layers → 2 continuous outputs (steering, boost)
- **Training:** 4× parallel `SubprocVecEnv`, auto-detects MPS/CUDA/CPU
