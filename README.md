# Slither.io Reinforcement Learning AI

A complete Gymnasium environment, Python physics engine, and Stable Baselines 3 training pipeline designed to learn how to play Slither.io from scratch using visual CNN inputs and Recurrent PPO (LSTM) memory.

---

## 1. Features 🌟

- **Recurrent PPO (LSTM):** Core architecture uses an LSTM hidden state (2048-step BPTT) to learn multi-step hunting strategies, coiling, and momentum.
- **Gladiator Reward Tuning:** Aggressive reward function that heavily incentivizes kills (5x food value) and makes boosting nearly free to encourage hunting, while remaining neutral on survival time to prevent passive foraging.
- **Modernized Vision:** 3x3 kernel CNN architecture applied to 5-channel egocentric observations with glowing (3px radius) food orbs to solve resolution-kernel mismatch.
- **Smart Scripted Bots:** 9 distinct algorithmic bot personalities (Bullies, Hunters, Foragers, Interceptors, etc.) that use spatial hashing to dodge bodies and fight intelligently.
- **Debug Training HUD:** Live PyGame overlay showing observation channel previews, steering magnitude, boost status, and episode stats.
- **Automatic Video Recording:** Periodically records a deterministic evaluation episode and pushes the video directly to TensorBoard.

---

## 2. Training the AI (`train.py`)

### Requirements
```bash
pip install -r requirements.txt
# (Includes sb3-contrib for LSTM support)
```

### Basic Training (LSTM)
```bash
python3 train.py
```

### All Flags
| Flag | Default | Description |
|---|---|---|
| `--stage <1\|2\|3>` | `3` | Curriculum stage (see below) |
| `--resume <path>` | — | Resume from `"latest"` or a specific checkpoint path |
| `--timesteps <N>` | `50,000,000` | Total training steps before auto-stop |
| `--num-envs <N>` | `4` | Parallel environment subprocesses |
| `--render` | off | Opens a live Pygame training HUD window (single env) |
| `--record-every <N>`| `500` | Record an eval episode to TensorBoard every N episodes (0=disabled) |
| `--no-lstm` | off | Fallback to standard feedforward PPO instead of RecurrentPPO |

### Curriculum Learning Stages
| Stage | Command | Arena | Goal |
|---|---|---|---|
| **1** | `python3 train.py --stage 1` | Agent + Food only | Learn movement, eating, wall avoidance |
| **2** | `python3 train.py --stage 2` | Agent + 20 scripted bots | Learn combat, maneuvering, hunting |
| **3** | `python3 train.py --stage 3` | Agent + 20 bots + 6 self-play clones | Develop robust, generalized strategies |

### Resuming Training
```bash
python3 train.py --resume latest --stage 2
```
Always picks `policy_final.zip` first (saved on Ctrl+C), then falls back to the highest numbered checkpoint. Auto-detects if the checkpoint is LSTM or standard PPO.

### TensorBoard
```bash
tensorboard --logdir logs/
```
- **Custom Metrics (`slither/`):** `reward_mean`, `episode_length`, `peak_mass_mean`, `kills_mean`, `food_eaten_mean`, `death_wall_pct`, `death_collision_pct`, `survival_rate`, `boost_pct`, `wall_close_pct`.
- **Gameplay Videos:** Check the **Images** tab in TensorBoard to watch the periodic `--record-every` eval episodes.

---

## 3. Testing Locally (`test_rl.py`)

Play against your trained models with your mouse. The script auto-detects if your checkpoints are LSTM or feedforward and manages hidden states automatically.

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

## 4. Web Deployment

### Export
```bash
python3 export_onnx.py          # produces slither_policy.onnx
python3 -m http.server 8080     # host it locally
```
*(Note: Export script may need updating for ONNX LSTM dynamic axes support depending on implementation).*

### Browser
1. Install `slither_ai.user.js` into **Tampermonkey**.
2. Open [slither.io](http://slither.io) and spawn in.
3. Press **Q** to toggle AI control on/off.

---

## 5. Architecture Details

- **Observation CNN:** 5-channel 84×84 ego-centric mini-map (Self, Enemies, Food, Boundary, Velocity Streaks) → processed by three 3x3 Conv2d layers.
- **Proprioception MLP:** 8-float vector (mass, turn rate, speed, boost state, wall distance, etc.).
- **Memory (LSTM):** 256-dim hidden state, `n_steps=2048`, tracking short-term maneuvers and persistent threats.
- **Policy:** RecurrentPPO (sb3-contrib) → 2 continuous outputs (steering [-1, 1], boost [0, 1]).
- **Hardware:** Auto-detects Apple Silicon (MPS), NVIDIA (CUDA), or CPU.
