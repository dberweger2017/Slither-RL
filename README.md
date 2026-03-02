# Slither.io Reinforcement Learning AI

A complete Gymnasium environment, Python physics engine, and Stable Baselines 3 training pipeline designed to learn how to play Slither.io from scratch using visual CNN inputs and Recurrent PPO (LSTM) memory.

---

## 1. Features 🌟

- **Recurrent PPO (LSTM):** Core architecture uses an LSTM hidden state (2048-step BPTT) to learn multi-step hunting strategies, coiling, and momentum.
- **Gladiator Reward Tuning:** Aggressive reward function that heavily incentivizes kills (5x food value) and makes boosting nearly free to encourage hunting, while remaining neutral on survival time to prevent passive foraging.
- **High-Resolution Vision:** 168×168 ego-centric 5-channel CNN observation map with `VIEW_RADIUS=500`, ensuring even single food pellets are visible to the agent (≥1px per orb).
- **4-Layer CNN Extractor:** Conv2d pipeline (32→64→128 channels) with stride-2 downsampling, purpose-built for the 168×168 input resolution.
- **Smart Scripted Bots:** 9 distinct algorithmic bot personalities (Bullies, Hunters, Foragers, Interceptors, Scavengers, etc.) that use spatial hashing to dodge bodies and fight intelligently.
- **Debug Vision Mode:** Press keys 1–6 during gameplay to cycle through the agent's raw CNN observation channels fullscreen for visual debugging.
- **Headless Bot Tournament:** `--simulate` flag runs a 60-second fast-forward match with 2 of each bot type and prints a ranked leaderboard.
- **Debug Training HUD:** Live PyGame overlay showing observation channel previews, steering magnitude, boost status, and episode stats.
- **Automatic Video Recording:** Periodically records a deterministic evaluation episode and pushes the video directly to TensorBoard.

---

## 2. Training the AI (`train.py`)

### Requirements
```bash
pip install stable-baselines3[extra] sb3-contrib gymnasium pygame numpy torch moviepy tensorboard tqdm rich
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
| `--record-every <N>`| `100,000` | Record an eval episode to TensorBoard every N timesteps (0=disabled) |
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
Always picks `policy_final.zip` first (saved on Ctrl+C), then falls back to the highest numbered checkpoint. Auto-detects if the checkpoint is LSTM or standard PPO. Incompatible checkpoints (e.g. trained with old 84×84 observations) are automatically rejected with an informative error.

### Behavior Cloning Warm Start
Use this when you want to initialize RL from a scripted bot policy.

1. Collect expert data from a bot:
```bash
python3 collect_expert_data.py \
  --bot-type interceptor \
  --frames 100000 \
  --output-dir datasets/expert_interceptor \
  --overwrite
```

2. Supervised pretraining (MSE on continuous actions):
```bash
python3 pretrain_bc.py \
  --dataset datasets/expert_interceptor \
  --epochs 10 \
  --batch-size 256 \
  --output checkpoints/policy_interceptor_bc_init
```

3. Resume RL from the BC checkpoint:
```bash
python3 train.py --resume checkpoints/policy_interceptor_bc_init.zip --stage 2
```

Notes:
- Collection and training use the same `168x168` map observation.
- The collector writes chunked compressed `.npz` files + `metadata.json` to handle large datasets.
- You can switch the expert style with `--bot-type` (`interceptor`, `hunter`, `scavenger`, etc.).

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

### All Flags
| Flag | Default | Description |
|---|---|---|
| `--rl <N>` | `6` | Number of golden RL agents from checkpoints |
| `--bots <N>` | `9` | Number of algorithmic scripted bots |
| `--bot-type <name>` | mixed | Force all bots to a specific personality (e.g. `scavenger`, `hunter`) |
| `--simulate` | off | Run a 60s headless bot tournament (no UI, no player) |
| `--debug-vision` | off | Enable CNN channel debug overlay |

### Examples

**Play against 20 scavengers:**
```bash
python3 test_rl.py --bots 20 --rl 0 --bot-type scavenger
```

**Pure RL arena (no scripted bots):**
```bash
python3 test_rl.py --rl 5 --bots 0
```

**Headless bot tournament:**
```bash
python3 test_rl.py --simulate --rl 0
```
Spawns 2 of each bot type, simulates 3600 frames (60 seconds at 60 FPS), and prints a ranked leaderboard by peak mass and kills.

### Controls
| Key | Action |
|---|---|
| Mouse | Steer |
| Left Click | Boost |
| 1 | Normal view (default) |
| 2–6 | Fullscreen CNN channels: Self, Enemy, Food, Boundary, Velocity |
| Escape | Quit |

---

## 4. Observation Space

The agent perceives the world through a **168×168 ego-centric rotated mini-map** with 5 channels:

| Channel | Color (debug) | Contents |
|---|---|---|
| 0 — Self | Cyan | Agent's own head and body segments (fading toward tail) |
| 1 — Enemies | Red | All enemy snake bodies (≥30% brightness) with bright heads (100%) |
| 2 — Food | Green | Food pellets (small orbs ≥1px, death drops 2–3px) |
| 3 — Boundary | Yellow | World edge proximity gradient (fades in within 300 units) |
| 4 — Velocity | Purple | Enemy heading streaks (lines showing movement direction and speed) |

Plus an **8-float proprioception vector**: mass, turn rate, speed, boost state, can-boost, wall distance, body length, kill count.

### Action Space

Continuous `Box[-1,1] × Box[0,1]`:
- **Steering** `[-1, 1]`: Relative turn (left/right) from current heading
- **Boost** `[0, 1]`: >0.5 activates boost

---

## 5. Web Deployment

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

## 6. Architecture Details

- **Observation CNN:** 5-channel 168×168 ego-centric mini-map (Self, Enemies, Food, Boundary, Velocity Streaks) → processed by four Conv2d layers (32→64→64→128 channels, 3×3 kernels, stride 2).
- **Proprioception MLP:** 8-float vector (mass, turn rate, speed, boost state, wall distance, etc.).
- **Memory (LSTM):** 256-dim hidden state, `n_steps=2048`, tracking short-term maneuvers and persistent threats.
- **Policy:** RecurrentPPO (sb3-contrib) → 2 continuous outputs (steering [-1, 1], boost [0, 1]).
- **Hardware:** Auto-detects Apple Silicon (MPS), NVIDIA (CUDA), or CPU.

---

## 7. Bot Personalities

| Bot | Strategy | Color |
|---|---|---|
| Random | Wanders, grabs nearby food | Gray |
| Forager | Efficient food collector, flees threats | Green |
| Bully | Cuts off paths with lead targeting | Red |
| Scavenger | Hunts death drops, gravitates toward fights | Orange |
| Patrol | Sweeps world edges, eats along the way | Blue |
| Parasite | Shadows biggest snake's tail | Magenta |
| Trapper | Circles smaller snakes when big enough | Dark Red |
| Interceptor | Adaptive lookahead, aborts if body-blocked | Cyan |
| Hunter | Pursues snakes ≤½ its size with heading prediction | Yellow |
