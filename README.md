# Slither.io Reinforcement Learning AI

A complete Gymnasium environment, Python physics engine, and Stable Baselines 3 PPO training pipeline designed to learn how to play Slither.io from scratch using visual CNN inputs.

## 1. Training the AI (`train.py`)

The main training script uses a combination of Curriculum Learning and Self-Play to teach the agent.

### Basic Training
```bash
python3 train.py
```
*By default, this runs headlessly (no Pygame window) across 4 parallel environment processes for maximum speed.*

### Command Line Flags
| Flag | Example | Description |
|---|---|---|
| `--render` | `python3 train.py --render` | Opens a live Pygame window to watch the agent train in real-time. (Slows down training significantly). |
| `--resume` | `python3 train.py --resume latest` | Resumes training from the most recent checkpoint in `checkpoints/`. You can also pass a specific path like `--resume checkpoints/policy_1000.zip`. |
| `--stage <1|2|3>` | `python3 train.py --stage 1` | Sets the Curriculum Learning stage. (See below). Default is `3`. |
| `--num-envs` | `python3 train.py --num-envs 8` | Number of parallel Python subprocesses to spawn. Default is `4`. |
| `--timesteps` | `python3 train.py --timesteps 1000000` | Total number of RL steps before the script terminates. Default is `20,000,000`. |

### Curriculum Learning Stages
- **Stage 1 (Movement & Mechanics):** `python3 train.py --stage 1`
  - Spawns only the learning agent and food (0 bots, 0 self-play). 
  - Goal: Learn how to steer, use boost, and avoid the red boundary circle without being interrupted.
- **Stage 2 (Hunting):** `python3 train.py --stage 2`
  - Spawns the agent + 9 scripted algorithm bots.
  - Goal: Learn how to intercept slower, predictable targets and survive in a crowded environment.
- **Stage 3 (Mastery):** `python3 train.py --stage 3`
  - Spawns the agent + 9 scripted bots + 6 Self-Play clones based on previous checkpoints.
  - Goal: Prevent strategy collapse by forcing the agent to fight dynamic, unpredictable clones of itself that counter its own behaviors.

---

## 2. Testing the AI Locally (`test_rl.py`)

Once you have trained the agent and have `.zip` files in the `checkpoints/` directory, you can drop into the arena and play against your creation using your mouse.

```bash
python3 test_rl.py
```
*Note: Your snake is Light Blue. The RL Agents are Golden Yellow.*

### Command Line Flags
You can control exactly what types of enemies spawn using flags:
| Flag | Example | Description |
|---|---|---|
| `--rl` | `python3 test_rl.py --rl 5` | The number of Golden RL agents to spawn from your checkpoints. Default is `6`. |
| `--bots` | `python3 test_rl.py --bots 0` | The number of random algorithmic bots to spawn. Default is `9`. |

*(e.g., `python3 test_rl.py --rl 5 --bots 0` drops you into an arena with exactly 5 neural networks and no other distractions).*

---

## 3. Web Deployment (`export_onnx.py` & `slither_ai.user.js`)

You can export the trained PyTorch brain into the browser and let it play on the real internet servers.

### Step 1: Export the Checkpoint
Convert the `.zip` checkpoint into a static ONNX graph.
```bash
python3 export_onnx.py
```
*(This produces a `slither_policy.onnx` file).*

### Step 2: Host the File
Open a terminal in the folder containing the `.onnx` file and run:
```bash
python3 -m http.server 8080
```

### Step 3: Run the Tampermonkey Script
1. Install the contents of `slither_ai.user.js` into the Tampermonkey browser extension.
2. Go to [slither.io](http://slither.io).
3. Spawn into the map.
4. Press **`Q`** on your keyboard to toggle the AI on/off. It will automatically download the ONNX file from your local `8080` server and take over the mouse controls.

---

## 4. Google Colab / Cloud Training

To accelerate training drastically, run this headlessly on a cloud GPU (e.g. Google Colab T4).

1. Zip the necessary files on your Mac:
   ```bash
   zip -r slither_rl.zip bot_ai.py observation.py slither_gym.py spatial_hash.py train.py
   ```
2. Upload `slither_rl.zip` to your Colab notebook.
3. Run this block:
   ```python
   !unzip -o slither_rl.zip
   !pip install stable-baselines3[extra] gymnasium pygame numpy torch
   !python3 train.py --timesteps 5000000 --stage 1
   ```
4. Download the resulting `checkpoints/` folder back to your local machine to test!
