"""
PPO Training Script for Slither.io with Fictitious Self-Play.

Architecture:
- 1 training agent vs 9 scripted bots + 6 self-play policy agents
- Checkpoints saved every N episodes
- Self-play opponents loaded with recency-weighted sampling

Usage:
    python train.py                  # Train from scratch
    python train.py --resume latest  # Resume from latest checkpoint
"""
import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

from slither_gym import SlitherEnv

# ─────────────────────────── Config ───────────────────────────

CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
SAVE_EVERY_EPISODES = 10_000
TOTAL_TIMESTEPS = 10_000_000
MAX_CHECKPOINTS = 50  # Keep at most this many checkpoint files


# ─────────────────── Custom Feature Extractor ─────────────────

class SlitherFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom CNN+MLP feature extractor for Dict observation.
    
    Processes:
      'map'   (5, 84, 84)  → Nature CNN → 512-dim
      'state' (8,)          → MLP → 64-dim
    Concatenated → 576-dim output
    """
    
    def __init__(self, observation_space, features_dim=576):
        super().__init__(observation_space, features_dim)
        
        n_channels = observation_space['map'].shape[0]  # 5
        
        # Nature CNN (same architecture as DeepMind Atari)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute CNN output size
        with torch.no_grad():
            sample = torch.zeros(1, n_channels, 84, 84)
            cnn_out_dim = self.cnn(sample).shape[1]
        
        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_out_dim, 512),
            nn.ReLU(),
        )
        
        # MLP for proprioception
        state_dim = observation_space['state'].shape[0]  # 8
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
        )
        
        # Final features dim = 512 + 64 = 576
        self._features_dim = 576
    
    def forward(self, observations):
        cnn_features = self.cnn_fc(self.cnn(observations['map']))
        state_features = self.state_mlp(observations['state'])
        return torch.cat([cnn_features, state_features], dim=1)


# ────────────────── Checkpoint Manager ──────────────────

class CheckpointManager:
    """Manages saving/loading policy checkpoints for self-play."""
    
    def __init__(self, checkpoint_dir=CHECKPOINT_DIR, max_checkpoints=MAX_CHECKPOINTS):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(self, model, episode_count):
        """Save current policy as a checkpoint."""
        path = os.path.join(self.checkpoint_dir, f"policy_{episode_count:08d}")
        model.save(path)
        print(f"  💾 Saved checkpoint: policy_{episode_count:08d}")
        self._prune_old()
    
    def _prune_old(self):
        """Remove oldest checkpoints if we exceed max_checkpoints."""
        files = sorted(self._list_checkpoints())
        while len(files) > self.max_checkpoints:
            oldest = files.pop(0)
            os.remove(os.path.join(self.checkpoint_dir, oldest))
            print(f"  🗑️  Pruned old checkpoint: {oldest}")
    
    def _list_checkpoints(self):
        """List checkpoint files sorted by episode number."""
        files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("policy_")]
        return sorted(files)
    
    def load_opponents(self, env, n=6):
        """
        Load n opponents with recency-weighted sampling.
        
        Newer checkpoints have higher probability of being selected.
        Uses exponential weighting: weight_i = 2^(i / total)
        """
        checkpoints = self._list_checkpoints()
        
        if not checkpoints:
            print("  ⚡ No checkpoints yet — self-play agents use random actions")
            env.set_selfplay_policies([])
            return
        
        # Recency-weighted sampling
        n_ckpts = len(checkpoints)
        weights = np.array([2.0 ** (i / max(n_ckpts, 1)) for i in range(n_ckpts)])
        weights /= weights.sum()
        
        # Sample with replacement (might pick same checkpoint twice)
        chosen_indices = np.random.choice(n_ckpts, size=min(n, n_ckpts), 
                                          replace=True, p=weights)
        
        policies = []
        for idx in chosen_indices:
            path = os.path.join(self.checkpoint_dir, checkpoints[idx])
            try:
                model = PPO.load(path, device='cpu')
                policies.append(model)
            except Exception as e:
                print(f"  ⚠️  Failed to load {checkpoints[idx]}: {e}")
        
        env.set_selfplay_policies(policies)
        names = [checkpoints[i] for i in chosen_indices]
        print(f"  🎯 Loaded {len(policies)} self-play opponents: {', '.join(names)}")


# ────────────────── Training Callback ──────────────────

class SelfPlayCallback(BaseCallback):
    """
    Callback that:
    1. Tracks episode count and rich per-episode metrics
    2. Logs custom metrics to tensorboard
    3. Saves checkpoints every SAVE_EVERY_EPISODES episodes
    4. Reloads self-play opponents after each save
    """
    
    def __init__(self, checkpoint_manager, env, save_every=SAVE_EVERY_EPISODES, verbose=1):
        super().__init__(verbose)
        self.ckpt_mgr = checkpoint_manager
        self.env = env
        self.save_every = save_every
        self.episode_count = 0
        self.last_save_episode = 0
        self.start_time = None
        
        # Rolling buffers for metrics
        self.ep_rewards = []
        self.ep_masses = []
        self.ep_peak_masses = []
        self.ep_kills = []
        self.ep_food_eaten = []
        self.ep_mass_per_frame = []
        self.ep_boost_pct = []
        self.ep_wall_close_pct = []
        self.ep_lengths = []
        self.death_causes = {'collision': 0, 'wall': 0, 'survived': 0}
    
    def _on_training_start(self):
        self.start_time = time.time()
        self.ckpt_mgr.load_opponents(self.env, n=6)
    
    def _on_step(self):
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.episode_count += 1
                ep_r = info['episode']['r']
                ep_l = info['episode']['l']
                
                # Collect metrics
                self.ep_rewards.append(ep_r)
                self.ep_lengths.append(ep_l)
                self.ep_masses.append(info.get('mass', 0))
                self.ep_peak_masses.append(info.get('peak_mass', 0))
                self.ep_kills.append(info.get('kills', 0))
                self.ep_food_eaten.append(info.get('food_eaten', 0))
                self.ep_mass_per_frame.append(info.get('mass_per_frame', 0))
                self.ep_boost_pct.append(info.get('boost_pct', 0))
                self.ep_wall_close_pct.append(info.get('wall_close_pct', 0))
                
                dc = info.get('death_cause', 'collision')
                if dc in self.death_causes:
                    self.death_causes[dc] += 1
                
                # Log to tensorboard every 10 episodes
                if self.episode_count % 10 == 0 and self.logger:
                    n = min(100, len(self.ep_rewards))
                    
                    # Core performance
                    self.logger.record("slither/reward_mean", np.mean(self.ep_rewards[-n:]))
                    self.logger.record("slither/episode_length", np.mean(self.ep_lengths[-n:]))
                    
                    # Mass metrics
                    self.logger.record("slither/final_mass_mean", np.mean(self.ep_masses[-n:]))
                    self.logger.record("slither/peak_mass_mean", np.mean(self.ep_peak_masses[-n:]))
                    self.logger.record("slither/peak_mass_max", np.max(self.ep_peak_masses[-n:]))
                    self.logger.record("slither/mass_per_frame", np.mean(self.ep_mass_per_frame[-n:]))
                    
                    # Combat
                    self.logger.record("slither/kills_mean", np.mean(self.ep_kills[-n:]))
                    self.logger.record("slither/kills_total", np.sum(self.ep_kills[-n:]))
                    self.logger.record("slither/food_eaten_mean", np.mean(self.ep_food_eaten[-n:]))
                    
                    # Behavior
                    self.logger.record("slither/boost_pct", np.mean(self.ep_boost_pct[-n:]))
                    self.logger.record("slither/wall_close_pct", np.mean(self.ep_wall_close_pct[-n:]))
                    
                    # Death causes (over last n episodes)
                    total_dc = sum(self.death_causes.values()) or 1
                    self.logger.record("slither/death_collision_pct", 
                                      self.death_causes['collision'] / total_dc)
                    self.logger.record("slither/death_wall_pct",
                                      self.death_causes['wall'] / total_dc)
                    self.logger.record("slither/survival_rate",
                                      self.death_causes['survived'] / total_dc)
                    
                    self.logger.record("slither/episodes", self.episode_count)
                
                # Console log every 100 episodes
                if self.episode_count % 100 == 0:
                    n = min(100, len(self.ep_rewards))
                    elapsed = time.time() - self.start_time
                    eps_per_sec = self.episode_count / elapsed if elapsed > 0 else 0
                    
                    print(f"\n📊 Episode {self.episode_count:,} "
                          f"({self.num_timesteps:,} steps, {elapsed/60:.1f}min, "
                          f"{eps_per_sec:.1f} ep/s)")
                    print(f"   Reward: {np.mean(self.ep_rewards[-n:]):+.2f} | "
                          f"Mass: {np.mean(self.ep_masses[-n:]):.0f} avg, "
                          f"{np.max(self.ep_peak_masses[-n:]):.0f} peak")
                    print(f"   Kills: {np.mean(self.ep_kills[-n:]):.2f}/ep | "
                          f"Food: {np.mean(self.ep_food_eaten[-n:]):.0f}/ep | "
                          f"Boost: {np.mean(self.ep_boost_pct[-n:])*100:.0f}%")
                    print(f"   Growth: {np.mean(self.ep_mass_per_frame[-n:]):.3f} mass/frame | "
                          f"Deaths: collision={self.death_causes['collision']}, "
                          f"wall={self.death_causes['wall']}, "
                          f"survived={self.death_causes['survived']}")
                
                # Checkpoint
                if self.episode_count - self.last_save_episode >= self.save_every:
                    print(f"\n{'='*60}")
                    print(f"🏁 Checkpoint at episode {self.episode_count:,}")
                    self.ckpt_mgr.save(self.model, self.episode_count)
                    self.ckpt_mgr.load_opponents(self.env, n=6)
                    self.last_save_episode = self.episode_count
                    # Reset death cause counters for next period
                    self.death_causes = {'collision': 0, 'wall': 0, 'survived': 0}
                    print(f"{'='*60}\n")
        
        return True


# ────────────────── Main Training ──────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Slither.io PPO Agent")
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint ("latest" or path)')
    parser.add_argument('--timesteps', type=int, default=TOTAL_TIMESTEPS,
                       help=f'Total training timesteps (default: {TOTAL_TIMESTEPS:,})')
    parser.add_argument('--render', action='store_true',
                       help='Enable rendering during training (slow!)')
    args = parser.parse_args()
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Create environment
    render_mode = 'human' if args.render else None
    env = SlitherEnv(num_scripted=9, num_selfplay=6, render_mode=render_mode)
    
    # Checkpoint manager
    ckpt_mgr = CheckpointManager()
    
    # Create or load model
    if args.resume:
        if args.resume == 'latest':
            checkpoints = ckpt_mgr._list_checkpoints()
            if checkpoints:
                path = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
                print(f"📂 Resuming from: {checkpoints[-1]}")
            else:
                print("⚠️  No checkpoints found, starting fresh")
                path = None
        else:
            path = args.resume
        
        if path:
            model = PPO.load(path, env=env, device='auto',
                           tensorboard_log=LOG_DIR)
            print(f"✅ Model loaded successfully")
        else:
            model = None
    else:
        model = None
    
    if model is None:
        # Fresh model
        policy_kwargs = {
            'features_extractor_class': SlitherFeatureExtractor,
            'features_extractor_kwargs': {'features_dim': 576},
            'net_arch': dict(pi=[256, 128], vf=[256, 128]),
        }
        
        model = PPO(
            'MultiInputPolicy',
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,         # Encourage exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=LOG_DIR,
            device='auto',
        )
        print("🆕 Created fresh PPO model")
    
    # Print model info
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"🧠 Model parameters: {total_params:,}")
    print(f"🎮 Environment: 1 agent + 9 scripted + 6 self-play = 16 snakes")
    print(f"📺 Render: {'ON' if args.render else 'OFF'}")
    print(f"🎯 Training for {args.timesteps:,} timesteps\n")
    
    # Create callback
    callback = SelfPlayCallback(
        checkpoint_manager=ckpt_mgr,
        env=env,
        save_every=SAVE_EVERY_EPISODES,
    )
    
    # Train!
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")
    finally:
        # Save final model
        final_path = os.path.join(CHECKPOINT_DIR, "policy_final")
        model.save(final_path)
        print(f"💾 Saved final model: {final_path}")
        env.close()
    
    print(f"\n✅ Training complete!")
    print(f"   Total episodes: {callback.episode_count:,}")
    print(f"   Total timesteps: {callback.num_timesteps:,}")
    if callback.episode_rewards:
        print(f"   Final avg reward: {np.mean(callback.episode_rewards[-100:]):+.2f}")


if __name__ == '__main__':
    main()
