"""
PPO Training Script for Slither.io with Fictitious Self-Play.
Supports both standard PPO and Recurrent PPO (LSTM).

Usage:
    python train.py                          # Train LSTM from scratch
    python train.py --resume latest          # Resume from latest checkpoint
    python train.py --no-lstm               # Use standard PPO instead
"""
import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

from slither_gym import SlitherEnv

CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
SAVE_EVERY_EPISODES = 200
TOTAL_TIMESTEPS = 50_000_000
MAX_CHECKPOINTS = 50


class SlitherFeatureExtractor(BaseFeaturesExtractor):
    """Refined 3x3 CNN for 5-channel map + MLP for 8-float proprioception → 576-dim."""

    def __init__(self, observation_space, features_dim=576):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space['map'].shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),  # 168 -> 84
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),          # 84 -> 42
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),          # 42 -> 21
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),         # 21 -> 11
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out_dim = self.cnn(torch.zeros(1, n_channels, 168, 168)).shape[1]

        self.cnn_fc = nn.Sequential(nn.Linear(cnn_out_dim, 512), nn.ReLU())

        state_dim = observation_space['state'].shape[0]
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU())
        self._features_dim = 576

    def forward(self, observations):
        cnn_features = self.cnn_fc(self.cnn(observations['map']))
        state_features = self.state_mlp(observations['state'])
        return torch.cat([cnn_features, state_features], dim=1)


class CheckpointManager:
    def __init__(self, checkpoint_dir=CHECKPOINT_DIR, max_checkpoints=MAX_CHECKPOINTS):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, model, episode_count):
        path = os.path.join(self.checkpoint_dir, f"policy_{episode_count:08d}")
        model.save(path)
        print(f"  💾 Saved checkpoint: policy_{episode_count:08d}")
        self._prune_old()

    def _prune_old(self):
        files = sorted(self._list_checkpoints())
        while len(files) > self.max_checkpoints:
            oldest = files.pop(0)
            os.remove(os.path.join(self.checkpoint_dir, oldest))

    def _list_checkpoints(self):
        if not os.path.exists(self.checkpoint_dir):
            return []
        files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("policy_")]
        return sorted(files)


class SelfPlayCallback(BaseCallback):
    def __init__(self, checkpoint_manager, env, save_every=SAVE_EVERY_EPISODES,
                 record_every=100000, stage=3, use_lstm=True,
                 save_videos_local=False, verbose=1):
        super().__init__(verbose)
        self.ckpt_mgr = checkpoint_manager
        self.env = env
        self.save_every = save_every
        self.record_every = record_every
        self.stage = stage
        self.use_lstm = use_lstm
        self.save_videos_local = save_videos_local
        self.episode_count = 0
        self.last_save_episode = 0
        self.last_record_episode = 0
        self.start_time = None

        self.ep_rewards = []
        self.ep_masses = []
        self.ep_peak_masses = []
        self.ep_kills = []
        self.ep_food_eaten = []
        self.ep_mass_per_frame = []
        self.ep_boost_pct = []
        self.ep_wall_close_pct = []
        self.ep_lengths = []
        self.ep_time_to_100 = []
        self.ep_safe_space = []
        self.death_causes = {'collision': 0, 'wall': 0, 'survived': 0}

        # Reward breakdown tracking
        self.ep_time_penalty = []
        self.ep_food_reward = []
        self.ep_boost_penalty = []
        self.ep_kill_reward = []
        self.ep_proximity_reward = []
        self.ep_loot_bonus_reward = []

    def _on_training_start(self):
        self.start_time = time.time()
        self.env.env_method('load_selfplay_from_dir', CHECKPOINT_DIR, 6)
        if self.record_every > 0:
            print(f"🎥 Video recording enabled: every {self.record_every:,} timesteps")
            if self.save_videos_local:
                print("💽 Local eval video saving: ON (logs/eval_videos/)")
            else:
                print("💽 Local eval video saving: OFF")
        else:
            print("🎥 Video recording disabled (--record-every 0)")

    def _on_step(self):
        infos = self.locals.get('infos', [])
        for info in infos:
            ep_info = info.get('episode', info)

            if not ('terminal_observation' in info or 'episode' in info):
                continue

            self.episode_count += 1
            if 'r' in ep_info:
                self.ep_rewards.append(ep_info['r'])
            if 'l' in ep_info:
                self.ep_lengths.append(ep_info['l'])

            self.ep_masses.append(info.get('mass', 0))
            self.ep_peak_masses.append(info.get('peak_mass', 0))
            self.ep_kills.append(info.get('kills', 0))
            self.ep_food_eaten.append(info.get('food_eaten', 0))
            self.ep_mass_per_frame.append(info.get('mass_per_frame', 0))
            self.ep_boost_pct.append(info.get('boost_pct', 0))
            self.ep_wall_close_pct.append(info.get('wall_close_pct', 0))
            self.ep_safe_space.append(info.get('safe_space_pct', 0))
            if info.get('time_to_100_mass') is not None:
                self.ep_time_to_100.append(info.get('time_to_100_mass'))

            # Store reward breakdown
            self.ep_time_penalty.append(info.get('time_penalty', 0))
            self.ep_food_reward.append(info.get('food_reward', 0))
            self.ep_boost_penalty.append(info.get('boost_penalty', 0))
            self.ep_kill_reward.append(info.get('kill_reward', 0))
            self.ep_proximity_reward.append(info.get('proximity_reward', 0))
            self.ep_loot_bonus_reward.append(info.get('loot_bonus_reward', 0))

            dc = info.get('death_cause', 'collision')
            if dc in self.death_causes:
                self.death_causes[dc] += 1

            # Tensorboard logging every 10 episodes
            if self.episode_count % 10 == 0 and self.logger:
                n = min(100, len(self.ep_rewards))
                self.logger.record("slither/reward_mean", np.mean(self.ep_rewards[-n:]))
                self.logger.record("slither/episode_length", np.mean(self.ep_lengths[-n:]))
                self.logger.record("slither/final_mass_mean", np.mean(self.ep_masses[-n:]))
                self.logger.record("slither/peak_mass_mean", np.mean(self.ep_peak_masses[-n:]))
                self.logger.record("slither/peak_mass_max", np.max(self.ep_peak_masses[-n:]))
                self.logger.record("slither/mass_per_frame", np.mean(self.ep_mass_per_frame[-n:]))
                self.logger.record("slither/kills_mean", np.mean(self.ep_kills[-n:]))
                self.logger.record("slither/kills_total", np.sum(self.ep_kills[-n:]))
                self.logger.record("slither/food_eaten_mean", np.mean(self.ep_food_eaten[-n:]))
                self.logger.record("slither/boost_pct", np.mean(self.ep_boost_pct[-n:]))
                self.logger.record("slither/loot_bonus_reward", np.mean(self.ep_loot_bonus_reward[-n:]))
                self.logger.record("slither/wall_close_pct", np.mean(self.ep_wall_close_pct[-n:]))
                self.logger.record("slither/safe_space_pct", np.mean(self.ep_safe_space[-n:]))
                if self.ep_time_to_100:
                    n_time = min(100, len(self.ep_time_to_100))
                    self.logger.record("slither/time_to_100_mass", np.mean(self.ep_time_to_100[-n_time:]))
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

                arch = "LSTM" if self.use_lstm else "FFN"
                print(f"\n📊 Episode {self.episode_count:,} [{arch}] "
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
                
                # Reward breakdown print
                avg_food = np.mean(self.ep_food_reward[-n:])
                avg_kill = np.mean(self.ep_kill_reward[-n:])
                avg_prox = np.mean(self.ep_proximity_reward[-n:])
                avg_loot = np.mean(self.ep_loot_bonus_reward[-n:])
                avg_time_pen = np.mean(self.ep_time_penalty[-n:])
                avg_boost_pen = np.mean(self.ep_boost_penalty[-n:])
                
                print(f"   Reward breakdown: Food: {avg_food:+.2f} | Loot: {avg_loot:+.2f} | Kills: {avg_kill:+.2f} | Prox: {avg_prox:+.2f}")
                print(f"   Penalty breakdown: Death/Time: {avg_time_pen:+.2f} | Boost: {avg_boost_pen:+.2f}")

                avg_time_100 = np.mean(self.ep_time_to_100[-100:]) if self.ep_time_to_100 else 0
                print(f"   Metrics: Safe Space: {np.mean(self.ep_safe_space[-n:])*100:.1f}% | Time to 100 Mass: {avg_time_100:.0f} steps")

                # Progress to next video
                if self.record_every > 0:
                    steps_since_record = self.num_timesteps - self.last_record_episode
                    progress_pct = (steps_since_record / self.record_every) * 100
                    print(f"   Next Video Progress: {progress_pct:.1f}% ({steps_since_record:,}/{self.record_every:,} steps)")

            # Checkpoint
            if self.episode_count - self.last_save_episode >= self.save_every:
                print(f"\n{'='*60}")
                print(f"🏁 Checkpoint at episode {self.episode_count:,}")
                self.ckpt_mgr.save(self.model, self.episode_count)

                base_env = self.training_env
                if hasattr(base_env, 'venv'):
                    base_env = base_env.venv

                base_env.env_method('load_selfplay_from_dir', CHECKPOINT_DIR, 6)
                self.last_save_episode = self.episode_count
                self.death_causes = {'collision': 0, 'wall': 0, 'survived': 0}
                print(f"{'='*60}\n")

        # Video recording every N timesteps (independent of episode endings)
        if (self.record_every > 0 and
                self.num_timesteps - self.last_record_episode >= self.record_every):
            print(f"\n🎥 Triggering eval video at {self.num_timesteps:,} timesteps")
            self._record_eval_episode()
            # Keep cadence stable even when recording fails
            self.last_record_episode = self.num_timesteps

        return True

    def _record_eval_episode(self):
        """Run one evaluation episode, capture frames, push video to TensorBoard."""
        try:
            num_scripted = 20 if self.stage >= 2 else 0
            eval_env = SlitherEnv(num_scripted=num_scripted, num_selfplay=0, max_steps=1500)
            obs, _ = eval_env.reset()

            frames = []
            max_frames = 600
            done = False

            # LSTM needs hidden state tracking
            lstm_states = None
            episode_start = np.ones((1,), dtype=bool)

            while not done and len(frames) < max_frames:
                if self.use_lstm:
                    action, lstm_states = self.model.predict(
                        obs, state=lstm_states,
                        episode_start=episode_start,
                        deterministic=True)
                    episode_start = np.zeros((1,), dtype=bool)
                else:
                    action, _ = self.model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated

                if len(frames) * 3 <= eval_env.step_count:
                    frame = eval_env.render_to_array(width=400, height=300)
                    frames.append(frame)

            eval_env.close()

            if len(frames) < 10:
                print(f"  ⚠️  Skipped video write: only {len(frames)} frames collected")
                return

            import torch
            video_tensor = torch.from_numpy(
                np.stack(frames)).permute(0, 3, 1, 2).unsqueeze(0)

            if self.save_videos_local:
                self._save_eval_video_local(frames)

            writer = None
            if hasattr(self.logger, 'output_formats'):
                from stable_baselines3.common.logger import TensorBoardOutputFormat
                for fmt in self.logger.output_formats:
                    if isinstance(fmt, TensorBoardOutputFormat):
                        writer = fmt.writer
                        break

            if writer is not None:
                writer.add_video(
                    'slither/gameplay', video_tensor,
                    global_step=self.num_timesteps, fps=20)
                writer.flush()
                print(f"  🎬 Recorded eval episode: {len(frames)} frames, "
                      f"mass={info.get('mass', 0):.0f}, "
                      f"kills={info.get('kills', 0)}, "
                      f"food={info.get('food_eaten', 0)}")
            else:
                print(f"  ⚠️  Could not find TensorBoard writer for video logging")

        except Exception as e:
            print(f"  ⚠️  Video recording failed: {e!r}")

    def _save_eval_video_local(self, frames):
        """Optional disk save for debugging when TensorBoard media doesn't show."""
        out_dir = os.path.join(LOG_DIR, "eval_videos")
        os.makedirs(out_dir, exist_ok=True)
        stem = f"eval_t{self.num_timesteps:09d}_ep{self.episode_count:07d}"
        mp4_path = os.path.join(out_dir, f"{stem}.mp4")
        try:
            import imageio.v2 as imageio
            with imageio.get_writer(mp4_path, fps=20) as writer:
                for frame in frames:
                    writer.append_data(frame)
            print(f"  💽 Saved eval video file: {mp4_path}")
            return
        except Exception as e:
            print(f"  ⚠️  Could not save mp4 video: {e!r}")

        gif_path = os.path.join(out_dir, f"{stem}.gif")
        try:
            import imageio.v2 as imageio
            imageio.mimsave(gif_path, frames, format='GIF', fps=20)
            print(f"  💽 Saved eval gif file: {gif_path}")
        except Exception as e:
            print(f"  ⚠️  Could not save gif video: {e!r}")


def main():
    parser = argparse.ArgumentParser(description="Train Slither.io PPO Agent")
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint ("latest" or path)')
    parser.add_argument('--timesteps', type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num-envs', type=int, default=4, help='Number of parallel envs')
    parser.add_argument('--stage', type=int, default=3, choices=[1, 2, 3],
                       help='Training stage (1: food only, 2: bots, 3: full self-play)')
    parser.add_argument('--record-every', type=int, default=100000,
                       help='Record an eval episode every N timesteps (0=disabled)')
    parser.add_argument('--save-videos-local', action='store_true',
                       help='Also save eval videos to logs/eval_videos for debugging')
    parser.add_argument('--no-lstm', action='store_true',
                       help='Use standard PPO instead of RecurrentPPO (LSTM)')
    args = parser.parse_args()

    use_lstm = not args.no_lstm

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = 'auto'
    if torch.backends.mps.is_available():
        device = 'mps'
        print("🍏 Using Apple Silicon MPS for acceleration")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("🚀 Using NVIDIA CUDA GPU for acceleration")
    else:
        print("🐌 Using CPU")

    def make_env(rank):
        def _init():
            render_mode = 'human' if (args.render and rank == 0) else None
            num_scripted = 10 if args.stage >= 2 else 0
            num_selfplay = 6 if args.stage == 3 else 0
            return SlitherEnv(num_scripted=num_scripted, num_selfplay=num_selfplay, render_mode=render_mode)
        return _init

    if args.render or args.num_envs == 1:
        env = DummyVecEnv([make_env(0)])
    else:
        env = SubprocVecEnv([make_env(i) for i in range(args.num_envs)])

    env = VecMonitor(env)

    ckpt_mgr = CheckpointManager()

    PPO_class = RecurrentPPO if use_lstm else PPO
    policy_name = 'MultiInputLstmPolicy' if use_lstm else 'MultiInputPolicy'
    arch_name = "RecurrentPPO (LSTM)" if use_lstm else "PPO (Feedforward)"

    custom_objects = {
        "learning_rate": 1e-4,
        "clip_range": 0.2,
    }

    model = None
    if args.resume:
        if args.resume == 'latest':
            if os.path.exists(os.path.join(CHECKPOINT_DIR, "policy_final.zip")):
                path = os.path.join(CHECKPOINT_DIR, "policy_final.zip")
                print(f"📂 Resuming from final save: policy_final.zip")
                model = PPO_class.load(path, env=env, custom_objects=custom_objects,
                                       device=device, tensorboard_log=LOG_DIR)
                expected_shape = model.observation_space.spaces['map'].shape
                if expected_shape != (5, 168, 168):
                    print(f"Error: Cannot resume training from {path}.")
                    print(f"       Incompatible observation shape {expected_shape} (expected (5, 168, 168)).")
                    print("       Please start a new training run without --resume, or clear checkpoints.")
                    import sys
                    sys.exit(1)
            else:
                checkpoints = ckpt_mgr._list_checkpoints()
                if checkpoints:
                    path = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
                    print(f"📂 Resuming from: {checkpoints[-1]}")
                    model = PPO_class.load(path, env=env, custom_objects=custom_objects,
                                           device=device, tensorboard_log=LOG_DIR)
                    expected_shape = model.observation_space.spaces['map'].shape
                    if expected_shape != (5, 168, 168):
                        print(f"Error: Cannot resume training from {path}.")
                        print(f"       Incompatible observation shape {expected_shape} (expected (5, 168, 168)).")
                        print("       Please start a new training run without --resume, or clear checkpoints.")
                        import sys
                        sys.exit(1)
                else:
                    print("⚠️  No checkpoints found, starting fresh")
        else:
            model = PPO_class.load(args.resume, env=env, custom_objects=custom_objects,
                                   device=device, tensorboard_log=LOG_DIR)
            expected_shape = model.observation_space.spaces['map'].shape
            if expected_shape != (5, 168, 168):
                print(f"Error: Cannot resume training from {args.resume}.")
                print(f"       Incompatible observation shape {expected_shape} (expected (5, 168, 168)).")
                print("       Please start a new training run without --resume, or clear checkpoints.")
                import sys
                sys.exit(1)

    if model is None:
        policy_kwargs = {
            'features_extractor_class': SlitherFeatureExtractor,
            'features_extractor_kwargs': {'features_dim': 576},
            'net_arch': dict(pi=[256, 128], vf=[256, 128]),
        }
        if use_lstm:
            policy_kwargs['lstm_hidden_size'] = 256
            policy_kwargs['n_lstm_layers'] = 1
            policy_kwargs['share_features_extractor'] = True

        # LSTM: longer unroll for temporal context, larger batch for GPU throughput
        if use_lstm:
            model = RecurrentPPO(
                policy_name, env,
                policy_kwargs=policy_kwargs,
                learning_rate=1e-4,
                n_steps=2048,
                batch_size=1024,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=1,
                tensorboard_log=LOG_DIR,
                device=device,
            )
        else:
            model = PPO(
                policy_name, env,
                policy_kwargs=policy_kwargs,
                learning_rate=1e-4,
                target_kl=0.015,
                n_steps=8192,
                batch_size=1024,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=1,
                tensorboard_log=LOG_DIR,
                device=device,
            )
        print(f"🆕 Created fresh {arch_name} model")

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"🧠 Model parameters: {total_params:,}")
    num_scripted = 10 if args.stage >= 2 else 0
    num_selfplay = 6 if args.stage == 3 else 0
    print(f"🏗️  Architecture: {arch_name}")
    print(f"🎮 Environment: {env.num_envs}x (1 agent + {num_scripted} scripted + {num_selfplay} self-play = {1 + num_scripted + num_selfplay} snakes)")
    print(f"📺 Render: {'ON' if args.render else 'OFF'}")
    print(f"📈 Stage {args.stage} Curriculum Active")
    print(f"🎯 Training for {args.timesteps:,} timesteps\n")

    callback = SelfPlayCallback(ckpt_mgr, env, save_every=SAVE_EVERY_EPISODES,
                                record_every=args.record_every, stage=args.stage,
                                use_lstm=use_lstm,
                                save_videos_local=args.save_videos_local)

    try:
        model.learn(total_timesteps=args.timesteps, callback=callback,
                    progress_bar=True, reset_num_timesteps=not bool(args.resume))
    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")
    finally:
        final_path = os.path.join(CHECKPOINT_DIR, "policy_final")
        model.save(final_path)
        print(f"💾 Saved final model: {final_path}")
        env.close()

    print(f"\n✅ Training complete!")
    print(f"   Total episodes: {callback.episode_count:,}")
    print(f"   Total timesteps: {callback.num_timesteps:,}")
    if callback.ep_rewards:
        print(f"   Final avg reward: {np.mean(callback.ep_rewards[-100:]):+.2f}")


if __name__ == '__main__':
    main()
