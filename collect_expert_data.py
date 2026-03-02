"""
Collect expert demonstrations from a scripted bot for behavior cloning.

Output format:
  <output_dir>/metadata.json
  <output_dir>/chunk_00000.npz
  <output_dir>/chunk_00001.npz
  ...

Each chunk stores:
  - map: (N, 5, 168, 168)
  - state: (N, 8)
  - action: (N, 2)  # [steering, boost]
  - episode_starts: (N,) bool
"""
import argparse
import json
import os
import random
import shutil
import time

import numpy as np
from tqdm.auto import tqdm

import bot_ai
from slither_gym import SlitherEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Collect expert data for BC")
    parser.add_argument("--bot-type", type=str, default="interceptor",
                        choices=bot_ai.BOT_TYPES,
                        help="Expert bot personality to imitate")
    parser.add_argument("--frames", type=int, default=50_000,
                        help="Total expert frames to record")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Frames per compressed chunk file")
    parser.add_argument("--output-dir", type=str, default="datasets/expert_interceptor",
                        help="Directory where chunked dataset will be written")
    parser.add_argument("--map-dtype", type=str, default="uint8",
                        choices=["uint8", "float16", "float32"],
                        help="Storage dtype for map observations")
    parser.add_argument("--num-scripted", type=int, default=10,
                        help="Number of scripted bots in the arena during collection")
    parser.add_argument("--world-radius", type=int, default=2000)
    parser.add_argument("--food-count", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=3000,
                        help="Episode length cap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true",
                        help="Render during collection (much slower)")
    parser.add_argument("--log-every", type=int, default=5000,
                        help="Log progress every N frames")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete output directory if it already exists")
    return parser.parse_args()


def _encode_map(obs_map, map_dtype):
    if map_dtype == np.uint8:
        return np.clip(obs_map * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    return obs_map.astype(map_dtype, copy=False)


def _flush_chunk(output_dir, chunk_idx, used, map_buffer, state_buffer, action_buffer, episode_buffer):
    chunk_name = f"chunk_{chunk_idx:05d}.npz"
    chunk_path = os.path.join(output_dir, chunk_name)
    np.savez_compressed(
        chunk_path,
        map=map_buffer[:used],
        state=state_buffer[:used],
        action=action_buffer[:used],
        episode_starts=episode_buffer[:used],
    )
    return chunk_name, used


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.frames <= 0:
        raise ValueError("--frames must be > 0")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")

    if os.path.exists(args.output_dir):
        if not args.overwrite:
            raise FileExistsError(
                f"Output dir already exists: {args.output_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    env = SlitherEnv(
        num_scripted=args.num_scripted,
        num_selfplay=0,
        world_radius=args.world_radius,
        food_count=args.food_count,
        max_steps=args.max_steps,
        render_mode="human" if args.render else None,
    )

    map_shape = env.observation_space["map"].shape
    state_shape = env.observation_space["state"].shape
    if map_shape != (5, 168, 168):
        raise RuntimeError(f"Unexpected map shape {map_shape}; expected (5, 168, 168)")
    if state_shape != (8,):
        raise RuntimeError(f"Unexpected state shape {state_shape}; expected (8,)")

    map_dtype = np.dtype(args.map_dtype)
    map_buffer = np.empty((args.chunk_size, *map_shape), dtype=map_dtype)
    state_buffer = np.empty((args.chunk_size, state_shape[0]), dtype=np.float32)
    action_buffer = np.empty((args.chunk_size, 2), dtype=np.float32)
    episode_buffer = np.empty((args.chunk_size,), dtype=np.bool_)

    metadata = {
        "version": 1,
        "bot_type": args.bot_type,
        "total_frames": int(args.frames),
        "chunk_size": int(args.chunk_size),
        "map_shape": list(map_shape),
        "state_shape": list(state_shape),
        "action_shape": [2],
        "map_dtype": args.map_dtype,
        "num_scripted": int(args.num_scripted),
        "world_radius": int(args.world_radius),
        "food_count": int(args.food_count),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "chunks": [],
    }

    obs, _ = env.reset(seed=args.seed)
    episode_start = True
    episodes = 1
    chunk_idx = 0
    chunk_used = 0
    collected = 0
    start_t = time.time()
    progress = tqdm(
        total=args.frames,
        desc=f"collect[{args.bot_type}]",
        unit="frame",
        dynamic_ncols=True,
    )

    while collected < args.frames:
        map_buffer[chunk_used] = _encode_map(obs["map"], map_dtype)
        state_buffer[chunk_used] = obs["state"]
        episode_buffer[chunk_used] = episode_start

        action = env.expert_action_for_player(bot_type=args.bot_type)
        action_buffer[chunk_used] = action

        obs, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        if args.render:
            env.render()

        if done:
            obs, _ = env.reset()
            episode_start = True
            episodes += 1
        else:
            episode_start = False

        chunk_used += 1
        collected += 1
        progress.update(1)

        if chunk_used == args.chunk_size or collected == args.frames:
            chunk_name, frame_count = _flush_chunk(
                args.output_dir, chunk_idx, chunk_used,
                map_buffer, state_buffer, action_buffer, episode_buffer
            )
            metadata["chunks"].append({"file": chunk_name, "frames": int(frame_count)})
            chunk_idx += 1
            chunk_used = 0

        if collected % args.log_every == 0 or collected == args.frames:
            elapsed = max(1e-6, time.time() - start_t)
            fps = collected / elapsed
            progress.set_postfix(episodes=episodes, fps=f"{fps:.1f}")

    progress.close()

    env.close()

    metadata["num_chunks"] = len(metadata["chunks"])
    metadata["episodes"] = episodes
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start_t
    print(f"\nSaved expert dataset to: {args.output_dir}")
    print(f"  bot_type: {args.bot_type}")
    print(f"  frames: {collected:,}")
    print(f"  chunks: {metadata['num_chunks']}")
    print(f"  episodes: {episodes:,}")
    print(f"  duration: {elapsed/60.0:.1f} min")


if __name__ == "__main__":
    main()
