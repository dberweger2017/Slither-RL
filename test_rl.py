"""
Play against trained RL models + scripted bots.
Loads 6 policy agents from checkpoints (recency-weighted) + 9 scripted bots.
Supports both standard PPO and RecurrentPPO (LSTM) models.
"""
import pygame
import math
import random
import os
import numpy as np
from tqdm.auto import tqdm

import bot_ai
import observation as obs_module
from spatial_hash import SpatialHash
from stable_baselines3 import PPO

try:
    from sb3_contrib import RecurrentPPO
    HAS_RECURRENT = True
except ImportError:
    HAS_RECURRENT = False

import torch
from train import SlitherFeatureExtractor

device = 'auto'
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Track whether loaded models are LSTM
IS_LSTM = False

WIDTH, HEIGHT = 1280, 900
FPS = 60
BASE_ZOOM = 0.75
BG_COLOR = (20, 20, 20)
GRID_COLOR = (40, 40, 40)
FOOD_COLORS = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100), (255, 100, 255)]

BASE_SPEED = 3.0
BOOST_SPEED = 6.0
BASE_TURN_RATE = 0.1
START_LENGTH = 10
START_MASS = 50
MASS_PER_FOOD = 5
FOOD_FRICTION = 0.92
CHECKPOINT_DIR = "checkpoints"


class Food:
    def __init__(self, x, y, value=1, color=None, vx=0, vy=0):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.value = value
        self.radius = 3 + value
        self.color = color or random.choice(FOOD_COLORS)

    def update(self):
        if abs(self.vx) > 0.1 or abs(self.vy) > 0.1:
            self.x += self.vx
            self.y += self.vy
            self.vx *= FOOD_FRICTION
            self.vy *= FOOD_FRICTION
        else:
            self.vx = self.vy = 0


class Snake:
    def __init__(self, x, y, is_player=False, bot_type=None):
        self.is_player = is_player
        self.bot_type = bot_type or random.choice(bot_ai.BOT_TYPES)
        self.head = [x, y]
        self.segments = [[x, y] for _ in range(START_LENGTH)]
        self.path_history = [[x, y] for _ in range(300)]
        self.angle = random.uniform(0, 2 * math.pi)
        self.target_angle = self.angle
        self.mass = START_MASS
        self.radius = 10
        self.is_boosting = False
        self.dead = False
        self.kills = 0
        self.role = 'player' if is_player else 'scripted'
        self._patrol_idx = 0
        self._wander_timer = 0

        if is_player:
            self.color = (0, 200, 255)
        elif self.role == 'rl':
            self.color = (255, 215, 0)
        else:
            self.color = bot_ai.BOT_COLORS.get(self.bot_type, (200, 200, 200))

    def get_speed(self):
        return BOOST_SPEED if self.is_boosting else BASE_SPEED

    def get_segment_dist(self):
        return max(6.0, self.radius * 0.8)

    def update(self):
        if self.dead:
            return
        if self.is_boosting and self.mass > (START_MASS + 10):
            self.mass -= max(0.5, self.mass * 0.001)
        else:
            self.is_boosting = False

        turn_rate = max(0.04, BASE_TURN_RATE / (1 + math.log10(self.mass / START_MASS) * 0.3))
        fam = self.mass / START_MASS
        self.radius = max(5, 5.0 * math.sqrt(fam + 1))

        diff = (self.target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
        if diff > turn_rate:
            self.angle += turn_rate
        elif diff < -turn_rate:
            self.angle -= turn_rate
        else:
            self.angle = self.target_angle
        self.angle %= (2 * math.pi)

        speed = self.get_speed()
        self.head[0] += math.cos(self.angle) * speed
        self.head[1] += math.sin(self.angle) * speed

        seg_dist = self.get_segment_dist()
        target_length = max(START_LENGTH, int(math.sqrt(self.mass) * 3))
        while len(self.segments) < target_length:
            self.segments.append(list(self.segments[-1]))
        while len(self.segments) > target_length:
            self.segments.pop()

        self.segments[0] = list(self.head)
        follow_speed = 0.25
        for i in range(1, len(self.segments)):
            prev, curr = self.segments[i - 1], self.segments[i]
            curr[0] += (prev[0] - curr[0]) * follow_speed
            curr[1] += (prev[1] - curr[1]) * follow_speed
            dx, dy = prev[0] - curr[0], prev[1] - curr[1]
            dist = math.hypot(dx, dy)
            if dist > seg_dist:
                ratio = seg_dist / dist
                curr[0] = prev[0] - dx * ratio
                curr[1] = prev[1] - dy * ratio
        self.segments = self.segments[:target_length]

        self.path_history.insert(0, list(self.head))
        self.path_history = self.path_history[:300]

    def draw(self, surface, camera_x, camera_y, zoom):
        if self.dead:
            return
        for i, seg in enumerate(self.segments):
            sx = int((seg[0] - camera_x) * zoom)
            sy = int((seg[1] - camera_y) * zoom)
            r = max(2, int(self.radius * 0.8 * zoom))
            if -r < sx < WIDTH + r and -r < sy < HEIGHT + r:
                pygame.draw.circle(surface, self.color, (sx, sy), r)
        hx = int((self.head[0] - camera_x) * zoom)
        hy = int((self.head[1] - camera_y) * zoom)
        hr = max(3, int(self.radius * zoom))
        pygame.draw.circle(surface, (255, 255, 255), (hx, hy), hr)
        eye_r = max(1, int(self.radius * 0.35 * zoom))
        ex = int(hx + math.cos(self.angle) * self.radius * 0.5 * zoom)
        ey = int(hy + math.sin(self.angle) * self.radius * 0.5 * zoom)
        pygame.draw.circle(surface, (0, 0, 0), (ex, ey), eye_r)


def make_obs_for(snake, snakes, foods, food_grid, world_radius):
    """Generate observation dict for an RL-controlled snake."""
    mini_map = obs_module.generate_observation(snake, snakes, foods, food_grid, world_radius)
    turn_rate = max(0.04, BASE_TURN_RATE / (1 + math.log10(snake.mass / START_MASS) * 0.3))
    dist_to_wall = world_radius - math.hypot(snake.head[0], snake.head[1])
    state = np.array([
        min(1.0, snake.mass / 5000.0),
        turn_rate / BASE_TURN_RATE,
        snake.get_speed() / BOOST_SPEED,
        1.0 if snake.is_boosting else 0.0,
        1.0 if snake.mass > START_MASS + 10 else 0.0,
        min(1.0, max(0.0, dist_to_wall / world_radius)),
        min(1.0, len(snake.segments) / 500.0),
        min(1.0, snake.kills / 20.0),
    ], dtype=np.float32)
    return {'map': mini_map, 'state': state}


def load_rl_opponents(n=6):
    """Load n models from checkpoints with recency-weighted sampling.
    Auto-detects RecurrentPPO (LSTM) vs standard PPO."""
    global IS_LSTM
    if not os.path.exists(CHECKPOINT_DIR):
        print("No checkpoints/ directory found. RL agents will act randomly.")
        return []

    files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("policy_")])
    if not files:
        print("No checkpoint files found. RL agents will act randomly.")
        return []

    n_ckpts = len(files)
    weights = np.array([2.0 ** (i / max(n_ckpts, 1)) for i in range(n_ckpts)])
    weights /= weights.sum()

    # Bypass version mismatch json/pickle unpickling errors
    # Note: If the checkpoint was trained on 84x84, it will crash when it
    # receives a 168x168 observation. This can't be fixed by custom_objects alone.
    custom_objects = {
        "learning_rate": 5e-5,
        "clip_range": 0.2,
        "lr_schedule": lambda _: 5e-5,
        "clip_range_schedule": lambda _: 0.2
    }

    chosen = np.random.choice(n_ckpts, size=min(n, n_ckpts), replace=True, p=weights)
    models = []
    for idx in chosen:
        path = os.path.join(CHECKPOINT_DIR, files[idx])
        loaded = False
        # Try RecurrentPPO first
        if HAS_RECURRENT:
            try:
                model = RecurrentPPO.load(path, device=device, custom_objects=custom_objects)
                
                # Check for observation shape compatibility
                expected_shape = model.observation_space.spaces['map'].shape
                if expected_shape != (5, 168, 168):
                    print(f"  Skipping {files[idx]}: Incompatible observation shape {expected_shape} (expected (5, 168, 168))")
                    continue
                    
                models.append(model)
                IS_LSTM = True
                loaded = True
                print(f"  Loaded (LSTM): {files[idx]}")
            except Exception as e:
                pass
        if not loaded:
            try:
                model = PPO.load(path, device=device, custom_objects=custom_objects)
                
                # Check for observation shape compatibility
                expected_shape = model.observation_space.spaces['map'].shape
                if expected_shape != (5, 168, 168):
                    print(f"  Skipping {files[idx]}: Incompatible observation shape {expected_shape} (expected (5, 168, 168))")
                    continue
                    
                models.append(model)
                loaded = True
                print(f"  Loaded (PPO): {files[idx]}")
            except Exception as e:
                print(f"  Failed to load {files[idx]}: {e}")

    arch = "LSTM" if IS_LSTM else "PPO"
    print(f"Loaded {len(models)} RL opponents ({arch})")
    return models

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl", type=int, default=6, help="Number of RL agents")
    parser.add_argument("--bots", type=int, default=9, help="Number of scripted bots")
    parser.add_argument("--bot-type", type=str, default=None, help="Force a specific bot personality (e.g. scavenger)")
    parser.add_argument("--simulate", action="store_true", help="Run 60s headless simulation with 2 bots of each type to find the winner")
    parser.add_argument("--simulate-frames", type=int, default=3600,
                        help="Frames to run in --simulate mode (3600 = 60s at 60 FPS)")
    parser.add_argument("--simulate-log-every", type=int, default=600,
                        help="Print simulation telemetry every N frames in --simulate mode (0=off)")
    parser.add_argument("--phase-watch", type=str, default="harvester",
                        help="Bot type to print phase telemetry for (e.g. harvester)")
    parser.add_argument("--debug-vision", action="store_true", help="View agent observation channels fullscreen")
    args = parser.parse_args()

    if not args.simulate:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Slither.io — Play vs Trained RL Agents")
        clock = pygame.time.Clock()

    WORLD_RADIUS = 2000
    FOOD_COUNT = 500

    food_grid = SpatialHash(cell_size=100)
    segment_grid = SpatialHash(cell_size=50)

    def random_point_in_circle(radius, margin=200):
        r = random.uniform(0, radius - margin)
        theta = random.uniform(0, 2 * math.pi)
        return r * math.cos(theta), r * math.sin(theta)

    # Load RL models
    print(f"Loading {args.rl} RL opponents...")
    rl_models = load_rl_opponents(n=args.rl)

    # Spawn player
    px, py = random_point_in_circle(WORLD_RADIUS)
    player = Snake(px, py, is_player=True)
    if args.simulate:
        player.dead = True
    snakes = [player]

    # Scripted bots
    if args.simulate:
        for bt in bot_ai.BOT_TYPES:
            for _ in range(2):
                bx, by = random_point_in_circle(WORLD_RADIUS)
                s = Snake(bx, by, bot_type=bt)
                s.role = 'scripted'
                snakes.append(s)
    else:
        for i in range(args.bots):
            bot_type = args.bot_type if args.bot_type else bot_ai.BOT_TYPES[i % len(bot_ai.BOT_TYPES)]
            bx, by = random_point_in_circle(WORLD_RADIUS)
            s = Snake(bx, by, bot_type=bot_type)
            s.role = 'scripted'
            snakes.append(s)

    # RL-controlled agents
    rl_indices = []
    for i in range(args.rl):
        bx, by = random_point_in_circle(WORLD_RADIUS)
        s = Snake(bx, by)
        s.role = 'rl'
        s.color = (255, 215, 0)
        rl_indices.append(len(snakes))
        snakes.append(s)

    # LSTM hidden states for RL agents
    rl_lstm_states = {}

    # Food
    foods = []
    for _ in range(FOOD_COUNT):
        fx, fy = random_point_in_circle(WORLD_RADIUS, margin=0)
        foods.append(Food(fx, fy))

    fps_min, fps_max = 999, 0
    fps_update_timer, fps_display = 0, 60

    vision_mode = 0  # 0: normal, 1-5: specific CNN channels

    step = 0
    running = True
    bot_peaks = {bt: 0 for bt in bot_ai.BOT_TYPES}
    bot_kills = {bt: 0 for bt in bot_ai.BOT_TYPES}
    bot_food_mass_gain = {bt: 0.0 for bt in bot_ai.BOT_TYPES}
    bot_food_mass_small = {bt: 0.0 for bt in bot_ai.BOT_TYPES}
    bot_food_mass_big = {bt: 0.0 for bt in bot_ai.BOT_TYPES}
    bot_kill_mass_potential = {bt: 0.0 for bt in bot_ai.BOT_TYPES}
    watch_phase_frames = {}
    watch_phase_food_mass = {}
    watch_phase_kill_mass = {}
    pbar = None

    if args.simulate:
        sim_seconds = args.simulate_frames / 60.0
        print(f"Starting headless simulation for {args.simulate_frames} frames ({sim_seconds:.1f} seconds)...")
        pbar = tqdm(total=args.simulate_frames, desc="simulate", unit="frame", dynamic_ncols=True)

    while running:
        if args.simulate and step >= args.simulate_frames:
            running = False
            continue
            
        step += 1
        if pbar is not None:
            pbar.update(1)
        
        if not args.simulate:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_1:
                        vision_mode = 0
                    elif event.key == pygame.K_2:
                        vision_mode = 1
                    elif event.key == pygame.K_3:
                        vision_mode = 2
                    elif event.key == pygame.K_4:
                        vision_mode = 3
                    elif event.key == pygame.K_5:
                        vision_mode = 4
                    elif event.key == pygame.K_6:
                        vision_mode = 5

        # Player input
        if not args.simulate and not player.dead:
            mx, my = pygame.mouse.get_pos()
            current_zoom = BASE_ZOOM / (1 + (player.mass - START_MASS) * 0.0003)
            current_zoom = max(0.25, min(BASE_ZOOM, current_zoom))
            camera_x = player.head[0] - (WIDTH / 2) / current_zoom
            camera_y = player.head[1] - (HEIGHT / 2) / current_zoom
            world_mx = mx / current_zoom + camera_x
            world_my = my / current_zoom + camera_y
            player.target_angle = math.atan2(world_my - player.head[1], world_mx - player.head[0])
            player.is_boosting = pygame.mouse.get_pressed()[0] and player.mass > (START_MASS + 10)

        # Boost food from boosting snakes
        for snake in snakes:
            if snake.is_boosting and not snake.dead and snake.mass > START_MASS + 10:
                if random.random() < 0.2:
                    tail = snake.segments[-1]
                    push = 4.0
                    vx = -math.cos(snake.angle) * push + random.uniform(-1, 1)
                    vy = -math.sin(snake.angle) * push + random.uniform(-1, 1)
                    foods.append(Food(tail[0], tail[1], value=1, vx=vx, vy=vy))

        for food in foods:
            food.update()

        # Rebuild spatial grids
        food_grid.clear()
        for food in foods:
            food_grid.insert(food, food.x, food.y)
        segment_grid.clear()
        for snake in snakes:
            if snake.dead:
                continue
            for seg in snake.segments[3:]:
                segment_grid.insert((snake, seg), seg[0], seg[1])

        # RL agent actions
        for i, idx in enumerate(rl_indices):
            snake = snakes[idx]
            if snake.dead:
                # Reset LSTM state on death
                if IS_LSTM and i in rl_lstm_states:
                    del rl_lstm_states[i]
                continue
            if rl_models:
                model = rl_models[i % len(rl_models)]
                obs = make_obs_for(snake, snakes, foods, food_grid, WORLD_RADIUS)
                if IS_LSTM:
                    lstm_state = rl_lstm_states.get(i, None)
                    episode_start = np.array([lstm_state is None], dtype=bool)
                    action, lstm_state = model.predict(
                        obs, state=lstm_state,
                        episode_start=episode_start,
                        deterministic=False)
                    rl_lstm_states[i] = lstm_state
                else:
                    action, _ = model.predict(obs, deterministic=False)
                steering = float(np.clip(action[0], -1, 1))
                boost = float(action[1]) > 0.5
                turn_rate = max(0.04, BASE_TURN_RATE / (1 + math.log10(snake.mass / START_MASS) * 0.3))
                snake.target_angle = snake.angle + steering * turn_rate
                snake.is_boosting = boost and snake.mass > (START_MASS + 10)
            else:
                # Random fallback
                snake.target_angle += random.uniform(-0.1, 0.1)

        # Scripted bot AI
        for snake in snakes:
            if snake.role == 'scripted' and not snake.dead:
                bot_ai.update(snake, foods, snakes, WORLD_RADIUS, segment_grid)
                if args.simulate and snake.bot_type == args.phase_watch:
                    phase = getattr(snake, "_ai_phase", "unknown")
                    watch_phase_frames[phase] = watch_phase_frames.get(phase, 0) + 1

        # Update all snakes
        for snake in snakes:
            snake.update()

        # Boundary death
        for snake in snakes:
            if not snake.dead:
                if math.hypot(snake.head[0], snake.head[1]) > WORLD_RADIUS:
                    snake.dead = True
                    cx = sum(s[0] for s in snake.segments) / len(snake.segments)
                    cy = sum(s[1] for s in snake.segments) / len(snake.segments)
                    for seg in snake.segments:
                        if random.random() < 0.5:
                            dx, dy = seg[0] - cx, seg[1] - cy
                            d = math.hypot(dx, dy) + 0.1
                            foods.append(Food(seg[0], seg[1], value=3, color=snake.color,
                                            vx=(dx/d)*random.uniform(2,5),
                                            vy=(dy/d)*random.uniform(2,5)))

        # Head-to-body collision
        for sa in snakes:
            if sa.dead:
                continue
            nearby = segment_grid.query(sa.head[0], sa.head[1], 50)
            for (owner, seg) in nearby:
                if owner is sa or owner.dead:
                    continue
                dist = math.hypot(sa.head[0] - seg[0], sa.head[1] - seg[1])
                if dist < owner.radius * 0.8:
                    sa.dead = True
                    owner.kills += 1
                    if owner.role == 'scripted':
                        bot_kills[owner.bot_type] += 1
                        kill_mass = max(0.0, sa.mass - START_MASS)
                        bot_kill_mass_potential[owner.bot_type] += kill_mass
                        if owner.bot_type == args.phase_watch:
                            phase = getattr(owner, "_ai_phase", "unknown")
                            watch_phase_kill_mass[phase] = watch_phase_kill_mass.get(phase, 0.0) + kill_mass
                    if sa.is_player and not args.simulate:
                        print(f"Killed by a {sa.role}! Mass: {int(sa.mass)}")
                    if owner.is_player and not args.simulate:
                        print(f"You killed a {sa.role}! (Kill #{owner.kills})")
                    cx = sum(s[0] for s in sa.segments) / len(sa.segments)
                    cy = sum(s[1] for s in sa.segments) / len(sa.segments)
                    for seg2 in sa.segments:
                        if random.random() < 0.5:
                            dx, dy = seg2[0] - cx, seg2[1] - cy
                            d = math.hypot(dx, dy) + 0.1
                            foods.append(Food(seg2[0], seg2[1], value=3, color=sa.color,
                                            vx=(dx/d)*random.uniform(2,5),
                                            vy=(dy/d)*random.uniform(2,5)))
                    break

        # Food collision
        eaten = []
        for snake in snakes:
            if snake.dead:
                continue
            nearby_food = food_grid.query(snake.head[0], snake.head[1], snake.radius + 10)
            for food in nearby_food:
                dist = math.hypot(snake.head[0] - food.x, snake.head[1] - food.y)
                if dist < snake.radius + food.radius:
                    eaten.append(food)
                    gain = food.value * MASS_PER_FOOD
                    snake.mass += gain
                    if snake.role == 'scripted':
                        bt = snake.bot_type
                        bot_food_mass_gain[bt] += gain
                        if food.value >= 3:
                            bot_food_mass_big[bt] += gain
                        else:
                            bot_food_mass_small[bt] += gain
                        if bt == args.phase_watch:
                            phase = getattr(snake, "_ai_phase", "unknown")
                            watch_phase_food_mass[phase] = watch_phase_food_mass.get(phase, 0.0) + gain
        for food in eaten:
            if food in foods:
                foods.remove(food)

        while len(foods) < FOOD_COUNT:
            fx, fy = random_point_in_circle(WORLD_RADIUS, margin=0)
            foods.append(Food(fx, fy))

        # Respawn dead bots
        for i, snake in enumerate(snakes):
            if snake.dead and not snake.is_player:
                bx, by = random_point_in_circle(WORLD_RADIUS)
                # Keep same role and bot_type
                btype = snake.bot_type
                role = snake.role
                models_idx = getattr(snake, '_rl_idx', None) # Use getattr for safety

                new_snake = Snake(bx, by, bot_type=btype)
                new_snake.role = role
                if models_idx is not None:
                    new_snake._rl_idx = models_idx
                if role == 'rl':
                    new_snake.color = (255, 215, 0)
                snakes[i] = new_snake

        if player.dead and not args.simulate:
            px, py = random_point_in_circle(WORLD_RADIUS)
            player = Snake(px, py, is_player=True)
            snakes[0] = player

        # Track stats
        if args.simulate:
            for snake in snakes:
                if not snake.dead and snake.role == 'scripted':
                    if snake.mass > bot_peaks[snake.bot_type]:
                        bot_peaks[snake.bot_type] = snake.mass

            if args.simulate_log_every > 0 and (step % args.simulate_log_every == 0):
                log = tqdm.write if pbar is not None else print
                scripted_alive = [s for s in snakes if not s.dead and s.role == 'scripted']
                leader = max(scripted_alive, key=lambda s: s.mass) if scripted_alive else None

                log(f"\n[sim {step:,}/{args.simulate_frames:,}]")
                if leader is not None:
                    log(f" leader_now={leader.bot_type} mass={leader.mass:.0f} kills={leader.kills}")
                else:
                    log(" leader_now=none")

                top_food = sorted(bot_food_mass_gain.items(), key=lambda kv: kv[1], reverse=True)[:3]
                food_str = " | ".join(f"{bt}:{val:.0f}" for bt, val in top_food)
                log(f" food_mass_top={food_str}")

                top_kill = sorted(bot_kill_mass_potential.items(), key=lambda kv: kv[1], reverse=True)[:3]
                kill_str = " | ".join(f"{bt}:{val:.0f}" for bt, val in top_kill)
                log(f" kill_mass_top={kill_str}")

                if args.phase_watch in bot_ai.BOT_TYPES:
                    total_phase_frames = sum(watch_phase_frames.values())
                    if total_phase_frames > 0:
                        phase_rows = []
                        for phase, frames in sorted(watch_phase_frames.items(),
                                                    key=lambda kv: kv[1], reverse=True):
                            pct = 100.0 * frames / total_phase_frames
                            food_gain = watch_phase_food_mass.get(phase, 0.0)
                            kill_gain = watch_phase_kill_mass.get(phase, 0.0)
                            phase_rows.append(f"{phase}:{pct:.0f}% food={food_gain:.0f} kill={kill_gain:.0f}")
                        log(f" {args.phase_watch}_phases=" + " | ".join(phase_rows[:5]))

        # --- DRAW ---
        if not args.simulate:
            screen.fill(BG_COLOR)

            if vision_mode == 0:
                current_zoom = BASE_ZOOM / (1 + (player.mass - START_MASS) * 0.0003)
                current_zoom = max(0.25, min(BASE_ZOOM, current_zoom))
                camera_x = player.head[0] - (WIDTH / 2) / current_zoom
                camera_y = player.head[1] - (HEIGHT / 2) / current_zoom

                grid_size = 50
                start_x = int((-camera_x % grid_size) * current_zoom)
                start_y = int((-camera_y % grid_size) * current_zoom)
                scaled_grid = int(grid_size * current_zoom)
                if scaled_grid > 0:
                    for x in range(start_x, WIDTH, scaled_grid):
                        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, HEIGHT))
                    for y in range(start_y, HEIGHT, scaled_grid):
                        pygame.draw.line(screen, GRID_COLOR, (0, y), (WIDTH, y))

                cx_screen = int(-camera_x * current_zoom)
                cy_screen = int(-camera_y * current_zoom)
                cr_screen = int(WORLD_RADIUS * current_zoom)
                pygame.draw.circle(screen, (200, 50, 50), (cx_screen, cy_screen), cr_screen, 3)

                for food in foods:
                    fx = int((food.x - camera_x) * current_zoom)
                    fy = int((food.y - camera_y) * current_zoom)
                    fr = max(1, int(food.radius * current_zoom))
                    pygame.draw.circle(screen, food.color, (fx, fy), fr)

                for snake in snakes:
                    snake.draw(screen, camera_x, camera_y, current_zoom)

                # CNN observation debug overlay
                if not player.dead:
                    obs = obs_module.generate_observation(player, snakes, foods, food_grid, WORLD_RADIUS)
                    previews = obs_module.obs_to_surfaces(obs, preview_size=100)
                    preview_x = WIDTH - 110
                    preview_y = 5
                    small_font = pygame.font.SysFont(None, 16)
                    for surf, label in previews:
                        pygame.draw.rect(screen, (0, 0, 0), (preview_x - 2, preview_y - 2, 104, 118))
                        screen.blit(surf, (preview_x, preview_y))
                        label_surf = small_font.render(label, True, (200, 200, 200))
                        screen.blit(label_surf, (preview_x + 2, preview_y + 102))
                        preview_y += 120
            else:
                if not player.dead:
                    # Clear screen completely for debug vision map
                    screen.fill((0, 0, 0))

                    obs = obs_module.generate_observation(player, snakes, foods, food_grid, WORLD_RADIUS)
                    # Convert to viewable surfaces
                    surfaces = obs_module.obs_to_surfaces(obs, preview_size=600)

                    channel_idx = vision_mode - 1
                    surf, label = surfaces[channel_idx]

                    font_lg = pygame.font.SysFont(None, 48)
                    label_surf = font_lg.render(f"CNN View: {label} [1 to 6 to toggle]", True, (255, 255, 255))
                    screen.blit(label_surf, (20, 20))

                    # Center the 600x600 surface
                    x_offset = (WIDTH - 600) // 2
                    y_offset = (HEIGHT - 600) // 2
                    screen.blit(surf, (x_offset, y_offset))

            # FPS counter
            current_fps = clock.get_fps()
            if current_fps > 0:
                fps_update_timer += 1
                if current_fps < fps_min:
                    fps_min = current_fps
                if current_fps > fps_max:
                    fps_max = current_fps
                if fps_update_timer >= 30:
                    fps_display = current_fps
                    fps_update_timer = 0

            font = pygame.font.SysFont(None, 24)
            alive_count = sum(1 for s in snakes if not s.dead)
            rl_alive = sum(1 for i in rl_indices if not snakes[i].dead)
            status = "DEAD" if player.dead else f"Mass: {int(player.mass)}"
            info_text = f"{status} | Kills: {player.kills} | Alive: {alive_count}/{len(snakes)} | RL alive: {rl_alive}/6"
            screen.blit(font.render(info_text, True, (255, 255, 255)), (10, 10))

            fps_color = (80, 255, 80) if fps_display >= 50 else (255, 255, 80) if fps_display >= 30 else (255, 80, 80)
            fps_text = f"FPS: {fps_display:.0f}  (min:{fps_min:.0f} max:{fps_max:.0f})"
            screen.blit(font.render(fps_text, True, fps_color), (WIDTH - 310, HEIGHT - 30))

            # Legend
            small_font = pygame.font.SysFont(None, 18)
            y_off = 30
            for bt in bot_ai.BOT_TYPES:
                count = sum(1 for s in snakes if not s.dead and s.role == 'scripted' and s.bot_type == bt)
                col = bot_ai.BOT_COLORS.get(bt, (200, 200, 200))
                pygame.draw.circle(screen, col, (15, y_off + 4), 4)
                screen.blit(small_font.render(f"{bt}: {count}", True, (180, 180, 180)), (25, y_off - 2))
                y_off += 16
            pygame.draw.circle(screen, (255, 215, 0), (15, y_off + 4), 4)
            screen.blit(small_font.render(f"RL agent: {rl_alive}", True, (180, 180, 180)), (25, y_off - 2))

            pygame.display.flip()
            clock.tick(FPS)

    if not args.simulate:
        pygame.quit()
    else:
        if pbar is not None:
            pbar.close()
        sim_seconds = args.simulate_frames / 60.0
        print(f"\n--- SIMULATION RESULTS ({sim_seconds:.1f} SECONDS) ---")
        scores = []
        for bt in bot_ai.BOT_TYPES:
            scores.append((bt, bot_peaks[bt], bot_kills[bt], bot_food_mass_gain[bt], bot_kill_mass_potential[bt]))
        scores.sort(key=lambda x: x[1], reverse=True) # sort by peak mass

        print(f"{'BOT TYPE':15} | {'PEAK MASS':<10} | {'KILLS':<5} | {'FOOD_MASS':<9} | {'KILL_MASS'}")
        print("-" * 74)
        for bt, mass, kills, food_mass, kill_mass in scores:
            print(f"{bt:15} | {int(mass):<10} | {kills:<5} | {int(food_mass):<9} | {int(kill_mass)}")
        print("-" * 74)
        winner = scores[0][0]
        print(f"WINNER (Highest Peak Mass): {winner.upper()}!")

        top_food = sorted(bot_food_mass_gain.items(), key=lambda kv: kv[1], reverse=True)[:3]
        print("Top food farmers: " + " | ".join(f"{bt}={val:.0f}" for bt, val in top_food))
        top_kill = sorted(bot_kill_mass_potential.items(), key=lambda kv: kv[1], reverse=True)[:3]
        print("Top kill pressure: " + " | ".join(f"{bt}={val:.0f}" for bt, val in top_kill))

        if args.phase_watch in bot_ai.BOT_TYPES:
            total_phase_frames = sum(watch_phase_frames.values())
            if total_phase_frames > 0:
                print(f"\n{args.phase_watch.upper()} PHASE BREAKDOWN")
                phase_keys = set(watch_phase_frames) | set(watch_phase_food_mass) | set(watch_phase_kill_mass)
                phase_rows = []
                for phase in phase_keys:
                    frames = watch_phase_frames.get(phase, 0)
                    pct = 100.0 * frames / total_phase_frames if total_phase_frames > 0 else 0.0
                    food_gain = watch_phase_food_mass.get(phase, 0.0)
                    kill_gain = watch_phase_kill_mass.get(phase, 0.0)
                    phase_rows.append((phase, pct, food_gain, kill_gain))
                phase_rows.sort(key=lambda x: x[1], reverse=True)
                for phase, pct, food_gain, kill_gain in phase_rows:
                    print(f"  {phase:16} time={pct:5.1f}%  food_mass={food_gain:7.1f}  kill_mass={kill_gain:7.1f}")


if __name__ == "__main__":
    main()
