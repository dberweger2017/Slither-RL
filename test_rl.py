"""
Play against trained RL models + scripted bots.
Loads 6 policy agents from checkpoints (recency-weighted) + 9 scripted bots.
"""
import pygame
import math
import random
import os
import numpy as np

import bot_ai
import observation as obs_module
from spatial_hash import SpatialHash
from stable_baselines3 import PPO

import torch
from train import SlitherFeatureExtractor

device = 'auto'
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

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
    """Load n models from checkpoints with recency-weighted sampling."""
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

    chosen = np.random.choice(n_ckpts, size=min(n, n_ckpts), replace=True, p=weights)
    models = []
    for idx in chosen:
        path = os.path.join(CHECKPOINT_DIR, files[idx])
        try:
            model = PPO.load(path, device='cpu')
            models.append(model)
            print(f"  Loaded: {files[idx]}")
        except Exception as e:
            print(f"  Failed to load {files[idx]}: {e}")

    print(f"Loaded {len(models)} RL opponents")
    return models


def main():
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
    print("Loading RL opponents...")
    rl_models = load_rl_opponents(n=6)

    # Spawn player
    px, py = random_point_in_circle(WORLD_RADIUS)
    player = Snake(px, py, is_player=True)
    snakes = [player]

    # 9 scripted bots (one per personality)
    for btype in bot_ai.BOT_TYPES:
        bx, by = random_point_in_circle(WORLD_RADIUS)
        s = Snake(bx, by, bot_type=btype)
        s.role = 'scripted'
        snakes.append(s)

    # 6 RL-controlled agents
    rl_indices = []
    for i in range(6):
        bx, by = random_point_in_circle(WORLD_RADIUS)
        s = Snake(bx, by)
        s.role = 'rl'
        s.color = (255, 215, 0)
        rl_indices.append(len(snakes))
        snakes.append(s)

    # Food
    foods = []
    for _ in range(FOOD_COUNT):
        fx, fy = random_point_in_circle(WORLD_RADIUS, margin=0)
        foods.append(Food(fx, fy))

    fps_min, fps_max = 999, 0
    fps_update_timer, fps_display = 0, 60

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Player input
        if not player.dead:
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
                continue
            if rl_models:
                model = rl_models[i % len(rl_models)]
                obs = make_obs_for(snake, snakes, foods, food_grid, WORLD_RADIUS)
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
                bot_ai.update(snake, foods, snakes, WORLD_RADIUS)

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
                    if sa.is_player:
                        print(f"Killed by a {sa.role}! Mass: {int(sa.mass)}")
                    if owner.is_player:
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
                    snake.mass += food.value * MASS_PER_FOOD
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
                role = snake.role
                new_snake = Snake(bx, by)
                new_snake.role = role
                if role == 'rl':
                    new_snake.color = (255, 215, 0)
                snakes[i] = new_snake

        if player.dead:
            px, py = random_point_in_circle(WORLD_RADIUS)
            player = Snake(px, py, is_player=True)
            snakes[0] = player

        # --- DRAW ---
        screen.fill(BG_COLOR)
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

    pygame.quit()


if __name__ == "__main__":
    main()
