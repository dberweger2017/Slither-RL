"""
Gymnasium Environment for Slither.io RL Training — Self-Play Edition.

Snake composition: 1 training agent + 9 scripted bots + 6 self-play policy agents = 16 snakes
"""
import math
import random
import numpy as np
import gymnasium
from gymnasium import spaces

from spatial_hash import SpatialHash
import bot_ai
import observation as obs_module

BASE_SPEED = 3.0
BOOST_SPEED = 6.0
BASE_TURN_RATE = 0.1
START_LENGTH = 10
START_MASS = 50
MASS_PER_FOOD = 5
FOOD_FRICTION = 0.92


class FoodItem:
    __slots__ = ('x', 'y', 'vx', 'vy', 'value', 'radius', 'color')

    def __init__(self, x, y, value=1, color=None, vx=0, vy=0):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.value = value
        self.radius = 3 + value
        self.color = color or (255, 255, 255)

    def update(self):
        if abs(self.vx) > 0.1 or abs(self.vy) > 0.1:
            self.x += self.vx
            self.y += self.vy
            self.vx *= FOOD_FRICTION
            self.vy *= FOOD_FRICTION
        else:
            self.vx = self.vy = 0


class SnakeEntity:
    def __init__(self, x, y, is_player=False, bot_type=None):
        self.is_player = is_player
        self.bot_type = bot_type or random.choice(bot_ai.BOT_TYPES)
        self.head = [x, y]
        self.segments = [[x, y] for _ in range(START_LENGTH)]
        self.angle = random.uniform(0, 2 * math.pi)
        self.target_angle = self.angle
        self.mass = START_MASS
        self.radius = 10
        self.is_boosting = False
        self.dead = False
        self.kills = 0
        self.color = bot_ai.BOT_COLORS.get(self.bot_type, (200, 200, 200))
        self._patrol_idx = 0
        self._wander_timer = 0
        self.role = 'player' if is_player else 'scripted'

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


class SlitherEnv(gymnasium.Env):
    """
    Slither.io Gymnasium Environment with Self-Play.
    Observation: Dict{'map': (5,84,84), 'state': (8,)}
    Action: Box[-1,1] x Box[0,1] = (steering, boost)
    """
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, num_scripted=9, num_selfplay=6, world_radius=2000,
                 food_count=500, max_steps=3000, render_mode=None):
        super().__init__()
        self.num_scripted = num_scripted
        self.num_selfplay = num_selfplay
        self.world_radius = world_radius
        self.food_count = food_count
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.observation_space = spaces.Dict({
            'map': spaces.Box(0, 1, shape=(5, obs_module.MAP_SIZE, obs_module.MAP_SIZE),
                            dtype=np.float32),
            'state': spaces.Box(0, 1, shape=(8,), dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        self.food_grid = SpatialHash(cell_size=100)
        self.segment_grid = SpatialHash(cell_size=50)

        self.player = None
        self.snakes = []
        self.foods = []
        self.step_count = 0
        self.prev_mass = START_MASS
        self.pending_kill_mass = 0

        # Episode-level metric tracking
        self.peak_mass = START_MASS
        self.food_eaten = 0
        self.boost_frames = 0
        self.wall_close_frames = 0
        self.death_cause = 'alive'

        self._selfplay_policies = []
        self._selfplay_indices = []
        self._screen = None

    def set_selfplay_policies(self, policies):
        self._selfplay_policies = policies

    def load_selfplay_from_dir(self, checkpoint_dir, n=6):
        """Load policies from checkpoint dir (used by SubprocVecEnv subprocesses)."""
        import os
        from stable_baselines3 import PPO
        if not os.path.exists(checkpoint_dir):
            self._selfplay_policies = []
            return
        files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("policy_")])
        if not files:
            self._selfplay_policies = []
            return
        n_ckpts = len(files)
        weights = np.array([2.0 ** (i / max(n_ckpts, 1)) for i in range(n_ckpts)])
        weights /= weights.sum()
        chosen = np.random.choice(n_ckpts, size=min(n, n_ckpts), replace=True, p=weights)
        policies = []
        for idx in chosen:
            path = os.path.join(checkpoint_dir, files[idx])
            try:
                policies.append(PPO.load(path, device='cpu'))
            except Exception:
                pass
        self._selfplay_policies = policies

    def _random_point(self, margin=200):
        r = random.uniform(0, self.world_radius - margin)
        theta = random.uniform(0, 2 * math.pi)
        return r * math.cos(theta), r * math.sin(theta)

    def _make_obs_for(self, snake):
        mini_map = obs_module.generate_observation(
            snake, self.snakes, self.foods, self.food_grid, self.world_radius)
        turn_rate = max(0.04, BASE_TURN_RATE / (1 + math.log10(
            snake.mass / START_MASS) * 0.3))
        dist_to_wall = self.world_radius - math.hypot(snake.head[0], snake.head[1])
        state = np.array([
            min(1.0, snake.mass / 5000.0),
            turn_rate / BASE_TURN_RATE,
            snake.get_speed() / BOOST_SPEED,
            1.0 if snake.is_boosting else 0.0,
            1.0 if snake.mass > START_MASS + 10 else 0.0,
            min(1.0, max(0.0, dist_to_wall / self.world_radius)),
            min(1.0, len(snake.segments) / 500.0),
            min(1.0, snake.kills / 20.0),
        ], dtype=np.float32)
        return {'map': mini_map, 'state': state}

    def _apply_action(self, snake, action):
        steering = float(np.clip(action[0], -1, 1))
        boost = float(action[1]) > 0.5
        turn_rate = max(0.04, BASE_TURN_RATE / (1 + math.log10(
            snake.mass / START_MASS) * 0.3))
        snake.target_angle = snake.angle + steering * turn_rate
        snake.is_boosting = boost and snake.mass > (START_MASS + 10)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.pending_kill_mass = 0
        self.peak_mass = START_MASS
        self.food_eaten = 0
        self.boost_frames = 0
        self.wall_close_frames = 0
        self.death_cause = 'alive'

        px, py = self._random_point()
        self.player = SnakeEntity(px, py, is_player=True)
        self.player.role = 'player'
        self.snakes = [self.player]

        for btype in bot_ai.BOT_TYPES:
            bx, by = self._random_point()
            s = SnakeEntity(bx, by, bot_type=btype)
            s.role = 'scripted'
            self.snakes.append(s)

        self._selfplay_indices = []
        for _ in range(self.num_selfplay):
            bx, by = self._random_point()
            s = SnakeEntity(bx, by)
            s.role = 'selfplay'
            s.color = (255, 215, 0)
            self._selfplay_indices.append(len(self.snakes))
            self.snakes.append(s)

        self.foods = []
        for _ in range(self.food_count):
            fx, fy = self._random_point(margin=0)
            self.foods.append(FoodItem(fx, fy))

        self.prev_mass = self.player.mass
        return self._make_obs_for(self.player), {}

    def step(self, action):
        self.step_count += 1
        self.pending_kill_mass = 0

        self._apply_action(self.player, action)
        if self.player.is_boosting:
            self.boost_frames += 1
        if self.player.is_boosting and self.player.mass > START_MASS + 10:
            if random.random() < 0.2:
                tail = self.player.segments[-1]
                vx = -math.cos(self.player.angle) * 4 + random.uniform(-1, 1)
                vy = -math.sin(self.player.angle) * 4 + random.uniform(-1, 1)
                self.foods.append(FoodItem(tail[0], tail[1], value=1, vx=vx, vy=vy))

        for food in self.foods:
            food.update()

        self.food_grid.clear()
        for food in self.foods:
            self.food_grid.insert(food, food.x, food.y)
        self.segment_grid.clear()
        for snake in self.snakes:
            if snake.dead:
                continue
            for seg in snake.segments[3:]:
                self.segment_grid.insert((snake, seg), seg[0], seg[1])

        # Self-play agents — skip obs generation when no policies loaded
        for idx in self._selfplay_indices:
            snake = self.snakes[idx]
            if snake.dead:
                continue
            if self._selfplay_policies:
                policy = random.choice(self._selfplay_policies)
                obs = self._make_obs_for(snake)
                sp_action, _ = policy.predict(obs, deterministic=False)
                self._apply_action(snake, sp_action)
            else:
                snake.target_angle += random.uniform(-0.15, 0.15)

        for snake in self.snakes:
            if snake.role == 'scripted' and not snake.dead:
                bot_ai.update(snake, self.foods, self.snakes, self.world_radius)

        for snake in self.snakes:
            snake.update()
            if not snake.dead:
                if math.hypot(snake.head[0], snake.head[1]) > self.world_radius:
                    snake.dead = True
                    self._explode_snake(snake)

        # Head-to-body collision
        for sa in self.snakes:
            if sa.dead:
                continue
            nearby = self.segment_grid.query(sa.head[0], sa.head[1], 50)
            for (owner, seg) in nearby:
                if owner is sa or owner.dead:
                    continue
                dist = math.hypot(sa.head[0] - seg[0], sa.head[1] - seg[1])
                if dist < owner.radius * 0.8:
                    sa.dead = True
                    owner.kills += 1
                    if sa.is_player:
                        self.death_cause = 'collision'
                    if owner.is_player:
                        self.pending_kill_mass += sa.mass
                    self._explode_snake(sa)
                    break

        # Food collision
        eaten = []
        for snake in self.snakes:
            if snake.dead:
                continue
            nearby_food = self.food_grid.query(snake.head[0], snake.head[1], snake.radius + 10)
            for food in nearby_food:
                dist = math.hypot(snake.head[0] - food.x, snake.head[1] - food.y)
                if dist < snake.radius + food.radius:
                    eaten.append(food)
                    snake.mass += food.value * MASS_PER_FOOD
                    if snake.is_player:
                        self.food_eaten += 1
        for food in eaten:
            if food in self.foods:
                self.foods.remove(food)

        while len(self.foods) < self.food_count:
            fx, fy = self._random_point(margin=0)
            self.foods.append(FoodItem(fx, fy))

        for i, snake in enumerate(self.snakes):
            if snake.dead and not snake.is_player:
                bx, by = self._random_point()
                role = snake.role
                new_snake = SnakeEntity(bx, by)
                new_snake.role = role
                if role == 'selfplay':
                    new_snake.color = (255, 215, 0)
                self.snakes[i] = new_snake

        if not self.player.dead:
            if self.player.mass > self.peak_mass:
                self.peak_mass = self.player.mass
            wall_dist = self.world_radius - math.hypot(
                self.player.head[0], self.player.head[1])
            if wall_dist < 200:
                self.wall_close_frames += 1

        reward = self._compute_reward()
        terminated = self.player.dead
        truncated = self.step_count >= self.max_steps

        if terminated and self.death_cause == 'alive':
            self.death_cause = 'wall'
        if truncated and not terminated:
            self.death_cause = 'survived'

        obs = self._make_obs_for(self.player)
        mass_growth_rate = (self.player.mass - START_MASS) / max(self.step_count, 1)
        info = {
            'mass': self.player.mass if not self.player.dead else 0,
            'peak_mass': self.peak_mass,
            'kills': self.player.kills,
            'step': self.step_count,
            'food_eaten': self.food_eaten,
            'mass_per_frame': mass_growth_rate,
            'boost_pct': self.boost_frames / max(self.step_count, 1),
            'wall_close_pct': self.wall_close_frames / max(self.step_count, 1),
            'death_cause': self.death_cause,
        }
        self.prev_mass = self.player.mass
        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        reward = 0.0
        mass_delta = (self.player.mass - self.prev_mass) / 100.0
        reward += mass_delta
        if self.pending_kill_mass > 0:
            reward += self.pending_kill_mass / 500.0
        reward += 0.001
        if self.player.dead:
            return -10.0
        dist_to_wall = self.world_radius - math.hypot(
            self.player.head[0], self.player.head[1])
        if dist_to_wall < 200:
            reward -= 0.01 * (1.0 - dist_to_wall / 200.0)
        return reward

    def _explode_snake(self, snake):
        cx = sum(s[0] for s in snake.segments) / len(snake.segments)
        cy = sum(s[1] for s in snake.segments) / len(snake.segments)
        for seg in snake.segments:
            if random.random() < 0.5:
                dx, dy = seg[0] - cx, seg[1] - cy
                d = math.hypot(dx, dy) + 0.1
                vx = (dx / d) * random.uniform(2, 5)
                vy = (dy / d) * random.uniform(2, 5)
                self.foods.append(FoodItem(seg[0], seg[1], value=3,
                                          color=snake.color, vx=vx, vy=vy))

    def render(self):
        if self.render_mode != 'human':
            return
        import pygame
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("SlitherEnv Training View")
            self._clock = pygame.time.Clock()
        self._screen.fill((20, 20, 20))
        if not self.player.dead:
            cam_x = self.player.head[0] - 400
            cam_y = self.player.head[1] - 300
            for food in self.foods:
                fx, fy = int(food.x - cam_x), int(food.y - cam_y)
                if 0 <= fx < 800 and 0 <= fy < 600:
                    pygame.draw.circle(self._screen, (100, 255, 100), (fx, fy), 3)
            for snake in self.snakes:
                if snake.dead:
                    continue
                color = (0, 200, 255) if snake.is_player else snake.color
                for seg in snake.segments:
                    sx, sy = int(seg[0] - cam_x), int(seg[1] - cam_y)
                    pygame.draw.circle(self._screen, color, (sx, sy),
                                     max(2, int(snake.radius * 0.8)))
        pygame.display.flip()
        self._clock.tick(60)

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None


if __name__ == '__main__':
    print("Creating SlitherEnv (self-play)...")
    env = SlitherEnv(num_scripted=9, num_selfplay=6)

    from stable_baselines3.common.env_checker import check_env
    check_env(env)
    print("✓ check_env passed!")

    print(f"\nSnake composition: {len(env.snakes)} total")
    roles = {}
    for s in env.snakes:
        roles[s.role] = roles.get(s.role, 0) + 1
    for role, count in roles.items():
        print(f"  {role}: {count}")

    obs, _ = env.reset()
    for _ in range(500):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            obs, _ = env.reset()
    print(f"✓ 500 steps. Mass: {info['mass']:.0f}, Kills: {info['kills']}")
    env.close()
