"""
Bot AI Personalities for Slither.io Engine.

Each function takes (snake, foods, snakes, world_size) and mutates
snake.target_angle and snake.is_boosting directly.
"""
import math
import random

BOT_TYPES = ['random', 'forager', 'bully', 'scavenger', 'patrol',
             'parasite', 'trapper', 'interceptor', 'hunter']

BOT_COLORS = {
    'random': (180, 180, 180),
    'forager': (80, 255, 80),
    'bully': (255, 80, 80),
    'scavenger': (255, 165, 0),
    'patrol': (100, 100, 255),
    'parasite': (255, 80, 255),
    'trapper': (200, 50, 50),
    'interceptor': (80, 255, 255),
    'hunter': (255, 255, 80),
}

# ──────────────────────────── Helpers ────────────────────────────

def nearest_food(snake, foods, max_dist=400, min_value=0):
    """Find closest food within max_dist, optionally filtering by value."""
    best, best_d = None, float('inf')
    for f in foods:
        if f.value < min_value:
            continue
        d = math.hypot(f.x - snake.head[0], f.y - snake.head[1])
        if d < best_d and d < max_dist:
            best_d = d
            best = f
    return best, best_d


def avoid_walls(snake, world_size, margin=150):
    """Returns an escape angle if near the circular boundary, else None."""
    dist = math.hypot(snake.head[0], snake.head[1])
    if dist > world_size - margin:
        return math.atan2(-snake.head[1], -snake.head[0])
    return None


def nearest_threat(snake, snakes, radius=150):
    """Find the closest other snake head within radius."""
    best, best_d = None, float('inf')
    for s in snakes:
        if s is snake or s.dead:
            continue
        d = math.hypot(s.head[0] - snake.head[0], s.head[1] - snake.head[1])
        if d < best_d and d < radius:
            best_d = d
            best = s
    return best, best_d


def any_body_nearby(snake, snakes, radius=100):
    """Check if any enemy body segment is within radius."""
    for s in snakes:
        if s is snake or s.dead:
            continue
        for seg in s.segments:
            d = math.hypot(seg[0] - snake.head[0], seg[1] - snake.head[1])
            if d < radius:
                return s, d
    return None, float('inf')


# ──────────────────────────── Dispatch ────────────────────────────

_AI_MAP = {}  # populated after function definitions


def update(snake, foods, snakes, world_size):
    """Main entry: dispatch to the correct personality function."""
    if snake.dead:
        return
    fn = _AI_MAP.get(snake.bot_type, ai_random)
    fn(snake, foods, snakes, world_size)


# ──────────────────────────── Personalities ────────────────────────────

def ai_random(snake, foods, snakes, world_size):
    """Random: Wanders aimlessly with random angle changes."""
    wall = avoid_walls(snake, world_size)
    if wall is not None:
        snake.target_angle = wall
    else:
        snake._wander_timer -= 1
        if snake._wander_timer <= 0:
            snake.target_angle = random.uniform(0, 2 * math.pi)
            snake._wander_timer = random.randint(30, 120)
    snake.is_boosting = False


def ai_forager(snake, foods, snakes, world_size):
    """Forager (Scared): Eats nearest food, panics near any snake."""
    wall = avoid_walls(snake, world_size, margin=200)
    if wall is not None:
        snake.target_angle = wall
        snake.is_boosting = True
        return

    # Panic: flee from any nearby snake
    threat, t_dist = nearest_threat(snake, snakes, radius=200)
    if threat:
        snake.target_angle = math.atan2(
            snake.head[1] - threat.head[1],
            snake.head[0] - threat.head[0])
        snake.is_boosting = t_dist < 100
        return

    # Also flee from body segments
    body_threat, b_dist = any_body_nearby(snake, snakes, radius=120)
    if body_threat:
        snake.target_angle = math.atan2(
            snake.head[1] - body_threat.head[1],
            snake.head[0] - body_threat.head[0])
        snake.is_boosting = b_dist < 60
        return

    # Otherwise eat nearest food
    food, dist = nearest_food(snake, foods, max_dist=500)
    if food:
        snake.target_angle = math.atan2(food.y - snake.head[1], food.x - snake.head[0])
        snake.is_boosting = False
    else:
        ai_random(snake, foods, snakes, world_size)


def ai_bully(snake, foods, snakes, world_size):
    """Bully: Tries to cut off other snakes' future path."""
    wall = avoid_walls(snake, world_size)
    if wall is not None:
        snake.target_angle = wall
        return

    target, t_dist = nearest_threat(snake, snakes, radius=400)
    if target and t_dist < 300:
        future_x = target.head[0] + math.cos(target.angle) * target.get_speed() * 20
        future_y = target.head[1] + math.sin(target.angle) * target.get_speed() * 20

        snake.target_angle = math.atan2(
            future_y - snake.head[1],
            future_x - snake.head[0])

        intercept_dist = math.hypot(future_x - snake.head[0], future_y - snake.head[1])
        snake.is_boosting = intercept_dist < 150 and t_dist < 200
    else:
        food, dist = nearest_food(snake, foods)
        if food:
            snake.target_angle = math.atan2(food.y - snake.head[1], food.x - snake.head[0])
        snake.is_boosting = False


def ai_scavenger(snake, foods, snakes, world_size):
    """Scavenger (Vulture): Ignores small food, waits for death drops."""
    wall = avoid_walls(snake, world_size)
    if wall is not None:
        snake.target_angle = wall
        return

    threat, t_dist = nearest_threat(snake, snakes, radius=120)
    if threat and t_dist < 80:
        snake.target_angle = math.atan2(
            snake.head[1] - threat.head[1],
            snake.head[0] - threat.head[0])
        snake.is_boosting = True
        return

    jackpot, dist = nearest_food(snake, foods, max_dist=800, min_value=2)
    if jackpot:
        snake.target_angle = math.atan2(jackpot.y - snake.head[1], jackpot.x - snake.head[0])
        snake.is_boosting = dist < 200
        return

    # Drift toward center
    snake.target_angle = math.atan2(-snake.head[1], -snake.head[0])
    snake.target_angle += random.uniform(-0.5, 0.5)
    snake.is_boosting = False


def ai_patrol(snake, foods, snakes, world_size):
    """Patrol (Recon): Sweeps outer map edges, massive threat radar."""
    r = world_size * 0.8
    waypoints = [(r, 0), (0, r), (-r, 0), (0, -r)]

    threat, t_dist = nearest_threat(snake, snakes, radius=300)
    if threat:
        enemy_vx = math.cos(threat.angle) * threat.get_speed()
        enemy_vy = math.sin(threat.angle) * threat.get_speed()
        flee_x = snake.head[0] - threat.head[0] - enemy_vx * 10
        flee_y = snake.head[1] - threat.head[1] - enemy_vy * 10
        snake.target_angle = math.atan2(flee_y, flee_x)
        snake.is_boosting = t_dist < 150
        return

    wx, wy = waypoints[snake._patrol_idx % len(waypoints)]
    if math.hypot(wx - snake.head[0], wy - snake.head[1]) < 100:
        snake._patrol_idx = (snake._patrol_idx + 1) % len(waypoints)

    snake.target_angle = math.atan2(wy - snake.head[1], wx - snake.head[0])
    snake.target_angle += random.uniform(-0.1, 0.1)

    food, fd = nearest_food(snake, foods, max_dist=100)
    if food:
        snake.target_angle = math.atan2(food.y - snake.head[1], food.x - snake.head[0])
    snake.is_boosting = False


def ai_parasite(snake, foods, snakes, world_size):
    """Parasite (Shadow): Follows the biggest snake's tail."""
    wall = avoid_walls(snake, world_size)
    if wall is not None:
        snake.target_angle = wall
        return

    biggest, biggest_mass = None, 0
    for s in snakes:
        if s is not snake and not s.dead and s.mass > biggest_mass:
            biggest = s
            biggest_mass = s.mass

    if biggest and biggest.mass > snake.mass * 0.5:
        tail = biggest.segments[-1]
        tail_dist = math.hypot(tail[0] - snake.head[0], tail[1] - snake.head[1])

        snake.target_angle = math.atan2(tail[1] - snake.head[1], tail[0] - snake.head[0])

        if tail_dist < biggest.radius * 3:
            snake.target_angle += math.pi * 0.1

        food, fd = nearest_food(snake, foods, max_dist=80)
        if food:
            snake.target_angle = math.atan2(food.y - snake.head[1], food.x - snake.head[0])

        snake.is_boosting = False
    else:
        ai_forager(snake, foods, snakes, world_size)


def ai_trapper(snake, foods, snakes, world_size):
    """Trapper (Boa): When big enough, circles smaller snakes."""
    wall = avoid_walls(snake, world_size)
    if wall is not None:
        snake.target_angle = wall
        return

    if snake.mass > 300:
        best_prey, best_dist = None, float('inf')
        for s in snakes:
            if s is snake or s.dead:
                continue
            if s.mass < snake.mass * 0.5:
                d = math.hypot(s.head[0] - snake.head[0], s.head[1] - snake.head[1])
                if d < 300 and d < best_dist:
                    best_prey = s
                    best_dist = d

        if best_prey:
            dx = best_prey.head[0] - snake.head[0]
            dy = best_prey.head[1] - snake.head[1]
            angle_to_prey = math.atan2(dy, dx)
            offset = math.pi / 2.5 if best_dist < 150 else math.pi / 3
            snake.target_angle = angle_to_prey + offset
            snake.is_boosting = best_dist < 100
            return

    food, dist = nearest_food(snake, foods)
    if food:
        snake.target_angle = math.atan2(food.y - snake.head[1], food.x - snake.head[0])

    threat, t_dist = nearest_threat(snake, snakes, radius=120)
    if threat and t_dist < 80:
        snake.target_angle = math.atan2(
            snake.head[1] - threat.head[1],
            snake.head[0] - threat.head[0])
    snake.is_boosting = False


def ai_interceptor(snake, foods, snakes, world_size):
    """Interceptor: Calculates future position and cuts off targets."""
    wall = avoid_walls(snake, world_size)
    if wall is not None:
        snake.target_angle = wall
        return

    best_target, best_score = None, float('inf')
    for s in snakes:
        if s is snake or s.dead:
            continue
        d = math.hypot(s.head[0] - snake.head[0], s.head[1] - snake.head[1])
        if d < 400:
            fx = s.head[0] + math.cos(s.angle) * s.get_speed() * 25
            fy = s.head[1] + math.sin(s.angle) * s.get_speed() * 25
            intercept_d = math.hypot(fx - snake.head[0], fy - snake.head[1])
            if intercept_d < best_score:
                best_target = s
                best_score = intercept_d

    if best_target and best_score < 300:
        fx = best_target.head[0] + math.cos(best_target.angle) * best_target.get_speed() * 25
        fy = best_target.head[1] + math.sin(best_target.angle) * best_target.get_speed() * 25
        snake.target_angle = math.atan2(fy - snake.head[1], fx - snake.head[0])
        snake.is_boosting = best_score < 120
    else:
        food, dist = nearest_food(snake, foods)
        if food:
            snake.target_angle = math.atan2(food.y - snake.head[1], food.x - snake.head[0])
        snake.is_boosting = False


def ai_hunter(snake, foods, snakes, world_size):
    """Hunter: Intercepts snakes ≤1/3 its size, otherwise forages safely."""
    wall = avoid_walls(snake, world_size)
    if wall is not None:
        snake.target_angle = wall
        return

    best_prey, best_dist = None, float('inf')
    for s in snakes:
        if s is snake or s.dead:
            continue
        if s.mass <= snake.mass / 3:
            d = math.hypot(s.head[0] - snake.head[0], s.head[1] - snake.head[1])
            if d < 350 and d < best_dist:
                best_prey = s
                best_dist = d

    if best_prey:
        fx = best_prey.head[0] + math.cos(best_prey.angle) * best_prey.get_speed() * 20
        fy = best_prey.head[1] + math.sin(best_prey.angle) * best_prey.get_speed() * 20
        snake.target_angle = math.atan2(fy - snake.head[1], fx - snake.head[0])
        snake.is_boosting = best_dist < 150
        return

    threat, t_dist = nearest_threat(snake, snakes, radius=150)
    if threat and threat.mass > snake.mass:
        snake.target_angle = math.atan2(
            snake.head[1] - threat.head[1],
            snake.head[0] - threat.head[0])
        snake.is_boosting = t_dist < 80
        return

    food, dist = nearest_food(snake, foods)
    if food:
        snake.target_angle = math.atan2(food.y - snake.head[1], food.x - snake.head[0])
    snake.is_boosting = False


# ──────────────────────────── Register ────────────────────────────

_AI_MAP = {
    'random': ai_random,
    'forager': ai_forager,
    'bully': ai_bully,
    'scavenger': ai_scavenger,
    'patrol': ai_patrol,
    'parasite': ai_parasite,
    'trapper': ai_trapper,
    'interceptor': ai_interceptor,
    'hunter': ai_hunter,
}
