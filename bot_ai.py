"""
Bot AI Personalities for Slither.io Engine.

Each function takes (snake, foods, snakes, world_size, segment_grid) and mutates
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

def _angle_diff(a, b):
    """Signed shortest difference between angles a and b."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d


def nearest_food(snake, foods, max_dist=400, min_value=0):
    """Find best food within max_dist, scored by value / distance."""
    best, best_score = None, -1
    for f in foods:
        if f.value < min_value:
            continue
        d = math.hypot(f.x - snake.head[0], f.y - snake.head[1])
        if d < max_dist and d > 0:
            score = f.value / d
            if score > best_score:
                best_score = score
                best = f
    return best


def best_food_cluster(snake, foods, max_dist=500, top_n=8):
    """Find the centroid of the densest nearby food cluster."""
    scored = []
    for f in foods:
        d = math.hypot(f.x - snake.head[0], f.y - snake.head[1])
        if 0 < d < max_dist:
            scored.append((f, d))
    if not scored:
        return None, None
    scored.sort(key=lambda x: x[1])
    cluster = scored[:top_n]
    cx = sum(f.x for f, _ in cluster) / len(cluster)
    cy = sum(f.y for f, _ in cluster) / len(cluster)
    cd = math.hypot(cx - snake.head[0], cy - snake.head[1])
    return (cx, cy), cd


def avoid_walls(snake, world_size, hard_margin=150, soft_margin=300):
    """Graduated wall avoidance. Returns (angle, urgency) or (None, 0).
    urgency: 0 = safe, 0-1 = soft nudge, 1 = hard turn needed."""
    dist = math.hypot(snake.head[0], snake.head[1])
    if dist > world_size - hard_margin:
        return math.atan2(-snake.head[1], -snake.head[0]), 1.0
    if dist > world_size - soft_margin:
        urgency = (dist - (world_size - soft_margin)) / (soft_margin - hard_margin)
        return math.atan2(-snake.head[1], -snake.head[0]), urgency
    return None, 0.0


def nearest_threat(snake, snakes, radius=150):
    """Find closest other snake head within radius, weighted by closing speed."""
    best, best_score = None, float('inf')
    best_dist = float('inf')
    for s in snakes:
        if s is snake or s.dead:
            continue
        dx = s.head[0] - snake.head[0]
        dy = s.head[1] - snake.head[1]
        d = math.hypot(dx, dy)
        if d < radius and d > 0:
            # How fast is this snake closing in on us?
            closing_vx = math.cos(s.angle) * s.get_speed()
            closing_vy = math.sin(s.angle) * s.get_speed()
            # Dot product of their velocity with the direction toward us
            dir_x, dir_y = -dx / d, -dy / d
            closing_speed = closing_vx * dir_x + closing_vy * dir_y
            # Score: lower = more dangerous. Subtract closing speed bonus
            score = d - max(0, closing_speed) * 15
            if score < best_score:
                best_score = score
                best = s
                best_dist = d
    return best, best_dist


def dodge_bodies(snake, snakes, segment_grid, radius=100):
    """Scan for enemy body segments in a forward cone and return a dodge angle.
    Returns (dodge_angle, danger_dist) or (None, inf)."""
    if segment_grid is None:
        return _dodge_bodies_fallback(snake, snakes, radius)

    nearby = segment_grid.query(snake.head[0], snake.head[1], radius)
    best_seg = None
    best_dist = float('inf')

    for owner, seg in nearby:
        if owner is snake or owner.dead:
            continue
        dx = seg[0] - snake.head[0]
        dy = seg[1] - snake.head[1]
        d = math.hypot(dx, dy)
        if d < 1 or d > radius:
            continue
        # Only care about segments in our forward cone (~120°)
        angle_to_seg = math.atan2(dy, dx)
        angle_diff = abs(_angle_diff(angle_to_seg, snake.angle))
        if angle_diff < math.pi / 3:  # 60° each side = 120° cone
            if d < best_dist:
                best_dist = d
                best_seg = seg

    if best_seg is None:
        return None, float('inf')

    # Steer perpendicular to the direction of the obstacle
    dx = best_seg[0] - snake.head[0]
    dy = best_seg[1] - snake.head[1]
    angle_to = math.atan2(dy, dx)

    # Choose the perpendicular direction that requires less turning
    perp_left = angle_to + math.pi / 2
    perp_right = angle_to - math.pi / 2
    if abs(_angle_diff(perp_left, snake.angle)) < abs(_angle_diff(perp_right, snake.angle)):
        dodge_angle = perp_left
    else:
        dodge_angle = perp_right

    return dodge_angle, best_dist


def _dodge_bodies_fallback(snake, snakes, radius=100):
    """Fallback body dodge without spatial hash — O(n*m) but still functional."""
    best_seg = None
    best_dist = float('inf')
    for s in snakes:
        if s is snake or s.dead:
            continue
        for seg in s.segments[3:]:
            dx = seg[0] - snake.head[0]
            dy = seg[1] - snake.head[1]
            d = math.hypot(dx, dy)
            if d < 1 or d > radius:
                continue
            angle_to_seg = math.atan2(dy, dx)
            angle_diff = abs(_angle_diff(angle_to_seg, snake.angle))
            if angle_diff < math.pi / 3:
                if d < best_dist:
                    best_dist = d
                    best_seg = seg
    if best_seg is None:
        return None, float('inf')
    dx = best_seg[0] - snake.head[0]
    dy = best_seg[1] - snake.head[1]
    angle_to = math.atan2(dy, dx)
    perp_left = angle_to + math.pi / 2
    perp_right = angle_to - math.pi / 2
    if abs(_angle_diff(perp_left, snake.angle)) < abs(_angle_diff(perp_right, snake.angle)):
        return perp_left, best_dist
    return perp_right, best_dist


def _blend_angle(current, target, weight):
    """Blend current angle toward target by weight (0 = keep current, 1 = full target)."""
    diff = _angle_diff(target, current)
    return current + diff * weight


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


def update(snake, foods, snakes, world_size, segment_grid=None):
    """Main entry: dispatch to the correct personality function."""
    if snake.dead:
        return
    fn = _AI_MAP.get(snake.bot_type, ai_random)
    fn(snake, foods, snakes, world_size, segment_grid)


# ──────────────────────────── Personalities ────────────────────────────

def ai_random(snake, foods, snakes, world_size, segment_grid=None):
    """Random: Wanders but grabs nearby food and dodges bodies."""
    wall_angle, wall_urg = avoid_walls(snake, world_size)
    if wall_urg > 0.5:
        snake.target_angle = wall_angle
        snake.is_boosting = False
        return

    # Dodge body segments
    dodge, d_dist = dodge_bodies(snake, snakes, segment_grid, radius=80)
    if dodge is not None and d_dist < 60:
        snake.target_angle = dodge
        snake.is_boosting = False
        return

    # Opportunistic food grab
    food = nearest_food(snake, foods, max_dist=150)
    if food:
        snake.target_angle = math.atan2(food.y - snake.head[1], food.x - snake.head[0])
        snake.is_boosting = False
        return

    # Wander
    snake._wander_timer -= 1
    if snake._wander_timer <= 0:
        snake.target_angle = random.uniform(0, 2 * math.pi)
        snake._wander_timer = random.randint(30, 120)

    # Soft wall nudge
    if wall_angle is not None and wall_urg > 0:
        snake.target_angle = _blend_angle(snake.target_angle, wall_angle, wall_urg)
    snake.is_boosting = False


def ai_forager(snake, foods, snakes, world_size, segment_grid=None):
    """Forager: Efficient food collector, flees only from closing threats."""
    wall_angle, wall_urg = avoid_walls(snake, world_size)
    if wall_urg > 0.7:
        snake.target_angle = wall_angle
        snake.is_boosting = wall_urg > 0.9
        return

    # Dodge body segments first
    dodge, d_dist = dodge_bodies(snake, snakes, segment_grid, radius=100)
    if dodge is not None and d_dist < 70:
        snake.target_angle = dodge
        snake.is_boosting = d_dist < 40
        return

    # Only flee from threats that are actually closing in
    threat, t_dist = nearest_threat(snake, snakes, radius=180)
    if threat and t_dist < 150:
        # Check if threat is actually heading toward us
        dx = snake.head[0] - threat.head[0]
        dy = snake.head[1] - threat.head[1]
        d = math.hypot(dx, dy)
        if d > 0:
            dir_x, dir_y = dx / d, dy / d
            threat_vx = math.cos(threat.angle) * threat.get_speed()
            threat_vy = math.sin(threat.angle) * threat.get_speed()
            closing = threat_vx * (-dir_x) + threat_vy * (-dir_y)
            if closing > 1.0:  # Actually approaching us
                snake.target_angle = math.atan2(dy, dx)
                snake.is_boosting = t_dist < 80
                return

    # Seek best food cluster
    cluster, c_dist = best_food_cluster(snake, foods, max_dist=600)
    food = nearest_food(snake, foods, max_dist=300)
    if food:
        snake.target_angle = math.atan2(food.y - snake.head[1], food.x - snake.head[0])
        snake.is_boosting = False
    elif cluster:
        snake.target_angle = math.atan2(cluster[1] - snake.head[1], cluster[0] - snake.head[0])
        snake.is_boosting = False
    else:
        ai_random(snake, foods, snakes, world_size, segment_grid)

    # Soft wall nudge
    if wall_angle is not None and wall_urg > 0:
        snake.target_angle = _blend_angle(snake.target_angle, wall_angle, wall_urg)


def ai_bully(snake, foods, snakes, world_size, segment_grid=None):
    """Bully: Aggressively cuts off other snakes' paths with lead targeting."""
    wall_angle, wall_urg = avoid_walls(snake, world_size)
    if wall_urg > 0.6:
        snake.target_angle = wall_angle
        snake.is_boosting = False
        return

    # Dodge bodies while chasing
    dodge, d_dist = dodge_bodies(snake, snakes, segment_grid, radius=80)
    if dodge is not None and d_dist < 50:
        snake.target_angle = dodge
        snake.is_boosting = False
        return

    target, t_dist = nearest_threat(snake, snakes, radius=450)
    if target and t_dist < 350:
        # Lead targeting — scale lookahead by distance
        lookahead = max(10, min(40, t_dist / target.get_speed()))
        future_x = target.head[0] + math.cos(target.angle) * target.get_speed() * lookahead
        future_y = target.head[1] + math.sin(target.angle) * target.get_speed() * lookahead

        intercept_dist = math.hypot(future_x - snake.head[0], future_y - snake.head[1])

        if t_dist < 100 and intercept_dist < 200:
            # Close range: orbit to cut off
            angle_to = math.atan2(target.head[1] - snake.head[1],
                                  target.head[0] - snake.head[0])
            snake.target_angle = angle_to + math.pi / 4
            snake.is_boosting = True
        else:
            snake.target_angle = math.atan2(
                future_y - snake.head[1],
                future_x - snake.head[0])
            snake.is_boosting = intercept_dist < 150 and t_dist < 250
    else:
        food = nearest_food(snake, foods)
        if food:
            snake.target_angle = math.atan2(food.y - snake.head[1], food.x - snake.head[0])
        snake.is_boosting = False

    if wall_angle is not None and wall_urg > 0:
        snake.target_angle = _blend_angle(snake.target_angle, wall_angle, wall_urg)


def ai_scavenger(snake, foods, snakes, world_size, segment_grid=None):
    """Scavenger: Hunts for death drops, gravitates toward nearby fights."""
    wall_angle, wall_urg = avoid_walls(snake, world_size)
    if wall_urg > 0.5:
        snake.target_angle = wall_angle
        snake.is_boosting = False
        return

    dodge, d_dist = dodge_bodies(snake, snakes, segment_grid, radius=100)
    if dodge is not None and d_dist < 60:
        snake.target_angle = dodge
        snake.is_boosting = d_dist < 35
        return

    # Flee from very close threats
    threat, t_dist = nearest_threat(snake, snakes, radius=100)
    if threat and t_dist < 70:
        snake.target_angle = math.atan2(
            snake.head[1] - threat.head[1],
            snake.head[0] - threat.head[0])
        snake.is_boosting = True
        return

    # Priority 1: high-value food (death drops)
    jackpot = nearest_food(snake, foods, max_dist=800, min_value=2)
    if jackpot:
        snake.target_angle = math.atan2(jackpot.y - snake.head[1], jackpot.x - snake.head[0])
        jd = math.hypot(jackpot.x - snake.head[0], jackpot.y - snake.head[1])
        snake.is_boosting = jd < 200 and snake.mass > 70
        return

    # Priority 2: gravitate toward fights (pairs of close snakes)
    best_fight = None
    best_fight_dist = float('inf')
    for i, s1 in enumerate(snakes):
        if s1 is snake or s1.dead:
            continue
        for s2 in snakes[i+1:]:
            if s2 is snake or s2.dead:
                continue
            pair_d = math.hypot(s1.head[0] - s2.head[0], s1.head[1] - s2.head[1])
            if pair_d < 200:  # They're fighting
                mid_x = (s1.head[0] + s2.head[0]) / 2
                mid_y = (s1.head[1] + s2.head[1]) / 2
                my_d = math.hypot(mid_x - snake.head[0], mid_y - snake.head[1])
                if my_d < 600 and my_d < best_fight_dist:
                    best_fight = (mid_x, mid_y)
                    best_fight_dist = my_d
    if best_fight:
        snake.target_angle = math.atan2(best_fight[1] - snake.head[1],
                                        best_fight[0] - snake.head[0])
        snake.is_boosting = False
        return

    # Drift toward center with food grabbing
    food = nearest_food(snake, foods, max_dist=200)
    if food:
        snake.target_angle = math.atan2(food.y - snake.head[1], food.x - snake.head[0])
    else:
        # Wander to prevent small tight loops
        snake._wander_timer -= 1
        if snake._wander_timer <= 0:
            snake.target_angle = random.uniform(0, 2 * math.pi)
            snake._wander_timer = random.randint(30, 90)
            
        snake.target_angle = _blend_angle(snake.target_angle, math.atan2(-snake.head[1], -snake.head[0]), 0.05)
    snake.is_boosting = False


def ai_patrol(snake, foods, snakes, world_size, segment_grid=None):
    """Patrol: Sweeps edges with randomized radius, eats along the way."""
    # Randomize patrol radius per snake (stable via _patrol_idx seed)
    r = world_size * random.uniform(0.6, 0.85) if snake._wander_timer == 0 else world_size * 0.75
    if snake._wander_timer == 0:
        snake._wander_timer = 1  # Mark as initialized
    n_waypoints = 6
    waypoints = [(r * math.cos(2 * math.pi * i / n_waypoints),
                  r * math.sin(2 * math.pi * i / n_waypoints))
                 for i in range(n_waypoints)]

    # Dodge bodies
    dodge, d_dist = dodge_bodies(snake, snakes, segment_grid, radius=90)
    if dodge is not None and d_dist < 60:
        snake.target_angle = dodge
        snake.is_boosting = d_dist < 35
        return

    # Flee from closing threats
    threat, t_dist = nearest_threat(snake, snakes, radius=250)
    if threat and t_dist < 160:
        dx = snake.head[0] - threat.head[0]
        dy = snake.head[1] - threat.head[1]
        enemy_vx = math.cos(threat.angle) * threat.get_speed()
        enemy_vy = math.sin(threat.angle) * threat.get_speed()
        flee_x = dx - enemy_vx * 10
        flee_y = dy - enemy_vy * 10
        snake.target_angle = math.atan2(flee_y, flee_x)
        snake.is_boosting = t_dist < 100
        return

    # Navigate waypoints
    wx, wy = waypoints[snake._patrol_idx % len(waypoints)]
    if math.hypot(wx - snake.head[0], wy - snake.head[1]) < 150:
        snake._patrol_idx = (snake._patrol_idx + 1) % len(waypoints)

    snake.target_angle = math.atan2(wy - snake.head[1], wx - snake.head[0])

    # Grab nearby food while patrolling
    food = nearest_food(snake, foods, max_dist=150)
    if food:
        snake.target_angle = math.atan2(food.y - snake.head[1], food.x - snake.head[0])

    # Graduated wall avoidance
    wall_angle, wall_urg = avoid_walls(snake, world_size)
    if wall_angle is not None and wall_urg > 0:
        snake.target_angle = _blend_angle(snake.target_angle, wall_angle, wall_urg)

    snake.is_boosting = False


def ai_parasite(snake, foods, snakes, world_size, segment_grid=None):
    """Parasite: Shadows the biggest snake's tail, scoops dropped food."""
    wall_angle, wall_urg = avoid_walls(snake, world_size)
    if wall_urg > 0.6:
        snake.target_angle = wall_angle
        snake.is_boosting = False
        return

    dodge, d_dist = dodge_bodies(snake, snakes, segment_grid, radius=90)
    if dodge is not None and d_dist < 50:
        snake.target_angle = dodge
        snake.is_boosting = False
        return

    biggest, biggest_mass = None, 0
    for s in snakes:
        if s is not snake and not s.dead and s.mass > biggest_mass:
            biggest = s
            biggest_mass = s.mass

    if biggest and biggest.mass > snake.mass * 0.5:
        tail = biggest.segments[-1]
        tail_dist = math.hypot(tail[0] - snake.head[0], tail[1] - snake.head[1])

        # If the target is boosting, rush to eat their dropped food
        if biggest.is_boosting:
            dropped = nearest_food(snake, foods, max_dist=200, min_value=1)
            if dropped:
                snake.target_angle = math.atan2(dropped.y - snake.head[1],
                                                dropped.x - snake.head[0])
                snake.is_boosting = True
                return

        # Follow at a safe distance (not too close, not too far)
        if tail_dist < biggest.radius * 2.5:
            # Too close — drift sideways
            snake.target_angle = math.atan2(tail[1] - snake.head[1],
                                            tail[0] - snake.head[0]) + math.pi / 3
        elif tail_dist > 300:
            # Too far — boost to catch up
            snake.target_angle = math.atan2(tail[1] - snake.head[1],
                                            tail[0] - snake.head[0])
            snake.is_boosting = tail_dist > 400 and snake.mass > 70
        else:
            # Good distance — follow
            snake.target_angle = math.atan2(tail[1] - snake.head[1],
                                            tail[0] - snake.head[0])
            snake.is_boosting = False

        # Grab food along the way
        food = nearest_food(snake, foods, max_dist=80)
        if food:
            snake.target_angle = math.atan2(food.y - snake.head[1], food.x - snake.head[0])
            snake.is_boosting = False
        return

    ai_forager(snake, foods, snakes, world_size, segment_grid)


def ai_trapper(snake, foods, snakes, world_size, segment_grid=None):
    """Trapper: When big enough, circles smaller snakes with tight orbits."""
    wall_angle, wall_urg = avoid_walls(snake, world_size)
    if wall_urg > 0.5:
        snake.target_angle = wall_angle
        snake.is_boosting = False
        return

    dodge, d_dist = dodge_bodies(snake, snakes, segment_grid, radius=80)
    if dodge is not None and d_dist < 45:
        snake.target_angle = dodge
        snake.is_boosting = False
        return

    # Lower threshold to 150 mass
    if snake.mass > 150:
        best_prey, best_dist = None, float('inf')
        for s in snakes:
            if s is snake or s.dead:
                continue
            if s.mass < snake.mass * 0.5:
                d = math.hypot(s.head[0] - snake.head[0], s.head[1] - snake.head[1])
                if d < 350 and d < best_dist:
                    best_prey = s
                    best_dist = d

        if best_prey:
            dx = best_prey.head[0] - snake.head[0]
            dy = best_prey.head[1] - snake.head[1]
            angle_to_prey = math.atan2(dy, dx)

            # Tighter circling at close range
            if best_dist < 80:
                offset = math.pi / 2  # Full perpendicular — tight circle
            elif best_dist < 150:
                offset = math.pi / 2.5
            else:
                offset = math.pi / 3

            # Detect if prey is escaping — reverse orbit direction
            prey_heading = best_prey.angle
            orbit_dir = _angle_diff(prey_heading, angle_to_prey)
            if orbit_dir > 0:
                snake.target_angle = angle_to_prey + offset
            else:
                snake.target_angle = angle_to_prey - offset

            snake.is_boosting = best_dist < 120 and snake.mass > 200
            return

    # Fallback: eat food, flee threats
    food = nearest_food(snake, foods)
    if food:
        snake.target_angle = math.atan2(food.y - snake.head[1], food.x - snake.head[0])

    threat, t_dist = nearest_threat(snake, snakes, radius=120)
    if threat and t_dist < 80 and threat.mass > snake.mass:
        snake.target_angle = math.atan2(
            snake.head[1] - threat.head[1],
            snake.head[0] - threat.head[0])
    snake.is_boosting = False


def ai_interceptor(snake, foods, snakes, world_size, segment_grid=None):
    """Interceptor: Adaptive lookahead, aborts if path is body-blocked."""
    wall_angle, wall_urg = avoid_walls(snake, world_size)
    if wall_urg > 0.5:
        snake.target_angle = wall_angle
        snake.is_boosting = False
        return

    dodge, d_dist = dodge_bodies(snake, snakes, segment_grid, radius=90)
    if dodge is not None and d_dist < 55:
        snake.target_angle = dodge
        snake.is_boosting = False
        return

    best_target, best_score = None, float('inf')
    best_future = None
    for s in snakes:
        if s is snake or s.dead:
            continue
        d = math.hypot(s.head[0] - snake.head[0], s.head[1] - snake.head[1])
        if d < 400:
            # Adaptive lookahead — closer = shorter prediction needed
            lookahead = max(10, min(35, d / s.get_speed()))
            fx = s.head[0] + math.cos(s.angle) * s.get_speed() * lookahead
            fy = s.head[1] + math.sin(s.angle) * s.get_speed() * lookahead
            intercept_d = math.hypot(fx - snake.head[0], fy - snake.head[1])
            if intercept_d < best_score:
                best_target = s
                best_score = intercept_d
                best_future = (fx, fy)

    if best_target and best_score < 300 and best_future:
        # Check if the intercept path is body-blocked
        mid_x = (snake.head[0] + best_future[0]) / 2
        mid_y = (snake.head[1] + best_future[1]) / 2
        block_dodge, block_dist = dodge_bodies(snake, snakes, segment_grid, radius=60)

        if block_dodge is not None and block_dist < 40:
            # Path is blocked — abort intercept, dodge instead
            snake.target_angle = block_dodge
            snake.is_boosting = False
        else:
            snake.target_angle = math.atan2(best_future[1] - snake.head[1],
                                            best_future[0] - snake.head[0])
            snake.is_boosting = best_score < 120
    else:
        food = nearest_food(snake, foods)
        if food:
            snake.target_angle = math.atan2(food.y - snake.head[1], food.x - snake.head[0])
        snake.is_boosting = False

    if wall_angle is not None and wall_urg > 0:
        snake.target_angle = _blend_angle(snake.target_angle, wall_angle, wall_urg)


def ai_hunter(snake, foods, snakes, world_size, segment_grid=None):
    """Hunter: Pursues snakes ≤1/2 its size with multi-step heading correction."""
    wall_angle, wall_urg = avoid_walls(snake, world_size)
    if wall_urg > 0.5:
        snake.target_angle = wall_angle
        snake.is_boosting = False
        return

    dodge, d_dist = dodge_bodies(snake, snakes, segment_grid, radius=90)
    if dodge is not None and d_dist < 55:
        snake.target_angle = dodge
        snake.is_boosting = d_dist < 30
        return

    # Expanded target range: ≤1/2 mass (from 1/3)
    best_prey, best_dist = None, float('inf')
    for s in snakes:
        if s is snake or s.dead:
            continue
        if s.mass <= snake.mass / 2:
            d = math.hypot(s.head[0] - snake.head[0], s.head[1] - snake.head[1])
            if d < 400 and d < best_dist:
                best_prey = s
                best_dist = d

    if best_prey:
        # Multi-step prediction: adjust lookahead based on closing rate
        lookahead = max(10, min(30, best_dist / snake.get_speed()))
        fx = best_prey.head[0] + math.cos(best_prey.angle) * best_prey.get_speed() * lookahead
        fy = best_prey.head[1] + math.sin(best_prey.angle) * best_prey.get_speed() * lookahead
        snake.target_angle = math.atan2(fy - snake.head[1], fx - snake.head[0])
        snake.is_boosting = best_dist < 200 and snake.mass > 80
        if wall_angle is not None and wall_urg > 0:
            snake.target_angle = _blend_angle(snake.target_angle, wall_angle, wall_urg)
        return

    # Flee from bigger threats
    threat, t_dist = nearest_threat(snake, snakes, radius=180)
    if threat and threat.mass > snake.mass:
        snake.target_angle = math.atan2(
            snake.head[1] - threat.head[1],
            snake.head[0] - threat.head[0])
        snake.is_boosting = t_dist < 80
        return

    # Forage
    food = nearest_food(snake, foods)
    if food:
        snake.target_angle = math.atan2(food.y - snake.head[1], food.x - snake.head[0])
    snake.is_boosting = False

    if wall_angle is not None and wall_urg > 0:
        snake.target_angle = _blend_angle(snake.target_angle, wall_angle, wall_urg)


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
