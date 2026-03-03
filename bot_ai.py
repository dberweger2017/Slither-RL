"""
Bot AI Personalities for Slither.io Engine.

Each function takes (snake, foods, snakes, world_size, segment_grid) and mutates
snake.target_angle and snake.is_boosting directly.
"""
import math
import random

BOT_TYPES = ['random', 'forager', 'bully', 'scavenger', 'patrol',
             'parasite', 'trapper', 'interceptor', 'hunter', 'harvester']

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
    'harvester': (80, 220, 140),
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


def best_food_patch(snake, foods, max_dist=700, top_n=14):
    """Find a rich nearby patch using value-weighted centroid and total value."""
    candidates = []
    for f in foods:
        d = math.hypot(f.x - snake.head[0], f.y - snake.head[1])
        if 0 < d < max_dist:
            # Prefer higher value and closer food for patch construction.
            rank = d / (f.value + 0.5)
            candidates.append((rank, f, d))

    if len(candidates) < 3:
        return None, None, 0.0

    candidates.sort(key=lambda x: x[0])
    patch = candidates[:top_n]
    total_weight = sum(f.value + 0.3 for _, f, _ in patch)
    if total_weight <= 0:
        return None, None, 0.0

    cx = sum(f.x * (f.value + 0.3) for _, f, _ in patch) / total_weight
    cy = sum(f.y * (f.value + 0.3) for _, f, _ in patch) / total_weight
    cd = math.hypot(cx - snake.head[0], cy - snake.head[1])
    total_value = sum(f.value for _, f, _ in patch)
    return (cx, cy), cd, total_value


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


def _nearest_live_enemies(snake, snakes, k=6, max_dist=900):
    """Return up to k closest living opponents for local safety prediction."""
    nearby = []
    sx, sy = snake.head
    for other in snakes:
        if other is snake or other.dead:
            continue
        d = math.hypot(other.head[0] - sx, other.head[1] - sy)
        if d < max_dist:
            nearby.append((d, other))
    nearby.sort(key=lambda x: x[0])
    return [other for _, other in nearby[:k]]


def find_fight_hotspot(snake, snakes, max_pair_dist=220, max_my_dist=900):
    """
    Detect nearby fight hotspots (two heads close together).
    Returns (x, y, my_dist, score) or None.
    """
    best = None
    best_score = -1.0
    for i, s1 in enumerate(snakes):
        if s1 is snake or s1.dead:
            continue
        for s2 in snakes[i + 1:]:
            if s2 is snake or s2.dead:
                continue
            pair_d = math.hypot(s1.head[0] - s2.head[0], s1.head[1] - s2.head[1])
            if pair_d > max_pair_dist:
                continue
            mx = (s1.head[0] + s2.head[0]) * 0.5
            my = (s1.head[1] + s2.head[1]) * 0.5
            my_d = math.hypot(mx - snake.head[0], my - snake.head[1])
            if my_d > max_my_dist:
                continue

            # Favor bigger nearby fights; distance and pair tightness matter.
            mass_term = min(s1.mass, s2.mass)
            tightness = (max_pair_dist - pair_d) / max_pair_dist
            score = (mass_term / (my_d + 90.0)) + 0.45 * tightness
            if score > best_score:
                best_score = score
                best = (mx, my, my_d, score)
    return best


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


def _turn_rate_estimate(snake):
    """Approximate per-step max turn rate used by physics."""
    mass_ratio = max(0.1, snake.mass / 50.0)
    return max(0.04, 0.1 / (1.0 + math.log10(mass_ratio) * 0.3))


def _step_pose_toward(x, y, angle, target_angle, turn_rate, speed):
    """Advance one predicted step while respecting turn-rate limits."""
    diff = _angle_diff(target_angle, angle)
    if diff > turn_rate:
        angle += turn_rate
    elif diff < -turn_rate:
        angle -= turn_rate
    else:
        angle = target_angle
    x += math.cos(angle) * speed
    y += math.sin(angle) * speed
    return x, y, angle


def _predict_heading_clearance(snake, target_angle, snakes, world_size,
                               segment_grid=None, steps=12, boost=False,
                               near_enemies=None):
    """
    Simulate a short horizon and return minimum clearance.
    Negative large value means high collision risk.
    """
    turn_rate = _turn_rate_estimate(snake)
    speed = snake.get_speed()
    if boost and not snake.is_boosting:
        speed *= 2.0
    elif (not boost) and snake.is_boosting:
        speed *= 0.5

    if near_enemies is None:
        near_enemies = _nearest_live_enemies(snake, snakes, k=7, max_dist=960)
    near_set = set(near_enemies)

    x, y = snake.head[0], snake.head[1]
    angle = snake.angle
    min_clear = float('inf')
    probe_r = max(34.0, snake.radius * 1.3 + 8.0)

    for t in range(1, steps + 1):
        x, y, angle = _step_pose_toward(x, y, angle, target_angle, turn_rate, speed)

        # Wall clearance
        wall_clear = world_size - math.hypot(x, y)
        min_clear = min(min_clear, wall_clear)
        if wall_clear < snake.radius * 0.8 + 8.0:
            return -1e6

        # Body segments
        if segment_grid is not None:
            nearby = segment_grid.query(x, y, probe_r)
            seg_iter = nearby
        else:
            seg_iter = []
            for s in near_enemies:
                for seg in s.segments[3:]:
                    seg_iter.append((s, seg))

        for owner, seg in seg_iter:
            if owner.dead or owner not in near_set:
                continue
            clear = math.hypot(seg[0] - x, seg[1] - y) - (owner.radius * 0.8 + snake.radius * 0.75)
            min_clear = min(min_clear, clear)
            if clear < 0:
                return -1e6

        # Opponent head prediction catches side/head cuts.
        for other in near_enemies:
            ov = other.get_speed()
            ox = other.head[0] + math.cos(other.angle) * ov * t
            oy = other.head[1] + math.sin(other.angle) * ov * t
            head_clear = math.hypot(ox - x, oy - y) - (other.radius + snake.radius + 6.0)
            min_clear = min(min_clear, head_clear)
            if head_clear < 0:
                return -1e6

    return min_clear


def _pick_safer_heading(snake, desired_angle, desired_boost, snakes, world_size,
                        segment_grid=None, wall_angle=None, wall_urg=0.0):
    """
    Evaluate a small heading set and choose a collision-safe option
    close to desired behavior.
    """
    candidates = [
        (desired_angle, desired_boost),
        (desired_angle, False),
        (_blend_angle(snake.angle, desired_angle, 0.55), False),
        (snake.angle + 0.45, False),
        (snake.angle - 0.45, False),
        (snake.angle + 0.85, False),
        (snake.angle - 0.85, False),
        (snake.angle + 1.15, False),
        (snake.angle - 1.15, False),
    ]
    if wall_angle is not None and wall_urg > 0:
        candidates.append((_blend_angle(desired_angle, wall_angle, max(0.45, wall_urg)), False))

    best_angle = desired_angle
    best_boost = False
    best_clear = -1e6
    best_score = -1e9
    seen = set()

    near_enemies = _nearest_live_enemies(snake, snakes, k=7, max_dist=980)

    for angle, boost in candidates:
        key = (int(angle * 100), int(boost))
        if key in seen:
            continue
        seen.add(key)

        clear = _predict_heading_clearance(
            snake, angle, snakes, world_size, segment_grid=segment_grid, steps=14, boost=boost,
            near_enemies=near_enemies,
        )
        turn_cost = abs(_angle_diff(angle, desired_angle))
        boost_penalty = 6.0 if boost and clear < 20.0 else 0.0
        score = clear - 14.0 * turn_cost - boost_penalty

        if score > best_score:
            best_score = score
            best_clear = clear
            best_angle = angle
            best_boost = boost

    return best_angle, best_boost, best_clear


def _food_approach_angle(snake, target_x, target_y):
    """
    Turn-radius-aware food approach.
    For very close hard-angle food, use a softer tangent-like approach.
    """
    dx = target_x - snake.head[0]
    dy = target_y - snake.head[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return snake.angle

    angle_to = math.atan2(dy, dx)
    turn_r = max(35.0, snake.get_speed() / max(_turn_rate_estimate(snake), 1e-4))
    ang_err = _angle_diff(angle_to, snake.angle)

    if dist < turn_r * 1.25 and abs(ang_err) > math.pi * 0.45:
        side = 1 if ang_err > 0 else -1
        return snake.angle + side * min(0.9, abs(ang_err) * 0.6)

    return angle_to


def _food_near_point(foods, x, y, max_dist=90, min_value=0):
    """Pick a food orb close to a remembered lock point."""
    best, best_score = None, -1.0
    for f in foods:
        if f.value < min_value:
            continue
        d = math.hypot(f.x - x, f.y - y)
        if 0 < d < max_dist:
            score = f.value / (d + 1e-3)
            if score > best_score:
                best_score = score
                best = f
    return best


def _build_food_graph_nodes(snake, foods, max_dist=1100, max_nodes=40):
    """
    Build a local food graph:
    - nodes: attractive food points
    - node weight: orb value + local neighborhood value
    """
    sx, sy = snake.head
    prelim = []
    for f in foods:
        dx = f.x - sx
        dy = f.y - sy
        d = math.hypot(dx, dy)
        if d <= 3 or d >= max_dist:
            continue
        heading = math.atan2(dy, dx)
        turn = abs(_angle_diff(heading, snake.angle))
        rank = (f.value + 0.35) / (d + 30.0 + 45.0 * turn)
        prelim.append((rank, f, d, heading))

    if not prelim:
        return []

    prelim.sort(key=lambda x: x[0], reverse=True)
    pool = prelim[:max_nodes * 2]
    core = pool[:max_nodes]

    nodes = []
    for _, f, d, heading in core:
        local_value = 0.0
        local_count = 0
        for _, g, _, _ in pool:
            if g is f:
                continue
            pd = math.hypot(g.x - f.x, g.y - f.y)
            if pd < 150:
                local_count += 1
                local_value += g.value * max(0.0, 1.0 - pd / 170.0)

        nodes.append({
            "x": f.x,
            "y": f.y,
            "value": float(f.value),
            "dist": d,
            "heading": heading,
            "cluster_value": float(f.value + local_value),
            "cluster_count": local_count + 1,
        })

    return nodes


def _food_graph_edge_score(ax, ay, a_heading, b_node, speed, turn_rate):
    """Score edge a->b by value density over travel cost."""
    dx = b_node["x"] - ax
    dy = b_node["y"] - ay
    d = math.hypot(dx, dy)
    if d < 4 or d > 540:
        return -1e6
    heading = math.atan2(dy, dx)
    turn = abs(_angle_diff(heading, a_heading))
    travel_steps = d / max(speed, 1e-3) + 0.7 * (turn / max(turn_rate, 1e-3))
    gain = (0.8 * b_node["cluster_value"] + 0.45 * b_node["value"]) / (3.5 + travel_steps)
    return gain


def _plan_food_graph_target(snake, foods, snakes, world_size, segment_grid=None):
    """
    Graph-based 2-hop planner:
    choose the first node of the best short route in local food graph.
    """
    nodes = _build_food_graph_nodes(snake, foods, max_dist=1100, max_nodes=40)
    if not nodes:
        return None

    speed = max(0.1, snake.get_speed())
    turn_rate = max(1e-3, _turn_rate_estimate(snake))
    near_enemies = _nearest_live_enemies(snake, snakes, k=7, max_dist=980)
    best = None
    best_score = -1e9

    for i, node in enumerate(nodes):
        desired = _food_approach_angle(snake, node["x"], node["y"])
        clear = _predict_heading_clearance(
            snake, desired, snakes, world_size, segment_grid=segment_grid, steps=14, boost=False,
            near_enemies=near_enemies,
        )
        if clear < 6.0:
            continue

        turn = abs(_angle_diff(desired, snake.angle))
        travel_steps = node["dist"] / speed + turn / turn_rate
        immediate_gain = (1.15 * node["cluster_value"] + 0.55 * node["value"])
        immediate_score = immediate_gain / (4.0 + travel_steps)

        # Graph lookahead: best 2nd and 3rd hops.
        second_best = -1e6
        second_idx = -1
        for j, nxt in enumerate(nodes):
            if j == i:
                continue
            s2 = _food_graph_edge_score(node["x"], node["y"], desired, nxt, speed, turn_rate)
            if s2 > second_best:
                second_best = s2
                second_idx = j

        third_best = 0.0
        if second_idx >= 0:
            n2 = nodes[second_idx]
            h2 = math.atan2(n2["y"] - node["y"], n2["x"] - node["x"])
            for k, n3 in enumerate(nodes):
                if k == i or k == second_idx:
                    continue
                s3 = _food_graph_edge_score(n2["x"], n2["y"], h2, n3, speed, turn_rate)
                if s3 > third_best:
                    third_best = s3

        route_score = immediate_score + 0.85 * max(0.0, second_best) + 0.55 * max(0.0, third_best)
        safety_bonus = 0.035 * min(40.0, clear)

        boost = (
            (node["cluster_value"] >= 14.0 and 260.0 < node["dist"] < 900.0 and snake.mass > 85) or
            (node["value"] >= 4.0 and 150.0 < node["dist"] < 850.0 and snake.mass > 80)
        )
        if boost:
            clear_boost = _predict_heading_clearance(
                snake, desired, snakes, world_size, segment_grid=segment_grid, steps=14, boost=True
            )
            if clear_boost < 10.0:
                boost = False
            else:
                route_score += 0.35

        total = route_score + safety_bonus
        if total > best_score:
            best_score = total
            best = {
                "x": node["x"],
                "y": node["y"],
                "boost": boost,
                "lock_frames": 16 if node["cluster_count"] >= 6 else 10,
            }

    return best


def _preselect_foods_for_mpc(snake, foods, max_dist=1200, max_count=80):
    """Keep only the most relevant nearby food for MPC scoring."""
    sx, sy = snake.head
    scored = []
    for idx, f in enumerate(foods):
        dx = f.x - sx
        dy = f.y - sy
        d = math.hypot(dx, dy)
        if d <= 3 or d >= max_dist:
            continue
        heading = math.atan2(dy, dx)
        turn = abs(_angle_diff(heading, snake.angle))
        score = (f.value + 0.35) / (d + 35.0 + 45.0 * turn)
        scored.append((score, idx, f))
    if not scored:
        return []
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(idx, f) for _, idx, f in scored[:max_count]]


def _simulate_harvest_candidate(snake, snakes, world_size, segment_grid, food_candidates,
                                target_angle, boost=False, horizon=16,
                                near_enemies=None):
    """
    Roll out one short trajectory and score it by
    expected food intake - risk - boost/maneuver cost.
    """
    turn_rate = _turn_rate_estimate(snake)
    speed = snake.get_speed()
    if boost and not snake.is_boosting:
        speed *= 2.0
    elif (not boost) and snake.is_boosting:
        speed *= 0.5
    if near_enemies is None:
        near_enemies = _nearest_live_enemies(snake, snakes, k=7, max_dist=980)
    near_set = set(near_enemies)

    x, y = snake.head[0], snake.head[1]
    angle = snake.angle
    eaten = set()
    food_gain = 0.0
    risk = 0.0
    nearest_wall = float('inf')
    nearest_body = float('inf')
    nearest_head = float('inf')

    body_probe_r = max(42.0, snake.radius * 1.55 + 12.0)
    collect_r_pad = max(4.0, snake.radius * 0.15)

    for t in range(1, horizon + 1):
        x, y, angle = _step_pose_toward(x, y, angle, target_angle, turn_rate, speed)

        wall_clear = world_size - math.hypot(x, y)
        nearest_wall = min(nearest_wall, wall_clear)
        if wall_clear < snake.radius * 0.85 + 8.0:
            return -1e6
        if wall_clear < 95.0:
            risk += (95.0 - wall_clear) * 0.07

        if segment_grid is not None:
            nearby = segment_grid.query(x, y, body_probe_r)
        else:
            nearby = []
            for s in near_enemies:
                for seg in s.segments[3:]:
                    nearby.append((s, seg))

        closest_body_t = float('inf')
        for owner, seg in nearby:
            if owner.dead or owner not in near_set:
                continue
            clear = math.hypot(seg[0] - x, seg[1] - y) - (owner.radius * 0.8 + snake.radius * 0.8)
            nearest_body = min(nearest_body, clear)
            closest_body_t = min(closest_body_t, clear)
            if clear < 0:
                return -1e6
        if closest_body_t < 64.0:
            risk += (64.0 - closest_body_t) * 0.09

        closest_head_t = float('inf')
        for other in near_enemies:
            ov = other.get_speed()
            ox = other.head[0] + math.cos(other.angle) * ov * t
            oy = other.head[1] + math.sin(other.angle) * ov * t
            clear_h = math.hypot(ox - x, oy - y) - (other.radius + snake.radius + 8.0)
            nearest_head = min(nearest_head, clear_h)
            closest_head_t = min(closest_head_t, clear_h)
            if clear_h < 0:
                return -1e6
        if closest_head_t < 95.0:
            risk += (95.0 - closest_head_t) * 0.05

        for idx, f in food_candidates:
            if idx in eaten:
                continue
            d = math.hypot(f.x - x, f.y - y)
            collect_r = snake.radius + getattr(f, "radius", 4.0) + collect_r_pad
            if d < collect_r:
                eaten.add(idx)
                food_gain += f.value * 6.2
            elif d < 130.0:
                food_gain += 0.05 * f.value * (1.0 - d / 130.0)

    endpoint_gain = 0.0
    for idx, f in food_candidates:
        if idx in eaten:
            continue
        d = math.hypot(f.x - x, f.y - y)
        if d < 280.0:
            endpoint_gain += f.value * (1.0 - d / 280.0) * 0.3

    turn_cost = abs(_angle_diff(target_angle, snake.angle)) * 2.2
    boost_cost = 0.0
    if boost:
        boost_cost = horizon * max(0.5, snake.mass * 0.001) * 0.65

    safety_bonus = 0.0
    if nearest_wall > 180.0:
        safety_bonus += 0.6
    if nearest_body > 85.0 and nearest_head > 130.0:
        safety_bonus += 0.5

    return food_gain + endpoint_gain + safety_bonus - risk - turn_cost - boost_cost


def _heading_route_value(snake, food_candidates, heading):
    """Projected value along a heading corridor for boost ROI checks."""
    total = 0.0
    sx, sy = snake.head
    for _, f in food_candidates:
        dx = f.x - sx
        dy = f.y - sy
        d = math.hypot(dx, dy)
        if d < 90.0 or d > 1000.0:
            continue
        a = math.atan2(dy, dx)
        if abs(_angle_diff(a, heading)) < 0.5:
            total += f.value * (1.0 - d / 1000.0)
    return total


def _mpc_harvest_action(snake, foods, snakes, world_size, segment_grid=None,
                        preferred_target=None):
    """
    MPC-style planner:
    sample headings, simulate short trajectories, and pick best first action.
    """
    food_candidates = _preselect_foods_for_mpc(snake, foods, max_dist=1200, max_count=56)
    if not food_candidates:
        return None
    near_enemies = _nearest_live_enemies(snake, snakes, k=7, max_dist=980)

    entries = []
    coarse_offsets = [0.0, 0.25, -0.25, 0.52, -0.52, 0.9, -0.9]
    for off in coarse_offsets:
        entries.append({
            "angle": snake.angle + off,
            "target": None,
            "intrinsic": 0.0,
            "lock_frames": 0,
        })

    for idx, f in food_candidates[:10]:
        d = math.hypot(f.x - snake.head[0], f.y - snake.head[1])
        entries.append({
            "angle": _food_approach_angle(snake, f.x, f.y),
            "target": (f.x, f.y),
            "intrinsic": f.value / (d + 25.0),
            "lock_frames": 8 if d < 220.0 else 12,
        })

    if preferred_target is not None:
        px, py, p_lock = preferred_target
        pd = math.hypot(px - snake.head[0], py - snake.head[1])
        entries.append({
            "angle": _food_approach_angle(snake, px, py),
            "target": (px, py),
            "intrinsic": 1.2 / (pd + 40.0),
            "lock_frames": p_lock,
        })

    seen = set()
    deduped = []
    for e in entries:
        key = int(e["angle"] * 20.0)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)

    best = None
    best_score = -1e9
    for e in deduped:
        a = e["angle"]
        base_score = _simulate_harvest_candidate(
            snake, snakes, world_size, segment_grid, food_candidates, a,
            boost=False, horizon=14, near_enemies=near_enemies,
        )
        score = base_score + e["intrinsic"]
        use_boost = False

        corridor_value = _heading_route_value(snake, food_candidates, a)
        boost_candidate = snake.mass > 80 and corridor_value > 4.2
        if boost_candidate:
            boost_score = _simulate_harvest_candidate(
                snake, snakes, world_size, segment_grid, food_candidates, a,
                boost=True, horizon=12, near_enemies=near_enemies,
            ) + e["intrinsic"] + 0.2
            # Boost only when route ROI is clearly better.
            if boost_score > score + 0.35:
                score = boost_score
                use_boost = True

        if score > best_score:
            best_score = score
            best = {
                "angle": a,
                "boost": use_boost,
                "target": e["target"],
                "lock_frames": e["lock_frames"],
                "score": score,
            }

    return best


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


def ai_harvester(snake, foods, snakes, world_size, segment_grid=None):
    """
    Harvester++: phase-driven food/loot optimizer.
    Uses safety filters, fight-loot opportunism, and MPC food routing.
    """
    wall_angle, wall_urg = avoid_walls(snake, world_size)
    snake._ai_phase = "init"

    def set_phase(name):
        snake._ai_phase = name

    def commit_heading(desired_angle, boost=False):
        base_angle = desired_angle
        if wall_angle is not None and wall_urg > 0:
            base_angle = _blend_angle(base_angle, wall_angle, min(1.0, wall_urg))
        chosen_angle, chosen_boost, clear = _pick_safer_heading(
            snake, base_angle, boost, snakes, world_size,
            segment_grid=segment_grid, wall_angle=wall_angle, wall_urg=wall_urg
        )
        snake.target_angle = chosen_angle
        snake.is_boosting = bool(boost and chosen_boost and clear > 8.0)
        return clear

    def commit_food_target(target_x, target_y, boost=False, lock_frames=0):
        desired_angle = _food_approach_angle(snake, target_x, target_y)
        target_dist = math.hypot(target_x - snake.head[0], target_y - snake.head[1])

        # Prevent endless tight circles around too-close food.
        realign_t = getattr(snake, "_harvest_realign_timer", 0)
        if realign_t > 0:
            snake._harvest_realign_timer = realign_t - 1
            desired_angle = getattr(snake, "_harvest_realign_angle", snake.angle)
            boost = False
        else:
            turn_r = max(35.0, snake.get_speed() / max(_turn_rate_estimate(snake), 1e-4))
            if (target_dist < turn_r * 1.03 and
                    abs(_angle_diff(desired_angle, snake.angle)) > math.pi * 0.55):
                side = 1 if _angle_diff(desired_angle, snake.angle) > 0 else -1
                snake._harvest_realign_timer = random.randint(8, 14)
                snake._harvest_realign_angle = snake.angle + side * 0.35
                desired_angle = snake._harvest_realign_angle
                boost = False

        commit_heading(desired_angle, boost=boost)
        if lock_frames > 0 and target_dist > 65.0:
            snake._harvest_lock = (target_x, target_y, lock_frames)

    # Phase 1: survival
    if wall_urg > 0.80:
        set_phase("wall_flee")
        commit_heading(wall_angle, boost=wall_urg > 0.9)
        return

    dodge, d_dist = dodge_bodies(snake, snakes, segment_grid, radius=122)
    if dodge is not None and d_dist < 76:
        set_phase("body_dodge")
        commit_heading(dodge, boost=d_dist < 44)
        return

    threat, t_dist = nearest_threat(snake, snakes, radius=240)
    if threat and ((threat.mass > snake.mass * 1.40 and t_dist < 190) or t_dist < 56):
        set_phase("threat_flee")
        dx = snake.head[0] - threat.head[0]
        dy = snake.head[1] - threat.head[1]
        tvx = math.cos(threat.angle) * threat.get_speed()
        tvy = math.sin(threat.angle) * threat.get_speed()
        flee_x = dx - tvx * 8.0
        flee_y = dy - tvy * 8.0
        commit_heading(
            math.atan2(flee_y, flee_x),
            boost=(t_dist < 100) or (threat.mass > snake.mass * 2.4)
        )
        return

    # Phase 1b: convert kills into mass quickly (corpse-loot sprint window).
    prev_kills = getattr(snake, "_harvest_prev_kills", snake.kills)
    if snake.kills > prev_kills:
        snake._harvest_post_kill_timer = 90
    snake._harvest_prev_kills = snake.kills
    post_kill_timer = getattr(snake, "_harvest_post_kill_timer", 0)
    if post_kill_timer > 0:
        snake._harvest_post_kill_timer = post_kill_timer - 1
        feast = nearest_food(snake, foods, max_dist=760, min_value=2)
        if feast is not None:
            set_phase("post_kill_loot")
            fd = math.hypot(feast.x - snake.head[0], feast.y - snake.head[1])
            commit_food_target(
                feast.x, feast.y,
                boost=(fd > 120 and snake.mass > 70),
                lock_frames=5,
            )
            return

    # Phase 2: high-value death drops first.
    jackpot = nearest_food(snake, foods, max_dist=760, min_value=3)
    if jackpot is not None:
        jd = math.hypot(jackpot.x - snake.head[0], jackpot.y - snake.head[1])
        ja = math.atan2(jackpot.y - snake.head[1], jackpot.x - snake.head[0])
        jturn = abs(_angle_diff(ja, snake.angle))
        cluster_bonus = 0.0
        cluster_count = 0
        for f in foods:
            if math.hypot(f.x - jackpot.x, f.y - jackpot.y) < 125:
                cluster_bonus += f.value
                cluster_count += 1
        drop_score = (jackpot.value + 0.28 * cluster_bonus) / (jd + 45.0 + 28.0 * jturn)
        chase_drop = (
            (jd < 220.0 and jackpot.value >= 3) or
            (cluster_count >= 3 and drop_score > 0.024) or
            (jackpot.value >= 5 and jd < 560.0 and drop_score > 0.020)
        )
        if chase_drop:
            set_phase("drop_hunt")
            commit_food_target(
                jackpot.x, jackpot.y,
                boost=(190.0 < jd < 760.0 and snake.mass > 78 and (cluster_count >= 4 or jackpot.value >= 5)),
                lock_frames=6,
            )
            return

    # Phase 2b: parasite-like tail shadowing when leader is boosting.
    lead_boost = None
    lead_mass = 0.0
    for s in snakes:
        if s is snake or s.dead:
            continue
        if s.is_boosting and s.mass > lead_mass:
            lead_boost = s
            lead_mass = s.mass
    if lead_boost is not None and lead_boost.mass > snake.mass * 1.12:
        tail = lead_boost.segments[-1]
        ld = math.hypot(tail[0] - snake.head[0], tail[1] - snake.head[1])
        if 160.0 < ld < 1220.0:
            set_phase("shadow_leader")
            commit_food_target(
                tail[0], tail[1],
                boost=(ld > 340 and snake.mass > 72),
                lock_frames=6,
            )
            return

    # Phase 3: continue lock to avoid target jitter.
    lock = getattr(snake, "_harvest_lock", None)
    if lock is not None:
        lx, ly, lt = lock
        if lt > 0:
            lock_food = _food_near_point(foods, lx, ly, max_dist=120)
            if lock_food is not None:
                lock_dist = math.hypot(lock_food.x - snake.head[0], lock_food.y - snake.head[1])
                if lock_food.value < 2 and lock_dist > 95.0:
                    snake._harvest_lock = None
                    snake._harvest_lock_stall = 0
                else:
                    prev_dist = getattr(snake, "_harvest_lock_prev_dist", lock_dist + 1.0)
                    lock_stall = getattr(snake, "_harvest_lock_stall", 0)
                    if lock_dist > prev_dist - 1.0:
                        lock_stall += 1
                    else:
                        lock_stall = max(0, lock_stall - 2)
                    snake._harvest_lock_prev_dist = lock_dist
                    snake._harvest_lock_stall = lock_stall
                    if lock_stall > 10:
                        snake._harvest_lock = None
                        snake._harvest_lock_stall = 0
                    else:
                        set_phase("locked_food")
                        snake._harvest_lock = (lock_food.x, lock_food.y, lt - 1)
                        commit_food_target(lock_food.x, lock_food.y, boost=False, lock_frames=max(0, lt - 1))
                        return
        snake._harvest_lock = None

    # Phase 4: rare control/encircle action.
    hunt_cd = getattr(snake, "_harvest_hunt_cooldown", 0)
    if hunt_cd > 0:
        hunt_cd -= 1
    snake._harvest_hunt_cooldown = hunt_cd

    if snake.mass > 220 and hunt_cd <= 0 and random.random() < 0.09:
        best_prey = None
        best_dist = float('inf')
        for s in snakes:
            if s is snake or s.dead:
                continue
            if snake.mass >= s.mass * 3.0:
                d = math.hypot(s.head[0] - snake.head[0], s.head[1] - snake.head[1])
                if d < 300 and d < best_dist:
                    best_prey = s
                    best_dist = d
        if best_prey is not None:
            set_phase("encircle_setup")
            angle_to_prey = math.atan2(best_prey.head[1] - snake.head[1],
                                       best_prey.head[0] - snake.head[0])
            side = 1 if _angle_diff(best_prey.angle, angle_to_prey) > 0 else -1
            offset = math.pi / 2.7 if best_dist < 130 else math.pi / 3.1
            commit_heading(
                angle_to_prey + side * offset,
                boost=(best_dist > 130 and snake.mass > 230)
            )
            snake._harvest_hunt_cooldown = random.randint(100, 200)
            return

    # Phase 5: immediate orb capture.
    instant_big = nearest_food(snake, foods, max_dist=320, min_value=2)
    if instant_big:
        set_phase("instant_big")
        bd = math.hypot(instant_big.x - snake.head[0], instant_big.y - snake.head[1])
        commit_food_target(
            instant_big.x, instant_big.y,
            boost=(instant_big.value >= 4 and bd > 110 and snake.mass > 85),
            lock_frames=4
        )
        return

    instant_food = nearest_food(snake, foods, max_dist=130)
    if instant_food:
        set_phase("instant_food")
        commit_food_target(instant_food.x, instant_food.y, boost=False, lock_frames=2)
        return

    # Phase 6: imitate top-performing scavenger/parasite behavior.
    hotspot = find_fight_hotspot(snake, snakes, max_pair_dist=230, max_my_dist=960)
    if hotspot is not None:
        hx, hy, hd, hs = hotspot
        if hs > 0.32 or hd < 520:
            set_phase("loot_rush")
            commit_food_target(
                hx, hy,
                boost=(hd > 220 and snake.mass > 70),
                lock_frames=6,
            )
            return

    leader = None
    leader_mass = 0.0
    for s in snakes:
        if s is snake or s.dead:
            continue
        if s.mass > leader_mass:
            leader = s
            leader_mass = s.mass
    if leader is not None and leader.mass > snake.mass * 1.25:
        tail = leader.segments[-1]
        ld = math.hypot(tail[0] - snake.head[0], tail[1] - snake.head[1])
        if ld < 1100 and (leader.is_boosting or ld < 480):
            set_phase("shadow_leader")
            commit_food_target(
                tail[0], tail[1],
                boost=(ld > 320 and snake.mass > 82),
                lock_frames=4,
            )
            return

    # Phase 7: forager-like patch sprint.
    patch, patch_dist, patch_value = best_food_patch(snake, foods, max_dist=1080, top_n=18)
    if patch is not None and patch_value >= 16.0:
        px, py = patch
        set_phase("patch_sprint")
        commit_food_target(
            px, py,
            boost=(patch_dist > 280 and patch_value >= 22.0 and snake.mass > 82),
            lock_frames=6 if patch_value >= 22.0 else 4,
        )
        return

    cluster, cluster_dist = best_food_cluster(snake, foods, max_dist=780, top_n=10)
    if cluster is not None:
        set_phase("forager_cluster")
        commit_food_target(
            cluster[0], cluster[1],
            boost=(cluster_dist > 260 and snake.mass > 76),
            lock_frames=3,
        )
        return

    # Phase 8: graph prior + MPC trajectory optimization.
    graph_plan = _plan_food_graph_target(
        snake, foods, snakes, world_size, segment_grid=segment_grid
    )
    preferred = None
    if graph_plan is not None:
        preferred = (graph_plan["x"], graph_plan["y"], graph_plan["lock_frames"])

    mpc_action = _mpc_harvest_action(
        snake, foods, snakes, world_size,
        segment_grid=segment_grid, preferred_target=preferred
    )
    if mpc_action is not None:
        set_phase("mpc_harvest")
        mpc_clear = _predict_heading_clearance(
            snake, mpc_action["angle"], snakes, world_size,
            segment_grid=segment_grid, steps=14, boost=mpc_action["boost"]
        )
        if mpc_clear > 14.0:
            direct_angle = mpc_action["angle"]
            if wall_angle is not None and wall_urg > 0:
                direct_angle = _blend_angle(direct_angle, wall_angle, min(0.50, wall_urg))
            snake.target_angle = direct_angle
            snake.is_boosting = bool(mpc_action["boost"])
        else:
            commit_heading(mpc_action["angle"], boost=mpc_action["boost"])
        if mpc_action["target"] is not None and mpc_action["lock_frames"] > 0:
            tx, ty = mpc_action["target"]
            snake._harvest_lock = (tx, ty, mpc_action["lock_frames"])
        return

    # Phase 9: fallback.
    fallback = nearest_food(snake, foods, max_dist=900)
    if fallback:
        set_phase("fallback_food")
        commit_food_target(
            fallback.x, fallback.y,
            boost=(fallback.value >= 3 and snake.mass > 80),
            lock_frames=3
        )
        return

    set_phase("scan_wander")
    snake._wander_timer -= 1
    if snake._wander_timer <= 0:
        snake.target_angle = random.uniform(0, 2 * math.pi)
        snake._wander_timer = random.randint(30, 90)
    desired = _blend_angle(snake.target_angle, math.atan2(-snake.head[1], -snake.head[0]), 0.1)
    commit_heading(desired, boost=False)


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
    'harvester': ai_harvester,
    'patrol': ai_patrol,
    'parasite': ai_parasite,
    'trapper': ai_trapper,
    'interceptor': ai_interceptor,
    'hunter': ai_hunter,
}
