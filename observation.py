"""
CNN Semantic Mini-Map Observation System (Vectorized).
5-channel ego-centric rotated mini-map (84×84) for RL training.
Channels: Self, Enemies, Food, Boundary, Enemy Velocity.
"""
import math
import numpy as np

MAP_SIZE = 84
VIEW_RADIUS = 500.0
PIXELS_PER_UNIT = MAP_SIZE / (2 * VIEW_RADIUS)
CENTER = MAP_SIZE // 2

_YY, _XX = np.mgrid[0:MAP_SIZE, 0:MAP_SIZE]


def _world_to_ego(wx, wy, head_x, head_y, cos_a, sin_a):
    dx = wx - head_x
    dy = wy - head_y
    rx = dx * cos_a + dy * sin_a
    ry = -dx * sin_a + dy * cos_a
    px = CENTER + rx * PIXELS_PER_UNIT
    py = CENTER - ry * PIXELS_PER_UNIT
    return px, py


def _stamp_circle(channel, cx, cy, radius, intensity):
    """Vectorized filled circle using numpy broadcasting."""
    r = max(1, int(radius + 0.5))
    y0, y1 = max(0, int(cy) - r - 1), min(MAP_SIZE, int(cy) + r + 2)
    x0, x1 = max(0, int(cx) - r - 1), min(MAP_SIZE, int(cx) + r + 2)
    if y0 >= y1 or x0 >= x1:
        return
    yy = _YY[y0:y1, x0:x1]
    xx = _XX[y0:y1, x0:x1]
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = dist_sq <= r * r
    np.maximum(channel[y0:y1, x0:x1], mask * intensity, out=channel[y0:y1, x0:x1])


def _stamp_line(channel, x0, y0, x1, y1, intensity, thickness=1):
    """Vectorized line using distance-to-segment."""
    dx, dy = x1 - x0, y1 - y0
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 0.5:
        return
    bx0 = max(0, int(min(x0, x1)) - thickness - 1)
    bx1 = min(MAP_SIZE, int(max(x0, x1)) + thickness + 2)
    by0 = max(0, int(min(y0, y1)) - thickness - 1)
    by1 = min(MAP_SIZE, int(max(y0, y1)) + thickness + 2)
    if bx0 >= bx1 or by0 >= by1:
        return
    xx = _XX[by0:by1, bx0:bx1].astype(np.float32)
    yy = _YY[by0:by1, bx0:bx1].astype(np.float32)
    t = ((xx - x0) * dx + (yy - y0) * dy) / seg_len_sq
    np.clip(t, 0, 1, out=t)
    proj_x = x0 + t * dx
    proj_y = y0 + t * dy
    dist_sq = (xx - proj_x) ** 2 + (yy - proj_y) ** 2
    mask = dist_sq <= thickness * thickness
    np.maximum(channel[by0:by1, bx0:bx1], mask * intensity, out=channel[by0:by1, bx0:bx1])


def generate_observation(player, snakes, foods, food_grid, world_radius):
    obs = np.zeros((5, MAP_SIZE, MAP_SIZE), dtype=np.float32)
    hx, hy = player.head
    heading = player.angle
    cos_a = math.cos(-heading + math.pi / 2)
    sin_a = math.sin(-heading + math.pi / 2)

    # Channel 0: Self
    _stamp_circle(obs[0], CENTER, CENTER, max(1, player.radius * PIXELS_PER_UNIT), 1.0)
    num_segs = len(player.segments)
    for i, seg in enumerate(player.segments[1:], 1):
        d = math.hypot(seg[0] - hx, seg[1] - hy)
        if d > VIEW_RADIUS:
            continue
        px, py = _world_to_ego(seg[0], seg[1], hx, hy, cos_a, sin_a)
        fade = 0.8 - 0.5 * (i / max(num_segs, 1))
        r_px = max(2, int(player.radius * 0.8 * PIXELS_PER_UNIT + 0.5))
        _stamp_circle(obs[0], px, py, r_px, max(0.2, fade))

    # Channel 1: Enemies
    for snake in snakes:
        if snake is player or snake.dead:
            continue
        head_d = math.hypot(snake.head[0] - hx, snake.head[1] - hy)
        if head_d > VIEW_RADIUS + 200:
            continue
        mass_intensity = min(1.0, snake.mass / 2000.0)
        for seg in snake.segments:
            d = math.hypot(seg[0] - hx, seg[1] - hy)
            if d > VIEW_RADIUS:
                continue
            px, py = _world_to_ego(seg[0], seg[1], hx, hy, cos_a, sin_a)
            r_px = max(2, int(snake.radius * 0.8 * PIXELS_PER_UNIT + 0.5))
            _stamp_circle(obs[1], px, py, r_px, mass_intensity * 0.7)
        if head_d <= VIEW_RADIUS:
            px, py = _world_to_ego(snake.head[0], snake.head[1], hx, hy, cos_a, sin_a)
            r_px = max(3, int(snake.radius * PIXELS_PER_UNIT + 0.5))
            _stamp_circle(obs[1], px, py, r_px, 1.0)

    # Channel 2: Food
    nearby_food = food_grid.query(hx, hy, VIEW_RADIUS)
    for f in nearby_food:
        d = math.hypot(f.x - hx, f.y - hy)
        if d > VIEW_RADIUS:
            continue
        px, py = _world_to_ego(f.x, f.y, hx, hy, cos_a, sin_a)
        intensity = min(1.0, f.value / 5.0)
        _stamp_circle(obs[2], px, py, 3, max(0.3, intensity))  # Fixed 3px glow radius

    # Channel 3: Boundary (fully vectorized — no Python pixel loops)
    dist_from_center = math.hypot(hx, hy)
    if dist_from_center + VIEW_RADIUS > world_radius - 300:
        rel_px = (_XX - CENTER).astype(np.float32) / PIXELS_PER_UNIT
        rel_py = -((_YY - CENTER).astype(np.float32)) / PIXELS_PER_UNIT
        wx = hx + rel_px * cos_a - rel_py * sin_a
        wy = hy + rel_px * sin_a + rel_py * cos_a
        world_dist = np.sqrt(wx * wx + wy * wy)
        d_to_edge = world_radius - world_dist
        obs[3] = np.clip(1.0 - d_to_edge / 300.0, 0.0, 1.0)

    # Channel 4: Enemy Velocity (streaks showing heading direction)
    for snake in snakes:
        if snake is player or snake.dead:
            continue
        head_d = math.hypot(snake.head[0] - hx, snake.head[1] - hy)
        if head_d > VIEW_RADIUS:
            continue
        speed = snake.get_speed()
        if speed < 0.5:
            continue
        streak_len = min(30, speed * 5)
        end_x = snake.head[0] + math.cos(snake.angle) * streak_len
        end_y = snake.head[1] + math.sin(snake.angle) * streak_len
        px0, py0 = _world_to_ego(snake.head[0], snake.head[1], hx, hy, cos_a, sin_a)
        px1, py1 = _world_to_ego(end_x, end_y, hx, hy, cos_a, sin_a)
        intensity = min(1.0, speed / 6.0)
        _stamp_line(obs[4], px0, py0, px1, py1, intensity, thickness=2)
        _stamp_circle(obs[4], px0, py0, 2, intensity)

    return obs


def obs_to_surfaces(obs, preview_size=100):
    """Convert 5-channel observation to pygame surfaces for debug display."""
    import pygame
    labels = ['Self', 'Enemies', 'Food', 'Boundary', 'Velocity']
    tints = [
        (0, 200, 255), (255, 80, 80), (80, 255, 80),
        (255, 200, 50), (255, 100, 255),
    ]
    surfaces = []
    for i in range(5):
        channel = obs[i]
        rgb = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
        for c in range(3):
            rgb[:, :, c] = (channel * tints[i][c]).clip(0, 255).astype(np.uint8)
        surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
        surf = pygame.transform.scale(surf, (preview_size, preview_size))
        surfaces.append((surf, labels[i]))
    return surfaces
