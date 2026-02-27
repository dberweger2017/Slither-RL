import pygame
import math
import random
import numpy as np
import bot_ai
import observation
from spatial_hash import SpatialHash

# Game Constants
WIDTH, HEIGHT = 1280, 900
FPS = 60
BASE_ZOOM = 0.75  # Starting zoom level

# Colors
BG_COLOR = (20, 20, 20)
GRID_COLOR = (40, 40, 40)
FOOD_COLORS = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100), (255, 100, 255)]

# Snake Physics
BASE_SPEED = 3.0
BOOST_SPEED = 6.0
BASE_TURN_RATE = 0.1  # Radians per frame
START_LENGTH = 10
START_MASS = 50
MASS_PER_FOOD = 5
FOOD_FRICTION = 0.92  # Velocity damping per frame for food inertia

class Food:
    def __init__(self, x, y, value=1, color=None, vx=0, vy=0):
        self.x = x
        self.y = y
        self.vx = vx  # Velocity for inertia
        self.vy = vy
        self.value = value
        self.radius = 3 + value
        if color:
            self.color = color
        else:
            self.color = random.choice(FOOD_COLORS)

    def update(self):
        """Apply velocity and friction for food inertia."""
        if abs(self.vx) > 0.1 or abs(self.vy) > 0.1:
            self.x += self.vx
            self.y += self.vy
            self.vx *= FOOD_FRICTION
            self.vy *= FOOD_FRICTION
        else:
            self.vx = 0
            self.vy = 0

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
        if is_player:
            self.color = (0, 200, 255)
        else:
            self.color = bot_ai.BOT_COLORS.get(self.bot_type, (200, 200, 200))
        self.radius = 10
        self.is_boosting = False
        self.dead = False
        self.kills = 0
        # State used by bot AI
        self._patrol_idx = 0
        self._wander_timer = 0

    def get_speed(self):
        return BOOST_SPEED if self.is_boosting else BASE_SPEED

    def get_segment_dist(self):
        """Dynamic segment spacing: scales with radius so big snakes look right."""
        return max(6.0, self.radius * 0.8)

    def update(self):
        if self.dead: return

        # Handling mass loss on boost (proportional to mass)
        if self.is_boosting and self.mass > (START_MASS + 10):
            self.mass -= max(0.5, self.mass * 0.001)  # Big snakes bleed faster
        else:
            self.is_boosting = False # Cannot boost if too small

        # The fatter you are, the slower you turn (very gradual via log scaling)
        # At mass 50: ~0.1, mass 500: ~0.077, mass 5000: ~0.062
        turn_rate = max(0.04, BASE_TURN_RATE / (1 + math.log10(self.mass / START_MASS) * 0.3))
        
        # Adjust radius based on real slither.io formula: w = 20 * sqrt(fam + 1)
        # At start (fam=1): radius ~7, diameter ~14 > SEGMENT_DIST(10) = segments overlap
        fam = self.mass / START_MASS
        self.radius = max(5, 5.0 * math.sqrt(fam + 1))

        # Turn towards target angle
        diff = (self.target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
        if diff > turn_rate:
            self.angle += turn_rate
        elif diff < -turn_rate:
            self.angle -= turn_rate
        else:
            self.angle = self.target_angle
            
        # Keep angle in bounds
        self.angle = self.angle % (2 * math.pi)

        # Move head
        speed = self.get_speed()
        self.head[0] += math.cos(self.angle) * speed
        self.head[1] += math.sin(self.angle) * speed

        # Dynamic segment distance based on current thickness
        seg_dist = self.get_segment_dist()

        # Non-linear length scaling using sqrt
        target_length = max(START_LENGTH, int(math.sqrt(self.mass) * 3))
        
        # Grow segments if needed
        while len(self.segments) < target_length:
            self.segments.append(list(self.segments[-1]))
            
        # Shrink segments if needed (due to boosting)
        while len(self.segments) > target_length:
            self.segments.pop()

        # Update Head position in segments
        self.segments[0] = list(self.head)

        # "Lagging Follower" physics (matches real slither.io)
        # Each segment Lerps toward the one in front, clamped to seg_dist.
        follow_speed = 0.25
        
        for i in range(1, len(self.segments)):
            prev_seg = self.segments[i-1]
            curr_seg = self.segments[i]
            
            # Lerp toward the segment ahead
            curr_seg[0] += (prev_seg[0] - curr_seg[0]) * follow_speed
            curr_seg[1] += (prev_seg[1] - curr_seg[1]) * follow_speed
            
            # Clamp: enforce max distance (dynamic based on thickness)
            dx = prev_seg[0] - curr_seg[0]
            dy = prev_seg[1] - curr_seg[1]
            dist = math.hypot(dx, dy)
            
            if dist > seg_dist:
                ratio = seg_dist / dist
                curr_seg[0] = prev_seg[0] - dx * ratio
                curr_seg[1] = prev_seg[1] - dy * ratio
                
        # Ensure segments array matches target length
        self.segments = self.segments[:target_length]
                
    def draw(self, surface, camera_x, camera_y, zoom):
        if self.dead: return
        
        # Draw body segments (back to front)
        rad_int = max(2, int(self.radius * 0.8 * zoom))
        for segment in reversed(self.segments):
            sx = int((segment[0] - camera_x) * zoom)
            sy = int((segment[1] - camera_y) * zoom)
            pygame.draw.circle(surface, self.color, (sx, sy), rad_int)
            
        # Draw head
        head_color = (255, 255, 255) if self.is_player else tuple(min(255, c + 80) for c in self.color)
        hx = int((self.head[0] - camera_x) * zoom)
        hy = int((self.head[1] - camera_y) * zoom)
        pygame.draw.circle(surface, head_color, (hx, hy), max(2, int(self.radius * zoom)))

    def ai_update(self, foods, snakes, world_size):
        """Delegate AI to the bot_ai module."""
        bot_ai.update(self, foods, snakes, world_size)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Slither.io Local Physics Engine")
    clock = pygame.time.Clock()

    # World space (circular arena centered at 0,0)
    WORLD_RADIUS = 2000
    NUM_BOTS = 15
    FOOD_COUNT = 500
    
    # Spatial hash grids (rebuilt each frame)
    food_grid = SpatialHash(cell_size=100)
    segment_grid = SpatialHash(cell_size=50)
    
    def random_point_in_circle(radius, margin=200):
        """Random point inside a circle with given radius, away from edge."""
        r = random.uniform(0, radius - margin)
        theta = random.uniform(0, 2 * math.pi)
        return r * math.cos(theta), r * math.sin(theta)
    
    px, py = 0, 0  # Player starts at center
    player = Snake(px, py, is_player=True)
    snakes = [player]
    foods = []
    
    # Spawn AI bots (one of each type + extras)
    for btype in bot_ai.BOT_TYPES:
        bx, by = random_point_in_circle(WORLD_RADIUS)
        snakes.append(Snake(bx, by, is_player=False, bot_type=btype))
    # Extra random bots to fill out
    for _ in range(NUM_BOTS - len(bot_ai.BOT_TYPES)):
        bx, by = random_point_in_circle(WORLD_RADIUS)
        snakes.append(Snake(bx, by, is_player=False))
    
    # Generate initial food (inside circle)
    for _ in range(FOOD_COUNT):
        fx, fy = random_point_in_circle(WORLD_RADIUS, margin=0)
        foods.append(Food(fx, fy))

    # FPS tracking
    fps_min = 999
    fps_max = 0
    fps_update_timer = 0
    fps_display = 60

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # --- INPUT ---
        mouse_x, mouse_y = pygame.mouse.get_pos()
        # Calculate target angle based on screen center (camera focuses on player)
        dx = mouse_x - (WIDTH // 2)
        dy = mouse_y - (HEIGHT // 2)
        
        if not player.dead:
            player.target_angle = math.atan2(dy, dx)
            
            buttons = pygame.mouse.get_pressed()
            player.is_boosting = buttons[0] or buttons[2] # Left or right click
            
            # Spawn food with backward velocity if boosting
            if player.is_boosting and player.mass > (START_MASS + 10):
                if random.random() < 0.2:
                    tail_pos = player.segments[-1]
                    # Push food backward (opposite of head direction)
                    push_speed = 4.0
                    vx = -math.cos(player.angle) * push_speed + random.uniform(-1, 1)
                    vy = -math.sin(player.angle) * push_speed + random.uniform(-1, 1)
                    foods.append(Food(tail_pos[0], tail_pos[1], value=1, color=player.color, vx=vx, vy=vy))


        # --- UPDATE ---
        # Update food physics (inertia)
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
        
        # AI brain for bots (pass food_grid for efficient queries)
        for snake in snakes:
            if not snake.is_player and not snake.dead:
                snake.ai_update(foods, snakes, WORLD_RADIUS)
        
        for snake in snakes:
            snake.update()
            
            if not snake.dead:
                # Circular Boundary Death Check
                dist_from_center = math.hypot(snake.head[0], snake.head[1])
                if dist_from_center > WORLD_RADIUS:
                    snake.dead = True
                    # Explode into food with outward velocity
                    cx = sum(s[0] for s in snake.segments) / len(snake.segments)
                    cy = sum(s[1] for s in snake.segments) / len(snake.segments)
                    for segment in snake.segments:
                        if random.random() < 0.5:
                            # Push outward from center of mass
                            dx = segment[0] - cx
                            dy = segment[1] - cy
                            d = math.hypot(dx, dy) + 0.1
                            vx = (dx / d) * random.uniform(2, 5)
                            vy = (dy / d) * random.uniform(2, 5)
                            foods.append(Food(segment[0], segment[1], value=3, color=snake.color, vx=vx, vy=vy))

        # Head-to-Body Collision using spatial hash
        for snake_a in snakes:
            if snake_a.dead:
                continue
            hx, hy = snake_a.head
            # Query only nearby segments (search radius = max possible body radius)
            nearby = segment_grid.query(hx, hy, 50)
            for (owner, seg) in nearby:
                if owner is snake_a or owner.dead:
                    continue
                dist = math.hypot(hx - seg[0], hy - seg[1])
                if dist < owner.radius * 0.8:
                    snake_a.dead = True
                    owner.kills += 1
                    if snake_a.is_player:
                        print(f"Killed by a bot! Mass: {int(snake_a.mass)}")
                    elif owner.is_player:
                        print(f"You killed a bot! (Kill #{owner.kills})")
                    # Explode into food with outward velocity
                    cx, cy = snake_a.head
                    for segment in snake_a.segments:
                        if random.random() < 0.5:
                            ddx = segment[0] - cx
                            ddy = segment[1] - cy
                            d = math.hypot(ddx, ddy) + 0.1
                            vx = (ddx / d) * random.uniform(2, 5)
                            vy = (ddy / d) * random.uniform(2, 5)
                            foods.append(Food(segment[0], segment[1], value=3, color=snake_a.color, vx=vx, vy=vy))
                    break

        # Food Collisions using spatial hash
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

        # Repopulate food (inside circle)
        while len(foods) < FOOD_COUNT:
            fx, fy = random_point_in_circle(WORLD_RADIUS, margin=0)
            foods.append(Food(fx, fy))
            
        # Respawn dead bots
        for i, snake in enumerate(snakes):
            if snake.dead and not snake.is_player:
                bx, by = random_point_in_circle(WORLD_RADIUS)
                snakes[i] = Snake(bx, by, is_player=False)
                
        # Respawn player on death
        if player.dead:
            px, py = random_point_in_circle(WORLD_RADIUS)
            player = Snake(px, py, is_player=True)
            snakes[0] = player

        # --- DRAW ---
        screen.fill(BG_COLOR)
        
        # Dynamic camera zoom: zooms out as you grow
        # BASE_ZOOM / (1 + mass * scale_factor)
        current_zoom = BASE_ZOOM / (1 + (player.mass - START_MASS) * 0.0003)
        current_zoom = max(0.25, min(BASE_ZOOM, current_zoom))  # Clamp
        
        # Camera follows player (centered, accounting for zoom)
        camera_x = player.head[0] - (WIDTH / 2) / current_zoom
        camera_y = player.head[1] - (HEIGHT / 2) / current_zoom
            
        # Draw Grid (scaled by zoom)
        grid_size = 50
        start_x = int((-camera_x % grid_size) * current_zoom)
        start_y = int((-camera_y % grid_size) * current_zoom)
        scaled_grid = int(grid_size * current_zoom)
        
        if scaled_grid > 0:
            for x in range(start_x, WIDTH, scaled_grid):
                pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, HEIGHT))
            for y in range(start_y, HEIGHT, scaled_grid):
                pygame.draw.line(screen, GRID_COLOR, (0, y), (WIDTH, y))

        # World bounds (circle)
        cx_screen = int(-camera_x * current_zoom)
        cy_screen = int(-camera_y * current_zoom)
        cr_screen = int(WORLD_RADIUS * current_zoom)
        pygame.draw.circle(screen, (200, 50, 50), (cx_screen, cy_screen), cr_screen, 3)

        # Draw Foods
        for food in foods:
            fx = int((food.x - camera_x) * current_zoom)
            fy = int((food.y - camera_y) * current_zoom)
            fr = max(1, int(food.radius * current_zoom))
            pygame.draw.circle(screen, food.color, (fx, fy), fr)

        # Draw Snakes
        for snake in snakes:
            snake.draw(screen, camera_x, camera_y, current_zoom)
            
        # --- CNN Observation Debug Overlay ---
        if not player.dead:
            obs = observation.generate_observation(player, snakes, foods, food_grid, WORLD_RADIUS)
            previews = observation.obs_to_surfaces(obs, preview_size=100)
            
            # Draw in top-right corner
            preview_x = WIDTH - 110
            preview_y = 5
            small_font = pygame.font.SysFont(None, 16)
            for surf, label in previews:
                # Dark background behind preview
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
            if fps_update_timer >= 30:  # Update display every 0.5s
                fps_display = current_fps
                fps_update_timer = 0
        
        # UI
        font = pygame.font.SysFont(None, 24)
        alive_count = sum(1 for s in snakes if not s.dead)
        status = "DEAD" if player.dead else f"Mass: {int(player.mass)}"
        info_text = f"{status} | Kills: {player.kills} | Alive: {alive_count}/{len(snakes)} | Click=Boost"
        text_surf = font.render(info_text, True, (255, 255, 255))
        screen.blit(text_surf, (10, 10))
        
        # FPS display
        fps_color = (80, 255, 80) if fps_display >= 50 else (255, 255, 80) if fps_display >= 30 else (255, 80, 80)
        fps_text = f"FPS: {fps_display:.0f}  (min:{fps_min:.0f} max:{fps_max:.0f})"
        fps_surf = font.render(fps_text, True, fps_color)
        screen.blit(fps_surf, (WIDTH - 310, HEIGHT - 30))
        
        # Bot type legend (small)
        small_font = pygame.font.SysFont(None, 18)
        y_off = 30
        for bt in bot_ai.BOT_TYPES:
            count = sum(1 for s in snakes if not s.dead and not s.is_player and s.bot_type == bt)
            col = bot_ai.BOT_COLORS.get(bt, (200,200,200))
            pygame.draw.circle(screen, col, (15, y_off + 4), 4)
            leg_surf = small_font.render(f"{bt}: {count}", True, (180, 180, 180))
            screen.blit(leg_surf, (25, y_off - 2))
            y_off += 16

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
