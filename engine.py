import sys
import math
from dataclasses import dataclass, field
import pygame


# -----------------------------------------------
#  Physical constants
# -----------------------------------------------

G = 6.674e-11        # gravitational constant  (m^3 kg^-1 s^-2)
C = 299_792_458.0    # speed of light          (m/s)


# -----------------------------------------------
#  Config
# -----------------------------------------------

@dataclass
class Config:
    title:    str   = "Black Hole Sim"
    width:    int   = 1280
    height:   int   = 720
    fps:      int   = 60
    bg_color: tuple = (2, 0, 10)


# -----------------------------------------------
#  Vec2
# -----------------------------------------------

@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other):  return Vec2(self.x + other.x, self.y + other.y)
    def __sub__(self, other):  return Vec2(self.x - other.x, self.y - other.y)
    def __mul__(self, scalar): return Vec2(self.x * scalar,  self.y * scalar)
    def length(self):          return math.sqrt(self.x ** 2 + self.y ** 2)


# -----------------------------------------------
#  BlackHole
# -----------------------------------------------

@dataclass
class BlackHole:
    position: Vec2  = field(default_factory=Vec2)
    mass:     float = 1e35

    event_horizon: float = field(init=False)

    # 1 pixel = 2.5e6 m
    PIXEL_SCALE:  float = field(init=False, repr=False, default=2.5e6)
    COLOR:        tuple = field(init=False, repr=False, default=(0, 0, 0))
    BORDER_COLOR: tuple = field(init=False, repr=False, default=(255, 160, 30))

    def __post_init__(self):
        # Schwarzschild radius: r_s = 2GM / c^2
        r_s = (2.0 * G * self.mass) / (C ** 2)
        self.event_horizon = r_s / self.PIXEL_SCALE
        print(f"[BlackHole] mass={self.mass:.2e} kg  r_s={r_s:.2e} m  screen_r={self.event_horizon:.1f} px")

    def draw(self, screen: pygame.Surface) -> None:
        cx = int(self.position.x)
        cy = int(self.position.y)
        r  = max(2, int(self.event_horizon))

        # filled black event horizon disk
        pygame.draw.circle(screen, self.COLOR, (cx, cy), r)

        # photon-sphere border: manual cos/sin loop
        num_steps = max(180, r * 4)
        angle_step = (2.0 * math.pi) / num_steps
        for i in range(num_steps):
            angle = i * angle_step
            px = cx + int(math.cos(angle) * r)
            py = cy + int(math.sin(angle) * r)
            screen.set_at((px, py), self.BORDER_COLOR)


# -----------------------------------------------
#  LightRay
# -----------------------------------------------

@dataclass
class LightRay:
    x:  float = 0.0
    y:  float = 0.0
    dx: float = 1.0    # unit direction (normalised by caller)
    dy: float = 0.0

    # Must match BlackHole.PIXEL_SCALE
    PIXEL_SCALE:  float = 2.5e6
    speed_scale:  float = 3.0
    TRAIL_LENGTH: int   = 120          # how many positions to remember
    COLOR:        tuple = (255, 240, 180)

    # trail stores (x, y) tuples; oldest at index 0, newest at -1
    trail: list = field(default_factory=list)

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the trail with a fade from dim (tail) to bright (tip).
        Each point is a small circle that grows slightly toward the tip.
        """
        n = len(self.trail)
        for i, (tx, ty) in enumerate(self.trail):
            # t: 0.0 at oldest tail, 1.0 at newest tip
            t = i / max(n - 1, 1)

            # brightness ramps 20 -> 255
            alpha = int(20 + 235 * t)

            # colour: dim blue-white at tail, warm white at tip
            r = min(255, int((180 + 75 * t) * alpha / 255))
            g = min(255, int((160 + 80 * t) * alpha / 255))
            b = min(255, int((100 + 80 * t) * alpha / 255))

            radius = max(1, int(2 * t) + 1)
            pygame.draw.circle(screen, (r, g, b), (int(tx), int(ty)), radius)

        # bright tip dot drawn last so it sits on top
        pygame.draw.circle(screen, self.COLOR, (int(self.x), int(self.y)), 3)

    def step(self, dt: float) -> None:
        """
        Move the ray forward one timestep and append the new position to the trail.
        Pop the oldest entry once the trail exceeds TRAIL_LENGTH.
        """
        dist_px = (C * dt) / self.PIXEL_SCALE * self.speed_scale
        self.x += self.dx * dist_px
        self.y += self.dy * dist_px

        self.trail.append((self.x, self.y))
        if len(self.trail) > self.TRAIL_LENGTH:
            self.trail.pop(0)


# -----------------------------------------------
#  Engine
# -----------------------------------------------

@dataclass
class Engine:
    config:  Config            = field(default_factory=Config)
    screen:  pygame.Surface    = field(init=False, default=None)
    clock:   pygame.time.Clock = field(init=False, default=None)
    running: bool              = field(init=False, default=False)
    dt:      float             = field(init=False, default=0.0)
    elapsed: float             = field(init=False, default=0.0)

    def init(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.config.width, self.config.height),
            pygame.DOUBLEBUF | pygame.HWSURFACE
        )
        pygame.display.set_caption(self.config.title)
        self.clock   = pygame.time.Clock()
        self.running = True
        self.elapsed = 0.0
        print(f"[Engine] {self.config.width}x{self.config.height} @ {self.config.fps} fps")

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.running = False

    def begin_frame(self) -> None:
        self.screen.fill(self.config.bg_color)

    def end_frame(self) -> None:
        pygame.display.flip()
        raw_ms    = self.clock.tick(self.config.fps)
        self.dt   = min(raw_ms / 1000.0, 0.05)
        self.elapsed += self.dt

    def shutdown(self) -> None:
        pygame.quit()
        print("[Engine] Shutdown.")
        sys.exit(0)

    def run(self) -> None:
        self.init()

        W, H = self.config.width, self.config.height
        font = pygame.font.SysFont("monospace", 14)

        black_hole = BlackHole(
            position=Vec2(W / 2, H / 2),
            mass=1e35,
        )

        # --- cluster of parallel rays ---
        # evenly spaced across the full screen height, all fired horizontally
        NUM_RAYS = 20
        SPACING  = H / (NUM_RAYS + 1)
        START_X  = 50.0

        def make_rays():
            return [
                LightRay(x=START_X, y=SPACING * (i + 1), dx=1.0, dy=0.0)
                for i in range(NUM_RAYS)
            ]

        rays = make_rays()

        while self.running:
            self.handle_events()
            self.begin_frame()

            # update all rays; reset individually when they leave the screen
            for i, ray in enumerate(rays):
                ray.step(self.dt)
                if ray.x > W or ray.y > H or ray.x < 0 or ray.y < 0:
                    ray.x  = START_X
                    ray.y  = SPACING * (i + 1)
                    ray.dx = 1.0
                    ray.dy = 0.0
                    ray.trail.clear()

            # draw rays first, black hole on top
            for ray in rays:
                ray.draw(self.screen)
            black_hole.draw(self.screen)

            # HUD
            hud = font.render(
                f"FPS: {self.clock.get_fps():.0f}   t={self.elapsed:.1f}s   rays={NUM_RAYS}",
                True, (80, 160, 255)
            )
            self.screen.blit(hud, (12, 12))

            self.end_frame()

        self.shutdown()


# -----------------------------------------------
#  Entry point
# -----------------------------------------------

if __name__ == "__main__":
    engine = Engine(config=Config())
    engine.run()