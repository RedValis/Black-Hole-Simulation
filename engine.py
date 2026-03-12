import sys
import math
from dataclasses import dataclass, field
import pygame


# ---------------------------------------------
#  Physical constants
# ---------------------------------------------

G = 6.674e-11        # gravitational constant  (m³ kg⁻¹ s⁻²)
C = 299_792_458.0    # speed of light          (m/s)


# ---------------------------------------------
#  Config
# ---------------------------------------------

@dataclass
class Config:
    title:    str   = "Black Hole Sim"
    width:    int   = 1280
    height:   int   = 720
    fps:      int   = 60
    bg_color: tuple = (2, 0, 10)


# ---------------------------------------------
#  Vec2  (lightweight 2-D vector)
# ---------------------------------------------

@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other):  return Vec2(self.x + other.x, self.y + other.y)
    def __sub__(self, other):  return Vec2(self.x - other.x, self.y - other.y)
    def __mul__(self, scalar): return Vec2(self.x * scalar,  self.y * scalar)
    def length(self):          return math.sqrt(self.x ** 2 + self.y ** 2)


# ---------------------------------------------
#  BlackHole
# ---------------------------------------------

@dataclass
class BlackHole:
    position: Vec2  = field(default_factory=Vec2)
    mass:     float = 1e35                          # kg  (stellar-mass scale)

    # Schwarzschild radius — computed from mass, not passed in
    event_horizon: float = field(init=False)

    # 1 pixel = 2.5e6 m  →  r_s lands at ~60 px (medium, clearly visible)
    PIXEL_SCALE:   float = field(init=False, repr=False, default=2.5e6)
    COLOR:         tuple = field(init=False, repr=False, default=(0, 0, 0))
    BORDER_COLOR:  tuple = field(init=False, repr=False, default=(255, 160, 30))
    BORDER_WIDTH:  int   = field(init=False, repr=False, default=2)

    def __post_init__(self):
        # Schwarzschild (sealed) radius:  r_s = 2GM / c²
        r_s = (2.0 * G * self.mass) / (C ** 2)
        self.event_horizon = r_s / self.PIXEL_SCALE   # convert to screen pixels
        print(f"[BlackHole] mass={self.mass:.2e} kg  "
              f"r_s={r_s:.2e} m  screen_r={self.event_horizon:.1f} px")

    def draw(self, screen: pygame.Surface) -> None:
        cx = int(self.position.x)
        cy = int(self.position.y)
        r  = max(2, int(self.event_horizon))

        # -- filled black disk (true event horizon) --------------
        # iterate over full circle: x = cos(θ)·r,  y = sin(θ)·r
        # then fill with pygame's built-in circle (most efficient)
        pygame.draw.circle(screen, self.COLOR, (cx, cy), r)

        # -- photon-sphere glow ring ------------------------------
        # drawn by stepping around the circumference manually so we
        # can swap in custom per-pixel colour later if needed
        num_steps = max(180, r * 4)
        step      = (2.0 * math.pi) / num_steps
        for i in range(num_steps):
            angle = i * step
            px = cx + int(math.cos(angle) * r)
            py = cy + int(math.sin(angle) * r)
            screen.set_at((px, py), self.BORDER_COLOR)


# ---------------------------------------------
#  LightRay
# ---------------------------------------------

@dataclass
class LightRay:
    x:  float = 0.0
    y:  float = 0.0
    dx: float = 1.0    # unit direction  (normalised by caller)
    dy: float = 0.0

    # Must match BlackHole.PIXEL_SCALE — 1 px = 2.5e6 m
    PIXEL_SCALE: float = 2.5e6
    # C / PIXEL_SCALE = ~120 px/s raw; * speed_scale = ~360 px/s
    # ray crosses 1280 px in ~3.5 s — visibly smooth
    speed_scale: float = 3.0
    COLOR:       tuple = (255, 240, 180)   # warm white

    def draw(self, screen: pygame.Surface) -> None:
        """Render the ray's current position as a single bright pixel."""
        screen.set_at((int(self.x), int(self.y)), self.COLOR)

    def step(self, dt: float) -> None:
        """
        Advance the ray by one timestep.

        Real-space displacement = C * dt  (metres)
        Screen displacement     = (C * dt) / PIXEL_SCALE  (pixels)

        dt is in seconds — even a single frame (≈0.016 s) moves the
        ray ~4.8e6 km, so we keep PIXEL_SCALE large to keep it on screen.
        """
        dist_px = (C * dt) / self.PIXEL_SCALE * self.speed_scale
        self.x += self.dx * dist_px
        self.y += self.dy * dist_px


# ---------------------------------------------
#  Engine
# ---------------------------------------------

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
        self.screen  = pygame.display.set_mode(
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

        # -- Black hole centred on screen --------------------------
        black_hole = BlackHole(
            position = Vec2(W / 2, H / 2),
            mass     = 1e35,
        )

        # -- Test ray — fires from left edge, aimed horizontally ---
        ray = LightRay(
            x  = 50.0,
            y  = H / 2,
            dx = 1.0,
            dy = 0.0,
        )

        while self.running:
            self.handle_events()
            self.begin_frame()

            # -- update ---------------------------------------------
            ray.step(self.dt)

            # reset ray when it leaves the screen (loop for testing)
            if ray.x > W or ray.y > H or ray.x < 0 or ray.y < 0:
                ray.x, ray.y = 50.0, H / 2

            # -- draw -----------------------------------------------
            ray.draw(self.screen)
            black_hole.draw(self.screen)

            # -- HUD ------------------------------------------------
            hud = font.render(
                f"FPS: {self.clock.get_fps():.0f}   "
                f"t={self.elapsed:.1f}s   "
                f"ray=({ray.x:.0f}, {ray.y:.0f})",
                True, (80, 160, 255)
            )
            self.screen.blit(hud, (12, 12))

            self.end_frame()

        self.shutdown()


# ---------------------------------------------
#  Entry point
# ---------------------------------------------

if __name__ == "__main__":
    engine = Engine(config=Config())
    engine.run()