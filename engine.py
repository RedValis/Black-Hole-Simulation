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
    speed_scale:  float = 400.0  # ~800 px/frame -> visibly fast
    TRAIL_LENGTH: int   = 2000         # long enough to show the full curved path
    COLOR:        tuple = (255, 240, 180)

    # polar velocities (set lazily on first step from dx/dy)
    dr:   float = 0.0    # radial velocity       dr/dlambda
    dphi: float = 0.0    # angular velocity    dphi/dlambda
    _polar_init: bool = False   # flag: have we converted dx/dy -> dr/dphi yet?

    # trail stores (x, y) tuples; oldest at index 0, newest at -1
    trail: list = field(default_factory=list)

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the trail as a connected line that fades from dim (tail) to bright (tip).
        Segments are drawn in batches of 8 to keep per-frame draw calls low.
        """
        n = len(self.trail)
        if n < 2:
            return

        SEGMENT = 8   # points per polyline chunk
        for start in range(0, n - 1, SEGMENT):
            chunk = self.trail[start: start + SEGMENT + 1]
            if len(chunk) < 2:
                break

            # t at midpoint of this chunk -> 0.0 (tail) .. 1.0 (tip)
            t = (start + SEGMENT / 2) / max(n - 1, 1)
            t = min(t, 1.0)

            alpha = int(20 + 235 * t)
            r = min(255, int((180 + 75 * t) * alpha / 255))
            g = min(255, int((160 + 80 * t) * alpha / 255))
            b = min(255, int((100 + 80 * t) * alpha / 255))

            pts = [(int(px), int(py)) for px, py in chunk]
            pygame.draw.lines(screen, (r, g, b), False, pts, 2)

        # bright tip
        pygame.draw.circle(screen, self.COLOR, (int(self.x), int(self.y)), 3)

    def step(self, dt: float, black_hole: "BlackHole", black_holes: list = None) -> None:
        """
        Advance the ray using RK4 integration of the Schwarzschild geodesic.

        h is the affine parameter step size — controls both speed and
        integration accuracy. Smaller h = slower but more accurate curves.
        """
        H = 3.5   # affine step size — tune for speed, curve accuracy unchanged

        rx  = self.x - black_hole.position.x
        ry  = self.y - black_hole.position.y
        r   = math.sqrt(rx ** 2 + ry ** 2)
        phi = math.atan2(ry, rx)

        if r < black_hole.event_horizon:
            return

        # init polar velocities from Cartesian direction
        if not self._polar_init:
            cos_p = math.cos(phi)
            sin_p = math.sin(phi)
            self.dr   = self.dx * cos_p  + self.dy * sin_p
            self.dphi = (-self.dx * sin_p + self.dy * cos_p) / r
            self._polar_init = True

        # RK4: takes 4 educated trial steps and blends them
        bhs = black_holes if black_holes else [black_hole]
        r, phi, self.dr, self.dphi = rk4(
            r, phi, self.dr, self.dphi,
            black_hole, bhs, H
        )

        self.x = black_hole.position.x + r * math.cos(phi)
        self.y = black_hole.position.y + r * math.sin(phi)

        self.trail.append((self.x, self.y))
        if len(self.trail) > self.TRAIL_LENGTH:
            self.trail.pop(0)


# -----------------------------------------------
#  Geodesic  (Schwarzschild null geodesic equations)
# -----------------------------------------------

def geodesic(r: float, dr: float, dphi: float, rs: float):
    """
    Schwarzschild null geodesic accelerations.

      d2phi = -2/r * dr * dphi
      d2r   = -(C2_PX * rs) / (2r^2)  +  r * dphi^2
    """
    C2_PX = 15

    d2phi = (-2.0 / r) * dr * dphi
    d2r   = -(C2_PX * rs) / (2.0 * r ** 2) + r * (dphi ** 2)
    return d2r, d2phi


def rk4(r: float, phi: float, dr: float, dphi: float,
        primary_bh, black_holes: list, h: float):
    """
    Runge-Kutta 4 integrator for multi-body Schwarzschild geodesic.

    State vector: (r, phi, dr, dphi) — polar coords centred on primary_bh.

    Extra black holes are handled by converting their gravitational pull
    to Cartesian, summing, then projecting back into the primary BH polar frame.
    """
    C2_PX = 15

    def derivs(r_, phi_, dr_, dphi_):
        if r_ <= 0:
            return 0.0, 0.0, 0.0, 0.0

        # --- primary BH geodesic (polar) ---
        d2r_, d2phi_ = geodesic(r_, dr_, dphi_, primary_bh.event_horizon)

        # --- extra BH gravity (Cartesian, then projected) ---
        # ray position in screen space
        ray_x = primary_bh.position.x + r_ * math.cos(phi_)
        ray_y = primary_bh.position.y + r_ * math.sin(phi_)

        ax_extra = 0.0
        ay_extra = 0.0
        for bh in black_holes:
            if bh is primary_bh:
                continue
            rx2 = ray_x - bh.position.x
            ry2 = ray_y - bh.position.y
            r2  = math.sqrt(rx2 ** 2 + ry2 ** 2)
            if r2 < bh.event_horizon:
                continue
            # gravitational acceleration toward this BH
            mag = C2_PX * bh.event_horizon / (2.0 * r2 ** 2)
            ax_extra -= mag * (rx2 / r2)
            ay_extra -= mag * (ry2 / r2)

        # project Cartesian extras onto primary BH polar frame
        cos_p = math.cos(phi_)
        sin_p = math.sin(phi_)
        d2r_   += ax_extra * cos_p  + ay_extra * sin_p
        d2phi_ += (-ax_extra * sin_p + ay_extra * cos_p) / r_

        return dr_, dphi_, d2r_, d2phi_

    k1_r, k1_phi, k1_dr, k1_dphi = derivs(r, phi, dr, dphi)
    k2_r, k2_phi, k2_dr, k2_dphi = derivs(
        r    + h*0.5*k1_r,   phi  + h*0.5*k1_phi,
        dr   + h*0.5*k1_dr,  dphi + h*0.5*k1_dphi)
    k3_r, k3_phi, k3_dr, k3_dphi = derivs(
        r    + h*0.5*k2_r,   phi  + h*0.5*k2_phi,
        dr   + h*0.5*k2_dr,  dphi + h*0.5*k2_dphi)
    k4_r, k4_phi, k4_dr, k4_dphi = derivs(
        r    + h*k3_r,        phi  + h*k3_phi,
        dr   + h*k3_dr,       dphi + h*k3_dphi)

    r_new    = r    + h/6.0 * (k1_r    + 2*k2_r    + 2*k3_r    + k4_r)
    phi_new  = phi  + h/6.0 * (k1_phi  + 2*k2_phi  + 2*k3_phi  + k4_phi)
    dr_new   = dr   + h/6.0 * (k1_dr   + 2*k2_dr   + 2*k3_dr   + k4_dr)
    dphi_new = dphi + h/6.0 * (k1_dphi + 2*k2_dphi + 2*k3_dphi + k4_dphi)

    return r_new, phi_new, dr_new, dphi_new


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

        BH_SPEED    = 4.0
        mode        = 1
        dual_mode   = False   # toggled with B

        black_hole2 = BlackHole(
            position=Vec2(W * 0.65, H * 0.4),
            mass=1e35,
        )

        # ── Mode 1: parallel cluster ─────────────────────────────
        NUM_RAYS = 60
        SPACING  = H / (NUM_RAYS + 1)
        START_X  = 50.0

        def make_parallel_rays():
            return [
                LightRay(x=START_X, y=SPACING * (i + 1), dx=1.0, dy=0.0)
                for i in range(NUM_RAYS)
            ]

        # ── Mode 2: point-source fan from top-left ───────────────
        NUM_FAN  = 60
        def make_fan_rays():
            rays = []
            for i in range(NUM_FAN):
                # spread angles from ~0 to ~90 degrees
                angle = math.radians(i * 90.0 / (NUM_FAN - 1))
                rays.append(LightRay(
                    x=10.0, y=10.0,
                    dx=math.cos(angle),
                    dy=math.sin(angle),
                ))
            return rays

        # ── Mode 3: orbital ray at photon sphere ─────────────────
        def make_orbital_ray():
            rs        = black_hole.event_horizon
            r_orb     = 1.5 * rs * 1.03
            dphi_circ = math.sqrt(15.0 * rs / (2.0 * r_orb ** 3))
            ray = LightRay(
                x     = black_hole.position.x + r_orb,
                y     = black_hole.position.y,
                COLOR = (80, 220, 255),
            )
            ray.dr          =  0.004
            ray.dphi        =  dphi_circ
            ray._polar_init =  True
            return ray

        # initialise starting mode
        rays = make_parallel_rays()

        def reset_rays():
            nonlocal rays
            if mode == 1:
                rays = make_parallel_rays()
            elif mode == 2:
                rays = make_fan_rays()
            else:
                rays = [make_orbital_ray()]

        def reset_ray(i, ray):
            """Reset a single ray back to its spawn position for the current mode."""
            if mode == 1:
                ray.x = START_X;  ray.y = SPACING * (i + 1)
                ray.dx = 1.0;     ray.dy = 0.0
                ray.dr = 0.0;     ray.dphi = 0.0
                ray._polar_init = False;  ray.trail.clear()
            elif mode == 2:
                angle = math.radians(i * 90.0 / (NUM_FAN - 1))
                ray.x = 10.0;  ray.y = 10.0
                ray.dx = math.cos(angle);  ray.dy = math.sin(angle)
                ray.dr = 0.0;  ray.dphi = 0.0
                ray._polar_init = False;   ray.trail.clear()
            else:
                # rebuild orbital ray in place
                rs        = black_hole.event_horizon
                r_orb     = 1.5 * rs * 1.03
                dphi_circ = math.sqrt(15.0 * rs / (2.0 * r_orb ** 3))
                ray.x           = black_hole.position.x + r_orb
                ray.y           = black_hole.position.y
                ray.dx          = 1.0;  ray.dy = 0.0
                ray.dr          = 0.004
                ray.dphi        = dphi_circ
                ray._polar_init = True
                ray.trail.clear()

        MODE_LABELS = {1: "1: Parallel cluster", 2: "2: Point source fan", 3: "3: Orbital"}

        while self.running:
            # ── events ───────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        self.running = False
                    elif event.key == pygame.K_1 and mode != 1:
                        mode = 1;  reset_rays()
                    elif event.key == pygame.K_2 and mode != 2:
                        mode = 2;  reset_rays()
                    elif event.key == pygame.K_3 and mode != 3:
                        mode = 3;  reset_rays()
                    elif event.key == pygame.K_b:
                        dual_mode = not dual_mode;  reset_rays()

            # ── arrow keys move black hole ────────────────────────
            keys = pygame.key.get_pressed()
            moved = False
            if keys[pygame.K_LEFT]:  black_hole.position.x -= BH_SPEED;  moved = True
            if keys[pygame.K_RIGHT]: black_hole.position.x += BH_SPEED;  moved = True
            if keys[pygame.K_UP]:    black_hole.position.y -= BH_SPEED;  moved = True
            if keys[pygame.K_DOWN]:  black_hole.position.y += BH_SPEED;  moved = True
            if moved:
                reset_rays()   # clear trails when black hole moves

            self.begin_frame()

            # ── update ───────────────────────────────────────────
            for i, ray in enumerate(rays):
                bh_list = [black_hole, black_hole2] if dual_mode else [black_hole]
                ray.step(self.dt, black_hole, bh_list)
                if ray.x > W + 50 or ray.x < -50 or ray.y > H + 50 or ray.y < -50:
                    reset_ray(i, ray)

            # ── draw ─────────────────────────────────────────────
            for ray in rays:
                ray.draw(self.screen)
            if dual_mode:
                black_hole2.draw(self.screen)
            black_hole.draw(self.screen)

            # ── HUD ──────────────────────────────────────────────
            hud = font.render(
                f"FPS: {self.clock.get_fps():.0f}   {MODE_LABELS[mode]}   "
                f"[1/2/3] mode   [B] dual BH: {'ON' if dual_mode else 'OFF'}   arrows: move BH1",
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