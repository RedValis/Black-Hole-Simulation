# black hole 3D ray tracer
# port of black_hole.cpp + geodesic.comp
#
# install: pip install moderngl pygame numpy
#
# controls:
#   left mouse drag  - orbit camera
#   scroll           - zoom
#   G                - toggle n-body gravity
#   Q / ESC          - quit

import sys, math
import pygame
import moderngl
import numpy as np


G_CONST = 6.67430e-11
C_LIGHT = 299792458.0
SAGA_RS = 1.269e10      # schwarzschild radius of sgr A*
D_LAMBDA = 1e7          # affine step size
STEPS   = 60000         # rk4 steps per ray
ESCAPE_R = 1e30

WIN_W, WIN_H   = 800, 600
COMP_W, COMP_H = 400, 300   # actual render res, gets upscaled to window


# simple passthrough vert for the fullscreen quad
VERT = """
#version 330
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); vUV = aUV; }
"""

QUAD_FRAG = """
#version 330
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D tex;
void main() { fragColor = texture(tex, vUV); }
"""

RAYTRACE_VERT = """
#version 330
layout(location=0) in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }
"""

# the actual ray tracer — runs entirely on GPU, one thread per pixel
RAYTRACE_FRAG = """
#version 330
out vec4 fragColor;

uniform vec3  camPos;
uniform vec3  camRight;
uniform vec3  camUp;
uniform vec3  camForward;
uniform float tanHalfFov;
uniform float aspect;
uniform vec2  resolution;
uniform float disk_r1;
uniform float disk_r2;
uniform int   numObjects;
uniform vec4  objPosRadius[4];
uniform vec4  objColor[4];

const float SagA_rs  = 1.269e10;
const float D_LAMBDA = 1e7;
const int   STEPS    = 60000;

// spherical coords, y-up convention
// theta = angle from +Y, phi = azimuth in XZ plane
struct Ray {
    float x, y, z;
    float r, theta, phi;
    float dr, dtheta, dphi;
    float E, L;
};

Ray initRay(vec3 pos, vec3 dir) {
    Ray ray;
    ray.x = pos.x; ray.y = pos.y; ray.z = pos.z;
    ray.r     = length(pos);
    ray.theta = acos(clamp(pos.y / ray.r, -1.0, 1.0));
    ray.phi   = atan(pos.z, pos.x);

    float dx = dir.x, dy = dir.y, dz = dir.z;
    // y-up spherical basis projection
    ray.dr     =  sin(ray.theta)*cos(ray.phi)*dx
                + cos(ray.theta)*dy
                - sin(ray.theta)*sin(ray.phi)*dz;
    // TODO: clean this up, still has the old z-up leftover coords below
    ray.theta = acos(clamp(pos.z / ray.r, -1.0, 1.0));
    ray.phi   = atan(pos.y, pos.x);

    ray.dr     = sin(ray.theta)*cos(ray.phi)*dx
               + sin(ray.theta)*sin(ray.phi)*dy
               + cos(ray.theta)*dz;
    ray.dtheta = (cos(ray.theta)*cos(ray.phi)*dx
               +  cos(ray.theta)*sin(ray.phi)*dy
               -  sin(ray.theta)*dz) / ray.r;
    ray.dphi   = (-sin(ray.phi)*dx + cos(ray.phi)*dy)
               / (ray.r * sin(ray.theta) + 1e-10);

    ray.L = ray.r * ray.r * sin(ray.theta) * ray.dphi;
    float f = 1.0 - SagA_rs / ray.r;
    float dt_dL = sqrt(abs(
        ray.dr*ray.dr/f
        + ray.r*ray.r*(ray.dtheta*ray.dtheta
        + sin(ray.theta)*sin(ray.theta)*ray.dphi*ray.dphi)
    ));
    ray.E = f * dt_dL;
    return ray;
}

void geodesicRHS(Ray ray, out vec3 d1, out vec3 d2) {
    float r = ray.r, theta = ray.theta;
    float dr = ray.dr, dtheta = ray.dtheta, dphi = ray.dphi;
    float f = 1.0 - SagA_rs / r;
    float dt_dL = ray.E / (f + 1e-30);

    d1 = vec3(dr, dtheta, dphi);

    d2.x = - (SagA_rs / (2.0*r*r)) * f * dt_dL * dt_dL
           + (SagA_rs / (2.0*r*r*f + 1e-30)) * dr * dr
           + r * (dtheta*dtheta + sin(theta)*sin(theta)*dphi*dphi);

    d2.y = -2.0*dr*dtheta/r + sin(theta)*cos(theta)*dphi*dphi;

    d2.z = -2.0*dr*dphi/r
           - 2.0*cos(theta)/(sin(theta) + 1e-10) * dtheta * dphi;
}

void rk4Step(inout Ray ray, float dL) {
    vec3 k1a, k1b;
    geodesicRHS(ray, k1a, k1b);

    ray.r      += dL * k1a.x;
    ray.theta  += dL * k1a.y;
    ray.phi    += dL * k1a.z;
    ray.dr     += dL * k1b.x;
    ray.dtheta += dL * k1b.y;
    ray.dphi   += dL * k1b.z;

    // back to cartesian
    ray.x = ray.r * sin(ray.theta) * cos(ray.phi);
    ray.y = ray.r * sin(ray.theta) * sin(ray.phi);
    ray.z = ray.r * cos(ray.theta);
}

// disk sits in the equatorial plane (z=0), radius measured in XY
bool crossesDisk(vec3 oldP, vec3 newP) {
    bool crossed = (oldP.z * newP.z < 0.0);
    float r = length(vec2(newP.x, newP.y));
    return crossed && (r >= disk_r1 && r <= disk_r2);
}

bool interceptObject(Ray ray, out vec4 color) {
    vec3 P = vec3(ray.x, ray.y, ray.z);
    for (int i = 0; i < numObjects; i++) {
        vec3  center = objPosRadius[i].xyz;
        float radius = objPosRadius[i].w;
        if (distance(P, center) <= radius) {
            vec3 N = normalize(P - center);
            vec3 V = normalize(camPos - P);
            float ambient = 0.1;
            float diff    = max(dot(N, V), 0.0);
            float intensity = ambient + (1.0 - ambient) * diff;
            color = vec4(objColor[i].rgb * intensity, 1.0);
            return true;
        }
    }
    return false;
}

// procedural star field so escaped rays aren't just black
vec3 starfield(vec3 dir) {
    dir = normalize(dir);
    ivec3 cell = ivec3(floor(dir * 20.0));
    int h = abs(cell.x*127 + cell.y*311 + cell.z*541) % 997;
    if (h > 980) {
        float b = 0.4 + 0.6*float(h-980)/17.0;
        int t = h % 3;
        if (t == 0) return vec3(b);
        if (t == 1) return vec3(b, b*0.9, b*0.6);
        return vec3(b*0.7, b*0.85, b);
    }
    return vec3(0.004, 0.004, 0.012);
}

void main() {
    vec2 pix = gl_FragCoord.xy;
    float u  = (2.0*(pix.x+0.5)/resolution.x - 1.0) * aspect * tanHalfFov;
    float v  = (1.0 - 2.0*(pix.y+0.5)/resolution.y) * tanHalfFov;

    vec3 dir = normalize(u * camRight - v * camUp + camForward);
    Ray  ray = initRay(camPos, dir);

    vec3 prevPos = vec3(ray.x, ray.y, ray.z);
    vec4 color   = vec4(0.0);
    vec4 objHitColor;
    bool hit = false;

    for (int i = 0; i < STEPS; i++) {
        if (ray.r <= SagA_rs) {
            color = vec4(0.0, 0.0, 0.0, 1.0);
            hit = true;
            break;
        }

        rk4Step(ray, D_LAMBDA);

        vec3 newPos = vec3(ray.x, ray.y, ray.z);

        if (crossesDisk(prevPos, newPos)) {
            float r = length(vec2(ray.x, ray.y)) / disk_r2;
            color = vec4(1.0, r, 0.2, 1.0);
            hit = true;
            break;
        }

        if (interceptObject(ray, objHitColor)) {
            color = objHitColor;
            hit = true;
            break;
        }

        prevPos = newPos;

        if (ray.r > 1e30) break;
    }

    if (!hit) color = vec4(starfield(dir), 1.0);

    fragColor = color;
}
"""

# warped grid — deforms vertices on CPU using schwarzschild geometry
GRID_VERT = """
#version 330
layout(location=0) in vec3 aPos;
uniform mat4 viewProj;
void main() { gl_Position = viewProj * vec4(aPos, 1.0); }
"""

GRID_FRAG = """
#version 330
out vec4 fragColor;
void main() { fragColor = vec4(0.2, 0.5, 1.0, 0.45); }
"""

# sphere billboards so objects are visible on the grid
SPHERE_VERT = """
#version 330
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aColor;
layout(location=2) in float aRadius;
out vec3 vColor;
out vec2 vUV;
uniform mat4 viewProj;
uniform vec3 camRight;
uniform vec3 camUp;
void main() {
    int corner = gl_VertexID % 4;
    vec2 off = vec2(corner < 2 ? -1.0 : 1.0,
                    corner % 2 == 0 ? -1.0 : 1.0);
    vec3 world = aPos + (camRight * off.x + camUp * off.y) * aRadius;
    gl_Position = viewProj * vec4(world, 1.0);
    vColor = aColor;
    vUV    = off;
}
"""
SPHERE_FRAG = """
#version 330
in vec3 vColor;
in vec2 vUV;
out vec4 fragColor;
void main() {
    float d = dot(vUV, vUV);
    if (d > 1.0) discard;
    float rim = 1.0 - sqrt(d);
    float light = 0.3 + 0.7 * rim;
    fragColor = vec4(vColor * light, 1.0);
}
"""


class Camera:
    def __init__(self):
        self.radius    = 7e11
        self.min_r     = 1e10
        self.max_r     = 5e12
        self.azimuth   = 0.0
        self.elevation = math.pi / 2.0
        self.orbit_spd = 0.01
        self.zoom_spd  = 25e9
        self.dragging  = False
        self.last_x    = 0
        self.last_y    = 0

    def position(self):
        e = max(0.01, min(math.pi - 0.01, self.elevation))
        return (
            self.radius * math.sin(e) * math.cos(self.azimuth),
            self.radius * math.cos(e),
            self.radius * math.sin(e) * math.sin(self.azimuth),
        )

    def basis(self):
        px, py, pz = self.position()
        l  = math.sqrt(px*px + py*py + pz*pz)
        fx, fy, fz = -px/l, -py/l, -pz/l
        # right = cross(fwd, world_up) where world up is Y
        rx =  fz;   ry = 0.0;  rz = -fx
        rl  = math.sqrt(rx*rx + rz*rz) + 1e-12
        rx /= rl;  rz /= rl
        ux = ry*fz - rz*fy
        uy = rz*fx - rx*fz
        uz = rx*fy - ry*fx
        return (rx,ry,rz), (ux,uy,uz), (fx,fy,fz)

    def mouse_move(self, x, y):
        if self.dragging:
            dx = x - self.last_x
            dy = y - self.last_y
            self.azimuth   += dx * self.orbit_spd
            self.elevation -= dy * self.orbit_spd
            self.elevation  = max(0.01, min(math.pi - 0.01, self.elevation))
        self.last_x, self.last_y = x, y

    def scroll(self, dy):
        self.radius -= dy * self.zoom_spd
        self.radius  = max(self.min_r, min(self.max_r, self.radius))


class ObjectData:
    def __init__(self, pos, radius, color, mass):
        self.pos    = np.array(pos,   dtype=np.float32)
        self.radius = radius
        self.color  = np.array(color, dtype=np.float32)
        self.mass   = mass
        self.vel    = np.zeros(3, dtype=np.float64)


def make_objects():
    return [
        ObjectData((4e11, 0, 0),    4e10, (1,1,0), 1.98892e30),
        ObjectData((0, 0, 4e11),    4e10, (1,0,0), 1.98892e30),
        # black hole — just here for gravity, shader handles the rendering
        ObjectData((0, 0, 0), SAGA_RS, (0,0,0), 8.54e36),
    ]


def generate_grid(objects, grid_size=25, spacing=4e10):
    verts = []
    for zi in range(grid_size + 1):
        for xi in range(grid_size + 1):
            wx = (xi - grid_size // 2) * spacing
            wz = (zi - grid_size // 2) * spacing
            y  = 0.0
            for obj in objects:
                ox, oy, oz = obj.pos
                mass = obj.mass
                r_s  = 2.0 * G_CONST * mass / (C_LIGHT * C_LIGHT)
                dx   = wx - ox
                dz   = wz - oz
                dist = math.sqrt(dx*dx + dz*dz)
                if dist > r_s:
                    delta = 2.0 * math.sqrt(r_s * (dist - r_s))
                    y += delta - 3e10
                else:
                    y += 2.0 * r_s - 3e10
            verts.append((wx, y, wz))

    indices = []
    for zi in range(grid_size):
        for xi in range(grid_size):
            i = zi * (grid_size + 1) + xi
            indices += [i, i + 1, i, i + grid_size + 1]

    vdata = np.array(verts, dtype=np.float32).flatten()
    idata = np.array(indices, dtype=np.uint32)
    return vdata, idata


def look_at(eye, target, up):
    e = np.array(eye);  t = np.array(target);  u = np.array(up)
    f = (t - e);  f /= np.linalg.norm(f)
    r = np.cross(f, u);  r /= np.linalg.norm(r)
    u2 = np.cross(r, f)
    M = np.eye(4, dtype=np.float32)
    M[0,:3] =  r;   M[0,3] = -np.dot(r,  e)
    M[1,:3] =  u2;  M[1,3] = -np.dot(u2, e)
    M[2,:3] = -f;   M[2,3] =  np.dot(f,  e)
    return M


def perspective(fov_y, aspect, near, far):
    t = math.tan(fov_y * 0.5)
    M = np.zeros((4,4), dtype=np.float32)
    M[0,0] = 1.0 / (aspect * t)
    M[1,1] = 1.0 / t
    M[2,2] = -(far + near) / (far - near)
    M[2,3] = -2.0 * far * near / (far - near)
    M[3,2] = -1.0
    return M


def apply_gravity(objects, gravity_on):
    if not gravity_on:
        return
    for i, obj in enumerate(objects):
        for j, obj2 in enumerate(objects):
            if i == j:
                continue
            dx = float(obj2.pos[0] - obj.pos[0])
            dy = float(obj2.pos[1] - obj.pos[1])
            dz = float(obj2.pos[2] - obj.pos[2])
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist > 0:
                gforce = G_CONST * obj.mass * obj2.mass / (dist * dist)
                acc    = gforce / obj.mass
                obj.vel[0] += (dx / dist) * acc
                obj.vel[1] += (dy / dist) * acc
                obj.vel[2] += (dz / dist) * acc
        obj.pos[0] += obj.vel[0]
        obj.pos[1] += obj.vel[1]
        obj.pos[2] += obj.vel[2]


def main():
    pygame.init()
    pygame.display.set_mode((WIN_W, WIN_H), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Black Hole — Schwarzschild Ray Tracer")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont("monospace", 14)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    quad_data = np.array([
        -1,-1, 0,0,   1,-1, 1,0,   -1,1, 0,1,
        -1, 1, 0,1,   1,-1, 1,0,    1,1, 1,1,
    ], dtype=np.float32)
    quad_vbo = ctx.buffer(quad_data.tobytes())

    rt_prog  = ctx.program(vertex_shader=RAYTRACE_VERT, fragment_shader=RAYTRACE_FRAG)
    rt_vao   = ctx.simple_vertex_array(rt_prog, quad_vbo, 'aPos')

    blit_prog = ctx.program(vertex_shader=VERT, fragment_shader=QUAD_FRAG)
    blit_vao  = ctx.simple_vertex_array(blit_prog, quad_vbo, 'aPos', 'aUV')

    grid_prog = ctx.program(vertex_shader=GRID_VERT, fragment_shader=GRID_FRAG)

    # render at lower res and upscale — GPU can't do 60k steps at full 800x600
    tex = ctx.texture((COMP_W, COMP_H), 4)
    tex.filter = moderngl.LINEAR, moderngl.LINEAR
    fbo = ctx.framebuffer(color_attachments=[tex])

    camera  = Camera()
    objects = make_objects()
    gravity = False

    grid_vbo = ctx.buffer(reserve=1024*1024)
    grid_ibo = ctx.buffer(reserve=1024*1024)
    grid_vao = ctx.vertex_array(grid_prog,
                                [(grid_vbo, '3f', 'aPos')],
                                index_buffer=grid_ibo,
                                index_element_size=4)

    sphere_prog = ctx.program(vertex_shader=SPHERE_VERT, fragment_shader=SPHERE_FRAG)
    sphere_vbo  = ctx.buffer(reserve=4 * 7 * 4 * 10)

    print(f"[BH3D] {WIN_W}x{WIN_H} window  |  {COMP_W}x{COMP_H} render  |  {STEPS} RK4 steps/ray")
    print("[BH3D] G = toggle gravity  |  drag = orbit  |  scroll = zoom")

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()
                elif ev.key == pygame.K_g:
                    gravity = not gravity
                    print(f"[INFO] Gravity {'ON' if gravity else 'OFF'}")
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    camera.dragging = True
                    camera.last_x, camera.last_y = ev.pos
                elif ev.button == 4:
                    camera.scroll(+1)
                elif ev.button == 5:
                    camera.scroll(-1)
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1:
                    camera.dragging = False
            elif ev.type == pygame.MOUSEMOTION:
                camera.mouse_move(*ev.pos)

        apply_gravity(objects, gravity)

        cp = camera.position()
        (rx,ry,rz), (ux,uy,uz), (fx,fy,fz) = camera.basis()
        tan_hfov = math.tan(math.radians(60.0 * 0.5))
        asp      = WIN_W / WIN_H

        disk_r1 = SAGA_RS * 2.2
        disk_r2 = SAGA_RS * 5.2

        shade_objs = objects[:2]

        fbo.use()
        fbo.clear(0, 0, 0, 1)

        rt_prog['camPos'].value     = cp
        rt_prog['camRight'].value   = (rx, ry, rz)
        rt_prog['camUp'].value      = (ux, uy, uz)
        rt_prog['camForward'].value = (fx, fy, fz)
        rt_prog['tanHalfFov'].value = tan_hfov
        rt_prog['aspect'].value     = asp
        rt_prog['resolution'].value = (float(COMP_W), float(COMP_H))
        rt_prog['disk_r1'].value    = disk_r1
        rt_prog['disk_r2'].value    = disk_r2
        rt_prog['numObjects'].value = len(shade_objs)

        # moderngl doesn't let you write uniform arrays element by element
        pos_data   = np.zeros((4, 4), dtype=np.float32)
        color_data = np.zeros((4, 4), dtype=np.float32)
        for i, obj in enumerate(shade_objs):
            pos_data[i]   = [*obj.pos.tolist(), obj.radius]
            color_data[i] = [*obj.color.tolist(), 1.0]
        rt_prog['objPosRadius'].write(pos_data.tobytes())
        rt_prog['objColor'].write(color_data.tobytes())

        rt_vao.render(moderngl.TRIANGLES)

        ctx.screen.use()
        ctx.clear(0, 0, 0, 1)
        ctx.viewport = (0, 0, WIN_W, WIN_H)
        tex.use(location=0)
        blit_prog['tex'].value = 0
        blit_vao.render(moderngl.TRIANGLES)

        vdata, idata = generate_grid(objects)
        grid_vbo.write(vdata.tobytes())
        grid_ibo.write(idata.tobytes())

        view = look_at(cp, (0,0,0), (0,1,0))
        proj = perspective(math.radians(60.0), asp, 1e9, 1e15)
        vp   = (proj @ view).T.flatten()
        grid_prog['viewProj'].write(vp.astype(np.float32).tobytes())

        ctx.disable(moderngl.DEPTH_TEST)
        grid_vao.render(moderngl.LINES, vertices=len(idata))
        ctx.enable(moderngl.DEPTH_TEST)

        # draw the planets as billboards so they're visible on the grid
        sphere_prog['viewProj'].write(vp.astype(np.float32).tobytes())
        sphere_prog['camRight'].value = (rx, ry, rz)
        sphere_prog['camUp'].value    = (ux, uy, uz)

        for obj in objects[:-1]:
            grid_y = 0.0
            for src in objects:
                r_s_src = 2.0 * G_CONST * src.mass / (C_LIGHT * C_LIGHT)
                dx = float(obj.pos[0] - src.pos[0])
                dz = float(obj.pos[2] - src.pos[2])
                dist = math.sqrt(dx*dx + dz*dz)
                if dist > r_s_src:
                    grid_y += 2.0 * math.sqrt(r_s_src * (dist - r_s_src)) - 3e10
                else:
                    grid_y += 2.0 * r_s_src - 3e10

            cx, cy, cz = float(obj.pos[0]), grid_y + obj.radius, float(obj.pos[2])
            r, g, b    = obj.color.tolist()
            rad        = obj.radius
            inst = np.array([
                cx, cy, cz, r, g, b, rad,
                cx, cy, cz, r, g, b, rad,
                cx, cy, cz, r, g, b, rad,
                cx, cy, cz, r, g, b, rad,
            ], dtype=np.float32)
            sphere_vbo.write(inst.tobytes())
            sphere_vao = ctx.vertex_array(sphere_prog, [
                (sphere_vbo, '3f 3f 1f', 'aPos', 'aColor', 'aRadius')
            ])
            sphere_vao.render(moderngl.TRIANGLE_STRIP, vertices=4)

        fps = clock.get_fps()
        hud = font.render(
            f"FPS:{fps:.1f}  r={camera.radius:.2e}  G={'ON' if gravity else 'OFF'}  drag=orbit  scroll=zoom  G=gravity",
            True, (80, 200, 255)
        )

        pygame.display.flip()
        clock.tick(0)


if __name__ == "__main__":
    main()