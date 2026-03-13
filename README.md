# Black Hole Simulation

A real-time black hole simulation in Python, built in two stages — a 2D light bending visualiser and a 3D ray tracer. Both are based on the Schwarzschild solution to Einstein's field equations, which describes the spacetime geometry around a non-rotating, spherically symmetric mass.

## Physics

Light doesn't travel in straight lines near a massive object — it follows **null geodesics**, paths through curved spacetime where the proper time is zero. In the Schwarzschild metric, these paths are governed by two conserved quantities: the energy **E** and the angular momentum **L**. The equations of motion can be written as a system of coupled ODEs in terms of an affine parameter λ, and both simulations integrate these using **4th-order Runge-Kutta (RK4)**.

The key scale is the **Schwarzschild radius** r_s = 2GM/c², the radius at which the escape velocity equals the speed of light. Anything that crosses this boundary — the event horizon — cannot return. Just outside it, at r = 1.5 r_s, sits the **photon sphere**, where light can orbit in an unstable circular path. The glowing ring you see around the black hole is light that has looped around one or more times before escaping toward the camera.

The simulation uses **Sagittarius A\***, the black hole at the centre of the Milky Way, with a mass of 8.54 × 10³⁶ kg and a Schwarzschild radius of about 1.27 × 10¹⁰ metres (~0.08 AU).

The **spacetime grid** is a visual representation of spatial curvature. Each grid vertex is displaced vertically by the embedding function y = 2√(r_s(r − r_s)), which is the standard way of visualising the Schwarzschild geometry as a 2D surface embedded in 3D space — the classic "rubber sheet" picture, done properly.

The **2D engine** works entirely on the CPU and uses pixel-space polar coordinates centred on the black hole. It's good for visualising how parallel rays bend, how the photon sphere traps orbital light, and what happens when two black holes interact.

The **3D engine** moves the geodesic integration into a GLSL fragment shader running on the GPU. Every pixel launches one ray and runs up to 60,000 RK4 steps to trace it through the curved spacetime. At 200×150 pixels that's around 1.8 billion integration steps per frame — only feasible on the GPU.

---

## Install and run

You'll need Python 3.10+ and a GPU that supports OpenGL 3.3 for the 3D version.

```
pip install pygame numpy moderngl
```

**2D simulation** — parallel light rays, photon orbits, dual black hole mode:
```
python 2Dengine.py
```

Controls: `1` `2` `3` to switch ray modes, `B` to toggle a second black hole, arrow keys to move the black hole, `Q` to quit.

**3D ray tracer** — Schwarzschild geodesics, accretion disc, spacetime grid, n-body gravity:
```
python 3Dengine.py
```

Controls: left mouse drag to orbit, scroll wheel to zoom, `G` to toggle gravity between the objects, `Q` to quit.