# Black Hole Light Bending Simulation

A real-time Schwarzschild black hole simulation built with Python and Pygame.

This project numerically integrates null geodesics (light paths) in curved spacetime to simulate gravitational lensing around one or two black holes. It uses a fourth-order Runge–Kutta (RK4) integrator to evolve photon trajectories under relativistic gravity.

---

## Features

- Real-time light bending around a Schwarzschild black hole
- Optional dual black hole mode
- Multiple emission modes:
  - Parallel light beam (gravitational lensing)
  - Point-source fan
  - Photon sphere orbital mode
- Fourth-order Runge–Kutta numerical integration
- Interactive black hole movement
- Smooth trail rendering with fading light paths

---

## Controls

| Key | Action |
|-----|--------|
| 1 | Parallel light rays |
| 2 | Point-source fan |
| 3 | Photon sphere orbital mode |
| B | Toggle dual black holes |
| Arrow Keys | Move primary black hole |
| Q / ESC | Quit |

---

# Physics Background

## Schwarzschild Black Hole

The simulation models a non-rotating, uncharged black hole using the Schwarzschild metric.

The event horizon radius is computed using:

r_s = 2GM / c²

Where:

- G = gravitational constant  
- M = black hole mass  
- c = speed of light  

This radius defines the event horizon.

---

## Null Geodesics (Light Paths)

Light follows curved paths in spacetime described by the Schwarzschild null geodesic equations.

In polar coordinates:

d²φ/dλ² = -2/r · (dr/dλ)(dφ/dλ)  
d²r/dλ² = -(C2 · r_s) / (2r²) + r(dφ/dλ)²

Where:

- λ is an affine parameter
- r is radial distance
- φ is angular coordinate
- C2 is a scaling constant used for visual tuning

These equations represent:

- Radial acceleration due to gravitational curvature
- Angular momentum conservation
- Spacetime curvature–induced bending of light

---

## Photon Sphere

At:

r = (3/2) r_s

Light can orbit the black hole in unstable circular trajectories.

Mode 3 initializes a photon slightly outside this radius to demonstrate near-orbital behavior.

---

## Multi-Black-Hole Approximation

When dual mode is enabled:

- The primary black hole uses Schwarzschild geodesics
- Additional black holes apply Newtonian-like gravitational acceleration
- Acceleration is projected into the primary black hole’s polar coordinate system

This is an approximation. Exact multi-black-hole solutions in general relativity are significantly more complex.

---

# Numerical Method

The simulation uses fourth-order Runge–Kutta (RK4) integration:

- Computes four intermediate derivative estimates per step
- Combines them to achieve high accuracy
- Provides significantly improved stability over first-order methods

State vector:

(r, φ, dr, dφ)

---

# Code Structure

## Config

Stores window configuration such as resolution, frame rate, and background color.

---

## Vec2

Basic 2D vector utility providing:

- Addition
- Subtraction
- Scalar multiplication
- Length calculation

---

## BlackHole

Responsibilities:

- Compute Schwarzschild radius
- Convert physical radius to screen scale
- Render event horizon
- Render photon sphere boundary

Key properties:

- mass
- event_horizon
- PIXEL_SCALE

---

## LightRay

Represents a photon.

Handles:

- Position and direction
- Conversion between Cartesian and polar velocities
- RK4 geodesic integration
- Trail management
- Rendering

Core method:

step(dt, black_hole, black_holes)

This method:
1. Converts coordinates to polar form
2. Integrates the geodesic equations
3. Converts back to Cartesian coordinates
4. Updates the light trail

---

## geodesic()

Returns second derivatives (d²r, d²φ) according to the Schwarzschild null geodesic equations.

---

## rk4()

Implements fourth-order Runge–Kutta integration.

Handles:

- Primary black hole spacetime curvature
- Additional black hole gravitational influence
- Projection between coordinate systems

---

## Engine

Main simulation controller:

- Initializes Pygame
- Handles input
- Updates light rays
- Renders each frame
- Displays HUD information
- Manages simulation modes

---

# Rendering Details

- Each light ray maintains a fixed-length trail
- Trails are drawn in segments to reduce draw calls
- Color intensity increases toward the photon tip
- The leading photon is highlighted

---

# Scaling

To make relativistic curvature visually observable:

1 pixel = 2.5 × 10^6 meters

Additional curvature scaling factor:

C2_PX = 15

These scaling constants preserve qualitative relativistic behavior while ensuring visible curvature on screen.

---

# Requirements

- Python 3.9 or higher
- Pygame

Install dependencies:

pip install pygame

Run the simulation:

python main.py

---

# Limitations

- Uses visual scaling constants
- Multi-black-hole mode is an approximation
- Does not solve full Einstein field equations

This project is intended for educational and visualization purposes.

---

# References

- Schutz, A First Course in General Relativity
- Carroll, Spacetime and Geometry
- Einstein (1916), General Relativity
- Schwarzschild (1916), Exact Solution to Einstein Field Equations

---

# Author

Black Hole Simulation by RedValis

If you enjoyed the project or are interested, please feel free to expand it further to your liking