#  Model Predictive Control (MPC) for Autonomous Vehicle Lateral Dynamics

*A complete MPC-based lateral control and animation simulation in
Python*

This repository implements a Model Predictive Controller (MPC) for an
autonomous vehicle's lateral dynamics using a linearized bicycle model.
It generates a smooth double lane-change reference, computes MPC
prediction matrices from scratch, optimizes steering inputs, and
animates the vehicle motion.

------------------------------------------------------------------------
## Features

-   MPC for lateral control
-   Linear bicycle model
-   Continuous → discrete (ZOH)
-   Prediction matrices Φ, Γ, H, M
-   Tanh-based double lane change reference
-   Real‑time animation
-   Steering saturation ±30°
-   Pure NumPy/SciPy implementation

------------------------------------------------------------------------

## Technical Overview

### Vehicle Model States

-   `v_y` -- lateral velocity\
-   `ψ` -- yaw angle\
-   `ψ̇` -- yaw rate\
-   `Y` -- lateral position

Includes mass, inertia, cornering stiffnesses, geometry, and velocity.

### MPC

-   Adjustable horizon
-   Δu formulation
-   Quadratic cost
-   Steering limit enforcement
-   Closed-form optimal input calculation

### Simulation

-   `solve_ivp` numerical integration
-   Time step: 0.05 s (recommended)
-   Animated steering + vehicle path

## How to Run

### 1. Create environment

``` bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

``` bash
pip install -r requirements.txt
```

### 3. Run notebook

``` bash
jupyter notebook mpc_vehicle_control.ipynb
```

------------------------------------------------------------------------

## Dependencies

    numpy
    scipy
    matplotlib
    ipython
    jupyter

------------------------------------------------------------------------

## Author

Ömer Faruk Özbektaş


------------------------------------------------------------------------

## ⚖️ License

MIT License (optional)
