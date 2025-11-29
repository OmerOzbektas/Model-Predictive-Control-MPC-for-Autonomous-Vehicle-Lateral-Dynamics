import numpy as np
from dataclasses import dataclass

@dataclass
class VehicleParams:
    """
    Data class to hold vehicle physical parameters.
    Standard passenger vehicle parameters.
    """
    m: float = 1500.0      # Mass [kg]
    Iz: float = 3000.0     # Moment of inertia [kg*m^2]
    Caf: float = 19000.0   # Front tire cornering stiffness [N/rad]
    Car: float = 33000.0   # Rear tire cornering stiffness [N/rad]
    lf: float = 2.0        # Distance from CG to front axle [m]
    lr: float = 3.0        # Distance from CG to rear axle [m]
    vx: float = 20.0       # Longitudinal velocity [m/s] (assumed constant)

def vehicle_dynamics_continuous(t, state, vehicle, u):
    """
    Defines the continuous-time vehicle dynamics (Bicycle Model).
    Used by the ODE solver for simulation fidelity.
    """
    vy, psi, psi_dot, Y = state
    delta = u 
    m, vx, Caf, Car, lf, lr, Iz = vehicle.m, vehicle.vx, vehicle.Caf, vehicle.Car, vehicle.lf, vehicle.lr, vehicle.Iz

    # slip angles (Small angle approximation)
    alpha_f = delta - (vy + lf * psi_dot) / vx
    alpha_r = - (vy - lr * psi_dot) / vx
    
    # lateral tire forces
    Fyf = Caf * alpha_f
    Fyr = Car * alpha_r

    # Equations of motion (Newton-Euler)
    vy_dot = (Fyf + Fyr) / m - vx * psi_dot
    psi_dot_dot = (lf * Fyf - lr * Fyr) / Iz
    Y_dot = vx * np.sin(psi) + vy * np.cos(psi)

    return [vy_dot, psi_dot, psi_dot_dot, Y_dot]