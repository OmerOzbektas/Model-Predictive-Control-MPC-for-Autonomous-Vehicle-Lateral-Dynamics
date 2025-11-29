import numpy as np
from dataclasses import dataclass
from scipy.signal import cont2discrete
from .vehicle_model import VehicleParams

@dataclass
class MPCParams:
    """
    MPC Controller settings.
    """
    Ts: float = 0.05       # Sampling time [s]
    Horizon: int = 20      # Prediction horizon
    q_y: float = 10.0      # Penalty for Y position error
    q_psi: float = 100.0   # Penalty for Yaw angle error
    r_input: float = 1.0   # Penalty for steering input change (Delta u)

class LateralMPC:
    def __init__(self, vehicle: VehicleParams, mpc: MPCParams):
        self.vp = vehicle
        self.mp = mpc
        
        # 1. Build Continuous-time model matrices (State Space: x_dot = Ax + Bu)
        A = np.array([
            [-(2*self.vp.Caf + 2*self.vp.Car)/(self.vp.m*self.vp.vx), 0, -self.vp.vx - (2*self.vp.Caf*self.vp.lf - 2*self.vp.Car*self.vp.lr)/(self.vp.m*self.vp.vx), 0],
            [0, 0, 1, 0],
            [-(2*self.vp.lf*self.vp.Caf - 2*self.vp.lr*self.vp.Car)/(self.vp.Iz*self.vp.vx), 0, -(2*self.vp.lf**2*self.vp.Caf + 2*self.vp.lr**2*self.vp.Car)/(self.vp.Iz*self.vp.vx), 0],
            [1, self.vp.vx, 0, 0]
        ])
        B = np.array([
            [2*self.vp.Caf / self.vp.m], 
            [0], 
            [2*self.vp.lf*self.vp.Caf / self.vp.Iz], 
            [0]
        ])
        C = np.array([[0, 1, 0, 0], [0, 0, 0, 1]]) # Outputs: Psi and Y
        D = np.array([[0], [0]])

        # 2. Discretization using Zero-Order Hold (ZOH)
        sys_d = cont2discrete((A, B, C, D), self.mp.Ts, method='zoh')
        self.Ad, self.Bd, self.Cd = sys_d[0], sys_d[1], sys_d[2]

        # 3. State Augmentation
        # Augmenting the state vector to include the previous input u_{k-1}
        # This allows controlling delta_u instead of u directly for smoothness.
        n_x, n_u = self.Ad.shape[0], self.Bd.shape[1]
        self.A_aug = np.block([[self.Ad, self.Bd], [np.zeros((n_u, n_x)), np.eye(n_u)]])
        self.B_aug = np.vstack([self.Bd, np.eye(n_u)])
        self.C_aug = np.hstack([self.Cd, np.zeros((self.Cd.shape[0], n_u))])

        # 4. Pre-compute prediction matrices
        self._build_prediction_matrices()

    def _build_prediction_matrices(self):
        """
        Constructs the large Phi and Gamma matrices for the horizon.
        This enables the analytical solution of the MPC problem.
        """
        hz = self.mp.Horizon
        nx_aug, nu, ny = self.A_aug.shape[0], self.B_aug.shape[1], self.C_aug.shape[0]
        
        # Weight matrices
        Q_bar = np.kron(np.eye(hz), np.diag([self.mp.q_psi, self.mp.q_y]))
        R_bar = np.kron(np.eye(hz), np.array([[self.mp.r_input]]))

        self.Phi = np.zeros((ny * hz, nx_aug))
        self.Gamma = np.zeros((ny * hz, nu * hz))
        A_pow = np.eye(nx_aug)

        for i in range(hz):
            A_next = self.A_aug @ A_pow
            self.Phi[i*ny:(i+1)*ny, :] = self.C_aug @ A_next
            for j in range(i + 1):
                term = self.C_aug @ np.linalg.matrix_power(self.A_aug, i - j) @ self.B_aug
                self.Gamma[i*ny:(i+1)*ny, j*nu:(j+1)*nu] = term
            A_pow = A_next

        # Unconstrained Solution Matrices (U = -M * error)
        self.H = self.Gamma.T @ Q_bar @ self.Gamma + R_bar
        self.M = np.linalg.inv(self.H) @ self.Gamma.T @ Q_bar

    def solve(self, current_state, last_input, reference_trajectory):
        """
        Calculates the optimal control input for the current step.
        """
        x_aug = np.concatenate([current_state, [last_input]])
        ref_vec = reference_trajectory.reshape(-1)
        
        # Prediction Error
        prediction_error = self.Phi @ x_aug - ref_vec
        
        # Analytical Solution
        delta_u_opt = (- self.M @ prediction_error)[0]
        
        # Apply Input constraints (Clipping)
        u_opt = np.clip(last_input + delta_u_opt, -np.radians(30), np.radians(30))
        
        return u_opt