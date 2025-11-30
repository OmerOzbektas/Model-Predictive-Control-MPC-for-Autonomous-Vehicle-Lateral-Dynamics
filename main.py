import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp

# Import custom modules
from src.vehicle_model import VehicleParams, vehicle_dynamics_continuous
from src.mpc_controller import LateralMPC, MPCParams
from src.utils import generate_reference_path

def main():
    # --- 1. SETUP ---
    vp = VehicleParams()
    mpc_params = MPCParams(Ts=0.05, Horizon=15)
    controller = LateralMPC(vp, mpc_params)

    # Generate Reference Trajectory
    sim_duration = 10.0
    t_sim, x_sim, refs = generate_reference_path(sim_duration, mpc_params.Ts, vp.vx)

    # Simulation Arrays
    n_steps = len(t_sim)
    states_history = np.zeros((n_steps, 4))
    u_history = np.zeros(n_steps)

    # Initial Condition [vy, psi, psi_dot, Y]
    # Start exactly at the reference Y to avoid initial jump
    current_state = np.array([0.0, 0.0, 0.0, refs[0,1]])
    current_u = 0.0
    states_history[0] = current_state

    # --- 2. SIMULATION LOOP ---
    print("Starting simulation...")
    start_time = time.time()

    for i in range(n_steps - 1):
        # Handle Horizon Edges
        if i + mpc_params.Horizon < n_steps:
            ref_horizon = refs[i:i+mpc_params.Horizon]
        else:
            padding = (i + mpc_params.Horizon) - n_steps
            ref_horizon = np.vstack([refs[i:], np.tile(refs[-1], (padding, 1))])

        # Control Step
        u_opt = controller.solve(current_state, current_u, ref_horizon)

        # Physics Step (Integration)
        sol = solve_ivp(
            vehicle_dynamics_continuous, 
            [t_sim[i], t_sim[i+1]], 
            current_state, 
            args=(vp, u_opt), 
            method='RK45'
        )

        # Update
        current_state = sol.y[:, -1]
        current_u = u_opt
        
        # Log
        states_history[i+1] = current_state
        u_history[i+1] = u_opt

    print(f"Simulation done. Duration: {time.time() - start_time:.4f} s.")

    # --- 3. ANIMATION ---
    print("Preparing animation...")
    
    # Prepare data
    X_actual = x_sim
    Y_actual = states_history[:, 3]
    Psi_actual = states_history[:, 1]
    Steer_actual = u_history

    fig = plt.figure(figsize=(16, 9), facecolor='white')
    gs = gridspec.GridSpec(3, 2, height_ratios=[3, 1, 1])

    # Top Plot (Path)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_title('MPC Lateral Control Simulation', fontsize=14)
    ax0.plot(x_sim, refs[:, 1], 'b--', linewidth=2, label='Reference')
    car_body, = ax0.plot([], [], 'k-', linewidth=3)
    car_f_wheel, = ax0.plot([], [], 'r-', linewidth=4)
    car_traj, = ax0.plot([], [], 'r-', linewidth=1, alpha=0.5)
    
    ax0.set_xlim(x_sim[0], x_sim[-1])
    ax0.set_ylim(np.min(Y_actual) - 5, np.max(Y_actual) + 5)
    ax0.set_aspect('equal')
    ax0.legend()
    ax0.grid(True)

    # Bottom Plots
    ax1 = fig.add_subplot(gs[1, 0])
    steer_line, = ax1.plot([], [], 'r-', linewidth=2)
    ax1.set_xlim(0, t_sim[-1])
    ax1.set_ylim(np.degrees(np.min(Steer_actual))-5, np.degrees(np.max(Steer_actual))+5)
    ax1.set_ylabel('Steering [deg]')
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[1, 1])
    yaw_line, = ax2.plot([], [], 'b-', linewidth=2)
    ax2.set_xlim(0, t_sim[-1])
    ax2.set_ylabel('Yaw [deg]')
    ax2.grid(True)

    def update(frame):
        # Update Car Geometry
        curr_x, curr_y = X_actual[frame], Y_actual[frame]
        curr_psi = Psi_actual[frame]
        lf, lr = vp.lf, vp.lr
        
        # Body
        bx = [curr_x - lr*np.cos(curr_psi), curr_x + lf*np.cos(curr_psi)]
        by = [curr_y - lr*np.sin(curr_psi), curr_y + lf*np.sin(curr_psi)]
        car_body.set_data(bx, by)
        
        # Front Wheel (Visual only)
        w_angle = curr_psi + Steer_actual[frame]
        wx = [bx[1] - 0.5*np.cos(w_angle), bx[1] + 0.5*np.cos(w_angle)]
        wy = [by[1] - 0.5*np.sin(w_angle), by[1] + 0.5*np.sin(w_angle)]
        car_f_wheel.set_data(wx, wy)
        
        # Trajectory
        car_traj.set_data(X_actual[:frame], Y_actual[:frame])
        
        # Plots
        steer_line.set_data(t_sim[:frame], np.degrees(Steer_actual[:frame]))
        yaw_line.set_data(t_sim[:frame], np.degrees(Psi_actual[:frame]))
        
        return car_body, car_f_wheel, car_traj, steer_line, yaw_line

    ani = animation.FuncAnimation(fig, update, frames=n_steps, interval=mpc_params.Ts*1000, blit=True)
    ani.save('media/mpc_animation.gif', writer='pillow', fps=20)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    main()
