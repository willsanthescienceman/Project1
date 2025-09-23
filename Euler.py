# Euler's Method

import numpy as np
import matplotlib.pyplot as plt
import rungekutta as rk

# Debugging
msg_lvl = 5
def msg(lvl, msg):
    if msg_lvl < lvl:
        print(msg)

t_max = 10.0
dt = 0.1

# Calculate number of steps
num_steps = int(t_max / dt)

#arrays to store results
time = np.arange(num_steps)*dt
position = np.zeros(num_steps)
velocity = np.zeros(num_steps)
total_energy = np.zeros(num_steps)

# Initial conditions
position[0] = x0
velocity[0] = v0

# Acceleration
def calculate_acceleration(x):
    return -omega**2 * x

# Total Energy
def calculate_total_energy(x, v, m, k):
    kinetic_energy = 0.5 * m * v**2
    potential_energy = 0.5 * k * x**2
    return kinetic_energy + potential_energy

# Euler method simulation
def Euler(x0, v0, t_max=10.0, dt=0.1):

    num_steps = int(t_max / dt)

    time = np.arange(num_steps)*dt
    position = np.zeros(num_steps)
    velocity = np.zeros(num_steps)
    total_energy = np.zeros(num_steps)

    # Initial conditions
    position[0] = x0
    velocity[0] = v0

    for i in range(1, num_steps):
        time[i] = time[i-1] + dt
    
        # Calculate acceleration at current state
        a = calculate_acceleration(position[i-1])

        # Updated  position and velocity using Euler method
        velocity[i] = velocity[i-1] + a * dt
        position[i] = position[i-1] + velocity[i] * dt # Note: Using v[i-1] for position update

        # Calculate total energy
        total_energy[i] = calculate_total_energy(position[i], velocity[i], m, k)

    return time, position, velocity, total_energy

    
# Plotting the total energy
show_Euler = True
if show_Euler:
    plt.figure(figsize=(8, 6))
    plt.plot(time, total_energy, label='Euler')
    plt.xlabel('Time (s)')
    plt.ylabel('Total Energy (J)')
    plt.title('Total Energy of SHM System (Euler Method)')
    plt.grid(True)
    #plt.legend()
    #plt.show()
    
# Runge Kutta method
time, pos, vel, PE, KE, total_energy = rk.RungeKutta(x0, v0, dt, t_max, m, k)
total_energy = np.array(total_energy)

# Plotting the total energy
show_RK = True
if show_RK:
    plt.plot(time, total_energy, label='Runge Kutta')
    plt.legend()
    plt.show()


# SIMULATION FUNCTIONS
def analytic_trajectory(x_arr, xo, yo, vox, voy, g):
    # take initial position in two axes (xo and yo), x_arr is the array of x positions, vox is the initial velocity in the x direction, voy same in y direction
    #this is basically y in terms of x (making a quadratic out of it, removing time)
    return yo + (voy / vox) * (x_arr - xo) - (0.5 * g / vox**2) * (x_arr - xo)**2

def calculate_drag_force(velocity):
    #calculating drag force by multiplying velocity with the drag coefficient
    return np.array([-cd * velocity[0] * abs(velocity[0]),
                     -cd * velocity[1] * abs(velocity[1])])

def total_force(velocity):
    #adding the force due to g and drag
    Fd = calculate_drag_force(velocity)
    Fg = -m * g
    return np.array([Fd[0], Fd[1] + Fg])

def moment_later(position, velocity, force, dt):
    #finding the new posotion and velocity 
    acceleration = force / m
    new_pos = position + velocity * dt
    new_vel = velocity + acceleration * dt
    return new_pos, new_vel

def calculate_energy(position, velocity):
    #using standard formulae to find Kinetic and Potential Energy.
    KE = 0.5 * m * np.sum(velocity**2)
    PE = m * g * position[1]
    E = KE + PE
    return KE, PE, E

def write_positions(times, positions, filename="position.out"):
    #entering the positions into position.out file
    with open(filename, "w") as f:
        for t, pos in zip(times, positions):
            f.write(f"{t:.5f} {pos[0]:.5f} {pos[1]:.5f}\n")

def run_simulation():
    position = np.array([xo, yo], dtype=float)
    velocity = np.array([vox, voy], dtype=float)
    time = 0.0

    # STORING DATA
    times, positions, velocities = [], [], []
    kinetic_energy, potential_energy, total_energy = [], [], []

#making a loop to ensure that it runs till position is 0
    while position[1] > 0:
        force = total_force(velocity)
        position, velocity = moment_later(position, velocity, force, dt)
        ke, pe, te = calculate_energy(position, velocity)

        times.append(time)
        positions.append(position.copy())
        velocities.append(np.linalg.norm(velocity))
        kinetic_energy.append(ke)
        potential_energy.append(pe)
        total_energy.append(te)

        time += dt
        msg(5, f"Time: {time:.4f}, Pos: {position}, Vel: {velocity}, KE: {ke:.2f}, PE: {pe:.2f}, E: {te:.2f}")

    return np.array(times), np.array(positions), np.array(kinetic_energy), np.array(potential_energy), np.array(total_energy)

# PLOTTING
def plot_results(times, positions, kinetic, potential, total, xo, yo, vox, voy, g, dt):
    x_vals = positions[:, 0]
    y_vals = positions[:, 1]

    # TRAJECTORY
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    ax1.plot(x_vals, y_vals, label="Numeric", color="blue")

    x_analytic = np.linspace(xo, x_vals[-1], 500)
    y_analytic = analytic_trajectory(x_analytic, xo, yo, vox, voy, g)
    ax1.plot(x_analytic, y_analytic, '--', label="Analytic", color="red")
    #plot specs
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("y vs x (no drag)")
    ax1.set_ylim(bottom=0)
    ax1.legend()
    ax1.text(
        0.05, 0.95,  # x and y in *axes fraction* (0 to 1)
        f"dt = {dt}",  # Text string
        transform=ax1.transAxes,  # Use axes coordinates
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
    )
    ax1.grid()

    plt.tight_layout()
    fig1.savefig(f"trajectory_dt_{dt}.png", dpi=300)
    plt.close(fig1)  # Close figure after saving to avoid overlap

    # ENERGY
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    ax2.plot(times, kinetic, label="Kinetic", color="blue")
    ax2.plot(times, potential, label="Potential", color="orange")
    ax2.plot(times, total, label="Total", color="green")

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Energy (J)")
    ax2.set_title(f"Energy vs Time")
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    fig2.savefig("energy.png", dpi=300)
    plt.close(fig2)  # Close figure to free memory

# MAIN FUNCTION
def main():
    times, positions, kinetic, potential, total = run_simulation()
    write_positions(times, positions)
    plot_results(times, positions, kinetic, potential, total, xo, yo, vox, voy, g, dt)

