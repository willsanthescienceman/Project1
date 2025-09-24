# 4th Order Runge Kutta
import numpy as np
import matplotlib.pyplot as plt


# Debugging
msg_lvl = 5
def msg(lvl, msg):
    if msg_lvl > lvl:
        print(msg)

# --- UPDATING VALUES --- #

# Update Derivatives
def f(yaux, t, m, k, func):
    f = [yaux[1]]
    f.append(func(yaux[1], m, k))
    return f

# Updates the y array
def update_y(y, t, func, step=0.1, mass=10, k=1):
    # --- COEFFICIENTS --- #
    #k1 = k_1(y, t, mass, k, func)
    
    k1 = [y[1]] # velocity, dy_0/dt
    k1.append(func(y[0], mass, k)) # acceleration, dy_1/dt
    msg(4, f"f: {f}")

    yaux = [y[0] + step * k1[0]/2]
    yaux.append(y[1] + step * k1[1]/2)
    k2 = f(yaux, t+step/2, mass, k, func)

    yaux = [y[0] + step * k2[0]/2]
    yaux.append(y[1] + step * k2[1]/2)
    k3 = f(yaux, t+step/2, mass, k, func)

    yaux = [y[0] + step * k3[0]]
    yaux.append(y[1] + step * k3[1])
    k4 = f(yaux, t+step/2, mass, k, func)

    msg(6, f"k1: {k1}")
    msg(8, f"k1[0]: {k1[0]}")

    # Update each index in y
    y[0] = update_value(0, y, k1, k2, k3, k4, step)
    y[1] = update_value(1, y, k1, k2, k3, k4, step)
    
    return y

# Updates a single value (one index of the y array)
def update_value(i, y, k_1, k_2, k_3, k_4, step):
    return (y[i] + step*(k_1[i] + 2*k_2[i] + 2*k_3[i] + k_4[i])/6)

# --- CREATE ARRAY --- #
    
def RungeKutta(pos_0, vel_0, func, step_size=0.1, max_time=10, mass=10, k=1):

    # User input
    x_0 = pos_0
    y = [x_0, vel_0]

    # Making time array
    time_arr = np.arange(0, max_time/step_size)
    time_arr = time_arr * step_size

    # Applying Runge Kutta approximation
    pos_arr = []
    vel_arr = []
    PE_arr = []
    KE_arr = []
    total_energy = []
    for time in time_arr:
        y = update_y(y, time, func)
        pos = y[0]
        vel = y[1]
        pos_arr.append(pos)
        vel_arr.append(vel)
        
        PE =  0.5 * k * pos**2
        PE_arr.append(PE)
        KE = 0.5 * mass * vel**2
        KE_arr.append(KE)
        E = PE + KE
        total_energy.append(E)
    msg(8, f"total_energy: {total_energy}")
    msg(4, f"KE: {KE_arr}")
    msg(8, f"pos_arr: {pos_arr}")
    
    return time_arr, pos_arr, vel_arr, PE_arr, KE_arr, total_energy
    
def plt_energy(time_arr, energy_arr, PE, show=True):
    plt.figure(figsize=(8, 6))
    plt.plot(time_arr, energy_arr, label='Runge Kutta')
    plt.plot(time_arr, energy_arr, label='PE')
    if show:
        plt.xlabel('Time (s)')
        plt.ylabel('Total Energy (J)')
        plt.title('Total Energy of SHM System (Runge Kutta Method)')
        plt.grid(True)
        plt.legend()
        plt.show()
