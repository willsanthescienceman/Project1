# 4th Order Runge Kutta
import numpy as np
import matplotlib.pyplot as plt

# Debugging
msg_lvl = 5
def msg(lvl, msg):
    if msg_lvl > lvl:
        print(msg)

# --- COEFFICIENTS --- #

def k_1(y, t, mass, k, func):
    f = [y[1]] # velocity, dy_0/dt
    f.append(func(y[0], mass, k)) # acceleration, dy_1/dt
    return f

def k_2(y, t, step, k_1, mass, k, func):
    yaux = [y[0]+step * k_1[0]/2]
    yaux.append(y[1]+step * k_1[1]/2)
    return f(yaux, t+step/2, mass, k, func)

def k_3(y, t, step, k_2, mass, k, func):
    yaux = [y[0]+step * k_2[0]/2]
    yaux.append(y[1]+step * k_2[1]/2)
    return f(yaux, t+step/2, mass, k, func)

def k_4(y, t, step, k_3, mass, k, func):
    yaux = [y[0]+step * k_3[0]]
    yaux.append(y[1]+step * k_3[1])
    return f(yaux, t+step/2, mass, k, func)

# --- UPDATING VALUES --- #

# Update Derivatives
def f(yaux, t, m, k, func):
    f = [yaux[1]]
    f.append(func(yaux[1], m, k))
    return f

# Updates the y array
def update_y(y, t, func, step=0.1, mass=10, k=1):
    # Get coefficients
    k1 = k_1(y, t, mass, k, func)
    msg(6, f"k1: {k1}")
    msg(8, f"k1[0]: {k1[0]}")
    k2 = k_2(y, t, step, k1, mass, k, func)
    k3 = k_3(y, t, step, k2, mass, k, func)
    k4 = k_4(y, t, step, k3, mass, k, func)

    # Update each index in y
    y[0] = update_value(0, y, k1, k2, k3, k4, step)
    y[1] = update_value(1, y, k1, k2, k3, k4, step)
    
    return y

# Updates a single value (one index of the y array)
def update_value(i, y, k_1, k_2, k_3, k_4, step):
    return (y[i] + (k_1[i] + 2*k_2[i] + 2*k_3[i] + k_4[i])/6)

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
        pos_arr.append(y[0])
        vel_arr.append(y[1])
        pos = pos_arr[-1]
        vel = vel_arr[-1]
        PE = -k * pos**2
        PE_arr.append(PE)
        KE = 0.5 * mass * vel**2
        KE_arr.append(KE)
        total_energy.append(PE + KE)

    return time_arr, pos_arr, vel_arr, PE_arr, KE_arr, total_energy
    
def plt_energy(time_arr, energy_arr, show=True):
    plt.figure(figsize=(8, 6))
    plt.plot(time_arr, energy_arr, label='Runge Kutta')
    if show:
        plt.xlabel('Time (s)')
        plt.ylabel('Total Energy (J)')
        plt.title('Total Energy of SHM System (Runge Kutta Method)')
        plt.grid(True)
        plt.legend()
        plt.show()
