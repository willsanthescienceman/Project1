# 4th Order Runge Kutta
import numpy as np

# --- USER MANAGED --- #
def acc(pos, mass, k):
    return -k*pos/mass

# --- COEFFICIENTS --- #

def k_1(y, t, mass, k):
    f = [y[1]] # velocity, dy_0/dt
    f.append(acc(y[0], mass, k)) # acceleration, dy_1/dt
    return f

def k_2(y, t, step, k_1, mass, k):
    yaux = [y[0]+step * k_1[0]/2]
    yaux.append(y[1]+step * k_1[1]/2)
    return compute_derivaties(yaux, t+step/2, mass, k)

def k_3(y, t, step, k_2, mass, k):
    yaux = [y[0]+step * k_2[0]/2]
    yaux.append(y[1]+step * k_2[1]/2)
    return compute_derivaties(yaux, t+step/2, mass, k)

def k_4(y, t, step, k_3, mass, k):
    yaux = [y[0]+step * k_3[0]]
    yaux.append(y[1]+step * k_3[1])
    return compute_derivaties(yaux, t+step/2, mass, k)

# --- UPDATING VALUES --- #

# Updates the y array
def update_y(y, t, step=0.1, mass=10, k=1):
    # Get coefficients
    k1 = k_1(y, t, mass, k)
    k2 = k_2(y, t, step, k1, mass, k)
    k3 = k_3(y, t, step, k2, mass, k)
    k4 = k_4(y, t, step, k3, mass, k)

    # Update each index in y
    y[0] = update_value(y[0], k_1, k_2, k_3, k_4, step)
    y[1] = update_value(y[1], k_1, k_2, k_3, k_4, step)
    
    return y

# Updates a single value (one index of the y array)
def update_value(y_index, k_1, k_2, k_3, k_4, step):
    return (y[0] + (k_1[0] + 2*k_2[0] + 2*k_3[0] + k4)/6)

# --- CREATE ARRAY --- #
    
def RungeKutta(pos_0, vel_0, step_size=0.1, max_time=10, mass=10, k=1):

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
        y = calc_next_step(y, time)
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
    
    
