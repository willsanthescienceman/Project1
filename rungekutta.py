# Claire O'Connor, Amelia Abruscato & Shreyan Goswami
# Runge Kutta
import numpy as np

def calc_next_step(y, t, step=0.1, mass=10, k=1):
    k1 = compute_derivaties(y, t, mass, k)
    k2 = compute_k2(y, t, step, k1, mass, k)
    return update_y(y, k1, k2, step)

def compute_derivaties(y, t, mass, k):
    derivative = [y[1]] # velocity, dy_0/dt
    derivative.append(-k*y[0]/mass) # acceleration, dy_1/dt
    return derivative

def compute_k2(y, t, step, k_1, mass, k):
    yaux = [y[0]+step * k_1[0]/2]
    yaux.append(y[1]+step * k_1[1]/2)
    return compute_derivaties(yaux, t+step/2, mass, k)

def update_y(y, k_1, k_2, step):
    y[0] = y[0] + step*(k_1[0] + 2*k_2[0])/6 # new position
    y[1] = y[1] + step*(k_1[1] + 2*k_2[1])/6 # new velocity
    return y
    
def RungeKutta(pos_0, vel_0, step_size=0.1, max_time=10, mass=10, k=1):

    # User input
    x_0 = pos_0
    y = [x_0, vel_0]

    # Making time array
    time_arr = np.arange(0, max_time/step_size)
    time_arr = time_arr * step_size

    # Applying Runge Kutta approximation
    pos_arr = [y[0]]
    vel_arr = [y[1]]
    PE_arr = [-k * x_0**2]
    KE_arr = [0.5 * mass * vel_0**2]
    #total_energy = [KE_arr[0] + PE_arr[0]]
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
    
    
