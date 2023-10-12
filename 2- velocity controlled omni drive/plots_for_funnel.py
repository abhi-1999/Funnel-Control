import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

epi_len = 10
h = 0.01
xd=[[-1.5+5.8*math.cos(0.24*h*t+1.5), 3*math.sin(0.24*t*h+1.5)] for t in range(epi_len)]
state_d = np.array(xd)


plt.plot(state_d[:,0],state_d[:,1])

x = -1.089
y = 3
theta = 0
time_int = h
xd=[[-1.5+5.8*math.cos(0.24*time_int*t+1.5), 3*math.sin(0.24*t*time_int+1.5)] for t in range(epi_len)]
state_d = np.array(xd)
l = np.array([0.7,0.7])
rho_f = np.array([0.2,0.2])
rho_0 = np.array(abs(np.array([x,y])- state_d[0,:]) + 0.01)
gamma = np.array([(rho_0 - rho_f)*np.exp(-l*time_int*t) + rho_f for t in range(epi_len)])
gamma_x = np.array([(rho_0[0] - rho_f[0])*np.exp(-l[0]*time_int*t) + rho_f[0] for t in range(epi_len)])
gamma_y = np.array([(rho_0[1] - rho_f[1])*np.exp(-l[1]*time_int*t) + rho_f[1] for t in range(epi_len)])
lb_soft_x = state_d[:,0] - gamma_x
lb_soft_y = state_d[:,1] - gamma_y
ub_soft_x = state_d[:,0] + gamma_x
ub_soft_y = state_d[:,1] + gamma_y
lb_soft = state_d - gamma
ub_soft = state_d + gamma
phi_ini_L = [0,0]
phi_ini_U = [0,0]

time = np.linspace(0,h*epi_len,epi_len)
plt.plot(time,lb_soft[:,0],time,ub_soft[:,0])
plt.grid()

plt.plot(time,lb_soft[:,1],time,ub_soft[:,1])
plt.grid()



def bound(phi,t,lb, ub, mu, kc):
    eta = ub-lb
    dphi_dt = 0.5*(1-np.sign(eta-mu))*(1/(eta+phi))-kc*phi
    return dphi_dt
  #    --------ADD FUNNEL HERE---------
def funnel(lb_soft, ub_soft, phi_ini_L, phi_ini_U):
    mu = np.array([3,3])   #For different states
    kc = np.array([3.0, 3.0])   #For different states
    # Hard Constraints - Boundary
    lb_hard = np.array([-6.58, -4.63])
    ub_hard = np.array([6.58, 4.63])

    # Soft Constraints - Trajectory tracking-as input
    t1 = [0,h]
    t, phi_L = odeint(bound, phi_ini_L, t1, args=(lb_soft, ub_hard, mu, kc))
    phi_sol_L = phi_L #np.abs(phi_L)
    t, phi_U = odeint(bound, phi_ini_U, t1, args=(lb_hard, ub_soft, mu, kc))
    phi_sol_U = phi_U #np.abs(phi_U)
    v = 10
    Lb = np.log(np.exp(v * (lb_soft - phi_sol_L)) + np.exp(v * lb_hard)) / v
    Ub = -np.log(np.exp(-v * (ub_soft + phi_sol_U)) + np.exp(-v * ub_hard)) / v
    return phi_sol_L, phi_sol_U, Lb, Ub

low_bound,upper_bound,time = [],[],[]
for j in range(epi_len):
  phi_sol_L,phi_sol_U,Lb,Ub = funnel(lb_soft[j,:],ub_soft[j,:],np.array(phi_ini_L),np.array(phi_ini_U))
  phi_ini_L = phi_sol_L
  phi_ini_U = phi_sol_U
  low_bound.append(Lb)
  upper_bound.append(Ub)
  time.append(j*h)

low_x = [inner_list[0] for inner_list in low_bound]
low_y = [inner_list[1] for inner_list in low_bound]
high_x = [inner_list[0] for inner_list in upper_bound]
high_y = [inner_list[1] for inner_list in upper_bound]

lb_hard = [-6.58 for i in range(epi_len)]
ub_hard = [6.58 for i in range(epi_len)]

plt.plot(time,low_x,time,high_x,time, lb_hard,time,ub_hard)
plt.grid()

lb_hard_y = [-4.63 for i in range(epi_len)]
ub_hard_y = [4.63 for i in range(epi_len)]
plt.plot(time,low_y,time,high_y,time,lb_hard_y,time,ub_hard_y)
plt.grid()
