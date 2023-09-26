import numpy as np
import math
from scipy.integrate import odeint
from numpy.linalg import norm
from scipy.integrate import solve_ivp as rk4
from scipy import integrate
from scipy.special import expit, logsumexp
import csv
from csv import writer
class RobotEnv:
    def __init__(self):
        self.x = -3.19
        self.y = 1.70
        self.theta = -0.33
        self.vel = 0.2
        self.ang_vel = -0.1
        self.psi = [self.x,self.y,self.theta,self.vel,self.ang_vel]
        self.max_action = [25,10]
        self.timestep = 0
        self.epi_len = 2500
        h = 25/self.epi_len
        self.t2 = self.timestep*h
        self.time_int = h
        xd=[[-1.5+5.8*math.cos(0.24*h*t+1.5), 3.8*math.sin(0.24*t*h+1.5)] for t in range(self.epi_len)]
        self.state_d = np.array(xd)
        l = np.array([0.1,0.1])
        rho_f = np.array([1.5,1.5])
        rho_0 = np.array(abs(np.array([self.x,self.y])- self.state_d[0,:]) + 0.2)
        gamma = np.array([(rho_0 - rho_f)*np.exp(-l*h*t) + rho_f for t in range(self.epi_len)])
        self.lb_soft = self.state_d - gamma
        self.ub_soft = self.state_d + gamma
        self.phi_ini_L = [0,0]
        self.phi_ini_U = [0,0]
        self.j = 0
#---------------------------------------------------
    def sample(self):
        x = np.linspace(-10,10,10000)
        y1 = 2*self.max_action[0]*(expit(0.5*x)-0.5)
        y2 = 2*self.max_action[1]*(expit(0.5*x)-0.5)
        random_index1 = np.random.randint(0,y1.shape[0])
        random_index2 = np.random.randint(0,y2.shape[0])
        return [y1[random_index1],y2[random_index2]]

    def reset(self):
        self.timestep = 0
        self.x = -3.19
        self.y = 1.70
        self.theta = -0.33
        self.vel = 0.2
        self.ang_vel = -0.1
        self.psi = [self.x, self.y, self.theta, self.vel, self.ang_vel]
        h = 25/self.epi_len
        self.t2 = self.timestep*h
        self.phi_ini_L = [0,0]
        self.phi_ini_U = [0,0]
        self.j = 0
        return np.array([self.x, self.y, self.theta, self.t2])

    def bound(self,phi,t,lb, ub, mu, kc):
        eta = ub-lb
        dphi_dt = 0.5*(1-np.sign(eta-mu))*(1/(eta+phi))-kc*phi
        return dphi_dt
  #    --------ADD FUNNEL HERE---------
    def funnel(self,lb_soft, ub_soft, phi_ini_L, phi_ini_U):
        mu = np.array([6,9])   #For different states
        kc = np.array([5.0, 4.0])   #For different states
        # Hard Constraints - Boundary
        lb_hard = np.array([-6.58, -4.63])
        ub_hard = np.array([6.58, 4.63])

        # Soft Constraints - Trajectory tracking-as input
        t1 = [0,25/self.epi_len]
        t, phi_L = odeint(self.bound, phi_ini_L, t1, args=(lb_soft, ub_hard, mu, kc))
        phi_sol_L = np.abs(phi_L)
        t, phi_U = odeint(self.bound, phi_ini_U, t1, args=(lb_hard, ub_soft, mu, kc))
        phi_sol_U = np.abs(phi_U)
        v = 10
        Lb = np.log(np.exp(v * (lb_soft - phi_sol_L)) + np.exp(v * lb_hard)) / v
        Ub = -np.log(np.exp(-v * (ub_soft + phi_sol_U)) + np.exp(-v * ub_hard)) / v
        return phi_sol_L, phi_sol_U, Lb, Ub

    def step(self,action):
        done = False
        m = 3.6; I=0.0405; L=0.15
        h = 25/self.epi_len
        self.t2 = (self.timestep*h)
        d = [0.75*np.sin(2*self.t2+math.pi/3) + 1.5*np.cos(3*self.t2+3*math.pi/7), 0.75*np.sin(5*self.t2-math.pi/3) + 0.25*np.cos(3*self.timestep+3*math.pi/6)]
        # https://towardsdatascience.com/solving-non-linear-differential-equations-numerically-using-the-finite-difference-method-1532d0863755
        x_old, y_old, theta_old, vel_old, ang_vel_old = self.x, self.y, self.theta, self.vel, self.ang_vel

        self.vel = vel_old + (action[0] + d[0] - 0.3*vel_old)*self.time_int/m
        self.ang_vel = ang_vel_old + (action[1] + d[1] -0.004*ang_vel_old)*self.time_int/I

        #new position and orientation depends on the old + current change
        self.theta = theta_old + (self.ang_vel*self.time_int)
        self.x = x_old + (self.vel*math.cos(self.theta) - L*self.ang_vel*math.sin(self.theta))*self.time_int
        self.y = y_old + (self.vel*math.sin(self.theta) + L*self.ang_vel*math.cos(self.theta))*self.time_int
        
# ------------------------------------------------------------------------
        # To keep the angle theta between -pi to pi
        if(self.theta > math.pi or self.theta < -math.pi ):
           self.theta = ((self.theta + math.pi)%(2*math.pi))-math.pi
        self.psi = [self.x,self.y, self.theta, self.vel, self.ang_vel]

        ### -------------ADD REWARDS HERE--------------
        self.phi_sol_L,self.phi_sol_U,Lb,Ub = self.funnel(self.lb_soft[self.j,:],self.ub_soft[self.j,:],np.array(self.phi_ini_L),np.array(self.phi_ini_U))
        self.x_min, self.x_max, self.y_min, self.y_max = Lb[0],Ub[0], Lb[1],Ub[1]
        
        # data = [Lb[0],Ub[0],Lb[1],Ub[1]] 
        # with open('dataset1.csv', 'a', encoding='UTF8') as f:
        #     file = csv.writer(f)
        #     # write the data
        #     file.writerow(data)
        #     f.close()

        self.phi_ini_L = self.phi_sol_L
        self.phi_ini_U = self.phi_sol_U
        self.j += 1
        reward = 0
        Rew_max = 5
        robust1 = Rew_max - ((self.x - (self.x_min + self.x_max)/2)**2)*(4*Rew_max/((self.x_max-self.x_min)**2))
        robust2 = Rew_max - ((self.y - (self.y_min + self.y_max)/2)**2)*(4*Rew_max/((self.y_max-self.y_min)**2))
        # print("typ:",type(robust1),type(action),robust1,action)
        reward = np.clip(min(robust1, robust2), -0.5,Rew_max)
        # reward = min(robust1, robust2)
        # print("rew", reward)
        # if (self.x_min <= self.x and self.x<=self.x_max and self.y_min<=self.y and self.y<=self.y_max):
        #     print(".")
        #     # print("within constraints",reward)
        #     # print("action", action, self.timestep)
        #     # print("states: ", self.x,self.y, self.x_min,self.x_max, self.y_min, self.y_max)
        # else:
        #     print("outside constraints", self.x,self.y,self.x_min, self.x_max,self.y_min, self.y_max)
        #     print("actions",action,self.timestep)
        self.timestep+=1
        if(self.timestep >= self.epi_len):
            done = True
        #sself.t2 = self.tipsi_init = [self.x,self.y,self.theta,self.vel,self.ang_vel]mestep/self.epi_len
        return np.array([self.x, self.y, self.theta, self.t2]), reward, done, None


    def get_action_dim(self):
        return 2        # 2 actions - u1_bar and u2_bar
    
    def get_state_dim(self):
        return 4
