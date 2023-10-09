import numpy as np
import torch
import math
import gymnasium as gym
from gymnasium.spaces import Box
from scipy.integrate import odeint
class RobotEnv(gym.Env):
    def __init__(self):
        """
        Must define self.acion_space and self.observation_space
        """
        # Normalize action space
        self.action_space = Box(low = np.array([-1,-1,-1]), #lower bounds for vel_x,vel_y,omega
                                     high=np.array([1,1,1])) #upper bounds for vel_x,vel_y,omega

        self.observation_space = Box(low = np.array([-6.58,-4.63,-math.pi]), #lower bounds of state
                                      high=np.array([6.58,4.63,math.pi]),dtype=np.float64) #upper bounds of state
        # self.epi_len = int(input("EpiLen"))
        self.epi_len = 10


        #states
        self.x = -3.19
        self.y = 3
        self.theta = 0
        self.state = torch.tensor([self.x, self.y, self.theta], dtype=torch.float64)

        # RL constants
        self.ep_t = 0
        self.time_int = 0.01

        # Reference trajectory
        ref_trajectory = [[-1.5 + 5.8 * math.cos(0.24 * self.time_int * t + 1.5),
                           5.8 * math.sin(0.24 * t * self.time_int + 1.5)] for t in range(self.epi_len)]
        self.state_d = torch.tensor(ref_trajectory, dtype=torch.float64)

        # Funnel for soft constraint currently values are according to paper
        l = torch.tensor([0.7, 0.7], dtype=torch.float64)
        ini_width = 0.2  # higher the value more is the initial funnel width
        rho_f = torch.tensor([0.2, 0.2], dtype=torch.float64)
        rho_0 = (torch.abs(torch.tensor([self.x, self.y], dtype=torch.float64) - self.state_d[0, :]) + ini_width).clone().detach()
        diff = rho_0 - rho_f

        gamma = torch.tensor([])
        for _ in range(self.epi_len):
            x = diff * torch.exp(-l*self.time_int*_) + rho_f
            gamma = torch.cat((gamma, x.unsqueeze(0)))

        #gamma_tensor = (rho_0.unsqueeze(0) - rho_f) * torch.exp(-l * time_int * torch.arange(epi_len, dtype=torch.float64).unsqueeze(1)) + rho_f
        self.lb_soft = self.state_d - gamma
        self.ub_soft = self.state_d + gamma
        self.phi_ini_L = torch.tensor([0.0, 0.0], dtype=torch.float64)
        self.phi_ini_U = torch.tensor([0.0, 0.0], dtype=torch.float64)

        # Creating final funnel
        mu = torch.tensor([3.0, 3.0], dtype=torch.float64)
        kc = torch.tensor([3.0, 3.0], dtype=torch.float64)
        self.lb_hard = torch.tensor([-6.58, -4.63], dtype=torch.float64)
        self.ub_hard = torch.tensor([6.58, 4.63], dtype=torch.float64)
        t1 = torch.linspace(0, self.time_int,100)
        self.phi_L, self.phi_U, self.Lb, self.Ub = [], [], [], []
        self.j = []
        for i in range(self.epi_len):
            phi_Lo = odeint(self.bound, self.phi_ini_L, t1, args=(self.lb_soft[i, :], self.ub_hard, mu, kc))
            phi_sol_L = torch.abs(torch.tensor(phi_Lo[-1]))  # this condition to be checked i.e, psi(modification signal) is always positive
            phi_U = odeint(self.bound, self.phi_ini_U, t1, args=(self.lb_hard, self.ub_soft[i, :], mu, kc))
            phi_sol_U = torch.abs(torch.tensor(phi_U[-1]))  # this condition to be checked i.e, psi(modification signal) is always positive
            v = 10.0
            lower_bo = torch.log(torch.exp(v * (self.lb_soft[i, :] - phi_sol_L)) + torch.exp(v * self.lb_hard)) / v
            upper_bo = -torch.log(torch.exp(-v * (self.ub_soft[i, :] + phi_sol_U)) + torch.exp(-v * self.ub_hard)) / v
            self.phi_ini_L = phi_sol_L  # will change according to comment in line 55
            self.phi_ini_U = phi_sol_U
            # self.phi_L.append(phi_sol_L)
            # self.phi_U.append(phi_sol_U)
            self.Lb.append(lower_bo)
            self.Ub.append(upper_bo)
            self.j.append(i)

    def bound(self, phi, t, lb, ub, mu, kc):
        eta = ub - lb
        dphi_dt = 0.5 * (1 - torch.sign(eta - mu)) * (1 / (eta + phi)) - kc * phi
        return dphi_dt
    
    def step(self, action):
        # Flag to check if episode is complete or not
        done = False

        # Get new position
        vel_x, vel_y, omega = 0.22 * action[0], 0.22 * action[1], 2.84 * action[2]
        x_old, y_old, theta_old = self.x, self.y, self.theta

        self.x = x_old + vel_x * self.time_int
        self.y = y_old + vel_y * self.time_int
        self.theta = theta_old + omega * self.time_int
        self.state = np.array([self.x, self.y, self.theta])

        # To keep the angle theta between -pi to pi
        if (self.theta > math.pi or self.theta < -math.pi):
            self.theta = ((self.theta + math.pi) % (2 * math.pi)) - math.pi

        #check if new position is within hard constraint
        x_min,y_min = self.Lb[self.ep_t]
        x_max,y_max = self.Ub[self.ep_t]

        Rew_max = 100
        if self.lb_hard[0] <= self.x <= self.ub_hard[0] and self.lb_hard[1] <= self.y <= self.ub_hard[1] :
            
            #check if within funnel and reward accordingly

 

            robust1 = Rew_max - ((self.x - (x_min + x_max)/2)**2)*(4*Rew_max/((x_max-x_min)**2))
            robust2 = Rew_max - ((self.y - (y_min + y_max)/2)**2)*(4*Rew_max/((y_max-y_min)**2))
            reward = np.clip(min(robust1, robust2), -10,Rew_max)
        else:
            #terminate the episode or restart the episode?
            reward = -100
            done = True

        self.ep_t +=1

        if self.ep_t == self.epi_len:
            done = True
        info ={}
        info['x_min'] = x_min
        info['x_max'] = x_max
        info['y_min'] = y_min
        info['y_max'] = y_max
        self.ep_t +=1

        if self.ep_t == self.epi_len:
            done = True

        info = {}
        truncated = done
        return self.state, reward, done, truncated, info

    def render(self, mode="human"):
        pass
    

    def reset(self, seed=None, options=None):
        self.x = -3.19
        self.y = 3
        self.theta = 0
        self.state = np.array([self.x, self.y, self.theta])
        self.ep_t = 0
        info = {}
        return self.state, info

    def close(self):
        pass

    def seed(self, seed=None):
        pass
