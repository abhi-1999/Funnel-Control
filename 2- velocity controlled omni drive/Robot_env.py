import numpy as np

import math

import gymnasium as gym

from gymnasium.spaces import Box

from scipy.integrate import odeint

import matplotlib.pyplot as plt

import random

 

class RobotEnv(gym.Env):

    def __init__(self, **kwargs):

        """

        Must define self.acion_space and self.observation_space

        """

        #normalize action space

        self.action_space = Box(low = np.array([-1,-1,-1]), #lower bounds for vel_x,vel_y,omega

                                     high=np.array([1,1,1])) #upper bounds for vel_x,vel_y,omega

 

        self.observation_space = Box(low = np.array([-6.58,-4.63,-math.pi]), #lower bounds of state

                                      high=np.array([6.58,4.63,math.pi]),dtype=np.float64) #upper bounds of state

        self.epi_len = kwargs.get('epi_len')

        

        #RL constants

        self.ep_t = 0

        self.time_int = 0.01


        # self.t2 = self.timestep*self.time_int

        #reference trajectory

        ref_trajectory=[[-1.5+5.8*math.cos(0.24*self.time_int*t+1.5), 3*math.sin(0.24*t*self.time_int+1.5)] for t in range(self.epi_len)]

        self.state_d = np.array(ref_trajectory)

        

        #states

        self.x = np.clip(self.state_d[0,0],self.observation_space.low[0],self.observation_space.high[0]) + random.uniform(-1,1)
        self.y = np.clip(self.state_d[0,1],self.observation_space.low[1],self.observation_space.high[1]) + random.uniform(-1,1)
        self.theta = random.uniform(self.observation_space.low[2], self.observation_space.high[2])

        # self.x = -1.089

        # self.y = 3

        # self.theta = 0

 

        self.state = np.array([self.x,self.y,self.theta])

        self.funnel()

 

    def funnel(self):

        #Funnel for soft constraint currently values are according to paper

        l = np.array([0.7,0.7])

        ini_width = 0.07 #higher the value more is the initial funnel width

        rho_f = np.array([0.2,0.2])

        rho_0 = np.array(abs(np.array([self.x,self.y])- self.state_d[0,:]) + ini_width )

        gamma = np.array([(rho_0 - rho_f)*np.exp(-l*self.time_int*t) + rho_f for t in range(self.epi_len)])

        self.lb_soft = self.state_d - gamma

        self.ub_soft = self.state_d + gamma

        self.phi_ini_L = [0,0]

        self.phi_ini_U = [0,0]

 

        #creating final funnel

        mu = np.array([5,5])

        kc = np.array([3,3])

        self.lb_hard = np.array([-6.58,-4.63])

        self.ub_hard = np.array([6.58,4.63])

        t1 = np.linspace(0,self.time_int)

        self.phi_L,self.phi_U,self.Lb,self.Ub = [],[],[],[]

        self.j = []

        for i in range(self.epi_len):

            phi_Lo = odeint(self.bound, self.phi_ini_L, t1, args=(self.lb_soft[i,:], self.ub_hard, mu, kc))

            phi_sol_L = np.abs(phi_Lo[-1]) #this condition to be checked i.e,psi(modification signal) is always positive

            phi_U = odeint(self.bound, self.phi_ini_U, t1, args=(self.lb_hard, self.ub_soft[i,:], mu, kc))

            phi_sol_U = np.abs(phi_U[-1]) #this condition to be checked i.e,psi(modification signal) is always positive

            v = 10

            lower_bo = np.log(np.exp(v * (self.lb_soft[i,:] - phi_sol_L)) + np.exp(v * self.lb_hard)) / v

            upper_bo = -np.log(np.exp(-v * (self.ub_soft[i,:] + phi_sol_U)) + np.exp(-v * self.ub_hard)) / v

            self.phi_ini_L = phi_sol_L #will change according to comment in line 55

            self.phi_ini_U = phi_sol_U

            # self.phi_L.append(phi_sol_L)

            # self.phi_U.append(phi_sol_U)

            self.Lb.append(lower_bo)

            self.Ub.append(upper_bo)

            # self.j.append(i)

 

    def bound(self,phi,t,lb, ub, mu, kc):
        eta = ub-lb
        dphi_dt = 0.5*(1-np.sign(eta-mu))*(1/(eta+phi))-kc*phi
        return dphi_dt

 

    def step(self, action):

        #flag to check if episode is complete or not
        done = False

        #get new position
        vel_x,vel_y,omega = 3*action[0], 5*action[1], 1*action[2]
        x_old,y_old,theta_old = self.x,self.y,self.theta
        self.theta = theta_old + omega * self.time_int
        self.x = x_old + (vel_x * math.cos(self.theta) - vel_y * math.sin(self.theta)) * self.time_int
        self.y = y_old + (vel_x * math.sin(self.theta) + vel_y * math.cos(self.theta)) * self.time_int

        # To keep the angle theta between -pi to pi

        # if(self.theta > math.pi or self.theta < -math.pi ):

        #    self.theta = ((self.theta + math.pi)%(2*math.pi))-math.pi

        self.state = np.array([self.x,self.y,self.theta])

 

        #check if new position is within hard constraint

        x_min,y_min = self.Lb[self.ep_t]

        x_max,y_max = self.Ub[self.ep_t]

        clip = -5

        Rew_max = 1

        if self.lb_hard[0] <= self.x <= self.ub_hard[0] and self.lb_hard[1] <= self.y <= self.ub_hard[1] :

 

            robust1 = Rew_max - ((self.x - (x_min + x_max)/2)**2)*(4*Rew_max/((x_max-x_min)**2))

            robust2 = Rew_max - ((self.y - (y_min + y_max)/2)**2)*(4*Rew_max/((y_max-y_min)**2))

            reward = np.clip(min(robust1, robust2), clip,Rew_max)

        else:

            #terminate the episode or restart the episode?

            reward = 0

            done = True

 

        self.ep_t +=1

 

        if self.ep_t == self.epi_len:

            done = True

        info ={}

        info['x_min'] = x_min

        info['x_max'] = x_max

        info['y_min'] = y_min

        info['y_max'] = y_max

 

        truncated = done

        return self.state, reward, done, truncated, info

 

    def render(self, mode="human"):

        pass

 

    def reset(self,seed = None,options=None):

 

        # self.x = random.uniform(self.observation_space.low[0], self.observation_space.high[0])

        # self.y = random.uniform(self.observation_space.low[1], self.observation_space.high[1])

        # self.theta = random.uniform(self.observation_space.low[2], self.observation_space.high[2])

 

        self.x = np.clip(self.state_d[0,0],self.observation_space.low[0],self.observation_space.high[0]) + random.uniform(-1,1)

        self.y = np.clip(self.state_d[0,1],self.observation_space.low[1],self.observation_space.high[1]) + random.uniform(-1,1)

        self.theta = random.uniform(self.observation_space.low[2], self.observation_space.high[2])

 

        self.state = np.array([self.x,self.y,self.theta])

 

        #RL constants

        self.ep_t = 0

        self.funnel()

        x_max,y_max = self.Ub[0]
        x_min,y_min = self.Lb[0]

        info ={}
        info['x_min'] = x_min
        info['x_max'] = x_max
        info['y_min'] = y_min
        info['y_max'] = y_max

 

        return self.state,info

 

    def  close(self):

        pass

 

    def seed(self, seed=None):

        pass