import tensorflow as tf
import numpy as np
import math
import gymnasium as gym
from gymnasium.spaces import Box
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class RobotEnv(gym.Env):
    def __init__(self):

        """
        Must define self.acion_space and self.observation_space
        """
        #normalize action space
        self.action_space = Box(low = np.array([-1,-1,-1]), #lower bounds for vel_x,vel_y,omega
                                     high=np.array([1,1,1])) #upper bounds for vel_x,vel_y,omega

        self.observation_space = Box(low = np.array([-6.58,-4.63,-math.pi]), #lower bounds of state
                                      high=np.array([6.58,4.63,math.pi]),dtype=np.float64) #upper bounds of state
        self.epi_len = 10

        #states
        self.x = tf.Variable(-3.19, dtype=tf.float32)
        self.y = tf.Variable(3.0, dtype=tf.float32)
        self.theta = tf.Variable(0.0, dtype=tf.float32)
        self.state = tf.stack([self.x, self.y, self.theta])

        # RL constants
        self.ep_t = tf.Variable(0, dtype=tf.int32, name='episode_time')

        self.time_int = 0.01

        ref_trajectory_np = np.array(
            [[-1.5 + 5.8 * np.cos(0.24 * self.time_int * t + 1.5), 5.8 * np.sin(0.24 * t * self.time_int + 1.5)]
             for t in range(self.epi_len)], dtype=np.float32)

        self.ref_trajectory = tf.convert_to_tensor(ref_trajectory_np, dtype=tf.float32, name='ref_trajectory')
    
        self.state_d = self.ref_trajectory

        # NOT WORKING DIRECT TENSOR

        # l = tf.constant([0.7, 0.7], dtype=tf.float32, name='l')
        # ini_width = 0.2
        # rho_f = tf.constant([0.2, 0.2], dtype=tf.float32, name='rho_f')
        # rho_0 = tf.abs(self.state[:2] - self.state_d[0, :]) + ini_width
        # gamma = (rho_0 - rho_f) * tf.exp(-l * self.time_int * tf.range(self.epi_len, dtype=tf.float32))

        l = np.array([0.7,0.7])
        ini_width = 0.2 #higher the value more is the initial funnel width
        rho_f = np.array([0.2,0.2])
        rho_0 = np.array(abs(np.array([self.x,self.y])- self.state_d[0,:]) + ini_width )
        gamma_np = np.array([(rho_0 - rho_f)*np.exp(-l*self.time_int*t) + rho_f for t in range(self.epi_len)])
        gamma = tf.convert_to_tensor(gamma_np, dtype=tf.float32, name='gamma')

        self.lb_soft = self.state_d - gamma
        self.ub_soft = self.state_d + gamma
        self.phi_ini_L = tf.constant([0.0, 0.0], dtype=tf.float32, name='phi_ini_L')
        self.phi_ini_U = tf.constant([0.0, 0.0], dtype=tf.float32, name='phi_ini_U')

        # creating final funnel
        mu = tf.constant([3.0, 3.0], dtype=tf.float32, name='mu')
        kc = tf.constant([3.0, 3.0], dtype=tf.float32, name='kc')
        self.lb_hard = tf.constant([-6.58, -4.63], dtype=tf.float32, name='lb_hard')
        self.ub_hard = tf.constant([6.58, 4.63], dtype=tf.float32, name='ub_hard')
        t1 = tf.linspace(0.0, self.time_int, num=2)
        self.phi_L, self.phi_U, self.Lb, self.Ub = [], [], [], []
        self.j = []
        for i in range(self.epi_len):
            phi_Lo = tf.numpy_function(self.bound, [self.phi_ini_L, t1, self.lb_soft[i, :], self.ub_hard, mu, kc],
                                       tf.float32)
            phi_sol_L = tf.abs(phi_Lo[-1])
            phi_U = tf.numpy_function(self.bound, [self.phi_ini_U, t1, self.lb_hard, self.ub_soft[i, :], mu, kc],
                                      tf.float32)
            phi_sol_U = tf.abs(phi_U[-1])
            v = 10.0
            lower_bo = tf.math.log(tf.exp(v * (self.lb_soft[i, :] - phi_sol_L)) + tf.exp(v * self.lb_hard)) / v
            upper_bo = -tf.math.log(tf.exp(-v * (self.ub_soft[i, :] + phi_sol_U)) + tf.exp(-v * self.ub_hard)) / v
            self.phi_ini_L = phi_sol_L
            self.phi_ini_U = phi_sol_U
            self.Lb.append(lower_bo)
            self.Ub.append(upper_bo)
            self.j.append(i)
    
    @tf.function
    def bound(self, phi, t, lb, ub, mu, kc):
        eta = ub - lb
        dphi_dt = 0.5 * (1 - tf.sign(eta - mu)) * (1 / (eta + phi)) - kc * phi
        return dphi_dt
    
    @tf.function
    def step(self, action):
        done = tf.constant(False, dtype=tf.bool, name='done')

        #get new position
        vel_x, vel_y, omega = 0.22 * action[0], 0.22 * action[1], 2.84 * action[2]
        
        self.x.assign_add(vel_x * self.time_int)
        self.y.assign_add(vel_y * self.time_int)
        self.theta.assign_add(omega * self.time_int)
        self.state = tf.stack([self.x, self.y, self.theta])

        # To keep the angle theta between -pi to pi
        if(self.theta > math.pi or self.theta < -math.pi ):
           self.theta = ((self.theta + math.pi)%(2*math.pi))-math.pi

        if self.lb_hard[0] <= self.x <= self.ub_hard[0] and self.lb_hard[1] <= self.y <= self.ub_hard[1] :
            Rew_max =100
            #check if within funnel and reward accordingly
            ep_t_val = self.ep_t.numpy().item()
            x_min,y_min = self.Lb[ep_t_val]
            x_max,y_max = self.Ub[ep_t_val]
            
            robust1 = Rew_max - ((self.x - (x_min + x_max)/2)**2)*(4*Rew_max/((x_max-x_min)**2))
            robust2 = Rew_max - ((self.y - (y_min + y_max)/2)**2)*(4*Rew_max/((y_max-y_min)**2))
            reward = np.clip(min(robust1, robust2), -10,Rew_max)
        else:
            #terminate the episode or restart the episode?
            reward = -100
            done = tf.constant(True, dtype=tf.bool, name='done')
        self.ep_t.assign_add(1)

        if tf.equal(self.ep_t, self.epi_len):
            done = tf.constant(True, dtype=tf.bool, name='done')

        info = {}
        truncated = done
        return self.state, reward, done, truncated, info
    
    def render(self, mode="human"):
        pass

    def reset(self, seed= None, options=None):
        self.x = tf.Variable(-3.19, dtype=tf.float32)
        self.y = tf.Variable(3.0, dtype=tf.float32)
        self.theta = tf.Variable(0.0, dtype=tf.float32)
        self.state = tf.stack([self.x, self.y, self.theta])

        self.ep_t.assign(0)
        info = {}
        return self.state, info

    def close(self):
        pass

    def seed(self, seed=None):
        pass