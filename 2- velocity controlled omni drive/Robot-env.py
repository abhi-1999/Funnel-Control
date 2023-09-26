import numpy as np
import math
import gym
from gym.spaces import Box

class RobotEnv(gym.Env):
    def __init__(self):
        """
        Must define self.acion_space and self.observation_space  
        """
        self.action_space = Box(low = np.array([-0.22,-0.22,-2.84]), #lower bounds for vel_x,vel_y,omega
                                     high=np.array([0.22,0.22,2.84]),dtype=np.float32) #upper bounds for vel_x,vel_y,omega
        
        self.observation_space = Box(low = np.array([-6.58,-4.63,-math.pi]), #lower bounds of state
                                      high=np.array([6.58,4.63,math.pi])) #upper bounds of state
        self.epi_len = 2500


        #states
        self.x = -3.19
        self.y = 1.70
        self.theta = 0
        self.state = np.array([self.x,self.y,self.theta])

        #RL constants
        self.timestep = 0
        
        self.time_int = 25/self.epi_len
        self.t2 = self.timestep*self.time_int
        #reference trajectory
        xd=[[-1.5+5.8*math.cos(0.24*self.time_int*t+1.5), 3.8*math.sin(0.24*t*self.time_int+1.5)] for t in range(self.epi_len)]
        self.state_d = np.array(xd)

        #Funnel specs ALL VALUES TO BE CHECKED
        l = np.array([0.1,0.1]) 
        rho_f = np.array([1.5,1.5])
        rho_0 = np.array(abs(np.array([self.x,self.y])- self.state_d[0,:]) + 0.2)
        gamma = np.array([(rho_0 - rho_f)*np.exp(-l*h*t) + rho_f for t in range(self.epi_len)])
        self.lb_soft = self.state_d - gamma
        self.ub_soft = self.state_d + gamma
        self.phi_ini_L = [0,0]
        self.phi_ini_U = [0,0]

    def reset(self):
        return super().reset()
    
    def step(self, action):
        vel_x,vel_y,omega = action[0],action[1],action[2]
        return super().step(action)
    
    def render(self, mode="human"):
        pass

    def  close(self):
        return super().close()
    
    def seed(self, seed=None):
        return super().seed(seed)