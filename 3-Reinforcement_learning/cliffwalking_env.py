import gym
from gym import error, spaces, utils
from gym.utils import seeding

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

class CliffWalkingEnv(gym.Env):
    ''' Cliff Walking Environment

        See the README.md file from https://github.com/caburu/gym-cliffwalking
    '''
    # There is no renderization yet
    # metadata = {'render.modes': ['human']}

    def observation(self, state):
        return state[0] * self.cols + state[1]

    def __init__(self, ncols: int=12, nrows: int=4):
        self.cols = ncols
        self.rows = nrows
        self.start = [0,0]
        self.goal = [0,ncols-1]
        self.current_state = None

        # There are four actions: up, down, left and right
        self.action_space = spaces.Discrete(4)

         # observation is the x, y coordinate of the grid
        self.observation_space = spaces.Discrete(self.rows*self.cols)


    def step(self, action):
        new_state = deepcopy(self.current_state)

        if action == 0: #right
            new_state[1] = min(new_state[1]+1, self.cols-1)
        elif action == 1: #down
            new_state[0] = max(new_state[0]-1, 0)
        elif action == 2: #left
            new_state[1] = max(new_state[1]-1, 0)
        elif action == 3: #up
            new_state[0] = min(new_state[0]+1, self.rows-1)
        else:
            raise Exception("Invalid action.")
        self.current_state = new_state

        reward = -1.0
        is_terminal = False
        if self.current_state[0] == 0 and self.current_state[1] > 0:
            if self.current_state[1] < self.cols - 1:
                reward = -100.0
                self.current_state = deepcopy(self.start)
            else:
                is_terminal = True

        return self.observation(self.current_state), reward, is_terminal, {}

    def reset(self):
        self.current_state = self.start
        return self.observation(self.current_state)

    def render(self, mode='human'):
        pass

    def close(self):
        pass
