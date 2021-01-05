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
		#  s = state[0] * self.cols + state[1]
		if state == self.start:
			return 0
		if state == self.goal:
			return 1
		return (state[0]-1) * self.cols + state[1]+2

	def __init__(self):
		self.action_names = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
		self.cols = 12
		self.rows = 4
		self.start = [0,0]
		self.goal = [0, 11]
		self.current_state = None
		self.need_reset = True

		# There are four actions: up, down, left and right
		self.action_space = spaces.Discrete(4)

		 # observation is the x, y coordinate of the grid
		self.observation_space = spaces.Discrete(self.rows*self.cols-10)


	def step(self, action):
		if self.need_reset:
			raise Exception("Episode is terminated, call env.reset()")

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
				self.need_reset = True

		return self.observation(self.current_state), reward, is_terminal, {}

	def reset(self):
		self.current_state = self.start
		self.need_reset = False
		return self.observation(self.start)

	def get_scene(self):
		scene = np.full(shape=(4, 12), fill_value=0)
		scene[0, 1:self.cols] = np.full(shape=(1, self.cols-1), fill_value=-1)
		scene[0, -1] = 1
		scene[self.current_state[0], self.current_state[1]] = 2
		return np.flip(scene, axis=0)

	def render(self, mode='human'):
		plt.matshow(self.get_scene())
		plt.show()

	def close(self):
		pass

