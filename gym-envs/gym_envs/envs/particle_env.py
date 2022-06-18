import gym
from gym import spaces
import cv2
import numpy as np

class ParticleEnv(gym.Env):
	def __init__(self, height=84, width=84, n_channels=1, step_size=10, reward_type='dense', reward_scale=None, start=None, goal=None, block=None):
		super(ParticleEnv, self).__init__()

		self.height = height
		self.width = width
		self.n_channels = n_channels
		self.step_size = step_size
		self.reward_type = reward_type
		self.reward_scale = np.sqrt(height**2 + width**2) if reward_scale is None else reward_scale
		
		# Set start and goal
		self.start = np.array([0,0]).astype(np.int) if start is None else np.array(start).astype(np.int)
		self.goal = np.array([height-15, height-1, width-15, width-1]).astype(np.int) if goal is None else np.array(goal).astype(np.int) # (hmin, hmax, wmin, wmax)
		self.goal_mean = np.array([int((self.goal[0] + self.goal[1])/2), int((self.goal[2] + self.goal[3])/2)]).astype(np.int)

		# Set state
		self.state = None
		self._step = None

		'''
		Define observation space which blocked in between.
		0: Traversable blocks
		1: Goal
		2: Blocked
		'''
		self.observation_space = spaces.Box(low = np.array([0,0],dtype=np.float32), 
									   		high = np.array([255,255],dtype=np.float32),
									  		dtype = np.float32)
		self.observation = np.zeros((height, width, n_channels)).astype(np.uint8)
		
		# Set blocked regions
		if block is not None:
			for region in block:
				block_hmin, block_hmax = int(region[0]), int(region[1])
				block_wmin, block_wmax = int(region[2]), int(region[3])
				for h in range(block_hmin, block_hmax+1):
					for w in range(block_wmin, block_wmax+1):
						for c in range(n_channels):
							self.observation[h,w,c] = 2

		# Set goal regions
		goal_hmin, goal_hmax = int(self.goal[0]), int(self.goal[1])
		goal_wmin, goal_wmax = int(self.goal[2]), int(self.goal[3])
		for h in range(goal_hmin, goal_hmax+1):
			for w in range(goal_wmin, goal_wmax+1):
				for c in range(n_channels):
					self.observation[h,w,c] = 1

		# if block is None:
		# 	block_hmin, block_hmax = int(height/3), int(2/3*height)
		# 	block_wmin, block_wmax = int(width/3), int(2/3*width)
		# else:
		# 	block_hmin, block_hmax = int(block[0]), int(block[1])
		# 	block_wmin, block_wmax = int(block[2]), int(block[3])
		# for h in range(height):
		# 	for w in range(width):
		# 		if h>=block_hmin and h<=block_hmax and w>=block_wmin and w<=block_wmax:
		# 			for c in range(n_channels):
		# 				self.observation[h,w,c] = 2
		# 		elif h>=self.goal[0] and h<=self.goal[1] and w>=self.goal[2] and w<=self.goal[3]:
		# 			for c in range(n_channels):
		# 				self.observation[h,w,c] = 1
		# 		else:
		# 			for c in range(n_channels):
		# 				self.observation[h,w,c] = 0
		

		# # Save image
		# cv2.imwrite("environment.jpg", (self.observation * 127).astype(np.uint8))
		# reward_map = np.zeros(self.observation.shape)
		# for h in range(height):
		# 	for w in range(width):
		# 		if h>=block_hmin and h<=block_hmax and w>=block_wmin and w<=block_wmax:
		# 			for c in range(n_channels):
		# 				reward_map[h,w,c] = 0 #-1000
		# 		elif h>=self.goal[0] and h<=self.goal[1] and w>=self.goal[2] and w<=self.goal[3]:
		# 			for c in range(n_channels):
		# 				reward_map[h,w,c] = 20 #1000
		# 		else:
		# 			for c in range(n_channels):
		# 				reward_map[h,w,c] = self.reward_scale/np.sqrt((self.goal_mean[0]-h)**2 + (self.goal_mean[1]-w)**2) #-np.sqrt((self.goal_mean[0]-h)**2 + (self.goal_mean[1]-w)**2) / 3.60
		# 				# print(reward_map[h,w,c])	
		# min, max = np.min(reward_map), np.max(reward_map)
		# reward_map = (reward_map - min)/(max - min)*255
		# cv2.imwrite("reward.jpg", (reward_map).astype(np.uint8))

		self.action_space = spaces.Box(low = np.array([-1,-1],dtype=np.float32), 
									   high = np.array([1,1],dtype=np.float32),
									   dtype = np.float32)
	
	def step(self, action):
		prev_state = self.state
		self.state = np.array([int(self.state[0] + self.step_size * action[0]), int(self.state[1] + self.step_size * action[1])], dtype=np.float32)
		
		if self.state[0]<0 or self.state[0]>=self.height or self.state[1]<0 or self.state[1]>=self.width:
			reward = -1#000
			self.state = prev_state
			# done = False #True
		elif self.observation[int(self.state[0]), int(self.state[1])]==2:
			reward = -1#000
			self.state = prev_state
			# done = False #True
		elif self.observation[int(self.state[0]), int(self.state[1])] == 1:
			reward = 1 #1000
			# done = False
		else:
			if self.reward_type == 'sparse':
				reward = 0
			else:
				reward = -np.sqrt((self.goal_mean[0]-self.state[0])**2 + (self.goal_mean[1]-self.state[1])**2) / self.reward_scale
				# reward = self.reward_scale/(1 + np.sqrt((self.goal_mean[0]-self.state[0])**2 + (self.goal_mean[1]-self.state[1])**2))
		done = False
		self._step += 1
		
		info = {}
		info['is_success'] = 1 if reward==1 else 0 

		return self.state, reward, done, info
	
	def reset(self):
		self.state = self.start
		self._step = 0
		return self.state

	# def render(self, mode=''):
	# 	img = np.ones((self.observation.shape[0], self.observation.shape[1], 3)).astype(np.uint8) * 255

	# 	# Identify blocked region
	# 	blocked = np.where(self.observation == 2)
	# 	img[blocked] = np.array([0,0,0])

	# 	# Identify goal region
	# 	img[self.goal[0]:self.goal[1]+1, self.goal[2]:self.goal[3]+1] = [255,0,0]
	# 	# blocked = np.where(self.observation>=self.goal[0] and self.observation<=self.goal[1] or self.observation>=self.goal[2] and self.observation<=self.goal[2])
	# 	# img[blocked] = 64


	# 	# Mark state
	# 	img[max(0, self.state[0]-5):min(self.height-1, self.state[0]+5), max(0, self.state[1]-5):min(self.width-1, self.state[1]+5)] = [0,0,255]

	# 	cv2.imshow("Render", img)
	# 	cv2.waitKey(5)

	def render(self, mode='', width=None, height=None):
		img = np.ones(self.observation.shape).astype(np.uint8) * 255
		# Identify blocked region
		blocked = np.where(self.observation == 2)
		img[blocked] = 0

		# Identify goal region
		img[self.goal[0]:self.goal[1]+1, self.goal[2]:self.goal[3]+1] = 64

		# Mark state
		img[max(0, int(self.state[0])-5):min(self.height-1, int(self.state[0])+5), max(0, int(self.state[1])-5):min(self.width-1, int(self.state[1])+5)] = 128

		if width is not None and height is not None:
			dim = (int(width), int(height))
			img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
			if self.n_channels == 1:
				img = np.expand_dims(img, axis=2)

		if mode=='rgb_array':
			return img
		else:
			cv2.imshow("Render", img)
			cv2.waitKey(5)