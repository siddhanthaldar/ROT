from cgitb import enable
import time
import gym
from gym import spaces
from pygments import highlight
from gym_envs.envs import robot_env

import numpy as np

class RobotEraseBoardEnv(robot_env.RobotEnv):
	def __init__(self, height=84, width=84, step_size=10, enable_arm=True, enable_gripper=True, enable_camera=True, camera_view='side',
				 use_depth=False, dist_threshold=0.05, random_start=True, x_limit=None, y_limit=None, z_limit=None, pitch=0, roll=180, keep_gripper_closed=True, highest_start=True, start_at_the_back=False):
		robot_env.RobotEnv.__init__(
			self,
			home_displacement=[1.525, 0, 0.81],
			height=height,
			width=width,
			step_size=step_size,
			enable_arm=enable_arm, 
			enable_gripper=enable_gripper,
			enable_camera=enable_camera,
			camera_view=camera_view,
			use_depth=use_depth,
			keep_gripper_closed=keep_gripper_closed,
			highest_start=highest_start,
			x_limit=x_limit,
			y_limit=y_limit,
			z_limit=z_limit,
			pitch=pitch,
			roll=roll
		)

		self.highest_start = highest_start
		self.start_at_the_back = start_at_the_back
		self.x_limit = x_limit
		self.y_limit = y_limit
		self.z_limit = z_limit

		self.action_space = spaces.Box(low = np.array([-1,-1,-1],dtype=np.float32), 
									   high = np.array([1, 1, 1],dtype=np.float32),
									   dtype = np.float32)
		self.dist_threshold = dist_threshold
		self.random_start = random_start

	def arm_refresh(self, reset=True):
		self.arm.clear_errors()
		self.arm.set_mode_and_state()
		if reset:
			self.arm.reset(home=True)
		time.sleep(2)

	def reset(self):
		if not self.enable_arm:
			return np.array([0,0,0], dtype=np.float32)
		self.arm_refresh(reset=False)
		if self.random_start:
			self.set_random_pos()
		time.sleep(0.4)		
		obs = {}
		obs['features'] = np.array(self.arm.get_position(), dtype=np.float32)
		obs['pixels'] = self.render(mode='rgb_array', width=self.width, height=self.height)
		return obs

	def step(self, action):
		new_pos = self.arm.get_position() 
		new_pos[:3] += action[:3] * 0.25
		if self.enable_arm:
			self.arm.set_position(new_pos)
			time.sleep(0.4)

		self.reward = self.get_reward()
		
		done = False
		
		info = {}
		info['is_success'] = 1 if self.reward==1 else 0

		obs = {}
		obs['features'] = np.array(self.arm.get_position(), dtype=np.float32)
		obs['pixels'] = self.render(mode='rgb_array', width=self.width, height=self.height)

		return obs, self.reward, done, info

	def get_reward(self):
		return 0 if self.goal_reached() else -1

	def goal_reached(self):
		pos = self.arm.get_position()[:3]
		dispX = abs(pos[0] - self.home_displacement[0])
		dispY = abs(pos[1] - self.home_displacement[1])
		dispZ = abs(pos[2] - self.home_displacement[2])
		return (dispX<=self.dist_threshold) and \
			   (dispY<=self.dist_threshold) and \
			   (dispZ<=self.dist_threshold)

	def set_random_pos(self):
		self.arm.clear_errors()
		self.arm.set_mode_and_state()
		pos = self.arm.get_position()
		
		pos[2] = self.z_limit[1]
		self.arm.set_position(pos)

		pos = self.get_random_pos(pos)

		self.arm.set_position(pos)
		if self.keep_gripper_closed:
			self.arm.close_gripper_fully()
		else:
			self.arm.open_gripper_fully()

	def get_random_pos(self, pos=None):
		pos[0] = self.x_limit[0] + np.random.rand()*(self.x_limit[1] - self.x_limit[0])
		pos[1] = self.y_limit[0] + np.random.rand()*(self.y_limit[1] - self.y_limit[0])
		pos[2] = self.z_limit[0] + np.random.rand()*(self.z_limit[1] - self.z_limit[0])

		if self.highest_start:
			pos[2] = self.z_limit[1]

		if self.start_at_the_back:
			pos[0] = self.x_limit[0]

		return pos