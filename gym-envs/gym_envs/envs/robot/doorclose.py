from cgitb import enable
import time
import gym
from gym import spaces
from gym_envs.envs import robot_env

import numpy as np

class RobotDoorCloseEnv(robot_env.RobotEnv):
	def __init__(self, height=84, width=84, step_size=10, enable_arm=True, enable_gripper=True, enable_camera=True, camera_view='side',
				 use_depth=False, dist_threshold=0.05, random_start=True, x_limit=None, y_limit=None, z_limit=None, pitch=0, roll=180, keep_gripper_closed=True):
		robot_env.RobotEnv.__init__(
			self,
			height=height,
			width=width,
			step_size=step_size,
			enable_arm=enable_arm, 
			enable_gripper=enable_gripper,
			enable_camera=enable_camera,
			camera_view=camera_view,
			use_depth=use_depth,
			keep_gripper_closed=keep_gripper_closed,
			highest_start=False,
			x_limit=x_limit,
			y_limit=y_limit,
			z_limit=z_limit,
			pitch=pitch,
			roll=roll
		)
		self.action_space = spaces.Box(low = np.array([-1,-1,-1],dtype=np.float32), 
									   high = np.array([1, 1, 1],dtype=np.float32),
									   dtype = np.float32)
		self.dist_threshold = dist_threshold
		self.random_start = random_start

		if self.random_start:
			self.random_limits = [
				[0.5, 3],
				[2.25, self.y_limit[1]],
				[self.z_limit[0], self.z_limit[1]]
			]
			self.home_displacement = [
				(self.random_limits[0][0]+self.random_limits[0][1])/2,
				(self.random_limits[1][0]+self.random_limits[1][1])/2,
				(self.random_limits[2][0]+self.random_limits[2][1])/2,
			]

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

	def get_random_pos(self, pos=None):
		x_disp = self.random_limits[0][0] + np.random.rand()*(self.random_limits[0][1] - self.random_limits[0][0])
		y_disp = self.random_limits[1][0] + np.random.rand()*(self.random_limits[1][1] - self.random_limits[1][0])
		z_disp = self.random_limits[2][0] + np.random.rand()*(self.random_limits[2][1] - self.random_limits[2][0])
		
		if pos is None:
			pos = np.zeros(3).astype(np.float32)
		pos[0] = x_disp
		pos[1] = y_disp 
		pos[2] = z_disp if not self.arm.highest_start else self.z_limit[1] 									
		return pos

	def set_random_pos(self):
		self.arm.clear_errors()
		self.arm.set_mode_and_state()
		pos = self.arm.get_position()
		
		pos[2] = self.arm.z_limit[1]
		self.arm.set_position(pos)
		pos[1] = (self.random_limits[1][0] + self.random_limits[1][1])/2
		self.arm.set_position(pos)

		pos = self.get_random_pos(pos)

		self.set_position(pos)
		if self.arm.keep_gripper_closed:
			self.arm.close_gripper_fully()
		else:
			self.arm.open_gripper_fully()

	def step(self, action):
		new_pos = self.arm.get_position() 
		new_pos[:3] += action[:3] * 0.25
		if self.enable_arm:
			self.set_position(new_pos)
			time.sleep(0.4)

		self.reward = 0
		done = False	
		info = {}
		info['is_success'] = 1 if self.reward==1 else 0
		obs = {}
		obs['features'] = np.array(self.arm.get_position(), dtype=np.float32)
		obs['pixels'] = self.render(mode='rgb_array', width=self.width, height=self.height)

		return obs, self.reward, done, info

	def set_position(self, pos, wait=False):
		pos = self.limit_pos(pos)
		x = (pos[0] + self.arm.zero[0])*100
		y = (pos[1] + self.arm.zero[1])*100
		z = (pos[2] + self.arm.zero[2])*100
		self.arm.arm.set_position(x=x, y=y, z=z, roll=180, pitch=self.arm.pitch, yaw=0, wait=wait)

	def limit_pos(self, pos):
		if pos[1] >= self.random_limits[1][0]:
			pos[0] = max(self.random_limits[0][0], pos[0])
			pos[0] = min(self.random_limits[0][1], pos[0])
		else:
			pos[0] = max(self.x_limit[0], pos[0])
			pos[0] = min(self.y_limit[1], pos[0])
		pos[1] = max(self.y_limit[0], pos[1])
		pos[1] = min(self.y_limit[1], pos[1])
		pos[2] = max(self.z_limit[0], pos[2])
		pos[2] = min(self.z_limit[1], pos[2])
		return pos