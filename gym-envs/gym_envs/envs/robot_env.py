from cgitb import enable
import gym
from gym import spaces
import cv2
import numpy as np
import time

from pygame import init

from gym_envs.envs.hardware.camera import Camera
from gym_envs.envs.hardware.robot import XArm

class RobotEnv(gym.Env):
	def __init__(self, home_displacement=[2,-0.21,.4], height=84, width=84, step_size=10, enable_arm=True, enable_gripper=True, enable_camera=True,
				 camera_view='side', use_depth=False, keep_gripper_closed=False, highest_start=False, 
				 x_limit=None, y_limit=None, z_limit=None, pitch=0, roll=180, goto_zero_at_init=False, start_at_the_back=False):
		super(RobotEnv, self).__init__()
		self.height = height
		self.width = width
		self.use_depth = use_depth
		self.step_size = step_size
		self.enable_arm = enable_arm
		self.enable_camera = enable_camera
		self.camera_view = camera_view
		self.home_displacement = np.array(home_displacement, dtype=np.float32)
		self.keep_gripper_closed = keep_gripper_closed
		self.highest_start = highest_start
		self.x_limit = x_limit
		self.y_limit = y_limit
		self.z_limit = z_limit
		self.pitch = pitch
		self.roll = roll
		self.goto_zero_at_init = goto_zero_at_init
		self.start_at_the_back = start_at_the_back
		
		if self.use_depth:
			self.n_channels = 4
		else:
			self.n_channels = 3  

		if self.enable_arm:
			self.enable_gripper = enable_gripper
		else:
			self.enable_gripper = False
		self.reward = 0
		

		self.observation_space = spaces.Box(low = np.array([0,0],dtype=np.float32), high = np.array([255,255],dtype=np.float32), dtype = np.float32)

		self.observation = np.zeros((self.n_channels, height, width)).astype(np.uint8)
		
		if self.enable_arm:
			self.init_arm()

		if self.enable_camera:
			# Realsense Camera
			self.cam = Camera(width=self.width, height=self.height, view=self.camera_view)
	
	def init_arm(self):
		self.arm = XArm(home_displacement=self.home_displacement, keep_gripper_closed=self.keep_gripper_closed, highest_start=self.highest_start, 
						x_limit=self.x_limit, y_limit=self.y_limit, z_limit=self.z_limit, pitch=self.pitch, roll=self.roll, start_at_the_back=self.start_at_the_back)
		self.arm.start_robot()
		self.arm.clear_errors()
		self.arm.set_mode_and_state()
		self.arm.reset(home=False)
		if self.goto_zero_at_init:
			self.arm.move_to_zero()
		time.sleep(2)

	def step(self, action):
		new_pos = self.arm.get_position()
		new_pos[:3] += action[:3] * 0.25
		gripper_movement = 0
		
		if self.enable_gripper:
			if action[3]>0.5:
				self.arm.open_gripper_fully()
				time.sleep(0.2)
			elif action[3]<-0.5:
				self.arm.close_gripper_fully()
				time.sleep(0.2)

		if self.enable_arm:
			self.arm.set_position(new_pos)
			time.sleep(0.4)

		done = False
		
		info = {}
		info['is_success'] = 1 if self.reward==1 else 0

		obs = {}
		obs['features'] = np.array(self.arm.get_position(), dtype=np.float32)
		obs['pixels'] = self.render(mode='rgb_array', width=self.width, height=self.height)

		self.reward = self.get_reward()

		return obs, self.reward, done, info

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
		self.arm.set_random_pos()
		time.sleep(0.4)		
		obs = {}
		obs['features'] = np.array(self.arm.get_position(), dtype=np.float32)
		obs['pixels'] = self.render(mode='rgb_array', width=self.width, height=self.height)
		return obs

	def render(self, mode='rgb_array', width=84, height=84):
		if not self.enable_camera:
			return np.random.rand(height, width, self.n_channels)
		obs = self.cam.get_frame()
		return obs[:,:,:self.n_channels]

	def get_reward(self):
		pass
