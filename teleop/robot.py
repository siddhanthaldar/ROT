import sys
import time
import numpy as np
import random
from configparser import ConfigParser
from xarm.wrapper import XArmAPI

class XArm:

	def __init__(self, config_file='./robot.conf', home_displacement = (0,0,0), low_range=(1,1,0.2) , high_range=(2,2,1),
				 keep_gripper_closed=False, highest_start=False, x_limit=None, y_limit=None, z_limit=None, pitch = 0, roll=180, gripper_action_scale=200):
		self.arm = None
		self.gripper_max_open = 600
		self.gripper_min_open = 0 #348
		self.zero = (206/100,0/100,120.5/100)	# Units: .1 meters 
		self.home = home_displacement
		self.keep_gripper_closed = keep_gripper_closed
		self.highest_start = highest_start
		self.low_range = low_range
		self.high_range = high_range
		self.joint_limits = None
		self.ip = '192.168.1.246'
		self.gripper_action_scale = gripper_action_scale

		# Limits
		self.x_limit = [0.5, 3.5] if x_limit is None else x_limit
		self.y_limit = [-1.7, 1.3] if y_limit is None else y_limit
		self.z_limit = [1.4, 3.4] if z_limit is None else z_limit # PressBlock

		# Pitch value - Horizontal or vertical orientation
		self.pitch = pitch
		self.roll = roll

	def start_robot(self):
		if self.ip is None:
			raise Exception('IP not provided.')
		self.arm = XArmAPI(self.ip, is_radian=False)
		self.arm.motion_enable(enable=False)
		self.arm.motion_enable(enable=True)
		if self.arm.error_code != 0:
			self.arm.clean_error()
		self.set_mode_and_state()

	def set_mode_and_state(self, mode=0, state=0):
		self.arm.set_mode(mode)
		self.arm.set_state(state=state)

	def clear_errors(self):
		self.arm.clean_warn()
		self.arm.clean_error()

	def has_error(self):
		return self.arm.has_err_warn

	def reset(self, home = False, reset_at_home=True):
		if self.arm.has_err_warn:
			self.clear_errors()
		if home:
			if reset_at_home:
				self.move_to_home()
			else:
				self.move_to_zero()
			if self.keep_gripper_closed:
				self.close_gripper_fully()
			else:
				self.open_gripper_fully()

	def move_to_home(self, open_gripper=False):
		pos = self.get_position()
		pos[0] = self.home[0]
		pos[1] = self.home[1]
		pos[2] = self.home[2]
		self.set_position(pos)
		if open_gripper and not self.keep_gripper_closed:
			self.open_gripper_fully()
	
	def set_random_pos(self):
		self.clear_errors()
		self.set_mode_and_state()
		pos = self.get_position()
		
		# Move up
		pos[2] = self.z_limit[1]
		self.set_position(pos)

		# Set random pos
		x_disp = self.low_range[0] + np.random.rand()*(self.high_range[0] - self.low_range[0])
		y_disp = self.low_range[1] + np.random.rand()*(self.high_range[1] - self.low_range[1])
		z_disp = self.low_range[2] + np.random.rand()*(self.high_range[2] - self.low_range[2])
		
		pos[0] = self.home[0] + x_disp * np.random.choice([-1,1])		# Here we sample in a square ring around the home 
		pos[1] = self.home[1] + y_disp * np.random.choice([-1,1])		# Here we sample in a square ring around the home 
		pos[2] = self.home[2] + z_disp if not self.highest_start else self.z_limit[1] 									# For z we jsut sample from [a,b]
		self.set_position(pos)
		if self.keep_gripper_closed:
			self.close_gripper_fully()
		else:
			self.open_gripper_fully()

	def move_to_zero(self):
		pos = self.get_position()
		pos[0] = min(max(self.x_limit[0],0), self.x_limit[1])# 0
		pos[1] = min(max(self.y_limit[0],0), self.y_limit[1])# 0
		pos[2] = min(max(self.z_limit[0],0), self.z_limit[1]) if not self.highest_start else self.z_limit[1] # 0
		self.set_position(pos)

	def set_position(self, pos, wait=False):
		pos = self.limit_pos(pos)
		x = (pos[0] + self.zero[0])*100
		y = (pos[1] + self.zero[1])*100
		z = (pos[2] + self.zero[2])*100
		self.arm.set_position(x=x, y=y, z=z, roll=self.roll, pitch=self.pitch, yaw=0, wait=wait)

	def get_position(self):
		pos = self.arm.get_position()[1]
		x = (pos[0]/100.0 - self.zero[0])
		y = (pos[1]/100.0 - self.zero[1])
		z = (pos[2]/100.0 - self.zero[2])
		return np.array([x,y,z, pos[3], pos[4], pos[5]]).astype(np.float32)

	def get_gripper_position(self):
		code, pos = self.arm.get_gripper_position()
		if code!=0:
			raise Exception('Correct gripper angle cannot be obtained.')
		return pos

	def open_gripper_fully(self):
		self.set_gripper_position(self.gripper_max_open)

	def close_gripper_fully(self):
		self.set_gripper_position(self.gripper_min_open)

	def open_gripper(self):
		self.set_gripper_position(self.get_gripper_position() + self.gripper_action_scale)

	def close_gripper(self):
		self.set_gripper_position(self.get_gripper_position() - self.gripper_action_scale)

	def set_gripper_position(self, pos, wait=False):
		'''
		wait: To wait till completion of action or not
		'''
		if pos<self.gripper_min_open:
			pos = self.gripper_min_open
		if pos>self.gripper_max_open:
			pos = self.gripper_max_open
		self.arm.set_gripper_position(pos, wait=wait, auto_enable=True)

	def get_servo_angle(self):
		code, angles = self.arm.get_servo_angle()
		if code!=0:
			raise Exception('Correct servo angles cannot be obtained.')
		return angles

	def set_servo_angle(self, angles, is_radian=None):
		'''
		angles: List of length 8
		'''
		self.arm.set_servo_angle(angle=angles, is_radian=is_radian)
	
	def limit_pos(self, pos):
		pos[0] = max(self.x_limit[0], pos[0])
		pos[0] = min(self.x_limit[1], pos[0])
		pos[1] = max(self.y_limit[0], pos[1])
		pos[1] = min(self.y_limit[1], pos[1])
		pos[2] = max(self.z_limit[0], pos[2])
		pos[2] = min(self.z_limit[1], pos[2])
		return pos