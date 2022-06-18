import os
import time
import pickle
import numpy as np
import cv2
import datetime

from robot import XArm
from joy import Joy
from camera import Camera

class BaseClass:
	def __init__(self, 
				 servo_angle=None,
				 scale_factor_pos=0.5, 
				 scale_factor_gripper=300, 
				 scale_factor_rotation=120,
				 image_width=84,
				 image_height=84,
				 camera_view='side',
				 home_displacement=(0,0,0),
				 random_start=False,
				 keep_gripper_closed=False,
				 highest_start=False,
				 x_limit = None,
				 y_limit = None,
				 z_limit = None,
				 pitch = 0,
				 roll = 180,
				 sleep=0.6,
				 is_radian=False):
		self.servo_angle = servo_angle
		self.sleep = sleep
		self.init_arm(home_displacement=home_displacement, keep_gripper_closed=keep_gripper_closed,
					  highest_start=highest_start, x_limit=x_limit, y_limit=y_limit, z_limit=z_limit, pitch=pitch, roll=roll)
		self.joy = Joy(self.arm,
					   scale_factor_pos, 
					   scale_factor_gripper, 
					   scale_factor_rotation,
					   sleep,
					   random_start=random_start,
					   is_radian=is_radian)
		self.cam = Camera(width=image_width, height=image_height, view=camera_view)

	def init_arm(self, home_displacement=(0,0,0), keep_gripper_closed=False, highest_start=False,
				 x_limit=None, y_limit=None, z_limit=None, pitch=0, roll=180):
		self.arm = XArm(home_displacement=home_displacement, keep_gripper_closed=keep_gripper_closed, highest_start=highest_start, 
						x_limit=x_limit, y_limit=y_limit, z_limit=z_limit, pitch=pitch, roll=roll)

		self.arm.start_robot()
		self.arm.set_mode_and_state()
		self.arm.reset(home=True)
		time.sleep(1)

	def record_trajectories(self, 
							num_trajectories,
							episode_len,
							save=True,
							directory_path=None,
							exp_name=""):

		args = locals()
		cfg = {}
		for key, item in args.items():
			if key != 'self':
				cfg[key] = item	
		
		now = datetime.datetime.now()
		cfg['time'] = str(now.year)+'_'+str(now.month)+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)
		directory_path = directory_path / (cfg['time']+'_'+exp_name)
		os.makedirs(directory_path, exist_ok=True)
		with open(directory_path / 'config.pickle', 'wb') as f:
			pickle.dump(cfg, f)

		print("Ready to record trajectories")
		
		trajectories = []

		for traj_index in range(num_trajectories):
			traj = {}
			state_obs = None
			while state_obs is None:
				state_obs, image_obs, actions, rewards = self._record_trajectory(episode_len)

			traj['state_observation'] = state_obs
			traj['image_observation'] = image_obs
			traj['actions'] = actions
			traj['start_loc'] = state_obs[0]
			traj['goal_loc'] = state_obs[-1]
			traj['reward'] = rewards
			trajectories.append(traj)

			if save:
				print("Start:", traj['start_loc'], "\tGoal:", traj['goal_loc'])
				self._save_trajectory(traj, traj_index, directory_path)

		return trajectories

	def _save_trajectory(self, traj, traj_index, directory_path):
		traj_dir = directory_path / str(traj_index)
		image_dir = directory_path / str(traj_index) / 'images'
		os.makedirs(traj_dir, exist_ok=True)
		os.makedirs(image_dir, exist_ok=True)
		for j, img in enumerate(traj['image_observation']):
			image_path = str(image_dir / (str(j)+'.png'))
			cv2.imwrite(image_path,img[:,:,:3])
		
		pickle_path = traj_dir / 'traj.pickle'
		dictionary = {
			'state_observation': traj['state_observation'],
			'image_observation': traj['image_observation'],
			'action': traj['actions'],
			'start_loc': traj['start_loc'],
			'goal_loc': traj['goal_loc'],
			'reward': traj['reward']
		}
		with open(pickle_path, 'wb') as outfile:
			pickle.dump(dictionary, outfile)
		print("Saved trajectory {%d}"%(traj_index))

class Reach(BaseClass):
	def __init__(self, 
				 scale_factor_pos=0.25,
				 image_width=84,
				 image_height=84,
				 home_displacement=(0,0,0),
				 random_start=True,
				 keep_gripper_closed=False,
				 highest_start=False,
				 x_limit = None,
				 y_limit = None,
				 z_limit = None,
				 **kwargs):
		super().__init__(scale_factor_pos=scale_factor_pos, image_width=image_width, image_height=image_height, home_displacement=home_displacement, random_start=random_start,
						 keep_gripper_closed=keep_gripper_closed, highest_start=highest_start, x_limit=x_limit, y_limit=y_limit, z_limit=z_limit, 
						 **kwargs)

	def _record_trajectory(self, episode_len):
		
		actions = []
		state_obs = []
		image_obs = []
		rewards = []
		step = 0
		start = False
		action_types = ['move_forward', 'move_backward', 'move_left',
						'move_right', 'move_up', 'move_down']

		while(True):
			obs = self.arm.get_position()[:3]
			image = self.cam.get_frame()

			self.joy.detect_event()
			pos, action = self.joy.move()

			if action == 'start':
				start = True
			elif action =='stop' or step == episode_len:
				for _ in range(episode_len - step):
					state_obs.append(obs)
					image_obs.append(image)
					actions.append(np.zeros(3).astype(np.float32))
					rewards.append(1)
				break
			elif action == 'cancel':
				return None, None, None, None
			elif start and action in action_types:
				pos = pos[:3]
				if action in action_types:
					state_obs.append(obs)
					image_obs.append(image)
					actions.append(pos-obs)
					rewards.append(0)
				step += 1

		return state_obs, image_obs, actions, rewards

class HorizontalReach(BaseClass):
	def __init__(self, 
				 scale_factor_pos=0.25,
				 image_width=84,
				 image_height=84,
				 home_displacement=(0,0,0),
				 random_start=True,
				 keep_gripper_closed=False,
				 highest_start=False,
				 x_limit = None,
				 y_limit = None,
				 z_limit = None,
				 **kwargs):
		super().__init__(scale_factor_pos=scale_factor_pos, image_width=image_width, image_height=image_height, home_displacement=home_displacement, random_start=random_start,
						 keep_gripper_closed=keep_gripper_closed, highest_start=highest_start, x_limit=x_limit, y_limit=y_limit, z_limit=z_limit, pitch=-90, **kwargs)

	def _record_trajectory(self, episode_len):
		
		actions = []
		state_obs = []
		image_obs = []
		rewards = []
		step = 0
		start = False
		action_types = ['move_forward', 'move_backward', 'move_left',
						'move_right', 'move_up', 'move_down']

		while(True):
			# Get observations
			obs = self.arm.get_position()[:3]
			image = self.cam.get_frame()
			self.joy.detect_event()
			pos, action = self.joy.move()

			if action == 'start':
				start = True
			elif action =='stop' or step == episode_len:
				for _ in range(episode_len - step):
					state_obs.append(obs)
					image_obs.append(image)
					actions.append(np.zeros(3).astype(np.float32))
					rewards.append(1)
				break
			elif action == 'cancel':
				return None, None, None, None
			elif start and action in action_types:
				pos = pos[:3]
				if action in action_types:
					state_obs.append(obs)
					image_obs.append(image)
					actions.append(pos-obs)
					rewards.append(0)
				step += 1

		return state_obs, image_obs, actions, rewards

class SideReach(BaseClass):
	def __init__(self, 
				 scale_factor_pos=0.25,
				 image_width=84,
				 image_height=84,
				 home_displacement=(0,0,0),
				 random_start=True,
				 keep_gripper_closed=False,
				 highest_start=False,
				 x_limit = None,
				 y_limit = None,
				 z_limit = None,
				 **kwargs):
		super().__init__(scale_factor_pos=scale_factor_pos, image_width=image_width, image_height=image_height, home_displacement=home_displacement, random_start=random_start,
						 keep_gripper_closed=keep_gripper_closed, highest_start=highest_start, x_limit=x_limit, y_limit=y_limit, z_limit=z_limit, roll=90, **kwargs)
		
	def init_arm(self, home_displacement=(0,0,0), keep_gripper_closed=False, highest_start=False,
				 x_limit=None, y_limit=None, z_limit=None, pitch=0, roll=180):
		self.arm = XArm(home_displacement=home_displacement, keep_gripper_closed=keep_gripper_closed, highest_start=highest_start, 
						x_limit=x_limit, y_limit=y_limit, z_limit=z_limit, pitch=pitch, roll=roll)
		self.arm.start_robot()
		self.arm.set_mode_and_state()
		self.arm.reset(home=True, reset_at_home=True)
		time.sleep(1)

	def _record_trajectory(self, episode_len):
		
		actions = []
		state_obs = []
		image_obs = []
		rewards = []
		step = 0
		start = False
		action_types = ['move_forward', 'move_backward', 'move_left',
						'move_right', 'move_up', 'move_down']

		while(True):
			obs = self.arm.get_position()[:3]
			image = self.cam.get_frame()

			self.joy.detect_event()
			pos, action = self.joy.move()

			if action == 'start':
				start = True
			elif action =='stop' or step == episode_len:
				for _ in range(episode_len - step):
					state_obs.append(obs)
					image_obs.append(image)
					actions.append(np.zeros(3).astype(np.float32))
					rewards.append(1)
				break
			elif action == 'cancel':
				return None, None, None, None
			elif start and action in action_types:
				pos = pos[:3]
				if action in action_types:
					state_obs.append(obs)
					image_obs.append(image)
					actions.append(pos-obs)
					rewards.append(0)
				step += 1

		return state_obs, image_obs, actions, rewards


class PickandPlace(BaseClass):
	def __init__(self, 
				 scale_factor_pos=0.25,
				 scale_factor_gripper=200,
				 image_width=84,
				 image_height=84,
				 home_displacement=(0,0,0),
				 random_start=False,
				 keep_gripper_closed=False,
				 highest_start=False,
				 x_limit = None,
				 y_limit = None,
				 z_limit = None,
				 **kwargs):
				super().__init__(scale_factor_pos=scale_factor_pos, image_width=image_width, image_height=image_height, home_displacement=home_displacement, random_start=random_start,
						 keep_gripper_closed=keep_gripper_closed, highest_start=highest_start, x_limit=x_limit, y_limit=y_limit, z_limit=z_limit, roll=180, **kwargs)

	def _record_trajectory(self, episode_len):		
		actions = []
		state_obs = []
		image_obs = []
		rewards = []
		step = 0
		start = False
		action_types = ['move_forward', 'move_backward', 'move_left', 'move_right',
						'move_up', 'move_down', 'close_gripper', 'open_gripper']

		while(True):
			# Get observations
			obs = self.arm.get_position()[:3]
			gripper_obs = np.reshape(self.arm.get_gripper_position(), (1,)).astype(np.float32)
			obs = np.concatenate((obs, gripper_obs), axis=0)
			image = self.cam.get_frame()
			
			self.joy.detect_event()
			pos, action = self.joy.move()
			if action is not None:
				print(action)
			if action == 'start':
				start = True
			elif action =='stop' or step == episode_len:
				for _ in range(episode_len - step):
					state_obs.append(obs)
					image_obs.append(image)
					actions.append(np.zeros(4).astype(np.float32))
					rewards.append(1)
				break
			elif action == 'cancel':
				return None, None, None, None
			elif start and action in action_types:
				state_obs.append(obs)
				image_obs.append(image)
				if 'gripper' in action:
					pos = np.reshape(pos, (1,)).astype(np.float32)
					pos = np.concatenate((np.zeros(3), pos), axis=0)
					print(f"pos with gripper:{pos}")
				else:
					pos = pos[:3]
					pos = np.concatenate((pos-obs[:3], np.zeros(1)), axis=0)
					print(f"pos without gripper:{pos}")
				actions.append(pos)
				rewards.append(0)
				step += 1

		return state_obs, image_obs, actions, rewards

class Pour(BaseClass):
	def __init__(self, 
				 scale_factor_pos=0.25,
				 image_width=84,
				 image_height=84,
				 home_displacement=(0,0,0),
				 random_start=True,
				 keep_gripper_closed=False,
				 highest_start=False,
				 x_limit = None,
				 y_limit = None,
				 z_limit = None,
				 **kwargs):
		super().__init__(scale_factor_pos=scale_factor_pos, image_width=image_width, image_height=image_height, home_displacement=home_displacement, 
						 random_start=random_start, keep_gripper_closed=keep_gripper_closed, highest_start=highest_start, x_limit=x_limit,
						 y_limit=y_limit, z_limit=z_limit, pitch=90, roll=-90, scale_factor_rotation=40, **kwargs)
		
	def init_arm(self, home_displacement=(0,0,0), keep_gripper_closed=False, highest_start=False,
				 x_limit=None, y_limit=None, z_limit=None, pitch=90, roll=-90):
		self.arm = XArm(home_displacement=home_displacement, keep_gripper_closed=keep_gripper_closed, highest_start=highest_start, 
						x_limit=x_limit, y_limit=y_limit, z_limit=z_limit, pitch=pitch, roll=roll)
		self.arm.start_robot()
		self.arm.set_mode_and_state()
		self.arm.reset(home=True, reset_at_home=True)
		time.sleep(1)

	def _record_trajectory(self, episode_len):
		
		actions = []
		state_obs = []
		image_obs = []
		rewards = []
		step = 0
		start = False
		action_types = ['move_forward', 'move_backward', 'move_left', 'move_right', 
						'move_up', 'move_down', 'rotate_arm_cw', 'rotate_arm_ccw']

		while(True):
			obs = self.arm.get_position()[:3]
			arm_angle_obs = np.reshape(self.arm.get_servo_angle()[6], (1,)).astype(np.float32)
			obs = np.concatenate((obs, arm_angle_obs), axis=0)
			image = self.cam.get_frame()

			self.joy.detect_event()
			pos, action = self.joy.move()
			
			if action == 'start':
				start = True
			elif action =='stop' or step == episode_len:
				for _ in range(episode_len - step):
					state_obs.append(obs)
					image_obs.append(image)
					actions.append(np.zeros(4).astype(np.float32))
					rewards.append(1)
				break
			elif action == 'cancel':
				return None, None, None, None
			elif start and action in action_types:
				if action in action_types:
					state_obs.append(obs)
					image_obs.append(image)
					if 'rotate_arm' in action:
						time.sleep(3)
						pos = np.reshape(pos, (1,)).astype(np.float32)
						pos = np.concatenate((np.zeros(3), (pos-arm_angle_obs)/40), axis=0)
					else:
						pos = pos[:3]
						pos = np.concatenate((pos-obs[:3], np.zeros(1)), axis=0)
					actions.append(pos)
					rewards.append(0)
				step += 1

		return state_obs, image_obs, actions, rewards
