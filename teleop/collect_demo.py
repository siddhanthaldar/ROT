from task_params import TASK_PARAMS

import os
from pathlib import Path

if __name__ == '__main__':

	image_width = 640
	image_height = 480
	task_name = 'EraseBoard' # Reach, InsertPegEasy/Medium/Hard, TurnKnob, HangMug/Bag/Hanger, DoorClose, BoxOpen, Pour, CupStacking, ButtonPress, HangHanger, EraseBoard
	num_trajectories = 1
	exp_name = 'random_s_fixed_g'
	sleep = 0.4
	
	task_params = TASK_PARAMS[task_name]
	task = task_params['demo_class'](sleep=sleep, image_width=image_width, image_height=image_height, home_displacement=task_params['home_displacement'],
								   random_start=task_params['random_start'], keep_gripper_closed=task_params['keep_gripper_closed'], highest_start=task_params['highest_start'], 
								   x_limit=task_params['x_limit'], y_limit=task_params['y_limit'], z_limit=task_params['z_limit'])
	
	directory_path = Path('./data') / task_name
	os.makedirs(directory_path, exist_ok=True)

	task.record_trajectories(num_trajectories=num_trajectories,
							 episode_len=task_params['episode_len'],
							 save=True,
							 directory_path=directory_path,
							 exp_name=exp_name)