import pickle
import numpy as np
import cv2
import os
from pathlib import Path

root_dir = Path('/mnt/robotlab/vaibhav/Projects/copied/Regularized-OT-on-Robot/teleop/data/EraseBoard/2022_6_14_17_29_random_s_fixed_g')
num_traj = 1
use_depth = False
image_height = 84
image_width = 84

images_list = []
states_list = []
actions_list = []
rewards_list = []
for index in range(0,num_traj):
	file_name = f"{root_dir}/{index}/traj.pickle"

	with open(file_name, 'rb') as f:
		traj = pickle.load(f)

	images = np.array(traj['image_observation'], dtype=np.uint8)
	resized_images = []
	for img in images:
		resized_images.append(cv2.resize(img, (image_width, image_height), interpolation=cv2.INTER_AREA))
	images = np.array(resized_images, dtype=np.uint8)
	images = np.transpose(images, (0,3,1,2))
	if not use_depth:
		images = images[:, :3]
	images_list.append(images)
	states_list.append(np.array(traj['state_observation'], dtype=np.float32))
	action = np.array(traj['action'], dtype=np.float32)

	action[:,:3] /= 0.25
	actions_list.append(action)
	rewards_list.append(np.array(traj['reward'], dtype=np.float32))

images_list = np.array(images_list)
states_list = np.array(states_list)
actions_list = np.array(actions_list)
rewards_list = np.array(rewards_list)

SAVE_PATH = root_dir / f'expert_demos_{image_height}.pkl'
with open(SAVE_PATH, 'wb') as outfile:
	pickle.dump([images_list, states_list, actions_list, rewards_list], outfile)
print("Saved.")