from demo_class import *

TASK_PARAMS = {
		'Reach' : {
					'demo_class': Reach,
					'home_displacement': (2.16,-0.047,1.06),
					'keep_gripper_closed': False,
					'highest_start' : False,
					'random_start' : True,
					'x_limit' : (0.5, 3.5),
					'y_limit' : (-1.7, 1.3),
					'z_limit' : (0.2, 3.4),
					'episode_len': 30
				  },
		'InsertPegEasy' : {
							'demo_class': Reach,
							'home_displacement': (2.16,-0.047,2.06),
							'keep_gripper_closed': True,
							'highest_start' : False,
							'random_start' : True,
							'x_limit' : (0.5, 3.5),
							'y_limit' : (-1.7, 1.3),
							'z_limit' : (2.1, 3.4),
							'episode_len': 30
						  },
		'InsertPegMedium':{
							'demo_class': Reach,
							'home_displacement': (2.16,-0.047,2.06),
							'keep_gripper_closed': True,
							'highest_start' : False,
							'random_start' : True,
							'x_limit' : (0.5, 3.5),
							'y_limit' : (-1.7, 1.3),
							'z_limit' : (2.1,3.4),
							'episode_len': 30
						  },
		'InsertPegHard' : {
							'demo_class': Reach,
							'home_displacement': (2.16,-0.047,2.06),
							'keep_gripper_closed': True,
							'highest_start' : False,
							'random_start' : True,
							'x_limit' : (0.5, 3.5),
							'y_limit' : (-1.7, 1.3),
							'z_limit' : (2.1, 3.4),
							'episode_len': 30
						  },
        'TurnKnob' : {
						 'demo_class': Reach,
						 'home_displacement': (2.16,-0.047,2.06),
						 'keep_gripper_closed': True,
						 'highest_start' : True,
						 'random_start' : True,
						 'x_limit' : (0.5, 3.5),
						 'y_limit' : (-1.7, 1.3),
						 'z_limit' : (1.4, 2.5),
						 'episode_len': 40
					   },
        'HangMug' : {
						 'demo_class': HorizontalReach,
						 'home_displacement': (2.16,-0.047,2.06),
						 'keep_gripper_closed': True,
						 'highest_start' : False,
						 'random_start' : True,
						 'x_limit' : (2.05, 3.47),
						 'y_limit' : (-1.1, 1.5),
						 'z_limit' : (3.40, 4.55),
						 'episode_len': 30
					   },
		'HangBag' : {
						 'demo_class': HorizontalReach,
						 'home_displacement': (2.16,-0.047,2.06),
						 'keep_gripper_closed': True,
						 'highest_start' : False,
						 'random_start' : True,
						 'x_limit' : (2.05, 3.75),
						 'y_limit' : (-1.2, 1.),
						 'z_limit' : (3.65, 4.55),
						 'episode_len': 30
					   },
		'HangHanger' : {
						 'demo_class': HorizontalReach,
						 'home_displacement': (2.16,-0.047,2.06),
						 'keep_gripper_closed': True,
						 'highest_start' : False,
						 'random_start' : True,
						 'x_limit' : (1.5, 3.36),
						 'y_limit' : (-1.5, 2.26),
						 'z_limit' : (3.7, 4.22),
						 'episode_len': 30
					   },
		'DoorClose' : {
						 'demo_class': Reach,
						 'home_displacement': (2.77,2.88,1.62),
						 'keep_gripper_closed': True,
						 'highest_start' : False,
						 'random_start' : True,
						 'x_limit' : (2.3, 3.3),
						 'y_limit' : (-0.9, 2.88),
						 'z_limit' : (1.6, 4.55),
						 'episode_len': 30
					   },
		'BoxOpen' : {
						 'demo_class': SideReach,
						 'home_displacement': (2.2, 3, 0.5),
						 'keep_gripper_closed': False,
						 'highest_start' : True,
						 'random_start' : True,
						 'x_limit' : (0.4, 2.6),
						 'y_limit' : (1.4, 3.7),
						 'z_limit' : (0.39, 1.5),
						 'episode_len': 30
					   },
		'Pour' : {
						 'demo_class': Pour,
						 'home_displacement': (1.5, -1, 0),
						 'keep_gripper_closed': True,
						 'highest_start' : False,
						 'random_start' : True,
						 'x_limit' : (0.4, 2.8),
						 'y_limit' : (-2.4, 0.3),
						 'z_limit' : (-0.2, 0.8),
						 'episode_len': 30
					   },
		'CupStacking' : {
							'demo_class': Reach,
							'home_displacement': (2.2,-0.047,3.6),
							'keep_gripper_closed': True,
							'highest_start' : False,
							'random_start' : True,
							'x_limit' : (0.5, 3.5),
							'y_limit' : (-1.7, 1.3),
							'z_limit' : (2, 3.4),
							'episode_len': 30
						  },
		'ButtonPress' : {
						 'demo_class': Reach,
						 'home_displacement': (2.16,-0.047,2.06),
						 'keep_gripper_closed': True,
						 'highest_start' : True,
						 'random_start' : True,
						 'x_limit' : (0.7, 3.5),
						 'y_limit' : (-1.5, 1.5),
						 'z_limit' : (1.5, 2.7),
						 'episode_len': 30
					   },
		'EraseBoard' : {
						 'demo_class': Reach,
						 'home_displacement': (2.16,-0.047,2.06),
						 'keep_gripper_closed': True,
						 'highest_start' : True,
						 'random_start' : True,
						 'x_limit' : (0.25, 2.8),
						 'y_limit' : (-2, 2),
						 'z_limit' : (0.12, 1.5),
						 'episode_len': 40
					   },
	}