from gym.envs.registration import register 

register(
	id='particle-v0',
	entry_point='gym_envs.envs:ParticleEnv',
	max_episode_steps=65,
	)

register(
	id='RobotReach-v1',
	entry_point='gym_envs.envs:RobotReachEnv',
	max_episode_steps=20,
	)

register(
	id='RobotInsertPeg-v1',
	entry_point='gym_envs.envs:RobotInsertPegEnv',
	max_episode_steps=30,
	) 

register(
	id='RobotPressBlock-v1',
	entry_point='gym_envs.envs:RobotPressBlockEnv',
	max_episode_steps=30,
	)

register(
	id='RobotTurnKnob-v1',
	entry_point='gym_envs.envs:RobotTurnKnobEnv',
	max_episode_steps=40,
	) 

register(
	id='RobotHangMug-v1',
	entry_point='gym_envs.envs:RobotHangMugEnv',
	max_episode_steps=30,
	) 

register(
	id='RobotHangBag-v1',
	entry_point='gym_envs.envs:RobotHangBagEnv',
	max_episode_steps=30,
	) 

register(
	id='RobotHangHanger-v1',
	entry_point='gym_envs.envs:RobotHangHangerEnv',
	max_episode_steps=30,
	) 

register(
	id='RobotDoorClose-v1',
	entry_point='gym_envs.envs:RobotDoorCloseEnv',
	max_episode_steps=30,
	) 

register(
	id='RobotBoxOpen-v1',
	entry_point='gym_envs.envs:RobotBoxOpenEnv',
	max_episode_steps=40,
	) 

register(
	id='RobotPour-v1',
	entry_point='gym_envs.envs:RobotPourEnv',
	max_episode_steps=30,
	)

register(
	id='RobotCupStacking-v1',
	entry_point='gym_envs.envs:RobotCupStackingEnv',
	max_episode_steps=30,
	) 

register(
	id='RobotButtonPress-v1',
	entry_point='gym_envs.envs:RobotBottonPressEnv',
	max_episode_steps=30,
	) 

register(
	id='RobotEraseBoard-v1',
	entry_point='gym_envs.envs:RobotEraseBoardEnv',
	max_episode_steps=40,
	) 