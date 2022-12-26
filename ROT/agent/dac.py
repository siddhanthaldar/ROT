import hydra
import numpy as np
from torch import autograd
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.encoder import Encoder
import time
import copy


class Discriminator(nn.Module):
	def __init__(self, in_dim, hid_dim):
		super().__init__()
		self.trunk = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.ReLU(),
								   nn.Linear(hid_dim, hid_dim), nn.ReLU(),
								   nn.Linear(hid_dim, 1))

		self.apply(utils.weight_init)

	def forward(self, x):
		output = self.trunk(x)
		return output


def compute_gradient_penalty(discriminator, expert_data, policy_data):
	alpha = torch.rand(expert_data.size(0), 1)
	alpha = alpha.expand_as(expert_data).to(expert_data.device)

	mixup_data = alpha * expert_data + (1 - alpha) * policy_data
	mixup_data.requires_grad = True

	disc = discriminator(mixup_data)
	ones = torch.ones(disc.size()).to(disc.device)
	grad = autograd.grad(outputs=disc,
						 inputs=mixup_data,
						 grad_outputs=ones,
						 create_graph=True,
						 retain_graph=True,
						 only_inputs=True)[0]

	grad_pen = 10 * (grad.norm(2, dim=1) - 1).pow(2).sum()
	return grad_pen


class Actor(nn.Module):
	def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
		super().__init__()

		self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
								   nn.LayerNorm(feature_dim), nn.Tanh())

		self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, action_shape[0]))

		self.apply(utils.weight_init)

	def forward(self, obs, std):
		h = self.trunk(obs)

		mu = self.policy(h)
		mu = torch.tanh(mu)
		std = torch.ones_like(mu) * std

		dist = utils.TruncatedNormal(mu, std)
		return dist


class Critic(nn.Module):
	def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
		super().__init__()

		self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
								   nn.LayerNorm(feature_dim), nn.Tanh())

		self.Q1 = nn.Sequential(
			nn.Linear(feature_dim + action_shape[0], hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

		self.Q2 = nn.Sequential(
			nn.Linear(feature_dim + action_shape[0], hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

		self.apply(utils.weight_init)

	def forward(self, obs, action):
		h = self.trunk(obs)
		h_action = torch.cat([h, action], dim=-1)
		q1 = self.Q1(h_action)
		q2 = self.Q2(h_action)

		return q1, q2


class DACAgent:
	def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
				 hidden_dim, critic_target_tau, num_expl_steps,
				 update_every_steps, stddev_schedule, stddev_clip, use_tb,
				 augment, use_actions, suite_name, obs_type, bc_weight_type):
		self.device = device
		self.lr = lr
		self.critic_target_tau = critic_target_tau
		self.update_every_steps = update_every_steps
		self.use_tb = use_tb
		self.num_expl_steps = num_expl_steps
		self.stddev_schedule = stddev_schedule
		self.stddev_clip = stddev_clip
		self.use_actions = use_actions
		self.use_encoder = True if obs_type=='pixels' else False
		self.augment = augment and self.use_encoder
		self.bc_weight_type = bc_weight_type

		# models
		if self.use_encoder:
			self.encoder = Encoder(obs_shape).to(device)
			self.encoder_target = Encoder(obs_shape).to(device)
			repr_dim = self.encoder.repr_dim
		else:
			repr_dim = obs_shape[0]

		disc_dim = feature_dim + action_shape[0] if use_actions else feature_dim
		self.discriminator = Discriminator(disc_dim, hidden_dim).to(device)

		self.actor = Actor(repr_dim, action_shape, feature_dim,
						   hidden_dim).to(device)

		self.critic = Critic(repr_dim, action_shape, feature_dim,
							 hidden_dim).to(device)
		self.critic_target = Critic(repr_dim, action_shape,
									feature_dim, hidden_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())

		# optimizers
		if self.use_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
		self.discriminator_opt = torch.optim.Adam(
			self.discriminator.parameters(), lr=lr)
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
		self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

		# data augmentation
		self.aug = utils.RandomShiftsAug(pad=4)

		self.train()
		self.critic_target.train()

	def train(self, training=True):
		self.training = training
		if self.use_encoder:
			self.encoder.train(training)
		self.actor.train(training)
		self.critic.train(training)
		self.discriminator.train(training)

	def __repr__(self):
		return 'dac'

	def act(self, obs, step, eval_mode):
		obs = torch.as_tensor(obs, device=self.device)

		obs = self.encoder(obs.unsqueeze(0)) if self.use_encoder else obs.unsqueeze(0)
		stddev = utils.schedule(self.stddev_schedule, step)

		dist = self.actor(obs, stddev)

		if eval_mode:
			action = dist.mean
		else:
			action = dist.sample(clip=None)
			if step < self.num_expl_steps:
				action.uniform_(-1.0, 1.0)
		return action.cpu().numpy()[0]

	def update_critic(self, obs, action, reward, discount, next_obs, step):
		metrics = dict()

		with torch.no_grad():
			stddev = utils.schedule(self.stddev_schedule, step)

			dist = self.actor(next_obs, stddev)
			next_action = dist.sample(clip=self.stddev_clip)
			target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
			target_V = torch.min(target_Q1, target_Q2)
			target_Q = reward + (discount * target_V)

		Q1, Q2 = self.critic(obs, action)

		critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

		if self.use_tb:
			metrics['critic_target_q'] = target_Q.mean().item()
			metrics['critic_q1'] = Q1.mean().item()
			metrics['critic_q2'] = Q2.mean().item()
			metrics['critic_loss'] = critic_loss.item()

		# optimize encoder and critic
		if self.use_encoder:
			self.encoder_opt.zero_grad(set_to_none=True)
		self.critic_opt.zero_grad(set_to_none=True)
		critic_loss.backward()
		self.critic_opt.step()
		if self.use_encoder:
			self.encoder_opt.step()

		return metrics

	def update_actor(self, obs, expert_obs, obs_qfilter, expert_action, bc_regularize, step):
		metrics = dict()

		stddev = utils.schedule(self.stddev_schedule, step)

		dist = self.actor(obs, stddev)
		action = dist.sample(clip=self.stddev_clip)
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)

		Q1, Q2 = self.critic(obs, action)
		Q = torch.min(Q1, Q2)

		# Compute bc weight
		if not bc_regularize:
			bc_weight = 0.0
		elif self.bc_weight_type == "linear":
			bc_weight = utils.schedule(self.bc_weight_schedule, step)
		elif self.bc_weight_type == "qfilter":
			"""
			Soft Q-filtering inspired from 			
			Nair, Ashvin, et al. "Overcoming exploration in reinforcement 
			learning with demonstrations." 2018 IEEE international 
			conference on robotics and automation (ICRA). IEEE, 2018.
			"""
			with torch.no_grad():
				stddev = 0.1
				dist_qf = self.actor_bc(obs_qfilter, stddev)
				action_qf = dist_qf.mean
				Q1_qf, Q2_qf = self.critic(obs_qfilter.clone(), action_qf)
				Q_qf = torch.min(Q1_qf, Q2_qf)
				bc_weight = (Q_qf>Q).float().mean().detach()

		actor_loss = - Q.mean() * (1-bc_weight)

		stddev = 0.1
		dist_bc = self.actor(expert_obs, stddev)
		log_prob_bc = dist_bc.log_prob(expert_action).sum(-1, keepdim=True)
		if bc_regularize:
			actor_loss += - log_prob_bc.mean()*bc_weight*0.03

		# optimize actor
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.actor_opt.step()

		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()
			metrics['actor_logprob'] = log_prob.mean().item()
			metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
			metrics['actor_q'] = Q.mean().item()
			if bc_regularize and self.bc_weight_type == "qfilter":
				metrics['actor_qf'] = Q_qf.mean().item()
			metrics['bc_weight'] = bc_weight
			metrics['regularized_rl_loss'] = -Q.mean().item()* (1-bc_weight)
			metrics['rl_loss'] = -Q.mean().item()
			if bc_regularize:
				metrics['regularized_bc_loss'] = - log_prob_bc.mean().item()*bc_weight*0.03
				metrics['bc_loss'] = - log_prob_bc.mean().item()*0.03
			
		return metrics

	def update(self, replay_iter, expert_replay_iter, step, bc_regularize=False):
		metrics = dict()

		if step % self.update_every_steps != 0:
			return metrics

		obs, action, reward, discount, next_obs = utils.to_torch(
			next(replay_iter), self.device)

		obs = obs.float()
		next_obs = next_obs.float()

		expert_obs, expert_action = utils.to_torch(next(expert_replay_iter),
												   self.device)

		expert_obs = expert_obs.float()

		# augment
		if self.use_encoder and self.augment:
			obs_qfilter = self.aug(obs.clone())
			obs = self.aug(obs)
			next_obs = self.aug(next_obs)
			expert_obs = self.aug(expert_obs)
		else:
			obs_qfilter = obs.clone()

		# encode
		if self.use_encoder:
			obs = self.encoder(obs)
			with torch.no_grad():
				next_obs = self.encoder(next_obs)
				expert_obs = self.encoder(expert_obs)

		results = self.update_discriminator(obs, action, expert_obs,
											expert_action)
		metrics.update(results)

		if self.use_tb:
			metrics['batch_reward'] = reward.mean().item()

		if bc_regularize and self.bc_weight_type=="qfilter":
			obs_qfilter = self.encoder_bc(obs_qfilter) if self.use_encoder else obs_qfilter
			obs_qfilter = obs_qfilter.detach()
			expert_obs = expert_obs.detach()
			expert_action = expert_action.detach()
		else:
			obs_qfilter = None
			
		# update critic
		metrics.update(
			self.update_critic(obs, action, reward, discount, next_obs, step))

		# update actor
		metrics.update(self.update_actor(obs.detach(), expert_obs, obs_qfilter, expert_action, bc_regularize, step))

		# update critic target
		utils.soft_update_params(self.critic, self.critic_target,
								 self.critic_target_tau)

		return metrics

	def dac_rewarder(self, obses, actions):
		obses = torch.tensor(obses).to(self.device)
		obses = self.critic.trunk(self.encoder(obses)) if self.use_encoder else self.critic.trunk(obses)
		if self.use_actions:
			actions = torch.tensor(actions).to(self.device)
			obses = torch.cat([obses, actions], dim=1)
		with torch.no_grad():
			with utils.eval_mode(self.discriminator):
				d = self.discriminator(obses)
			s = torch.sigmoid(d)
			reward = s.log() - (1 - s).log()
			return reward.flatten().detach().cpu().numpy()

	def update_discriminator(self, policy_obs, policy_action, expert_obs,
							 expert_action):
		metrics = dict()
		batch_size = expert_obs.shape[0]
		# policy batch size is 2x
		policy_obs = policy_obs[:batch_size]
		policy_action = policy_action[:batch_size]

		ones = torch.ones(batch_size, device=self.device)
		zeros = torch.zeros(batch_size, device=self.device)

		disc_input = torch.cat([expert_obs, policy_obs], dim=0)
		if self.use_actions:
			disc_action = torch.cat([expert_action, policy_action], dim=0)
			disc_input = torch.cat([disc_input, disc_action], dim=1)
		disc_label = torch.cat([ones, zeros], dim=0).unsqueeze(dim=1)

		with torch.no_grad():
			disc_input = self.critic.trunk(disc_input)

		disc_output = self.discriminator(disc_input)
		dac_loss = F.binary_cross_entropy_with_logits(disc_output,
													  disc_label,
													  reduction='sum')

		expert_obs, policy_obs = torch.split(disc_input, batch_size, dim=0)
		grad_pen = compute_gradient_penalty(self.discriminator, expert_obs,
											policy_obs)

		dac_loss /= batch_size
		grad_pen /= batch_size

		metrics['disc_loss'] = dac_loss.mean().item()
		metrics['disc_grad_pen'] = grad_pen.mean().item()

		self.discriminator_opt.zero_grad(set_to_none=True)
		dac_loss.backward()
		grad_pen.backward()
		self.discriminator_opt.step()
		return metrics

	def save_snapshot(self):
		keys_to_save = ['actor', 'critic']
		if self.use_encoder:
			keys_to_save += ['encoder']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		return payload

	def load_snapshot(self, payload):
		for k, v in payload.items():
			self.__dict__[k] = v
		self.critic_target.load_state_dict(self.critic.state_dict())
		if self.use_encoder:
			self.encoder_target.load_state_dict(self.encoder.state_dict())
		
		if self.bc_weight_type == "qfilter":
			# Store a copy of the BC policy with frozen weights
			if self.use_encoder:
				self.encoder_bc = copy.deepcopy(self.encoder)
				for param in self.encoder_bc.parameters():
					param.requires_grad = False
			self.actor_bc = copy.deepcopy(self.actor)
			for param in self.actor_bc.parameters():
				param.required_grad = False

		# Update optimizers
		if self.use_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
		self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
