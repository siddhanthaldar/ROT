import numpy as np
import torch
from torch import nn, optim, distributions
from torch.nn import functional as F

import utils
from agent.encoder import Encoder


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


class BCAgent:
	def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
				 hidden_dim, stddev_schedule, stddev_clip, use_tb, augment, suite_name, obs_type):
		self.device = device
		self.lr = lr
		self.stddev_schedule = stddev_schedule
		self.stddev_clip = stddev_clip
		self.use_tb = use_tb
		self.augment = augment
		self.use_encoder = True if (suite_name!="adroit" and obs_type=='pixels') else False

		# models
		if self.use_encoder:
			self.encoder = Encoder(obs_shape).to(device)
			repr_dim = self.encoder.repr_dim
		else:
			repr_dim = obs_shape[0]

		self.actor = Actor(repr_dim, action_shape, feature_dim,
						   hidden_dim).to(device)

		# optimizers
		if self.use_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

		# data augmentation
		self.aug = utils.RandomShiftsAug(pad=4)

		self.train()

	def __repr__(self):
		return "bc"
	
	def train(self, training=True):
		self.training = training
		if self.use_encoder:
			self.encoder.train(training)
		self.actor.train(training)

	def act(self, obs, step, eval_mode):
		obs = torch.as_tensor(obs, device=self.device)

		obs = self.encoder(obs.unsqueeze(0)) if self.use_encoder else obs.unsqueeze(0)
		stddev = utils.schedule(self.stddev_schedule, step)

		dist = self.actor(obs, stddev)

		if eval_mode:
			action = dist.mean
		else:
			action = dist.sample(clip=None)
		return action.cpu().numpy()[0]

	def update(self, replay_iter, expert_replay_iter, step, bc_regularize=False):
		metrics = dict()

		batch = next(expert_replay_iter)
		obs, action = utils.to_torch(batch, self.device)
		action = action.float()
		
		# augment
		if self.use_encoder and self.augment:
			obs = self.aug(obs.float())
			# encode
			obs = self.encoder(obs)
		else:
			obs = obs.float()

		stddev = utils.schedule(self.stddev_schedule, step)
		dist = self.actor(obs, stddev)
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)

		actor_loss = -log_prob.mean()
		
		if self.use_encoder:
			self.encoder_opt.zero_grad(set_to_none=True)
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		if self.use_encoder:
			self.encoder_opt.step()
		self.actor_opt.step()

		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()
			metrics['actor_logprob'] = log_prob.mean().item()
			metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

		return metrics

	def save_snapshot(self):
		keys_to_save = ['actor']
		if self.use_encoder:
			keys_to_save += ['encoder']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		return payload

	def load_snapshot(self, payload):
		for k, v in payload.items():
			self.__dict__[k] = v

		# Update optimizers
		if self.use_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
