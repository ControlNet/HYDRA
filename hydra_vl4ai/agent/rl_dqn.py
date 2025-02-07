import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


# MLP head
class MLPHead(nn.Module):

    def __init__(
        self,
        input_dim,
        mlp_hidden_dim,
        output_dim,
        layer_num
    ) -> None:
        super().__init__()

        mlp_head = []
        for idx in range(layer_num):
            if idx == 0:
                i_dim = input_dim
            else:
                i_dim = mlp_hidden_dim

            if idx == layer_num - 1:
                o_dim = output_dim
            else:
                o_dim = mlp_hidden_dim

            mlp_head.append(nn.Linear(i_dim, o_dim))

            # if idx != layer_num -1: 
            mlp_head.append(nn.Sigmoid())

        self.mlp_head = nn.Sequential(*mlp_head)

    def forward(self, x):
        return self.mlp_head(x)


class ReplayBuffer:

    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = int((self.pos + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DQN_EmbeddingViaLLM:

    def __init__(
        self,
        device,
        llm_embedding_dim_concat,
        mlp_hidden_dim,
        action_dim,
        critic_layer_num,
        critic_lr,
    ) -> None:
        self.device = device

        self.critic_head = MLPHead(input_dim=llm_embedding_dim_concat, mlp_hidden_dim=mlp_hidden_dim,
            output_dim=action_dim, layer_num=critic_layer_num).to(device)
        self.tar_critic_head = MLPHead(input_dim=llm_embedding_dim_concat, mlp_hidden_dim=mlp_hidden_dim,
            output_dim=action_dim, layer_num=critic_layer_num).to(device)
        for tar_param, param in zip(self.tar_critic_head.parameters(), self.critic_head.parameters()):
            tar_param.data.copy_(param.data)

        self.critic_optim = optim.Adam(self.critic_head.parameters(), lr=critic_lr)

    def get_action(self, obs, batch_input=False):
        if not batch_input:
            obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        else:
            obs = torch.FloatTensor(obs).to(self.device)

        qval = self.critic_head.forward(x=obs)

        return qval.detach().cpu().numpy().flatten()
        # if not batch_input: return qval.argmax().detach().cpu().numpy().flatten() 
        # else: return qval.argmax(dim=-1).detach().cpu().numpy().flatten() 

    def update(self, replay_buffer, batch_size, reward_scale=10., gamma=0.99, soft_tau=1e-2, is_clip_gradient=True,
        clip_gradient_val=40
    ):
        obs, action, reward, next_obs, done = replay_buffer.sample(batch_size)

        obs = torch.FloatTensor(obs).to(self.device)  # obs.size = (batch_size, 1+seq_dim)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(
            self.device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(
            dim=0) + 1e-6)  # normalize with batch mean and std; plus a small number to prevent numerical problem
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
        # print(f'self.critic_head.forward(x=obs) {self.critic_head.forward(x=obs).shape}')
        # print(f'action.long() {action.long().shape}')
        # qval = self.critic_head.forward(x=obs).gather(1, action.long())
        val_value_ = self.critic_head.forward(x=obs)
        qval = val_value_.gather(1, action.long().reshape(val_value_.size(0), 1))

        with torch.no_grad():
            max_next_qval = self.tar_critic_head.forward(x=next_obs).max(dim=-1)[0].unsqueeze(-1)
            tar_qval = reward + gamma * (1 - done) * max_next_qval

        loss_func = nn.MSELoss()
        qloss = loss_func(qval, tar_qval.detach())

        self.critic_optim.zero_grad()
        qloss.backward()
        if is_clip_gradient: clip_grad_norm_(self.critic_head.parameters(), clip_gradient_val)
        self.critic_optim.step()

        for tar_param, param in zip(self.tar_critic_head.parameters(), self.critic_head.parameters()):
            tar_param.data.copy_(param.data * soft_tau + tar_param.data * (1 - soft_tau))

        return qloss.detach().cpu().item()

    def save_model(self, path: str):
        torch.save(self.critic_head.state_dict(), path)

    def load_model(self, path: str):
        self.critic_head.load_state_dict(torch.load(path))

        for tar_param, param in zip(self.tar_critic_head.parameters(), self.critic_head.parameters()):
            tar_param.data.copy_(param.data)

        self.critic_head.eval()
        self.tar_critic_head.eval()

    def eval_mode(self):
        self.critic_head.eval()
        self.tar_critic_head.eval()

    def train_mode(self):
        self.critic_head.train()
        self.tar_critic_head.train()
