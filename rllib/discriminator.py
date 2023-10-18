#  Copyright: (C) ETAS GmbH 2019. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.nn.utils import spectral_norm

# from baselines.common.running_mean_std import RunningMeanStd
from common.multiprocessing_env import RunningMeanStd
from common import helper


def train_discriminator(
    discriminator,
    replay_buffer,
    expert_batch_generator, sample_batch_size,
    optimizer_discrim,
    discrim_criterion,
    discriminator_update_iterations):

    d_loss_values = []
    for _ in range(discriminator_update_iterations):
        expert_state, expert_action, _ = expert_batch_generator.sample(sample_batch_size)
        expert_state = torch.tensor(expert_state, dtype=torch.float).to(discriminator.device)
        expert_action = torch.tensor(expert_action, dtype=torch.float).to(discriminator.device)
        expert_state_action = torch.cat([expert_state, expert_action], 1)

        # state, action, reward, next_state, done, next_action, next_next_state, time_steps
        p_states, p_actions, _, _, _, _, _, _ = replay_buffer.sample(sample_batch_size)
        p_states = torch.tensor(p_states, dtype=torch.float).to(discriminator.device)
        p_actions = torch.tensor(p_actions, dtype=torch.float).to(discriminator.device)
        state_action = torch.cat([p_states, p_actions], 1)

        d_loss_value, grad_loss = update_discriminator(
            state_action=state_action,
            expert_state_action=expert_state_action,
            discriminator=discriminator,
            optimizer_discrim=optimizer_discrim,
            discrim_criterion=discrim_criterion,
        )
        d_loss_values.append(d_loss_value)
    return np.mean(d_loss_values)


def update_discriminator(
    state_action,
    expert_state_action,
    discriminator,
    optimizer_discrim,
    discrim_criterion,
    use_noisy_label=False,
    use_gp=False,
):

    fake = discriminator(state_action)
    real = discriminator(expert_state_action)
    optimizer_discrim.zero_grad()

    if use_noisy_label:
        policy_ones = np.random.normal(loc=1, scale=0.1, size=(state_action.size(0), 1))
        expert_zeros = np.random.normal(
            loc=0, scale=0.1, size=(expert_state_action.shape[0], 1)
        )
        policy_ones = torch.FloatTensor(policy_ones).to(discriminator.device)
        expert_zeros = torch.FloatTensor(expert_zeros).to(discriminator.device)
    else:
        policy_ones = torch.ones((state_action.shape[0], 1)).to(discriminator.device)
        expert_zeros = torch.zeros((expert_state_action.size(0), 1)).to(discriminator.device)

    discrim_loss = discrim_criterion(fake, policy_ones) + discrim_criterion(
        real, expert_zeros
    )

    d_loss_value = discrim_loss.item()

    if use_gp:
        grad_pen = discriminator.compute_grad_pen(expert_state_action, state_action)
        discrim_loss = discrim_loss + grad_pen
        grad_loss = grad_pen.item()
    else:
        grad_loss = 0

    discrim_loss.backward()
    optimizer_discrim.step()
    return d_loss_value, grad_loss


class Discriminator(nn.Module):
    def __init__(
        self,
        num_inputs,
        hidden_size,
        use_spec_norm=True,
        device="cpu",
        use_reward_normalization=False,
        use_std_reward=True,
    ):
        super(Discriminator, self).__init__()
        self.device = device
        self.use_reward_normalization = use_reward_normalization
        self.use_std_reward = use_std_reward
        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())
        # F.tanh  # nn.ReLU
        self.sigmoid = nn.Sigmoid()
        if use_spec_norm:
            self.forward_inside = nn.Sequential(
                spectral_norm(nn.Linear(num_inputs, hidden_size)),
                nn.LeakyReLU(negative_slope=0.1),
                spectral_norm(nn.Linear(hidden_size, hidden_size)),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(hidden_size, 1),
            )
        else:
            self.forward_inside = nn.Sequential(
                nn.Linear(num_inputs, hidden_size),
                nn.LeakyReLU(negative_slope=0.1),
                # nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(negative_slope=0.1),
                # nn.LeakyReLU(),
                # nn.Dropout(p=0.3),
                nn.Linear(hidden_size, 1),
            )
        self.apply(helper.init_weights)

    def forward(self, x):
        x = self.forward_inside(x)
        prob = self.sigmoid(x)
        return prob

    def compute_grad_pen(self, expert_state_action, policy_state_action, lambda_=10):
        alpha = torch.rand(expert_state_action.size(0), 1)
        expert_data = expert_state_action
        policy_data = policy_state_action

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + ((1 - alpha) * policy_data)
        mixup_data = autograd.Variable(
            mixup_data.to(expert_data.device), requires_grad=True
        )

        disc = self.forward_inside(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def expert_reward(self, state, action):
        self.eval()
        EPS = 1e-8
        state_action = torch.cat([state, action], 1).to(self.device)
        if self.use_std_reward:
            reward = -np.log(np.abs(self.forward(state_action).cpu().squeeze(1).data.numpy()) + EPS)
        else:
            reward = -np.log(np.abs(self.forward(state_action).cpu().squeeze(1).data.numpy()) + EPS) + \
                     np.log((np.abs(1-self.forward(state_action).cpu().squeeze(1).data.numpy())) + EPS)
        return reward

    def predict_reward(self, state, action, update_rms=True):
        with torch.no_grad():
            reward = self.expert_reward(state, action)
            # if self.returns is None:
            #     self.returns = reward.copy()  # clone()
            if update_rms:
                # self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(reward)
            if self.use_reward_normalization:
                reward = reward / np.sqrt(self.ret_rms.var + 1e-8)
            return reward

    def mgail_reward(self, state, action):
        self.eval()
        EPS = 1e-8
        state_action = torch.cat([state, action], 1).to(self.device)
        return -torch.log(torch.abs(self.forward(state_action).cpu()) + EPS)

    def save(self, name):
        self.forward_inside.to("cpu")
        torch.save(
            {
                "discriminator": self.forward_inside.state_dict(),
            },
            name,
        )
        print(f"saved  model at {name}")
        self.forward_inside.to(self.device)

    def load(self, name):
        print("loads model from ", name)
        state_dicts = torch.load(name, map_location="cpu")
        self.forward_inside.load_state_dict(state_dicts["discriminator"])
        self.forward_inside.to(self.device)
