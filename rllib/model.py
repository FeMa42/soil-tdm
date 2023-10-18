#  Copyright: (C) ETAS GmbH 2019. All rights reserved.

from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from common import helper
from common.helper import AddBias
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
from nflib.flows import NormalizingFlowModel


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, bias = 0):
        super(QNetwork, self).__init__()
        # print("ATTENTION!!! This was changed to support old models, change back!!!")
        self.q = nn.Sequential(
            nn.Linear(num_inputs+num_actions, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.bias = bias

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.q(x)+self.bias
        return x

# Normal
FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(
    self, actions.float()
).sum(-1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNNValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_size, bias=0):
        super(CNNValueNetwork, self).__init__()
        def init_(m): return init(m, nn.init.orthogonal_, lambda x: nn.init.
                                  constant_(x, 0), nn.init.calculate_gain('relu'))

        self.base = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(), Flatten())

        self.q = nn.Sequential(
            nn.Linear(64 * 2 * 2, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.bias = bias

    def forward(self, state):
        state_conv = self.base(state.transpose(1, 3) / 255.0)
        x = self.q(state_conv)+self.bias
        return x


class CNNActor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=512):
        super(CNNActor, self).__init__()
        self._hidden_size = hidden_size

        def init_(m): return init(m, nn.init.orthogonal_, lambda x: nn.init.
                                  constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(64 * 2 * 2, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, num_actions)))
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    def forward(self, inputs):
        x = self.main(inputs.transpose(1, 3) / 255.0)
        return x


class ActorCritic(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        hidden_size,
        std=0.0,
        device="cpu",
        use_batch_actor=False,
        act_limit=10.
    ):
        super(ActorCritic, self).__init__()

        # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
        #                        constant_(x, 0), np.sqrt(2))
        self.device = device
        self.use_batch_actor = use_batch_actor
        self.has_deterministic = False

        if len(num_inputs) == 3:
            self.critic = CNNValueNetwork(num_inputs[-1], hidden_size=hidden_size)
        else:
            self.critic = nn.Sequential(
                nn.Linear(num_inputs, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            ).to(device)

        if len(num_inputs) == 3:
            self.actor = CNNActor(num_inputs[-1], num_actions=num_outputs, hidden_size=hidden_size)
        else:
            self.actor = nn.Sequential(
                nn.Linear(num_inputs, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_outputs),
            ).to(device)

        if use_batch_actor:
            self.batch_actor = nn.Sequential(
                nn.Linear(num_inputs, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, num_outputs),
            ).to(device)
        else:
            self.batch_actor = None

        self.act_limit = act_limit
        num_actions = num_outputs
        self.action_shape = (num_outputs,)
        self.log_std = AddBias(torch.zeros(num_outputs)).to(device)

        self.train()

        self.apply(helper.init_weights)

    def get_policy_parameter(self):
        trainable_parameters = [p for p in self.actor.parameters() if p.requires_grad]
        trainable_parameters.extend([p for p in self.log_std.parameters() if p.requires_grad])
        trainable_parameters.extend([p for p in self.batch_actor.parameters() if p.requires_grad])
        return trainable_parameters

    def get_value_parameter(self):
        trainable_parameters = [p for p in self.critic.parameters() if p.requires_grad]
        return trainable_parameters

    def forward(self, x):
        value = self.critic(x)
        if self.use_batch_actor:
            mu = self.batch_actor(x)
        else:
            mu = self.actor(x)

        assert not torch.isnan(mu).any()
        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(mu.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        std = self.log_std(zeros).exp()  # .expand_as(mu)
        assert not torch.isnan(std).any()

        # std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value

    def forward_sampling(self, state, n_samples=50):
        repeated_state = state.repeat(n_samples, 1)

        mu = self.actor(repeated_state)
        assert not torch.isnan(mu).any()
        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(mu.size())
        if repeated_state.is_cuda:
            zeros = zeros.cuda()
        std = self.log_std(zeros).exp()
        assert not torch.isnan(std).any()
        normal = Normal(mu, std)
        action = normal.rsample()
        log_prob = normal.log_prob(action).view(action.size(0), -1).sum(-1)
        log_prob = log_prob.unsqueeze(-1)

        return action, log_prob

    def sum_actor(self):
        sum_up = 0
        for stuff in self.actor.named_parameters():
            sum_up += torch.abs(torch.sum(stuff[1][0])).detach().cpu().numpy()
        return sum_up

    def reset_hidden(self, batch_size=None):
        pass

    def save(self, name):
        self.actor.to("cpu")
        torch.save(
            {
                "actor": self.actor.state_dict(),
            },
            name
        )
        print(f"saved  model at {name}")
        self.actor.to(self.device)

    def load(self, name):
        print("load model from ", name)
        state_dicts = torch.load(name, map_location="cpu")
        self.actor.load_state_dict(state_dicts["actor"])
        self.actor.to(self.device)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class ActorCriticGRU(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        hidden_size,
        std=0.0,
        device="cpu",
        n_layers=2,
        drop_prob=0.2,
    ):
        super(ActorCriticGRU, self).__init__()

        self.hidden_size = hidden_size
        self.device = device
        self.critic = GRUNet(
            num_inputs, 1, hidden_size, device, n_layers, drop_prob
        ).to(device)
        self.actor = GRUNet(
            num_inputs, num_outputs, hidden_size, device, n_layers, drop_prob
        ).to(device)
        self.hidden_critic = self.critic.init_hidden(1)
        self.hidden_actor = self.actor.init_hidden(1)

        self.log_std = AddBias(torch.zeros(num_outputs)).to(device)

    def forward(self, x):
        x = x.unsqueeze(0)
        value, self.hidden_critic = self.critic(x, self.hidden_critic)
        mu, self.hidden_actor = self.actor(x, self.hidden_actor)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(mu.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        std = self.log_std(zeros).exp()

        dist = Normal(mu, std)
        return dist, value

    def sample(self, x, hidden_critic, hidden_actor):
        x = x.unsqueeze(0)
        value, hidden_critic = self.critic(x, hidden_critic)
        mu, hidden_actor = self.actor(x, hidden_actor)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(mu.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        std = self.log_std(zeros).exp()
        dist = Normal(mu, std)
        return dist, value, hidden_critic, hidden_actor

    def init_hidden(self, batch_size):
        self.hidden_critic = self.critic.init_hidden(batch_size)
        self.hidden_actor = self.actor.init_hidden(batch_size)
        return self.hidden_critic, self.hidden_actor

    def set_hidden(self, hidden_critic, hidden_actor):
        self.hidden_critic = hidden_critic
        self.hidden_actor = hidden_actor
        return self.hidden_critic, self.hidden_actor

    def get_hidden(self):
        return self.hidden_critic, self.hidden_actor

    def get_hidden_batch_first(self):
        hidden_critic = self.hidden_critic.view(
            -1, self.critic.n_layers, self.hidden_size
        )
        hidden_actor = self.hidden_actor.view(-1, self.actor.n_layers, self.hidden_size)
        return hidden_critic, hidden_actor

    def view_hidden(self, hidden_critic, hidden_actor):
        hidden_critic = hidden_critic.view(self.critic.n_layers, -1, self.hidden_size)
        hidden_actor = hidden_actor.view(self.actor.n_layers, -1, self.hidden_size)
        return hidden_critic, hidden_actor


class GRUNet(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        hidden_size,
        device="cpu",
        n_layers=2,
        drop_prob=0.2,
    ):
        super(GRUNet, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device
        self.gru = nn.GRU(num_inputs, self.hidden_size, n_layers, dropout=drop_prob).to(
            device
        )  # , batch_first=True
        self.fc = nn.Linear(hidden_size, num_outputs).to(device)
        self.relu = nn.ReLU().to(device)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = out[-1, :]
        out = self.fc(self.relu(out))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_size)
            .zero_()
            .to(self.device)
        )
        return hidden
