import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from torch import distributions

from torch.distributions import MultivariateNormal
from nflib.flows import NormalizingFlowModel
from nflib.freia_c_flow import CINN
from copy import deepcopy
from common.mlp_model import generate_snn
from nflib.straight_throug_clamp import StraightThrougClamp


class FlowDistribution(object):
    def __init__(self, action, log_prob, entropy, clip_actions=True):
        """
        Pseudo distribution to ease the implementation of flow policies alongside MLP policies
        """
        self._log_prob = log_prob
        self._entropy = entropy
        self._action = action
        self.clip_actions = clip_actions

    def sample(self):
        action = self.mode()
        return action.detach()

    def mode(self):
        # this is technically not correct!
        action = self._action
        if self.clip_actions:
            l = torch.tensor([[-7.0, -1.0]]).to(action.device)  # -6.63, -1.
            u = torch.tensor([[7.0, 1.0]]).to(action.device)  # 5.41, 0.41
            action = torch.max(torch.min(action, u), l)
        return action

    def rsample(self):
        return self._action

    def entropy(self):
        return self._entropy

    def log_prob(self, action):
        return self._log_prob


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

class FixedCategorical(torch.distributions.Categorical):
    '''from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail'''
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super().log_prob(actions.squeeze(-1))
            # .view(actions.size(0), -1)
            # .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_w = 5e-1
        self.linear = nn.Linear(num_inputs, num_outputs)
        self.linear.weight.data.uniform_(-init_w, init_w)
        self.linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)
        # return GumbelSoftmax(logits=x, temperature=torch.tensor(1.0))


class GumbelSoftmax(distributions.RelaxedOneHotCategorical):
    '''
    found in https://github.com/kengz/SLM-Lab for discrete SAC implementation
    A differentiable Categorical distribution using reparametrization trick with Gumbel-Softmax
    Explanation http://amid.fish/assets/gumbel.html
    NOTE: use this in place PyTorch's RelaxedOneHotCategorical distribution since its log_prob is not working right (returns positive values)
    Papers:
    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables (Maddison et al, 2017)
    [2] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017)
    '''

    def log_prob(self, value):
        return - torch.sum(- value * F.log_softmax(self.logits, -1), -1)


def to_box(a_t, scale):
    eps1 = (1-1e-4)
    #print("to_box a_t", a_t.abs().mean(0))
    a = torch.tanh(a_t)
    a = torch.clamp(a, min=-eps1, max=eps1)
    a_t = torch.atanh(a)

    ld = -(2 * (np.log(2.) - a_t - F.softplus(-2 * a_t))).sum(-1)

    a = scale * a
    ld -= math.log(scale) * a.shape[1]
    return a, ld

def from_box(a, scale):
    eps1 = (1 - 1e-4)
    #print("from_box a", a.abs().mean(0))
    a = a / scale
    a = torch.clamp(a, min=-eps1, max=eps1)
    a_t = torch.atanh(a)

    ld = -(2 * (np.log(2.) - a_t - F.softplus(-2 * a_t))).sum(-1)

    ld -= math.log(scale) * a.shape[1]
    return a_t, ld


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class CNNBase(nn.Module):
    def __init__(self, num_inputs, hidden_size=512):
        super(CNNBase, self).__init__()
        self._hidden_size = hidden_size

        def init_(m): return init(m, nn.init.orthogonal_, lambda x: nn.init.
                                  constant_(x, 0), nn.init.calculate_gain('relu'))

        # self.main = nn.Sequential(
        #     init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
        #     init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
        #     init_(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(), Flatten() , 
        #     init_(nn.Linear(64 * 2 * 2, hidden_size)), nn.ReLU())
        # self.main_1 = nn.Sequential(
        #     init_(nn.Conv2d(num_inputs, 32, 4, stride=2)), nn.ReLU())
        # self.main_2 = nn.Sequential(
        #     init_(nn.Conv2d(32, 32, 3, stride=2)), nn.ReLU())
        # self.main_3 = nn.Sequential(
        #     init_(nn.Conv2d(32, 32, 3, stride=2)), nn.ReLU())
        # self.main_4 = nn.Sequential(Flatten())
        # self.main_5 = nn.Sequential(
        #     init_(nn.Linear(32 * 5 * 5, hidden_size)), nn.ReLU())
        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, 3, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, 3, stride=2)), nn.ReLU(), Flatten() ,
            init_(nn.Linear(32 * 5 * 5, hidden_size)), nn.ReLU())

        self.train()
    
    @property
    def output_size(self):
        return self._hidden_size

    def forward(self, inputs):
        x = self.main(inputs.transpose(1, 3) / 255.0)
        # x = self.main_1(inputs.transpose(1, 3) / 255.0)
        # x = self.main_2(x)
        # x = self.main_3(x)
        # x = self.main_4(x)
        # x = self.main_5(x)
        return x

class QCNNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, bias=0):
        super(QCNNetwork, self).__init__()
        def init_(m): return init(m, nn.init.orthogonal_, lambda x: nn.init.
                                  constant_(x, 0), nn.init.calculate_gain('relu'))

        self.base = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(), Flatten())

        self.q = nn.Sequential(
            nn.Linear(64 * 2 * 2 + num_actions, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.bias = bias

    def forward(self, state, action):
        state_conv = self.base(state.transpose(1, 3) / 255.0)
        x = torch.cat([state_conv, action], 1)
        x = self.q(x)+self.bias
        return x

class MLPSAC(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, act_limit=10., device="cpu",
                 use_snn=False, is_discrete=False, hidden_size_policy=None):
        super(MLPSAC, self).__init__()
        self.device = device
        self.is_flow = False
        if is_discrete is False:
            self.has_deterministic = False
        else:
            self.has_deterministic = False

        self.epsilon = 1e-6
        self.act_limit = act_limit

        self.log_std_min = -10
        self.log_std_max = np.log(0.5)
        init_w = 3e-3

        if hidden_size_policy is None:
            hidden_size_policy = hidden_size
        if use_snn:
            self.actor_base = generate_snn(
                num_inputs, num_outputs, hidden_size_policy, identity_start=False).to(device)
        else:
            if isinstance(num_inputs, int): 
                self.actor_base = nn.Sequential(
                    nn.Linear(num_inputs, hidden_size_policy),
                    nn.ReLU(),
                    nn.Linear(hidden_size_policy, hidden_size_policy),
                    nn.ReLU(),
                    nn.Linear(hidden_size_policy, hidden_size_policy),
                    nn.ReLU(),
                ).to(device)
            elif len(num_inputs) == 3:
                self.actor_base = CNNBase(num_inputs[-1], hidden_size=hidden_size_policy)
            else:
                raise NotImplementedError

            # self.actor_base = generate_mlp(num_inputs, num_outputs, hidden_size, identity_start=False).to(device)

        self.is_discrete = is_discrete
        self.dist = Categorical(hidden_size_policy, num_outputs)
        self.clip_actions = False

        if self.is_discrete:
            num_actions = 1
            self.action_shape = (num_outputs,)
        else:
            num_actions = num_outputs
            self.action_shape = (num_outputs,)

        self.mean_linear = nn.Linear(hidden_size_policy, num_outputs)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.zero_()

        self.log_std_linear = nn.Linear(hidden_size_policy, num_outputs)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.zero_()

        if isinstance(num_inputs, int):
            self.soft_q_net_1 = QNetwork(num_inputs, num_actions, hidden_size, bias = -10).to(device)
            self.soft_q_net_2 = QNetwork(num_inputs, num_actions, hidden_size, bias = -10).to(device)
        elif len(num_inputs) == 3:
            self.soft_q_net_1 = QCNNetwork(num_inputs[-1], num_actions, hidden_size, bias = -10).to(device)
            self.soft_q_net_2 = QCNNetwork(
                num_inputs[-1], num_actions, hidden_size, bias=-10).to(device)
        else:
            raise NotImplementedError
            

        self.target_q_net_1 = deepcopy(self.soft_q_net_1)
        self.target_q_net_2 = deepcopy(self.soft_q_net_2)

        self.test_mode = False
        self.train()

        self.ndist = Normal(0,1)

    def _log_prob(self, x, mean, logstd):
        return self.ndist.log_prob((x-mean)/logstd.exp()).sum(-1) - logstd.sum(-1)

    def _sample(self, mean, logstd):
        x = torch.randn(mean.shape, requires_grad = True).to(mean.device)*logstd.exp()+mean
        return x

    def get_policy_parameter(self):
        trainable_parameters = [p for p in self.actor_base.parameters()]
        trainable_parameters.extend([p for p in self.mean_linear.parameters()])
        trainable_parameters.extend([p for p in self.log_std_linear.parameters()])
        trainable_parameters.extend([p for p in self.dist.parameters()])
        return trainable_parameters

    def policy_forward(self, state):
        base = self.actor_base(state)
        mean = self.mean_linear(base)
        log_std = self.log_std_linear(base)
        log_std = StraightThrougClamp.apply(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_value(self, state):
        value = torch.zeros((state.shape[0], 1))
        return value

    def forward_action(self, state, action):
        value = self.get_value(state)
        if self.is_discrete:
            raise Exception("not implemented")
        else:
            mean, log_std = self.policy_forward(state)
            #normal = Normal(mean, log_std.exp())
            #action_t = normal.rsample()

            action_t, ld = from_box(action, self.act_limit)

            log_prob = self._log_prob(action_t, mean, log_std)
            log_prob += ld

            log_prob = log_prob.unsqueeze(-1)
        return log_prob

    def forward(self, state, add_noise = False, deterministic=False):
        value = self.get_value(state)
        if self.is_discrete:
            if deterministic:
                raise Exception("not implemented")
            base = self.actor_base(state)
            dist = self.dist(base)
        else:
            if deterministic:
                mean, log_std = self.policy_forward(state)
                action_t = mean
                log_prob = self._log_prob(action_t, mean, log_std)
                log_prob = log_prob.unsqueeze(-1)
                entropy = -log_prob
                action, ld = to_box(action_t, self.act_limit)
                dist = FlowDistribution(action, log_prob, entropy, clip_actions=self.clip_actions)
            else:
                mean, log_std = self.policy_forward(state)
                #normal = Normal(mean, log_std.exp())
                #action_t = normal.rsample()
                action_t = self._sample(mean, log_std)
                log_prob = self._log_prob(action_t, mean, log_std)

                #if add_noise:
                #    action_t += torch.randn(action_t.shape).to(action_t.device)*0.1

                action, ld = to_box(action_t, self.act_limit)
                log_prob += ld

                #entropy = 0.5 + 0.5 * math.log(2 * math.pi) + log_std

                log_prob = log_prob.unsqueeze(-1)

                entropy = -log_prob

                dist = FlowDistribution(action, log_prob, entropy, clip_actions=self.clip_actions)
        return dist, value

    def forward_sampling(self, state, n_samples=50):
        repeated_state = state.repeat(n_samples, 1)

        mean, log_std = self.policy_forward(repeated_state)
        #normal = Normal(mean, log_std.exp())
        #action_t = normal.rsample()
        #log_prob = normal.log_prob(action_t).view(action_t.size(0), -1).sum(-1)

        action_t = self._sample(mean, log_std)
        log_prob = self._log_prob(action_t, mean, log_std)


        action, ld = to_box(action_t, self.act_limit)
        log_prob += ld

        log_prob = log_prob.unsqueeze(-1)

        return action, log_prob

    def evaluate(self, state, add_noise = False):
        if self.is_discrete:
            base = self.actor_base(state)
            dist = self.dist(base)
            action = dist.sample()
            log_prob = dist.log_prob(action.squeeze(1)).unsqueeze(1)
            action_t = action  # dummy
            mean = action  # dummy
            log_std = action  # dummy
        else:
            mean, log_std = self.policy_forward(state)
            #normal = Normal(mean, std)
            #action_t = normal.rsample()
            #log_prob = normal.log_prob(action_t).view(action_t.size(0), -1).sum(-1)
            action_t = self._sample(mean, log_std)
            log_prob = self._log_prob(action_t, mean, log_std)

            #if add_noise:
            #    action_t += torch.randn(action_t.shape).to(action_t.device)*0.1

            action, ld = to_box(action_t, self.act_limit)
            log_prob += ld
            log_prob = log_prob.unsqueeze(-1)
        return action, log_prob, action_t, mean, log_std

    def get_action(self, state):
        if self.is_discrete:
            base = self.actor_base(state)
            dist = self.dist(base)
            action = dist.sample().squeeze(1)
        else:
            action, _ = self.forward_sampling(state, 1)
        return action

    def reset_hidden(self, batch_size=None):
        pass

    def save(self, name):
        self.actor_base.to("cpu")
        self.log_std_linear.to("cpu")
        self.mean_linear.to("cpu")
        self.soft_q_net_1.to("cpu")
        self.soft_q_net_2.to("cpu")
        self.dist.to("cpu")
        torch.save(
            {
                "actor_base": self.actor_base.state_dict(),
                "log_std_linear": self.log_std_linear.state_dict(),
                "mean_linear": self.mean_linear.state_dict(),
                "soft_q_net_1": self.soft_q_net_1.state_dict(),
                "soft_q_net_2": self.soft_q_net_2.state_dict(),
                "dist": self.dist.state_dict(),
            },
            name
        )
        print(f"saved  model at {name}")
        self.actor_base.to(self.device)
        self.log_std_linear.to(self.device)
        self.mean_linear.to(self.device)
        self.soft_q_net_1.to(self.device)
        self.soft_q_net_2.to(self.device)
        self.dist.to(self.device)

    def load(self, name):
        print("load model from ", name)
        state_dicts = torch.load(name, map_location="cpu")
        self.actor_base.load_state_dict(state_dicts["actor_base"])
        self.log_std_linear.load_state_dict(state_dicts["log_std_linear"])
        self.mean_linear.load_state_dict(state_dicts["mean_linear"])
        self.soft_q_net_1.load_state_dict(state_dicts["soft_q_net_1"])
        self.soft_q_net_2.load_state_dict(state_dicts["soft_q_net_2"])
        self.dist.load_state_dict(state_dicts["dist"])
        self.target_q_net_1 = deepcopy(self.soft_q_net_1)
        self.target_q_net_2 = deepcopy(self.soft_q_net_2)
        self.actor_base.to(self.device)
        self.soft_q_net_1.to(self.device)
        self.soft_q_net_2.to(self.device)
        self.target_q_net_1.to(self.device)
        self.target_q_net_2.to(self.device)
        self.dist.to(self.device)


class StackedFlows(nn.Module):
    def __init__(self, n_data=2, n_cond=2, n_blocks=10, internal_width=64, hidden_size_cond=64,
                 exponent_clamping=2., y_dim_features=64, use_snn=False, identity_init=False,
                 device="cpu"):
        super(StackedFlows, self).__init__()
        self.device = device

        self.action_prior = MultivariateNormal(torch.zeros(n_data).to(device),
                                               torch.eye(n_data).to(device))

        self.action_flows_1 = CINN(n_data=n_data, n_cond=n_cond, n_blocks=n_blocks,
                                internal_width=internal_width, hidden_size_cond=hidden_size_cond,
                                exponent_clamping=exponent_clamping, y_dim_features=y_dim_features,
                                use_snn=use_snn, identity_init=True, model_device=device)

        self.action_flows_2 = CINN(n_data=n_data, n_cond=n_cond, n_blocks=n_blocks,
                                internal_width=internal_width, hidden_size_cond=hidden_size_cond,
                                exponent_clamping=exponent_clamping, y_dim_features=y_dim_features,
                                use_snn=use_snn, identity_init=True, model_device=device)

        self.active_second_block = False
        self.active_first_block = True
        for param in self.action_flows_2.parameters():
            param.requires_grad = False

    def activate_second_block(self):
        self.active_second_block = True
        for param in self.action_flows_2.parameters():
            param.requires_grad = True

    def freeze_first_block(self):
        self.active_first_block = False
        for param in self.action_flows_1.parameters():
            param.requires_grad = False

    def get_flow_parameter(self):
        trainable_parameters = [p for p in self.action_flows_1.parameters() if p.requires_grad]
        if self.active_second_block:
            trainable_parameters.append([p for p in self.action_flows_2.parameters() if p.requires_grad])
        return trainable_parameters

    def forward_sample(self, input, condition=None):
        if self.active_second_block:
            input_z, jac_2 = self.action_flows_2.forward_sample(input, condition)
        else:
            jac_2 = 0
            input_z = input
        z, jac_1 = self.action_flows_1.forward_sample(input_z, condition)
        jac = jac_1 + jac_2
        base_logprob = self.action_prior.log_prob(z[-1]).view(input.size(0), -1).sum(1)
        return z, base_logprob, jac

    def backward_sample(self, input, condition=None):
        z, jac = self.action_flows_1.backward_sample(input, condition)
        if self.active_second_block:
            z, jac_2 = self.action_flows_2.backward_sample(z[-1], condition)
            jac += jac_2
        return z, jac


class FlowSAC(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, n_flows=16, flow_hidden=16,
                 exp_clamping=5, hidden_size_cond=8, con_dim_features=64, act_limit=10.,
                 use_mc_entropy=False, use_value=False, use_tanh=True, use_snn=False,
                 use_stacked_blocks=False, device="cpu"):
        super(FlowSAC, self).__init__()
        self.device = device
        self.is_flow = True
        self.has_deterministic = False

        self.epsilon = 1e-6
        self.clip_actions = False
        self.use_tanh = use_tanh
        self.use_mc_entropy = use_mc_entropy
        self.act_limit = act_limit
        self.action_shape = (num_outputs,)
        self.use_stacked_blocks = use_stacked_blocks
        if use_stacked_blocks:
            self.actor_flow = StackedFlows(n_data=num_outputs, n_cond=num_inputs, n_blocks=n_flows,
                                internal_width=flow_hidden, hidden_size_cond=hidden_size_cond,
                                exponent_clamping=exp_clamping, y_dim_features=con_dim_features,
                                use_snn=use_snn, device=device)
            self.action_prior = self.actor_flow.action_prior
        else:
            self.action_prior = MultivariateNormal(torch.zeros(num_outputs).to(device),
                                              torch.eye(num_outputs).to(device))

            action_flows = CINN(n_data=num_outputs, n_cond=num_inputs, n_blocks=n_flows,
                                internal_width=flow_hidden, hidden_size_cond=hidden_size_cond,
                                exponent_clamping=exp_clamping, y_dim_features=con_dim_features,
                                use_snn=use_snn, model_device=device)

            self.actor_flow = NormalizingFlowModel(self.action_prior, action_flows, device=device,
                                                   prep_con=False, is_freia=True).to(device)
        self.use_value = use_value
        if use_value:
            self.value_net = nn.Sequential(
                nn.Linear(num_inputs, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, 1)
            ).to(device)

            self.soft_q_net_1 = None
            self.soft_q_net_2 = None

            self.target_q_net_1 = None
            self.target_q_net_2 = None
        else:
            self.soft_q_net_1 = QNetwork(num_inputs, num_outputs, hidden_size).to(device)
            self.soft_q_net_2 = QNetwork(num_inputs, num_outputs, hidden_size).to(device)

            self.target_q_net_1 = deepcopy(self.soft_q_net_1)
            self.target_q_net_2 = deepcopy(self.soft_q_net_2)

            self.value_net = None

        self.test_mode = False
        self.train()

    def activate_second_and_freeze_first(self):
        if self.use_stacked_blocks:
            self.actor_flow.activate_second_block()
            self.actor_flow.freeze_first_block()
        return self.get_policy_parameter()

    def get_policy_parameter(self):
        trainable_parameters = [p for p in self.actor_flow.parameters() if p.requires_grad]
        return trainable_parameters

    def get_value_parameter(self):
        if self.value_net is not None:
            trainable_parameters = [p for p in self.value_net.parameters() if p.requires_grad]
        else:
            trainable_parameters = None
        return trainable_parameters

    def get_value(self, state):
        if self.value_net is None:
            value = torch.zeros((state.shape[0], 1))
        else:
            value = self.value_net(state)
        return value

    def forward(self, state, deterministic=False):
        if deterministic:
            self.test_mode = True
        value = self.get_value(state)
        if self.test_mode:
            dist = self.forward_test(state)
        else:
            dist = self.forward_train(state)
        if deterministic:
            self.test_mode = False
        return dist, value

    def output_layer(self, action_t):
        if self.use_tanh:
            action, log_det = to_box(action_t, self.act_limit)
        else:
            action = action_t
            log_det = 0.
        return action, log_det

    def forward_train(self, state):
        normal = self.action_prior
        x_t = normal.rsample((state.shape[0],)).detach()
        action_, action_log_det = self.actor_flow.backward_sample(x_t, state)
        action_t = action_[-1]

        log_prob = normal.log_prob(x_t).view(action_t.size(0), -1).sum(-1)
        log_prob += action_log_det

        action, add_lp = self.output_layer(action_t)
        log_prob += add_lp
        if torch.isnan(action).any():
            print("found NaN in action!")
        if torch.isnan(log_prob).any():
            print("found NaN in log_prob!")

        if self.use_mc_entropy:
            entropy = self.estimate_entropy(state)
        else:
            entropy = torch.mean(-log_prob, dim=0)  # dummy entropy (should not be used)

        dist = FlowDistribution(action, log_prob.unsqueeze(-1), entropy, clip_actions=self.clip_actions)
        return dist

    def forward_test(self, state):
        action, log_prob = self.forward_sampling(state)

        action, add_lp = self.output_layer(action)
        log_prob += add_lp

        max_action_loc = torch.argmax(log_prob)
        max_action = action[max_action_loc.item()].unsqueeze(0)
        max_log_prob = log_prob[max_action_loc.item()].unsqueeze(0)

        dist = FlowDistribution(max_action, max_log_prob.unsqueeze(-1), 0., clip_actions=self.clip_actions)
        return dist

    def forward_sampling(self, state, n_samples=50):
        repeated_state = state.repeat(n_samples, 1)
        x_t = self.action_prior.sample((repeated_state.shape[0],))

        action_, log_det_tmp = self.actor_flow.backward_sample(x_t, repeated_state)
        action_t = action_[-1]

        log_prob = self.action_prior.log_prob(x_t).view(action_t.size(0), -1).sum(-1)
        log_prob += log_det_tmp

        return action_t, log_prob

    def evaluate(self, state, add_noise=False):
        mean = torch.ones((state.shape[0], 1))
        log_std = torch.zeros((state.shape[0], 1))
        normal = self.action_prior
        x_t = normal.sample((state.shape[0],)).detach()

        action_, action_log_det = self.actor_flow.backward_sample(x_t, state)
        action_t = action_[-1]

        log_prob = normal.log_prob(x_t).view(action_t.size(0), -1).sum(-1).clone().detach()
        log_prob += action_log_det

        action, add_lp = self.output_layer(action_t)
        log_prob += add_lp

        if torch.isnan(action).any():
            print("found NaN in action!")

        return action, log_prob.unsqueeze(-1), x_t, mean, log_std

    def eval_action(self, action, state):
        if self.use_tanh:
            action_t, ld = from_box(action, self.act_limit)
        else:
            action_t = action
        normal = self.action_prior
        zs, _, action_log_det = self.actor_flow.forward_sample(action_t.clone().detach(), state)
        log_prob = normal.log_prob(zs[-1]).view(action.size(0), -1).sum(-1)
        if self.use_tanh:
            log_prob += ld
        log_prob += action_log_det
        return zs, log_prob

    def estimate_entropy(self, state):
        normal = self.action_prior
        repeated_state = state.repeat(20, 1)
        base_dist_sample = normal.sample((repeated_state.shape[0],))
        action_, action_log_det = self.actor_flow.backward_sample(base_dist_sample, repeated_state)
        action_t = action_[-1]
        # action_log_det = action_log_det
        if self.use_tanh:
            action_log_det -= (2 * (np.log(2.) - action_t - F.softplus(-2 * action_t))).sum(-1)

        # base_log_prob = normal.log_prob(base_dist_sample).view(repeated_state.shape[0], -1).sum(-1).clone().detach()
        # entropy = (-(base_log_prob + action_log_det.detach())).mean()
        normal_entropy = normal.entropy().mean()
        entropy = normal_entropy - action_log_det.detach().mean()
        return entropy

    def log_std(self, input):
        '''This is a dummy method'''
        return torch.tensor(([[1., 1.]]))

    def reset_hidden(self, batch_size=None):
        pass

    def get_policy_state_dict(self):
        if self.use_stacked_blocks:
            if self.use_value:
                dict = {
                    "actor_flow_1": self.actor_flow.action_flows_1.cinn.state_dict(),
                    "actor_flow_2": self.actor_flow.action_flows_1.cinn.state_dict(),
                    "actor_flow_cond_net": self.actor_flow.flow.cond_net.state_dict(),
                    "value_net": self.value_net.state_dict(),
                }
            else:
                dict = {
                    "actor_flow_1": self.actor_flow.action_flows_1.cinn.state_dict(),
                    "actor_flow_2": self.actor_flow.action_flows_2.cinn.state_dict(),
                    "actor_flow_cond_net_1": self.actor_flow.action_flows_1.cond_net.state_dict(),
                    "actor_flow_cond_net_2": self.actor_flow.action_flows_2.cond_net.state_dict(),
                    "soft_q_net_1": self.soft_q_net_1.state_dict(),
                    "soft_q_net_2": self.soft_q_net_2.state_dict(),
                }
        else:
            if self.use_value:
                dict = {
                    "actor_flow": self.actor_flow.flow.cinn.state_dict(),
                    "actor_flow_cond_net": self.actor_flow.flow.cond_net.state_dict(),
                    "value_net": self.value_net.state_dict(),
                }
            else:
                dict = {
                    "actor_flow": self.actor_flow.flow.cinn.state_dict(),
                    "actor_flow_cond_net": self.actor_flow.flow.cond_net.state_dict(),
                    "soft_q_net_1": self.soft_q_net_1.state_dict(),
                    "soft_q_net_2": self.soft_q_net_2.state_dict(),
                }
        return dict

    def save(self, name):
        self.actor_flow.to("cpu")
        if self.use_value:
            self.value_net.to("cpu")
        else:
            self.soft_q_net_1.to("cpu")
            self.soft_q_net_2.to("cpu")

        torch.save(
            self.get_policy_state_dict(),
            name,
        )
        print(f"saved  model at {name}")
        self.actor_flow.to(self.device)
        if self.use_value:
            self.value_net.to(self.device)
        else:
            self.soft_q_net_1.to(self.device)
            self.soft_q_net_2.to(self.device)

    def load(self, name):
        print("loads model from ", name)
        state_dicts = torch.load(name, map_location="cpu")
        if self.use_stacked_blocks:
            self.actor_flow.action_flows_1.cinn.load_state_dict(state_dicts["actor_flow_1"])
            self.actor_flow.action_flows_2.cinn.load_state_dict(state_dicts["actor_flow_2"])
            self.actor_flow.action_flows_1.cond_net.load_state_dict(state_dicts["actor_flow_cond_net_1"])
            self.actor_flow.action_flows_2.cond_net.load_state_dict(state_dicts["actor_flow_cond_net_2"])
        else:
            self.actor_flow.flow.cinn.load_state_dict(state_dicts["actor_flow"])
            self.actor_flow.flow.cond_net.load_state_dict(state_dicts["actor_flow_cond_net"])
        self.actor_flow.to(self.device)

        if self.use_value:
            self.value_net.load_state_dict(state_dicts["value_net"])
            self.value_net.to(self.device)
        else:
            self.soft_q_net_1.load_state_dict(state_dicts["soft_q_net_1"])
            self.soft_q_net_2.load_state_dict(state_dicts["soft_q_net_2"])
            self.target_q_net_1 = deepcopy(self.soft_q_net_1)
            self.target_q_net_2 = deepcopy(self.soft_q_net_2)
            self.soft_q_net_1.to(self.device)
            self.soft_q_net_2.to(self.device)
            self.target_q_net_1.to(self.device)
            self.target_q_net_2.to(self.device)
