import numpy as np
import torch
import torch.nn as nn
import math

from nflib.freia_c_flow import CINN
from torch.distributions import MultivariateNormal
from nflib.flows import NormalizingFlowModel
import torch.optim as optim
from rllib.model_sac import to_box, from_box
from rllib.sac import compress
from nflib.straight_throug_clamp import StraightThrougClamp

class ForwardBackwardCondLL(torch.nn.Module):
    def __init__(self, state_dim, action_dim,
                 n_flows=16, flow_state_hidden=16, hidden_size_cond=64, y_dim_features=2, exp_clamping=3.,
                 flow_state_hidden_ns=16, hidden_size_cond_ns=64, y_dim_features_ns=2, exp_clamping_ns=2.,
                 max_nll=4., alpha_pol=3., alpha_a=0.1, alpha_ns=1.0,
                 grad_clip_val=-1, act_limit=1.0, device="cpu", use_delta_state=False, use_straight_throug_clamp=False):
        super(ForwardBackwardCondLL, self).__init__()

        # Log P(s'|s) = log p(s'|a,s) + log p(a|s) – log p(a|s', s)
        # log p(a|s) is the policy (not model it here)
        self.max_nll = max_nll
        self.alpha_pol = alpha_pol
        self.alpha_a = alpha_a
        self.alpha_ns = alpha_ns
        self.device = device
        self.act_limit = act_limit
        self.use_straight_throug_clamp = use_straight_throug_clamp

        # n_cond = state_dim + action_dim
        if grad_clip_val > 0.:
            self.grad_clipping = True
        else:
            self.grad_clipping = False
        self.grad_clip_val = grad_clip_val
        lr = 2.5e-4
        self.use_delta_state = use_delta_state

        # p(s'|a,s)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_flows = n_flows
        self.flow_state_hidden_ns = flow_state_hidden_ns
        self.hidden_size_cond_ns = hidden_size_cond_ns
        self.exp_clamping_ns = exp_clamping_ns
        self.y_dim_features_ns = y_dim_features_ns
        self.lr = lr
        self.build_ns_given_a_s(device)

        # p(a|s',s)
        self.flow_state_hidden = flow_state_hidden
        self.hidden_size_cond = hidden_size_cond
        self.exp_clamping = exp_clamping
        self.y_dim_features = y_dim_features
        self.build_a_given_ns_s(device)

        ###################### legacy models #############################
        self.a_s_model = None 
        self.ns_s_model = None

        self.old_policy = None

    def build_ns_given_a_s(self, device):
        # p(s'|a,s)
        n_cond = self.state_dim + self.action_dim
        ns_a_s_prior = MultivariateNormal(torch.zeros(self.state_dim).to(device),
                                          torch.eye(self.state_dim).to(device))

        ns_a_s_flow = CINN(n_data=self.state_dim, n_cond=n_cond, n_blocks=self.n_flows,
                           internal_width=self.flow_state_hidden_ns, hidden_size_cond=self.hidden_size_cond_ns,
                           exponent_clamping=self.exp_clamping_ns, y_dim_features=self.y_dim_features_ns,
                           model_device=device, use1h=True).to(device)
                
        self.ns_a_s_model = NormalizingFlowModel(ns_a_s_prior, ns_a_s_flow,
                                                 device=device, is_freia=True).to(device)
        self.ns_a_s_model.train()
        
        self.ns_a_s_optimizer = optim.Adam(self.ns_a_s_model.parameters(), lr=self.lr, amsgrad=True, betas=(0.9, 0.998))
        print("number of params in ns_a_s model: ", sum(p.numel() for p in self.ns_a_s_model.parameters()))
    
    def build_a_given_ns_s(self, device):
        # p(a|s',s)
        n_cond = self.state_dim + self.state_dim

        a_ns_s_prior = MultivariateNormal(torch.zeros(self.action_dim).to(device),
                                          torch.eye(self.action_dim).to(device))

        a_ns_s_flow = CINN(n_data=self.action_dim, n_cond=n_cond, n_blocks=self.n_flows,
                           internal_width=self.flow_state_hidden, hidden_size_cond=self.hidden_size_cond,
                           exponent_clamping=self.exp_clamping, y_dim_features=self.y_dim_features,
                           model_device=device, use1h = True).to(device)

        self.a_ns_s_model = NormalizingFlowModel(a_ns_s_prior, a_ns_s_flow,
                                            device=device, is_freia=True).to(device)
        self.a_ns_s_model.train()

        self.a_ns_s_optimizer = optim.Adam(self.a_ns_s_model.parameters(), lr=self.lr, amsgrad=True, betas=(0.9, 0.998))
        print("number of params in a_ns_s model: ", sum(p.numel() for p in self.a_ns_s_model.parameters()))
    
    def redefine_ns_a_s_optimizer(self):
        self.ns_a_s_optimizer = optim.Adam(
            self.ns_a_s_model.parameters(), lr=self.lr, amsgrad=True, betas=(0.9, 0.998))

    def redefine_a_ns_s_optimizer(self):
        self.a_ns_s_optimizer = optim.Adam(
            self.a_ns_s_model.parameters(), lr=self.lr, amsgrad=True, betas=(0.9, 0.998))


    def get_all_model_parameter(self):
        trainable_parameters = [p for p in self.ns_a_s_model.parameters()]
        trainable_parameters.extend([p for p in self.a_ns_s_model.parameters()])
        return trainable_parameters

    def set_alpha(self, alpha_pol=1., alpha_a=1., alpha_ns=1.):
        self.alpha_pol = alpha_pol
        self.alpha_a = alpha_a
        self.alpha_ns = alpha_ns

    def estimate_mod_reward(self, next_state, state, action, expert_reward, action_pol_log_prob=None):
        if self.use_straight_throug_clamp:
            creward = StraightThrougClamp.apply(
                expert_reward, -self.max_nll*1.5, self.max_nll*1.5)
            if action_pol_log_prob is not None:
                creward = StraightThrougClamp.apply(
                    action_pol_log_prob, -self.max_nll, self.max_nll)
        else:
            creward = compress(expert_reward, self.max_nll)
        if action_pol_log_prob is None:
            action_pol_log_prob = creward * 0
        mod_reward = -self.estimate_state_cod_prob(state, action,
                                                             next_state, creward * 0, 1,
                                                   self.max_nll) + creward + action_pol_log_prob
        return mod_reward
    
    def clamp_expert(self, expert_reward):
        if self.use_straight_throug_clamp:
            creward = StraightThrougClamp.apply(
                expert_reward, -self.max_nll*1.5, self.max_nll*1.5)
        else:
            creward = compress(expert_reward, self.max_nll)
        return creward

    def estimate_state_cod_prob(self, state, action, next_state, pol_log_prob, alpha_fac_nonpol, max_nll=10.):
        # estmate Log P(s'|s) = log p(s'|a,s) + log p(a|s) – log p(a|s', s)

        # log p(s'|a,s)
        lp_ns_given_a_s = self.estimete_lp_ns_given_a_s(state, action, next_state)
        #lp_ns_given_a_s = torch.clamp(lp_ns_given_a_s, min=-self.max_nll, max=1e9)  # self.alpha_2*

        # log p(a|s', s)
        lp_a_given_ns_s = self.estimete_lp_a_given_ns_s(state, action, next_state)
        #lp_a_given_ns_s = torch.clamp(lp_a_given_ns_s, min=-self.max_nll, max=1e9)

        if torch.isnan(lp_ns_given_a_s).any():
            print("found NaN in lp_ns_given_a_s!")
        if torch.isnan(pol_log_prob).any():
            print("found NaN in pol_log_prob!")
        if torch.isnan(lp_a_given_ns_s).any():
            print("found NaN in lp_a_given_ns_s!")
        if torch.isinf(lp_ns_given_a_s).any():
            print("found inf in lp_ns_given_a_s!")
        if torch.isinf(pol_log_prob).any():
            print("found inf in pol_log_prob!")
        if torch.isinf(lp_a_given_ns_s).any():
            print("found inf in lp_a_given_ns_s!")

        if self.use_straight_throug_clamp:
            lp_ns_given_a_s = StraightThrougClamp.apply(lp_ns_given_a_s, -max_nll, max_nll)
            lp_a_given_ns_s = StraightThrougClamp.apply(lp_a_given_ns_s, -max_nll*1.5, max_nll*1.5)
        else:
            lp_ns_given_a_s = -compress(-lp_ns_given_a_s, max_nll)
            lp_a_given_ns_s = compress(lp_a_given_ns_s, max_nll)
        pol_log_prob = pol_log_prob

        # Log P(s'|s) = log p(s'|a,s) + log p(a|s) – log p(a|s', s) # [256] [256, 1]
        lp_ns_given_s = (alpha_fac_nonpol*self.alpha_ns * lp_ns_given_a_s.unsqueeze(1) + \
                        #0*self.alpha_pol * pol_log_prob \
                        -alpha_fac_nonpol*self.alpha_a * lp_a_given_ns_s.unsqueeze(1))
        lp_ns_given_s = lp_ns_given_s
        return lp_ns_given_s

    def test_both_model(self, state, action, next_state, max_nll=10.):
            # log p(s'|a,s)
        lp_ns_given_a_s = self.estimete_lp_ns_given_a_s(state, action, next_state)
        lp_a_given_ns_s = self.estimete_lp_a_given_ns_s(state, action, next_state)

        if self.use_straight_throug_clamp:
            lp_ns_given_a_s = StraightThrougClamp.apply(
                lp_ns_given_a_s, -max_nll, max_nll)
            lp_a_given_ns_s = StraightThrougClamp.apply(
                lp_a_given_ns_s, -max_nll*1.5, max_nll*1.5)
        else:
            lp_ns_given_a_s = -compress(-lp_ns_given_a_s, max_nll)
            lp_a_given_ns_s = compress(lp_a_given_ns_s, max_nll)

        ns_given_a_s_test_loss = -torch.mean(lp_ns_given_a_s)
        a_given_ns_s_test_loss = -torch.mean(lp_a_given_ns_s)
        return ns_given_a_s_test_loss, a_given_ns_s_test_loss

    def estimete_lp_ns_given_a_s(self, state, action, next_state):
        # log p(s'|a,s)
        state_actions = torch.cat((action, state), dim=1)
        if self.use_delta_state: 
            next_state = next_state - state 
        _, ns_given_a_s_prior_logprob, ns_given_a_s_log_det = self.ns_a_s_model(next_state.clone(), state_actions)
        lp_ns_given_a_s = ns_given_a_s_prior_logprob + ns_given_a_s_log_det
        return lp_ns_given_a_s

    def estimete_lp_ns_given_s(self, state, next_state):
        # log p(s'|s)

        _, ns_given_s_prior_logprob, ns_given_s_log_det = self.ns_s_model(next_state.clone(), state)
        lp_ns_given_s = ns_given_s_prior_logprob + ns_given_s_log_det
        return lp_ns_given_s


    def estimete_lp_a_given_ns_s(self, state, action, next_state):
        # log p(a|s', s)
        state_next_states = torch.cat((state, next_state - state), dim=1)

        action_t, ld = from_box(action, self.act_limit)

        _, a_given_ns_s_prior_logprob, a_given_ns_s_log_det = self.a_ns_s_model(action_t, state_next_states)
        lp_a_given_ns_s = a_given_ns_s_prior_logprob + a_given_ns_s_log_det + ld

        return lp_a_given_ns_s

    def estimete_lp_a_given_s(self, state, action):

        action_t, ld = from_box(action, self.act_limit)

        _, a_given_s_prior_logprob, a_given_s_log_det = self.a_s_model(action_t, state)
        lp_a_given_s = a_given_s_prior_logprob + a_given_s_log_det + ld

        return lp_a_given_s


    def optimize_step_lp_ns_given_a_s(self, state, action, next_state):
        # log p(s'|a,s)
        lp_ns_given_a_s = self.estimete_lp_ns_given_a_s(state, action, next_state)
        lp_ns_loss = -torch.mean(lp_ns_given_a_s)  # NLL
        self.ns_a_s_model.zero_grad()
        lp_ns_loss.backward()
        if self.grad_clipping:
            norm = torch.nn.utils.clip_grad_value_(self.ns_a_s_model.parameters(), self.grad_clip_val)
        self.ns_a_s_optimizer.step()
        return lp_ns_loss.item()

    def optimize_step_lp_ns_given_s(self, state, next_state):
        # log p(s'|a,s)
        lp_ns_given_s = self.estimete_lp_ns_given_s(state, next_state)
        lp_ns_loss = -torch.mean(lp_ns_given_s)  # NLL

        self.ns_s_model.zero_grad()
        lp_ns_loss.backward()
        if self.grad_clipping:
            norm = torch.nn.utils.clip_grad_value_(self.ns_s_model.parameters(), self.grad_clip_val)
        self.ns_s_optimizer.step()
        return lp_ns_loss.item()


    def optimize_step_lp_a_given_ns_s(self, state, action, next_state, w = None):
        # log p(a|s', s)
        if w is None:
            w = 1/state.shape[0]
        lp_a_given_ns_s = self.estimete_lp_a_given_ns_s(state, action, next_state)

        lp_a_loss = -torch.sum(w*lp_a_given_ns_s)  # NLL

        self.a_ns_s_model.zero_grad()
        lp_a_loss.backward()
        if self.grad_clipping:
            norm = torch.nn.utils.clip_grad_value_(self.a_ns_s_model.parameters(), self.grad_clip_val)
        self.a_ns_s_optimizer.step()
        return lp_a_loss.item()

    def optimize_step_lp_a_given_s(self, state, action):
        # log p(a|s', s)
        lp_a_given_s = self.estimete_lp_a_given_s(state, action,)
        lp_a_loss = -torch.mean(lp_a_given_s)  # NLL

        self.a_s_model.zero_grad()
        lp_a_loss.backward()
        if self.grad_clipping:
            norm = torch.nn.utils.clip_grad_value_(self.a_s_model.parameters(), self.grad_clip_val)
        self.a_s_optimizer.step()
        return lp_a_loss.item()

    def generate_states_based_on_actions(self, state, action):
        state_actions = torch.cat((action, state), dim=1)
        next_states, _, _ = self.ns_a_s_model.r_sample(state.shape[0], con_in=state_actions)
        if self.use_delta_state:
            next_state = next_states[-1] + state
        else:
            next_state = next_states[-1]
        return next_state

    def generate_actions_based_on_transition(self, state, next_state):
        state_next_states = torch.cat((state, next_state - state), dim=1)

        action_t, a_given_ns_s_prior_lp, a_given_ns_s_ld = self.a_ns_s_model.r_sample(state_next_states.shape[0],
                                                                                       con_in=state_next_states)
        action_t = action_t[-1]
        lp_a_given_ns_s = a_given_ns_s_prior_lp + a_given_ns_s_ld

        action, ld = to_box(action_t, self.act_limit)
        lp_a_given_ns_s += ld

        return action, lp_a_given_ns_s

    def save(self, name):
        self.a_ns_s_model.to("cpu")
        self.ns_a_s_model.to("cpu")
        torch.save(
            {
                "a_ns_s_model": self.a_ns_s_model.flow.cinn.state_dict(),
                "ns_a_s_model": self.ns_a_s_model.flow.cinn.state_dict(),
            },
            name,
        )
        print(f"saved  model at {name}")
        self.a_ns_s_model.to(self.device)
        self.ns_a_s_model.to(self.device)

    def load(self, name):
        print("loads model from ", name)
        state_dicts = torch.load(name, map_location="cpu")

        self.a_ns_s_model.flow.cinn.load_state_dict(state_dicts["a_ns_s_model"])
        self.ns_a_s_model.flow.cinn.load_state_dict(state_dicts["ns_a_s_model"])
        self.a_ns_s_model.to(self.device)
        self.ns_a_s_model.to(self.device)

    def get_state_dict(self):
        self.ns_a_s_model.to("cpu")
        self.a_ns_s_model.to("cpu")
        ns_a_s_state_dict = self.ns_a_s_model.flow.cinn.state_dict()
        a_ns_s_state_dict = self.a_ns_s_model.flow.cinn.state_dict()
        self.ns_a_s_model.to(self.device)
        self.a_ns_s_model.to(self.device)
        return ns_a_s_state_dict, a_ns_s_state_dict
    
    def load_ns_a_s_state_dict(self, ns_a_s_state_dict):
        self.build_ns_given_a_s(self.device)
        self.ns_a_s_model.to("cpu")
        self.ns_a_s_model.flow.cinn.load_state_dict(ns_a_s_state_dict)
        self.ns_a_s_model.to(self.device)
    
    def load_a_ns_s_state_dict(self, a_ns_s_state_dict):
        self.build_a_given_ns_s(self.device)
        self.a_ns_s_model.to("cpu")
        self.a_ns_s_model.flow.cinn.load_state_dict(a_ns_s_state_dict)
        self.a_ns_s_model.to(self.device)


