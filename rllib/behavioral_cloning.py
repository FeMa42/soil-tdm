#  Copyright: (C) ETAS GmbH 2019. All rights reserved.
import torch
import torch.nn as nn
import torch.optim as optim


class BehavioralCloning():
    def __init__(self,
                 policy,
                 lr=None,
                 weight_decay=1e-4, discrete_actions=False,
                 flow_model=False, grad_clip_val=10.):
        self.steps = 0
        self.policy = policy
        self.policy.train()
        self.discrete_actions = discrete_actions
        if discrete_actions:
            self.opt_criterion = nn.CrossEntropyLoss()  # NLLLoss, CrossEntropyLoss
        else:
            # self.opt_criterion = nn.L1Loss(), nn.MSELoss(), nn.SmoothL1Loss()
            self.opt_criterion = nn.SmoothL1Loss()
        self.flow_model = flow_model
        self.clip_grad = False
        self.clip_value = grad_clip_val

        # beta1=0.5 beta2=0.999 ?
        parameter = policy.get_policy_parameter()
        self.optimizer = optim.Adam(parameter, lr=lr, weight_decay=weight_decay)

    def estimate_loss(self, state, expert_action, clamp_value=None):
        dist, _ = self.policy(state)
        policy_actions = dist.rsample()  # differentiable action with reparametrization trick

        if self.discrete_actions:
            # targets = expert_action.long().squeeze(1)  # .max(dim=1)  # .long().squeeze(1)
            _, targets = expert_action.max(dim=1)  # .long().squeeze(1)
        else:
            targets = expert_action
        if clamp_value is not None:
            targets = torch.clamp(targets, -clamp_value, clamp_value)
        loss = self.opt_criterion(policy_actions, targets)
        return loss

    def estimate_nll(self, state, expert_action):
        _, action_logprob = self.policy.eval_action(expert_action, state)
        loss = -torch.mean(action_logprob)
        return loss

    def update(self, state, expert_action):
        self.steps += 1

        if self.flow_model:
            loss = self.estimate_nll(state, expert_action)
        else:
            loss = self.estimate_loss(state, expert_action)

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_value)
        self.optimizer.step()

        return loss.item()

    def gpril_update(self, state, expert_action, prev_state, prev_action, logger=None, alpha=0.5, beta=0.5):
        self.steps += 1

        bc_loss = self.estimate_loss(state, expert_action)

        gpril_loss = self.estimate_loss(prev_state, prev_action)

        loss = alpha*bc_loss+beta*gpril_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if logger is not None:
            logger.log_additional_hist("BC_actions_acc_policy", policy_actions.detach().cpu().numpy()[:, 0], self.steps)
            logger.log_additional_hist("BC_actions_lc_policy", policy_actions.detach().cpu().numpy()[:, 1], self.steps)
            logger.log_additional_hist("BC_actions_acc_expert", expert_action.detach().cpu().numpy()[:, 0], self.steps)
            logger.log_additional_hist("BC_actions_lc_expert", expert_action.detach().cpu().numpy()[:, 1], self.steps)
        return loss.item()

    def estimate_error(self, state, expert_action):
        with torch.no_grad():
            if self.flow_model:
                loss = self.estimate_nll(state, expert_action)
            else:
                loss = self.estimate_loss(state, expert_action)
        return loss.item()
