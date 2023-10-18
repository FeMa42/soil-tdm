import torch.nn as nn


class ExpSurvivalEst(nn.Module):
    def __init__(self, time_steps_expert=100, penalty=-500):
        super().__init__()
        self.time_steps_expert = time_steps_expert
        if penalty > 0:
            penalty = -penalty
        self.penalty = penalty

    def forward(self, time_steps, dones):
        early_termination = (time_steps < self.time_steps_expert)
        rewards = dones*early_termination*self.penalty/(time_steps+1)
        return rewards