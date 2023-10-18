#  Copyright: (C) ETAS GmbH 2019. All rights reserved.
import torch
import numpy as np


class RolloutStorageGPRIL(object):
    def __init__(self, n_agents=0):
        self.trajectories = [Trajectory() for _ in range(n_agents)]
        self.finished_trejectories = []
        self.step = 0

    def append_experience(self, state, action, dones):
        for it, done in enumerate(dones):
            if done:
                if len(self.trajectories[it]) > 3:
                    self.finished_trejectories.append(self.trajectories[it].copy())
                self.trajectories[it].clear()
            else:
                state_pt = state[it, :].clone().detach()  # .requires_grad_(True)
                action_pt = action[it, :].clone().detach()  # .requires_grad_(True)
                state_action = torch.cat([state_pt, action_pt], 0)
                self.trajectories[it].add(state_action)

    def finish_trajectories(self):
        for trajectory in self.trajectories:
            if len(trajectory) > 3:  # 3:
                self.finished_trejectories.append(trajectory.copy())
            trajectory.clear()
        return self.finished_trejectories

    def clear(self):
        self.trajectories = []

    def reset(self, n_agents=0):
        self.trajectories = [Trajectory() for _ in range(n_agents)]

    def copy(self):
        finished_trajectory = self.finished_trejectories[:]
        return [finished_trajectory]


class Trajectory(object):
    def __init__(self):
        self.state_actions = []
        self.step = 0

    def __len__(self):
        return len(self.state_actions)

    def add(self, state_action):
        self.state_actions.append(state_action)

    def clear(self):
        self.state_actions = []

    def copy(self):
        if len(self.state_actions) > 0:
            traj = torch.stack(self.state_actions.copy())
        else:
            traj = None
        return traj
