#  Copyright: (C) ETAS GmbH 2019. All rights reserved.
import torch
import numpy as np
import random
from collections import deque
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from common.normalize import NormalizedActions


def gen_std_scaler(traj, scaler=0.1):
    std_values = np.std(np.reshape(traj, (-1, traj.shape[-1])), axis=0)
    std_scaler = std_values * scaler
    # define lane index separately
    std_scaler[1] = 0.1
    std_scaler[14] = 0.1
    std_scaler[21] = 0.1
    std_scaler[27] = 0.1
    std_scaler[33] = 0.1
    std_scaler[39] = 0.1
    std_scaler[45] = 0.1
    return std_scaler


def gen_correction(n_feature=49, batch_size=1):
    if batch_size > 1:
        shape = (batch_size, n_feature)
    else:
        shape = n_feature
    bias_corrections = np.zeros(shape)
    corrections = np.ones(shape)
    corrections[:, 4] = 50
    bias_corrections[:, 4] = -0.5
    corrections[:, 7] = 150
    corrections[:, 8] = 300
    return bias_corrections, corrections


class SimpleBatchGenerator(object):
    def __init__(self, data_x, data_y, batch_size=64, is_state_conditional=False, use_action_norm=False,
                 action_scale=7., noise_value=0.002):
        self.data_x = np.reshape(data_x, (-1, data_x.shape[-1]))
        self.data_y = np.reshape(data_y, (-1, data_y.shape[-1]))
        self.std_values = None
        self.mean_values = None
        self.is_state_conditional = is_state_conditional

        if use_action_norm:
            self.action_scaler = NormalizedActions(low=-action_scale, high=action_scale, use_numpy=True)
        else:
            self.action_scaler = None

        self.batch_size = batch_size
        self.noise_value = noise_value
        self.n_feature = self.data_x.shape[1]
        self.n_action = self.data_y.shape[1]

    def sample(self, batch_size=None, noise_value_y=0.002, noise_value_x=0.002):
        if batch_size is None:
            batch_size = self.batch_size
        indices = np.array(random.sample(range(0, len(self.data_x)-1), batch_size))
        indices = np.sort(indices)
        batch_x = self.data_x[indices, :]
        batch_y = self.data_y[indices, :]
        batch_y = self.transform_action(batch_y)
        batch_y += np.random.randn(*batch_y.shape) * noise_value_y
        batch_x += np.random.randn(*batch_x.shape) * noise_value_x
        keys = None
        return batch_x, batch_y, keys

    def transform_action(self, action):
        if self.action_scaler is not None:
            return self.action_scaler.normalize(action)
        else:
            return action

    def inv_transform_action(self, action):
        if self.action_scaler is not None:
            return self.action_scaler.reverse_normalize(action)
        else:
            return action

    def std_values(self):
        return np.std(self.data_x, 0)

    def mean_values(self):
        return np.mean(self.data_x, 0)


class GPRILBatchGenerator(object):
    def __init__(self, trajectories=None, batch_size=64, n_feature=49, n_action=2, capacity=100000, gamma=0.9,
                 add_corrections=False):
        self.capacity = capacity
        self.position = 0
        if trajectories is not None:
            if len(trajectories) > capacity:
                trajectories = trajectories[:capacity]
            self.traj = trajectories
        else:
            self.traj = []

        if add_corrections:
            self.bias_corrections, self.corrections = gen_correction(n_feature)
        else:
            self.corrections = None

        self.position = len(self.traj)
        self.batch_size = batch_size
        self.n_feature = n_feature
        self.n_action = n_action
        self.gamma = gamma

    def push(self, state_actions):
        for it in range(len(state_actions)):
            if len(self.traj) < self.capacity:
                self.traj.append(None)
            self.traj[self.position] = (state_actions[it])
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        start_states = []
        future_states = []
        actions = []
        for n in range(batch_size):
            indice = random.sample(range(0, len(self.traj)), 1)[0]
            single_traj = self.traj[indice]
            time_indices = self.sample_time_id(single_traj.shape[0] - 1)
            states = single_traj[time_indices, :self.n_feature]
            action = single_traj[time_indices[0], self.n_feature:]
            state = states[0, :]
            future_state = states[1, :]
            if self.corrections is not None:
                state = (state + self.bias_corrections)*self.corrections
                future_state = (future_state + self.bias_corrections) * self.corrections
            start_states.append(state)
            future_states.append(future_state)
            actions.append(action)
        start_states = torch.stack(start_states)
        future_states = torch.stack(future_states)
        actions = torch.stack(actions)
        return start_states, actions, future_states

    def sample_time_id(self, max):
        start_index = random.sample(range(0, max-1), 1)[0]
        future_index = np.random.geometric(p=self.gamma, size=1)[0]+start_index
        if future_index > max:
            future_index = max
        return np.array([start_index, future_index])

    def __len__(self):
        return len(self.traj)


class NFBatchGenerator(object):
    def __init__(self, trajectories, batch_size=64, n_feature=49, gamma=0.9,
                 add_data_point=False, add_noise=False, std_scaler=0.1, add_corrections=False):
        self.batch_size = batch_size
        self.n_feature = n_feature
        self.gamma = gamma
        self.traj = trajectories
        self.std_values = np.std(np.reshape(trajectories, (-1, trajectories.shape[-1])), 0)
        self.mean_values = np.mean(np.reshape(trajectories, (-1, trajectories.shape[-1])), 0)
        if add_data_point:
            zeros = np.zeros((self.traj.shape[0], self.traj.shape[1], 1))
            self.traj = np.concatenate((self.traj, zeros), axis=2)
            self.n_feature += 1

        if add_noise:
            self.std_scaler = gen_std_scaler(self.traj, scaler=std_scaler)
        else:
            self.std_scaler = None

        if add_corrections:
            self.bias_corrections, self.corrections = gen_correction(self.traj.shape[-1])
        else:
            self.corrections = None

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        start_states = []
        future_states = []
        for n in range(batch_size):
            indice = random.sample(range(0, self.traj.shape[0]), 1)[0]
            single_traj = self.traj[indice, :, :]
            time_indices = self.sample_time_id(single_traj.shape[0] - 1)
            states = single_traj[time_indices, :self.n_feature]
            state = states[0, :]
            future_state = states[1, :]
            if self.corrections is not None:
                state = (state + self.bias_corrections)*self.corrections
                future_state = (future_state + self.bias_corrections) * self.corrections
            if self.std_scaler is not None:
                state = state + np.random.normal(scale=self.std_scaler)
                future_state = future_state + np.random.normal(scale=self.std_scaler)
            start_states.append(state)
            future_states.append(future_state)
        start_states = np.stack(start_states)
        future_states = np.stack(future_states)
        return start_states, future_states

    def sample_time_id(self, max):
        start_index = random.sample(range(0, max-1), 1)[0]
        future_index = np.random.geometric(p=self.gamma, size=1)[0]+start_index
        if future_index > max:
            future_index = max
        return np.array([start_index, future_index])

    def __len__(self):
        return len(self.traj)


class TrajBatchGenerator(object):
    def __init__(self, trajectories, batch_size=64, n_feature=49, gamma=0.9,
                 add_data_point=False, add_noise=False, std_scaler=0.1, add_corrections=False):
        self.batch_size = batch_size
        self.n_feature = n_feature
        self.traj = trajectories

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        start_states = []
        future_states = []
        for n in range(batch_size):
            indice = random.sample(range(0, self.traj.shape[0]), 1)[0]
            single_traj = self.traj[indice, :, :]
            time_indices = self.sample_time_id(single_traj.shape[0] - 1)
            states = single_traj[time_indices, :self.n_feature]
            state = states[0, :]
            future_state = states[1, :]
            if self.corrections is not None:
                state = (state + self.bias_corrections)*self.corrections
                future_state = (future_state + self.bias_corrections) * self.corrections
            if self.std_scaler is not None:
                state = state + np.random.normal(scale=self.std_scaler)
                future_state = future_state + np.random.normal(scale=self.std_scaler)
            start_states.append(state)
            future_states.append(future_state)
        start_states = np.stack(start_states)
        future_states = np.stack(future_states)
        return start_states, future_states

    def __len__(self):
        return len(self.traj)


