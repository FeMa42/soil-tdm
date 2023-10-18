import numpy as np
import torch


class NormalizedActions():
    def __init__(self, low=-7., high=7., use_numpy=False, do_normalize=True):
        self.low = low
        self.high = high
        self.use_numpy = use_numpy
        self.do_normalize = do_normalize

    def reverse_normalize(self, action):
        '''[-1, 1] to [min, max]'''
        if self.do_normalize:
            low = self.low
            high = self.high

            action = low + (action + 1.0) * 0.5 * (high - low)
            if self.use_numpy:
                action = np.clip(action, low, high)
            else:
                action = torch.clamp(action, low, high)
        return action

    def normalize(self, action):
        '''[min, max] to [-1, 1]'''
        if self.do_normalize:
            low = self.low
            high = self.high

            if self.use_numpy:
                action = np.clip(action, low, high)
            else:
                action = torch.clamp(action, low, high)
            action = 2 * (action - low) / (high - low) - 1
        return action