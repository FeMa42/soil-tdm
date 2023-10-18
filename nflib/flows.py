"""
# Inspired from https://github.com/karpathy/pytorch-normalizing-flows

"""

import torch
from torch import nn

class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows, device="cpu", prep_con=False, cond_in=49, nh=64):
        super().__init__()
        self.flows = nn.ModuleList(flows).to(device)
        if prep_con:
            self.cond_layer = MLP(nin=cond_in, nout=nh, nh=nh)
        else:
            self.cond_layer = None
        self.device = device

    def forward_sample(self, x, con_in=None):
        m, _ = x.shape
        log_det = torch.zeros(m).to(self.device)
        zs = [x]
        if con_in is not None and self.cond_layer is not None:
            con_in = self.cond_layer(con_in)
        for flow in self.flows:
            if con_in is not None:
                x, ld = flow.forward_sample(x, con_in)
            else:
                x, ld = flow.forward_sample(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward_sample(self, z, con_in=None):
        m, _ = z.shape
        log_det = torch.zeros(m).to(self.device)
        xs = [z]
        if con_in is not None and self.cond_layer is not None:
            con_in = self.cond_layer(con_in)
        for flow in self.flows[::-1]:
            if con_in is not None:
                z, ld = flow.backward_sample(z, con_in)
            else:
                z, ld = flow.backward_sample(z)
            log_det += ld
            xs.append(z)
        return xs, -log_det


class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (base, flow) pair """
    
    def __init__(self, base, flows, device="cpu", prep_con=True, cond_in=49, nh=64,
                 is_freia=False):
        super().__init__()
        self.device = device
        self.base = base
        if is_freia:
            self.flow = flows
        else:
            self.flow = NormalizingFlow(flows, device, prep_con, cond_in, nh).to(device)
        self.freia_inn = is_freia

    def forward(self, x, con_in=None):
        return self.forward_sample(x, con_in)

    def log_prob(self, x, con_in=None):
        if con_in is not None:
            zs, log_det = self.flow.forward_sample(x, con_in)
        else:
            zs, log_det = self.flow.forward_sample(x)
        base_logprob = self.base.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return base_logprob+log_det

    def forward_sample(self, x, con_in=None):
        if con_in is not None:
            zs, log_det = self.flow.forward_sample(x, con_in)
        else:
            zs, log_det = self.flow.forward_sample(x)
        base_logprob = self.base.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, base_logprob, log_det

    def backward_sample(self, z, con_in=None):
        if con_in is not None:
            xs, log_det = self.flow.backward_sample(z, con_in)
        else:
            xs, log_det = self.flow.backward_sample(z)
        return xs, log_det
    
    def sample(self, num_samples, con_in=None):
        z = self.base.sample((num_samples,))
        if con_in is not None:
            xs, _ = self.flow.backward_sample(z, con_in)
        else:
            xs, _ = self.flow.backward_sample(z)
        return xs

    def r_sample(self, num_samples, con_in=None):
        z = self.base.sample((num_samples,))
        if con_in is not None:
            xs, log_det = self.flow.backward_sample(z, con_in)
        else:
            xs, log_det = self.flow.backward_sample(z)
        base_logprob = self.base.log_prob(z)
        base_logprob = base_logprob.view(xs[-1].size(0), -1).sum(1)
        return xs, base_logprob, log_det

    def sample_base(self, sample_size):
        return self.base.sample((sample_size,))
