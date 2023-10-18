"""Model for the cINN

get VLL-HD FrEIA from https://github.com/VLL-HD/FrEIA
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim
import numpy as np
from types import SimpleNamespace


import FrEIA.framework as Ff
import FrEIA.modules as Fm
from common.mlp_model import generate_mlp, generate_snn

class CondNet(nn.Module):
    """conditioning2 network"""

    # from experiments/colorization_minimal_example/model.py
    def __init__(self, input_size=2, hidden_size=8, n_blocks=10, y_dim_features=64, use_snn=False, use1h = False):
        super().__init__()
        if use_snn:
            self.condNets = generate_snn(input_size, hidden_size, hidden_size, identity_start=False)
        else:
            if use1h is False:
                self.condNets = nn.Sequential(
                    nn.Linear(input_size, hidden_size), nn.LeakyReLU(),
                    nn.Linear(hidden_size, hidden_size),  # 96
                    nn.LeakyReLU(),
                )
            else:
                self.condNets = nn.Sequential(
                    nn.Linear(input_size, hidden_size), nn.LeakyReLU()
                )

        self.subCondNets = nn.ModuleList([])
        for _ in range(n_blocks):
            self.subCondNets.append(
                nn.Sequential(
                    nn.Linear(hidden_size, y_dim_features),
                )
            )

    def forward(self, conditions):
        c = self.condNets(conditions)
        outputs = []
        for m in self.subCondNets:
            outputs.append(m(c))
        return outputs


class CINN(nn.Module):
    """cINN, including the conditioning network"""

    def __init__(self, n_data=2, n_cond=2, n_blocks=10, internal_width=64, hidden_size_cond=64, exponent_clamping=2.,
                 y_dim_features=64, use_snn=False, identity_init=False, model_device="cpu", use1h = False):
        super().__init__()

        # y_dim_features = 64
        self.cinn = self.build_inn(n_data, n_blocks, internal_width, exponent_clamping, y_dim_features,
                                   use_snn=use_snn)
        self.cond_net = CondNet(n_cond, hidden_size=hidden_size_cond, n_blocks=n_blocks,
                                y_dim_features=y_dim_features, use_snn=use_snn, use1h = use1h)
        self.cinn.to(model_device)
        self.cond_net.to(model_device)
        self.device = model_device
        init_scale = 0.075 / 16

        self.trainable_parameters = [
            p for p in self.cinn.parameters() if p.requires_grad
        ]

        if not identity_init:
            for p in self.trainable_parameters:
                p.data = init_scale * torch.randn_like(p)

        self.trainable_parameters += list(self.cond_net.parameters())

    def build_inn(self, n_data, n_blocks, internal_width, exponent_clamping, y_dim_features, use_snn):
        # TODO: use here your constructor for your networks inside the INN

        def fc_constr():
            def _thunk(ch_in, ch_out):
                if use_snn:
                    network = generate_snn(ch_in, ch_out, internal_width, identity_start=True)
                else:
                    self.linear3 = nn.Linear(internal_width, ch_out)
                    network = nn.Sequential(
                        nn.Linear(ch_in, internal_width),
                        nn.LeakyReLU(),
                        nn.Linear(internal_width, internal_width),
                        nn.LeakyReLU(),
                        self.linear3
                    )
                    # this makes the flow block an identity function after initialization
                    torch.nn.init.zeros_(self.linear3.weight.data)
                    torch.nn.init.zeros_(self.linear3.bias.data)
                    # network = generate_mlp(ch_in, ch_out, internal_width, identity_start=True)
                return network
            return _thunk

        nodes = [Ff.InputNode(n_data)]

        # nodes.append(Ff.Node(nodes[-1], Fm.ActNorm, {"clamp_value": exponent_clamping}, name=f"ActN{0}"))

        # outputs of the cond. net at different resolution levels
        conditions = []

        for i in range(n_blocks):
            conditions.append(Ff.ConditionNode(y_dim_features))

            # Normalizes the inputs
            # nodes.append(Ff.Node(nodes[-1], Fm.ActNorm, {"clamp_value": exponent_clamping}, name=f"ActN{i}"))

            nodes.append(
                Ff.Node(
                    nodes[-1],
                    Fm.GLOWCouplingBlock,
                    {
                        "subnet_constructor": fc_constr(),
                        "clamp": exponent_clamping,
                    },
                    conditions=conditions[i],
                    name=f"coupling_{i}",
                )
            )
            # Permutates Order after each block
            nodes.append(
                Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": i}, name=f"PERM_FC_{i}")
            )
            # Normalizes the results
            nodes.append(Ff.Node(nodes[-1], Fm.ActNorm, {"clamp_value": exponent_clamping}, name=f"ActN{i}"))

        nodes.append(Ff.OutputNode(nodes[-1]))

        return Ff.ReversibleGraphNet(nodes + conditions, verbose=False)
        # return Ff.ReversibleGraphNet(nodes, verbose=False)

    def forward_sample(self, input_x, condition=None):
        condition = self.cond_net(condition)
        z = self.cinn(input_x, c=condition)
        jac = self.cinn.log_jacobian(run_forward=False)
        return [z], jac

    def backward_sample(self, z, condition=None):
        condition = self.cond_net(condition)
        x_samples = self.cinn(z, c=condition, rev=True)
        jac = self.cinn.log_jacobian(run_forward=False)
        return [x_samples], jac

    def read_params(self):
        params = []
        names = []
        for name, param in self.cinn.named_parameters():
            params.append(torch.sum(param.data).detach().cpu().numpy())
            names.append(name)
        return params, names

    def save(self, name):
        self.cinn.to("cpu")
        self.cond_net.to("cpu")
        torch.save(
            {
                "net": self.cinn.state_dict(),
                "cond_net": self.cond_net.state_dict(),
            },
            name,
        )

        print(f"saved  model at {name}")
        self.cinn.to(self.device)
        self.cond_net.to(self.device)

    def load(self, name):
        print("loads model from ", name)
        state_dicts = torch.load(name, map_location="cpu")
        self.cinn.load_state_dict(state_dicts["net"])
        self.cond_net.load_state_dict(state_dicts["cond_net"])
        self.cinn.to(self.device)
        self.cond_net.to(self.device)
