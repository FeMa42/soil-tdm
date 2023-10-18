import torch
import torch.nn as nn


def generate_snn(ch_in, ch_out, internal_width, identity_start=True):
    if identity_start:
        last_linear = nn.Linear(internal_width, ch_out)
    else:
        last_linear = nn.Identity()
    network = nn.Sequential(
        nn.Linear(ch_in, internal_width),
        nn.SELU(),
        nn.AlphaDropout(p=0.2),
        nn.Linear(internal_width, internal_width),
        nn.SELU(),
        nn.AlphaDropout(p=0.2),
        nn.Linear(internal_width, internal_width),
        nn.SELU(),
        nn.AlphaDropout(p=0.2),
        nn.Linear(internal_width, internal_width),
        nn.SELU(),
        nn.AlphaDropout(p=0.2),
        # nn.Linear(internal_width, internal_width),
        # nn.SELU(),
        # nn.AlphaDropout(p=0.2),
        # nn.Linear(internal_width, internal_width),
        # nn.SELU(),
        # nn.AlphaDropout(p=0.2),
        last_linear
    )
    if identity_start:
        torch.nn.init.zeros_(last_linear.weight.data)
        torch.nn.init.zeros_(last_linear.bias.data)
    return network


def generate_mlp(ch_in, ch_out, internal_width, identity_start=True):
    if identity_start:
        last_linear = nn.Linear(internal_width, ch_out)
    else:
        last_linear = nn.Identity()
    network = nn.Sequential(
        nn.Linear(ch_in, internal_width),
        nn.LeakyReLU(),
        nn.Linear(internal_width, internal_width),
        nn.LeakyReLU(),
        last_linear
    )
    if identity_start:
        torch.nn.init.zeros_(last_linear.weight.data)
        torch.nn.init.zeros_(last_linear.bias.data)
    return network