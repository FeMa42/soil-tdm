import os
import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from common import helper


def get_norm_val(device):
    norm_val = [3.6, 1., 130., 10., 1., 1., 1., 3.6, 1., 3.6, 3.6,  # ego features (lat_pos_norm normally changes !!!)
                10., 3.6,  # last action features (acceleration and lateral change)
                100.0, 12.54, 130., 10., 1., 1., 1.,  # leader features last one is ttc (maybe we also need to change it
                100.0, 12.54, 1., 130., 1., 1.,  # veh 1 feature dist, lat_dist, l_ind, v, length, width
                100.0, 12.54, 1., 130., 1., 1.,  # veh 2 feature dist, lat_dist, l_ind, v, length, width
                100.0, 12.54, 1., 130., 1., 1.,  # veh 3 feature dist, lat_dist, l_ind, v, length, width
                100.0, 12.54, 1., 130., 1., 1.,  # veh 4 feature dist, lat_dist, l_ind, v, length, width
                100.0, 12.54, 1., 130., 1., 1.,  # veh 5 feature dist, lat_dist, l_ind, v, length, width
                ]
    norm_val = np.array(norm_val)
    norm_val = torch.tensor(norm_val, dtype=torch.float).to(device)
    norm_val.unsqueeze(0)
    return norm_val


def denormalize_features(features, device):
    norm_val = get_norm_val(device)
    features = features * norm_val
    return features


def normalize_features(features, device):
    norm_val = get_norm_val(device)
    features = features / norm_val
    return features


def forward_model(in_features, action, device):
    '''Sumo Transition model with acceleration and lateral change as action values'''
    # Action[:, 0] is v
    # Action[:, 1] is negative lateral change
    #   -> y_lc = abs(y(t)) - abs(y(t+1)) ; Bsp abs(-6.0) - abs(-6.1) = -0.1
    #   - normally it would be the other way round
    #   - since y values are negative (from 0 to -12) we calculate the lateral change like this
    # lateral change in feature vector is calculated differently
    #   -> y_lc = y(t+1) - y(t) ; Bsp: -6.1 - (-6.0) =  -0.1
    #   - the value should be the same however
    # Calculating d2r and d2l, both values are positive! -> towards right is negative lateral change
    #   -> d2r(t+1) =  d2r(t) + y_lc ;  Bsp: 2.1 + (-0.1) =  2.0
    #   -> d2l(t+1) =  d2l(t) - y_lc ;  Bsp: 2.1 - (-0.1) =  2.2
    features = denormalize_features(in_features, device)

    delta_t = 0.04
    delta_v = action[:, 0] * delta_t
    lateral_change = action[:, 1] - features[:, 0]  # features[:, 12] delta y (wished y übergeben)
    delta_lon = features[:, 2] * delta_t    # not considering distance covered by acceleration
    delta_lat = lateral_change  # features[:, 7].clone()      # features[:, 6] is already y(t+1) - y(t)

    features[:, 0] = action[:, 1]  # lateral pos
    features[:, 2] = features[:, 2] + delta_v  # v
    features[:, 3] = action[:, 0]  # a
    features[:, 7] = lateral_change  # action[:, 1]  # lc
    features[:, 9] = features[:, 9] + delta_lat  # dis_2_right
    features[:, 10] = features[:, 10] - delta_lat  # dis_2_left
    features[:, 11] = action[:, 0]  # a
    features[:, 12] = action[:, 1]  # lateral pos

    apply_lon = delta_lon * (features[:, 13] != 0.).int()
    apply_lat = delta_lat * (features[:, 13] != 0.).int()
    features[:, 13] = (features[:, 13] - apply_lon) + (features[:, 15] * delta_t)  # lon_dis_lead
    features[:, 14] = (features[:, 14] - apply_lat)  # lat_dis_lead,  positive->left, negative->right
    features[:, 15] = features[:, 15] + (features[:, 16] * delta_t)

    apply_lon = delta_lon * (features[:, 20] != 0.).int()
    apply_lat = delta_lat * (features[:, 20] != 0.).int()
    features[:, 20] = (features[:, 20] - apply_lon) + (features[:, 23] * delta_t)  # lon_dis_1
    features[:, 21] = (features[:, 21] - apply_lat)  # lat_dis_1

    apply_lon = delta_lon * (features[:, 26] != 0.).int()
    apply_lat = delta_lat * (features[:, 26] != 0.).int()
    features[:, 26] = (features[:, 26] - apply_lon) + (features[:, 29] * delta_t)  # lon_dis_2
    features[:, 27] = (features[:, 27] - apply_lat)  # lat_dis_2

    apply_lon = delta_lon * (features[:, 32] != 0.).int()
    apply_lat = delta_lat * (features[:, 32] != 0.).int()
    features[:, 32] = (features[:, 32] - apply_lon) + (features[:, 35] * delta_t)  # lon_dis_3
    features[:, 33] = (features[:, 33] - apply_lat)  # lat_dis_3

    apply_lon = delta_lon * (features[:, 38] != 0.).int()
    apply_lat = delta_lat * (features[:, 38] != 0.).int()
    features[:, 38] = (features[:, 38] - apply_lon) + (features[:, 41] * delta_t)  # lon_dis_4
    features[:, 39] = (features[:, 39] - apply_lat)  # lat_dis_4

    apply_lon = delta_lon * (features[:, 44] != 0.).int()
    apply_lat = delta_lat * (features[:, 44] != 0.).int()
    features[:, 44] = (features[:, 44] - apply_lon) + (features[:, 47] * delta_t)  # lon_dis_5
    features[:, 45] = (features[:, 45] - apply_lat)  # lat_dis_5

    features = normalize_features(features, device)

    return features


def test_transition_model(train_state, train_action, output_dir, device="cpu"):
    # test sumo_transition_model:
    test_trans_model = False
    if test_trans_model:
        timestamp = 10
        traj = 500  # neg: 908 (0), 887 (4) pos:  829 (0), 828 (0)
        lat_changes = train_state[0:1000, :, 6]
        state_subset = train_state[200:traj, :, :]
        action_subset = train_action[200:traj, :, :]
        next_state = torch.tensor(state_subset[:, timestamp, :]).to(device)
        first_action = torch.tensor(action_subset[:, timestamp, :]).to(device)
        diffs = []
        for it in range(20):
            current_action = torch.tensor(action_subset[:, timestamp+it, :]).to(device)
            # print(current_action)
            real_next_state = torch.tensor(state_subset[:, (timestamp + 1) + it, :]).to(device)
            time_diff = real_next_state - next_state
            next_state = forward_model(next_state, current_action, device)
            diff = real_next_state - next_state
            diffs.append(abs(diff).mean(0).unsqueeze(0))
            print("lateral position")
            print(abs(diff[:, 0]).max(0))
            print("Last lateral action")
            print(abs(diff[:, 12]).max(0))
        diffs = torch.cat(diffs).cpu().numpy()
        for iter in range(diffs.shape[1]):
            f, axes = plt.subplots(1, 1, sharex=True)
            plt.grid(b=True)
            sns.set_style("darkgrid")
            axes.set_title(helper.features[iter] + " error")
            axes.grid(b=True)
            axes.plot(diffs[:, iter])
            plt.savefig(os.path.join(output_dir, "Sumo_multi-v0" + "_error_series_" + helper.features[iter] + ".png"))
            plt.close()
            print(diffs[:, iter])