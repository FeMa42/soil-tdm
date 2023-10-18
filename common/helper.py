#  Copyright: (C) ETAS GmbH 2019. All rights reserved.
import os
import numpy as np
import torch
import torch.nn as nn
import gym
import h5py
import random
from scipy.special import rel_entr
from scipy.spatial import distance
import csv
from collections import deque
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

features = {
    0: "lat_pos",
    1: "l_ind",
    2: "v",
    3: "a",
    4: "heading",
    5: "length",
    6: "width",
    7: "lat_change",
    8: "head_change",
    9: "dis_2_right",
    10: "dis_2_left",
    11: "last_acc",
    12: "last_lc",
    13: "lon_dis_lead",
    14: "lat_dis_lead",
    15: "v_lead",
    16: "a_lead",
    17: "length_lead",
    18: "width_lead",
    19: "ttc",
    20: "lon_dis_1",
    21: "lat_dis_1",
    22: "l_ind_1",
    23: "v_1",
    24: "length_1",
    25: "width_1",
    26: "lon_dis_2",
    27: "lat_dis_2",
    28: "l_ind_2",
    29: "v_2",
    30: "length_2",
    31: "width_2",
    32: "lon_dis_3",
    33: "lat_dis_3",
    34: "l_ind_3",
    35: "v_3",
    36: "length_3",
    37: "width_3",
    38: "lon_dis_4",
    39: "lat_dis_4",
    40: "l_ind_4",
    41: "v_4",
    42: "length_4",
    43: "width_4",
    44: "lon_dis_5",
    45: "lat_dis_5",
    46: "l_ind_5",
    47: "v_5",
    48: "length_5",
    49: "width_5",
    50: "lon_dis_6",
    51: "lat_dis_6",
    52: "l_ind_6",
    53: "v_6",
    54: "length_6",
    55: "width_6",
    56: "-",
    57: "-",
    58: "-",
    59: "-",
    60: "-"}


class SumoEvaluation():
    def __init__(self, writer, epoch, test_envs, generate_data_new_method, reduce_state_method,
                 output_dir, actor_critic, expert_state_est_model, max_nll, device):
        self.writer = writer 
        self.epoch = epoch
        self.generate_data_new_method = generate_data_new_method
        self.reduce_state_method = reduce_state_method
        self.output_dir = output_dir
        self.test_envs = test_envs
        self.actor_critic = actor_critic
        self.device = device 
        self.expert_state_est_model = expert_state_est_model
        self.max_nll = max_nll

    
    def perform_sumo_test(self, n_test=100, num_env_steps=250, metric_file="highD_final_metrics.csv"):
        test_start = self.epoch + 1
        sumo_jsd_calc = SumoJSD(self.writer, clear_n_epoch=-1,
                                    dir_feature='./data/', run_dir=self.output_dir,
                                    metric_file=metric_file)
        sumo_jsd_calc.clear_all()
        state = self.test_envs.reset()
        state = torch.FloatTensor(state).to(self.device)
        length = torch.ones((state.shape[0],))
        for test_step in range(n_test):
            test_epoch = test_start + test_step
            state, _, rewards, _, \
                _, length, sim_length = self.generate_data_new_method(state, num_env_steps,
                                                                      self.actor_critic, self.test_envs,
                                                                      None,
                                                                      length,
                                                                      self.expert_state_est_model,
                                                                      max_nll=self.max_nll,
                                                                      is_sumo_gym=True,
                                                                      state_mod_function=self.reduce_state_method,
                                                                      sumo_jsd_calc=sumo_jsd_calc)
            sumo_jsd_calc.estimate_jsd(test_epoch)
            state = self.test_envs.reset()
            state = torch.FloatTensor(state).to(self.device)
            length = torch.ones((state.shape[0],))
            mean_rewards = np.mean(rewards)
            self.writer.add_scalar(
                'rl_training/current_policy_expert_rewards', mean_rewards, test_epoch)
            self.writer.add_scalar(
                'rl_training/mean_length', sim_length, test_epoch)
        self.epoch = test_epoch


def pretrain_bc(args, train_state, train_action, valid_state, valid_action, 
                BehavioralCloning, actor_critic, output_dir, env_name, 
                use_mlp, writer, device="cpu", num_bc_updates=1000, sumo_evaluation=None, noise_value_x=0.0):
    use_linear_lr_decay_bc = False
    weight_clipping = False
    weight_clip_val = 1.
    losses = deque(maxlen=100)
    agent = BehavioralCloning(actor_critic, args.lr_bc, args.weight_decay,
                              flow_model=(not use_mlp), grad_clip_val=args.grad_clip_val)
    for j in range(num_bc_updates):
        if use_linear_lr_decay_bc:
            update_linear_schedule(
                agent.optimizer, j, args.num_bc_updates, args.lr_bc)

        indices = np.array(random.sample(
            range(0, train_state.shape[0] - 1), args.bc_batch_size))
        indices = np.sort(indices)
        actions_trajs = torch.tensor(
            train_action[indices, :], dtype=torch.float).to(device)
        state_trajs = train_state[indices, :]
        state_trajs += np.random.randn(*state_trajs.shape) * noise_value_x
        state_trajs = torch.tensor(state_trajs, dtype=torch.float).to(device)

        loss = agent.update(state_trajs, actions_trajs)

        if weight_clipping:
            for p in actor_critic.get_policy_parameter():
                p.data.clamp_(-weight_clip_val, weight_clip_val)

        losses.append(loss)

        if j % 25 == 0:
            print("Update Steps: ", j)
            print("Mean BC Loss: ", np.mean(losses))
            writer.add_scalar('training/train_bc_loss', np.mean(losses), j)

            with torch.no_grad():
                test_indices = np.array(random.sample(
                    range(0, valid_state.shape[0] - 1), args.bc_batch_size))
                test_indices = np.sort(test_indices)
                test_actions = torch.tensor(
                    valid_action[test_indices, :], dtype=torch.float).to(device)
                test_state = torch.tensor(
                    valid_state[test_indices, :], dtype=torch.float).to(device)
                test_loss = agent.estimate_error(test_state, test_actions)
                print("Mean Test Loss: ", np.mean(test_loss))
                writer.add_scalar('training/test_bc_loss', np.mean(test_loss), j)
        if j % 100 == 0:
            actor_critic.save(os.path.join(output_dir, env_name + "_d.pt"))

        if j % 100000 == 0:
            if sumo_evaluation is not None: 
                sumo_evaluation.perform_sumo_test(n_test=10, metric_file="highD_" + str(j/100000) + "_metrics.csv")



def regularize_bco(train_state, train_next_state, bco_agent, state_cond_ll_model, 
                   bc_batch_size, device="cpu", num_bc_updates=100):
    losses = deque(maxlen=100)
    for j in range(num_bc_updates):
        indices = np.array(random.sample(
            range(0, train_state.shape[0] - 1), bc_batch_size))
        indices = np.sort(indices)
        state_trajs = torch.tensor(
            train_state[indices, :], dtype=torch.float).to(device)
        next_state_trajs = torch.tensor(
            train_next_state[indices, :], dtype=torch.float).to(device)

        # generate action using inverse dynamic model p(a|s',s):
        actions_trajs, _ = state_cond_ll_model.generate_actions_based_on_transition(
            state_trajs, next_state_trajs)

        loss = bco_agent.update(state_trajs, actions_trajs)

        losses.append(loss)

    return np.mean(losses)

def normality_test(valid_state, actor_critic, output_dir, env_name, device="cpu"):
    # test normality of policy distribution
    # valid_state, valid_action
    test_size = 1000

    alpha = 0.05
    shapiro_list = []
    normaltest_list = []
    shapiro_stat_list = []
    normaltest_stat_list = []
    anderson_list_sl = []
    anderson_list_cv = []
    anderson_list_statistics = []
    shapiro_count = 0
    normaltest_count = 0
    anderson_count = 0

    indices = np.array(random.sample(
        range(0, valid_state.shape[0] - 1), test_size))
    indices = np.sort(indices)
    valid_state_trajs = torch.tensor(
        valid_state[indices, :], dtype=torch.float).to(device)

    for indice in range(test_size):
        state_sample = torch.tensor(
            valid_state_trajs[indice, :], dtype=torch.float).unsqueeze(0).to(device)

        actions, log_prob = actor_critic.forward_sampling(
            state_sample, n_samples=4096)

        # Shapiro-Wilk Test
        shapiro_normal = False
        actions = actions.detach().cpu().numpy()
        stat_acc, p_acc = shapiro(actions[:, 0])
        print('Acc Statistics=%.3f, p=%.3f' % (stat_acc, p_acc))
        if p_acc > alpha:
            print('Acc Sample looks Gaussian (fail to reject H0)')
            shapiro_normal = True
        else:
            print('Acc Sample does not look Gaussian (reject H0)')
        shapiro_list.append(p_acc)
        shapiro_stat_list.append(stat_acc)

        stat_lc, p_lc = shapiro(actions[:, 1])
        print('LC Statistics=%.3f, p=%.3f' % (stat_lc, p_lc))
        if p_lc > alpha:
            print('LC Sample looks Gaussian (fail to reject H0)')
            shapiro_normal = True
        else:
            print('LC Sample does not look Gaussian (reject H0)')
        shapiro_list.append(p_lc)
        shapiro_stat_list.append(stat_lc)
        if shapiro_normal:
            shapiro_count += 1

        # D'Agostino and Pearson's Test
        normaltest_normal = False
        stat_n_acc, p_n_acc = normaltest(actions[:, 0])
        print('N ACC Statistics=%.3f, p=%.3f' % (stat_n_acc, p_n_acc))
        if p_n_acc > alpha:
            print('N ACC Sample looks Gaussian (fail to reject H0)')
            normaltest_normal = True
        else:
            print('N ACC Sample does not look Gaussian (reject H0)')
        normaltest_list.append(p_n_acc)
        normaltest_stat_list.append(stat_n_acc)
        stat_n_lc, p_n_lc = normaltest(actions[:, 0])
        print('N LC Statistics=%.3f, p=%.3f' % (stat_n_lc, p_n_lc))
        if p_n_lc > alpha:
            print('N ACC Sample looks Gaussian (fail to reject H0)')
            normaltest_normal = True
        else:
            print('N ACC Sample does not look Gaussian (reject H0)')
        normaltest_list.append(p_n_lc)
        normaltest_stat_list.append(stat_n_lc)
        if normaltest_normal:
            normaltest_count += 1

        # Anderson-Darling Test
        anderson_normal = False
        result_acc = anderson(actions[:, 0])
        print('Statistic: %.3f' % result_acc.statistic)
        anderson_list_statistics.append(result_acc.statistic)
        p = 0
        for i in range(len(result_acc.critical_values)):
            sl, cv = result_acc.significance_level[i], result_acc.critical_values[i]
            if result_acc.statistic < result_acc.critical_values[i]:
                print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
                anderson_normal = True
            else:
                print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
            anderson_list_sl.append(sl)
            anderson_list_cv.append(cv)

        if anderson_normal:
            anderson_count += 1
        elif result_acc.statistic > 400.0:
            fig = plt.figure()  # figsize=(9, 6)
            ax1 = fig.add_subplot(211)
            ax1.title.set_text('Acceleration')
            plt.hist(actions[:, 0])

            ax2 = fig.add_subplot(212)
            ax2.title.set_text('Lateral Change')
            plt.hist(actions[:, 1])

            plt.savefig(os.path.join(output_dir,
                                     env_name + str(result_acc.statistic) + "action_modality" + str(indice) + ".png"))
            plt.close()

            plt.figure(figsize=(6, 8))
            plt.scatter(actions[:, 0], actions[:, 1], c='g', s=5)
            plt.legend(['action'])
            plt.axis('scaled')
            plt.title('actions')
            plt.savefig(os.path.join(output_dir, env_name + str(result_acc.statistic) + "_2_dim_action_modality" + str(
                indice) + ".png"))
            plt.close()

    # mean results
    print('shapiro_count: ', shapiro_count)
    print('normaltest_count: ', normaltest_count)
    print('anderson_count: ', anderson_count)
    print('shapiro_list=%.3f' % np.mean(shapiro_list))
    print('normaltest_list=%.3f' % np.mean(normaltest_list))
    print('shapiro_stat_list=%.3f' % np.mean(shapiro_stat_list))
    print('normaltest_stat_list=%.3f' % np.mean(normaltest_stat_list))
    print('anderson_list_sl=%.3f' % np.mean(anderson_list_sl))
    print('anderson_list_cv=%.3f' % np.mean(anderson_list_cv))
    print('anderson_list_statistics=%.3f' % np.mean(anderson_list_statistics))



def make_env(seed, rank, env_name, use_env_id=True):
    def _thunk():
        env = gym.make(env_name)
        seed_val = 2 * (seed + rank) + 1
        env.seed(seed_val)
        if use_env_id: 
            env.reset(env_id=rank)
        else:
            env.reset()
        return env

    return _thunk


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


def load_pandas_data(load_dir, seq_length=1, filename="feature_pandas", skip_step=1):  # feature_pandas_interest
    import pickle
    n_feature = 49 

    print('loading data...')
    d = []
    with open(os.path.join(load_dir, filename), 'rb') as fp:
        data = pickle.load(fp)

    total_idx = 0
    for traj in data:
        idx = 0
        length = np.shape(traj)[0]
        while (idx + seq_length) < length:
            d.append(traj[idx:idx + seq_length, :])
            idx += skip_step
            total_idx += skip_step
        if total_idx > 10000:
            break
    d = np.asarray(d, dtype="float32")
    np.random.shuffle(d)
    s = d[:, :, 0:n_feature]
    a = d[:, :, -2:]

    train_size = int(np.shape(s)[0] * 0.8)
    train_x, valid_x = s[0:train_size, :, :], s[train_size:, :, :]
    train_y, valid_y = a[0:train_size, :, :], a[train_size:, :, :]

    print('loading data completed, num of trajectory:', train_size)
    return train_x, train_y, valid_x, valid_y


def load_hdf5_data(load_dir, filename="feature_pandas"):
    hdf5_store = h5py.File(os.path.join(load_dir, filename + ".hdf5"), "r")
    train_x = hdf5_store["train_x"]
    train_y = hdf5_store["train_y"]
    valid_x = hdf5_store["valid_x"]
    valid_y = hdf5_store["valid_y"]

    train_state = train_x[:].reshape((-1, train_x.shape[-1]))
    train_action = train_y[:].reshape((-1, train_y.shape[-1]))
    valid_state = valid_x[:].reshape((-1, valid_x.shape[-1]))
    valid_action = valid_y[:].reshape((-1, valid_y.shape[-1]))
    return train_state, train_action, valid_state, valid_action


def load_hdf5_data_wt(load_dir, filename="feature_pandas"):
    hdf5_store = h5py.File(os.path.join(load_dir, filename + ".hdf5"), "r")
    train_x = hdf5_store["train_x"]
    train_y = hdf5_store["train_y"]
    valid_x = hdf5_store["valid_x"]
    valid_y = hdf5_store["valid_y"]
    train_t = hdf5_store["train_t"]
    valid_t = hdf5_store["valid_t"]
    return train_x, train_y, valid_x, valid_y, train_t, valid_t


def load_hdf5_data_wns(load_dir, filename="feature_pandas", n_trajectories=-1, start_traj=0, test_mode=False, load_perc=1.,
                       reshape=True):
    hdf5_store = h5py.File(os.path.join(load_dir, filename + ".hdf5"), "r")
    train_x = hdf5_store["train_x"]
    train_y = hdf5_store["train_y"]
    valid_x = hdf5_store["valid_x"]
    valid_y = hdf5_store["valid_y"]
    train_n_x = hdf5_store["train_n_x"]
    valid_n_x = hdf5_store["valid_n_x"]

    start_trajectory = start_traj
    if n_trajectories > 0 and test_mode:
        test_x_add = train_x[start_trajectory + n_trajectories:, :, :]
        test_y_add = train_y[start_trajectory + n_trajectories:, :, :]
        test_n_x_add = train_n_x[start_trajectory + n_trajectories:, :, :]
        test_x_add_2 = train_x[:start_trajectory, :, :]
        test_y_add_2 = train_y[:start_trajectory, :, :]
        test_n_x_add_2 = train_n_x[:start_trajectory, :, :]

        valid_x = np.concatenate((valid_x, test_x_add), axis=0)
        valid_y = np.concatenate((valid_y, test_y_add), axis=0)
        valid_n_x = np.concatenate((valid_n_x, test_n_x_add), axis=0)
        valid_x = np.concatenate((valid_x, test_x_add_2), axis=0)
        valid_y = np.concatenate((valid_y, test_y_add_2), axis=0)
        valid_n_x = np.concatenate((valid_n_x, test_n_x_add_2), axis=0)

    if n_trajectories > 0:
        train_x = train_x[start_trajectory:start_trajectory+n_trajectories, :, :]
        train_y = train_y[start_trajectory:start_trajectory+n_trajectories, :, :]
        # valid_x = valid_x[start_trajectory:n_trajectories, :, :]
        # valid_y = valid_y[start_trajectory:n_trajectories, :, :]
        train_n_x = train_n_x[start_trajectory:start_trajectory+n_trajectories, :, :]
        # valid_n_x = valid_n_x[start_trajectory:n_trajectories, :, :]
    else:
        if load_perc > 1.:
            if load_perc >= train_x.shape[0]:
                load_perc = 1.
                num_datapoints = int(train_x.shape[0] * load_perc)
            else:
                num_datapoints = int(load_perc)
        else:
            num_datapoints = int(train_x.shape[0]*load_perc)
        print("Amount of expert samples: ", num_datapoints)
        num_test_datapoints = int(valid_x.shape[0])
        if num_datapoints > 5000:
            use_random_datapoints = False
        else:
            use_random_datapoints = True
        if use_random_datapoints:
            indices = np.array(random.sample(range(0, len(train_x) - 1), num_datapoints))
            indices = np.sort(indices)
            train_x = train_x[indices, :]
            train_y = train_y[indices, :]
            train_n_x = train_n_x[indices, :]
            valid_x = valid_x[:num_test_datapoints, :]
            valid_y = valid_y[:num_test_datapoints, :]
            valid_n_x = valid_n_x[:num_test_datapoints, :]
        else:
            train_x = train_x[:num_datapoints, :]
            train_y = train_y[:num_datapoints, :]
            train_n_x = train_n_x[:num_datapoints, :]

    print("Expert samples loaded. ")
    if reshape:
        print("reshape and laoding all into RAM... ")
        train_state = train_x[:].reshape((-1, train_x.shape[-1]))
        train_action = train_y[:].reshape((-1, train_y.shape[-1]))
        valid_state = valid_x[:].reshape((-1, valid_x.shape[-1]))
        valid_action = valid_y[:].reshape((-1, valid_y.shape[-1]))
        train_next_state = train_n_x[:].reshape((-1, train_n_x.shape[-1]))
        valid_next_state = valid_n_x[:].reshape((-1, valid_n_x.shape[-1]))
    else:
        print("Loading all into RAM... ")
        train_state = train_x[:]
        train_action = train_y[:]
        valid_state = valid_x[:]
        valid_action = valid_y[:]
        train_next_state = train_n_x[:]
        valid_next_state = valid_n_x[:]

    return train_state, train_action, valid_state, valid_action, train_next_state, valid_next_state


def aval_hdf5_data_wns(load_dir, filename="feature_pandas"):
    hdf5_store = h5py.File(os.path.join(load_dir, filename + ".hdf5"), "r")
    train_x = hdf5_store["train_x"]
    train_y = hdf5_store["train_y"]
    valid_x = hdf5_store["valid_x"]
    valid_y = hdf5_store["valid_y"]
    train_n_x = hdf5_store["train_n_x"]
    valid_n_x = hdf5_store["valid_n_x"]


def eval_env(env, model, test_steps=500, device="cpu", state_scaler=None, use_gru=False, deterministic=True):
    state = env.reset()
    state = np.stack(state)
    model.reset_hidden(batch_size=state.shape[0])
    if state_scaler is not None:
        state = state_scaler.transform(state)
    total_reward = []
    speeds = []
    accs = []
    n_ls = 0
    n_hard_brake = 0
    turn_rates = []
    jerks = []
    actions = []
    # states = []
    # states.append(state)
    collisions = 0
    total_n_traj = 0
    n_traj = 0
    total_driven_dist = 0
    driven_dist = 0
    ittc = []
    l_s = []
    d_a = []
    sba = 0
    sflc = 0
    for i in range(test_steps):
        state = torch.FloatTensor(state).to(device)
        dist, _ = model(state)
        # if deterministic:
        #     action = dist.mode().detach().clone().cpu().numpy()
        # else:
        action = dist.sample().clone().cpu().numpy()
        actions.append(action.copy())
        state, reward, done, infos = env.step(action)
        if state_scaler is not None:
            state = np.stack(state)
            state = state_scaler.transform(state)
        n_traj = 0
        driven_dist = 0
        for it in range(env.nenvs):
            info = infos[it]
            collision_occoured, tmp_speed, tmp_acc, tmp_n_ls, tmp_n_hard_brake, \
            tmp_turn_rates, tmp_jerks, n_traj_se, tmp_collisions, driven_dist_se, tmp_ittc, \
            tmp_l_s, tmp_d_a, tmp_sflc, tmp_sba = gather_information(info)
            speeds.extend(tmp_speed)
            accs.extend(tmp_acc)
            n_ls += tmp_n_ls
            # collisions += tmp_collisions
            n_hard_brake += tmp_n_hard_brake
            sflc += tmp_sflc
            sba += tmp_sba
            turn_rates.extend(tmp_turn_rates)
            jerks.extend(tmp_jerks)
            ittc.extend(tmp_ittc)
            l_s.extend(tmp_l_s)
            d_a.extend(tmp_d_a)
            total_reward.extend(reward)
            n_traj += n_traj_se
            driven_dist += driven_dist_se
            if collision_occoured:
                collisions += 1
                total_n_traj += n_traj
                total_driven_dist += driven_dist_se
        # if collision_occoured:
        #     collisions += 1
        #     state = env.reset()
        #     state = np.stack(state)
        #     total_n_traj += n_traj
        #     total_driven_dist += driven_dist
        # if False not in done:
        #     state = env.reset()
        #     state = np.stack(state)
        #     total_n_traj += n_traj
        #     total_driven_dist += driven_dist
    total_n_traj += n_traj
    total_driven_dist += driven_dist
    return np.sum(total_reward), speeds, accs, collisions, np.concatenate(actions, 0), \
           total_n_traj, n_ls, n_hard_brake, turn_rates, jerks, total_driven_dist, ittc, l_s, d_a, sflc, sba


def gather_information(info):
    speeds = info["speed"]
    accs = info["acceleration"]
    collision_occoured = info["ego_collision"]
    collisions = info["collision_ids"]
    lane_change_ids = info["lane_changes"]
    hard_brake_ids = info["hard_brake"]
    turn_rates = info["turn_rate"]
    jerks = info["jerk"]
    number_trajectories = info["n_traj"]
    driven_dist = info["total_dist"]
    ittc = info["ittc"]
    l_s = info["lane_and_speed"]
    d_a = info["dist_acc"]
    sflc_active = info["sflc_active"]
    sba_active = info["sba_active"]
    return collision_occoured, speeds, accs, len(lane_change_ids), \
           len(hard_brake_ids), turn_rates, jerks, number_trajectories, len(collisions), driven_dist, ittc, l_s, d_a, sflc_active, sba_active


class SumoJSD():
    def __init__(self, writer, clear_n_epoch=5, dir_feature='./data/', run_dir="./",
                 metric_file="highD_training_metrics.csv", expert_actions=None):
        self.writer = writer
        self.run_dir = run_dir
        self.csv_path = os.path.join(run_dir, metric_file)
        with open(self.csv_path, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["episode", "total_km", "n_lc", "hard_brake", "collisions",
                             "jsd_speed", "jsd_all_accs", "jsd_all_turn_rates",
                             "jsd_all_jerks", "jsd_all_ittc", "jsd_all_lane_and_speed", "jsd_all_dist_acc",
                             "sum_of_all_jsd", "jsd_all_act_accs", "jsd_all_act_turn_rates"])

        self.data_all_speeds = np.load(os.path.join(dir_feature, "all_speeds" + ".npy"))
        self.data_all_accs = np.load(os.path.join(dir_feature, "all_accs" + ".npy"))
        self.data_all_ittc = np.load(os.path.join(dir_feature, "all_ittc" + ".npy"))
        self.data_all_dist_acc = np.load(os.path.join(dir_feature, "all_dist_acc" + ".npy"))
        self.data_all_lane_and_speed = np.load(os.path.join(dir_feature, "all_lane_and_speed" + ".npy"))
        self.data_all_turn_rates = np.load(os.path.join(dir_feature, "all_turn_rates" + ".npy"))
        self.data_all_jerks = np.load(os.path.join(dir_feature, "all_jerks" + ".npy"))
        self.expert_actions = expert_actions
        self.speeds = []  # deque(maxlen=10000)
        self.accs = []  # deque(maxlen=10000)
        self.ittc = []  # deque(maxlen=10000)
        self.l_s = []  # deque(maxlen=10000)
        self.d_a = []  # deque(maxlen=10000)
        self.a_a = []
        self.a_lc = []
        self.turn_rates = []
        self.jerks = []
        self.collisions = 0
        self.n_lane_change = 0
        self.n_hard_brake = 0
        self.total_driven_dist = 0
        self.clear_n_epoch = clear_n_epoch

    def append_infos(self, infos):
        for info in infos:
            self.speeds.extend(info["speed"])
            self.accs.extend(info["acceleration"])
            self.turn_rates.extend(info["turn_rate"])
            self.ittc.extend(info["ittc"])
            self.l_s.extend(info["lane_and_speed"])
            self.d_a.extend(info["dist_acc"])
            self.n_lane_change += len(info["lane_changes"])
            self.n_hard_brake += len(info["hard_brake"])
            self.jerks.extend(info["jerk"])
            if info["ego_collision"]:
                self.collisions += 1

    def append_actions(self, actions):
        self.a_a.extend(actions[:, 0])
        self.a_lc.extend(actions[:, 1])

    def calculate_distance(self, infos):
        for info in infos:
            self.add_drive_dist(info["total_dist"])

    def add_drive_dist(self, distance):
        self.total_driven_dist += distance

    def clear_all(self):
        self.speeds = []
        self.accs = []
        self.turn_rates = []
        self.ittc = []
        self.l_s = []
        self.d_a = []
        self.jerks = []
        self.a_a = []
        self.a_lc = []
        self.collisions = 0

    def estimate_jsd(self, epoch):

        if self.total_driven_dist > 0:
            total_km = float(self.total_driven_dist) / 1000.0
            print("collision_rate: ", self.collisions / total_km)
            print("lane_change_rate: ", self.n_lane_change / total_km)
            self.writer.add_scalar('rl_training/lane_change_rate', self.n_lane_change / total_km, epoch)
            self.writer.add_scalar('rl_training/hard_brake_rate', self.n_hard_brake / total_km, epoch)
            self.writer.add_scalar('rl_training/collision_rate', self.collisions / total_km, epoch)
        else:
            total_km = 0

        if self.expert_actions is not None and len(self.a_a) > 0:
            jsd_all_act_accs = calculate_js_dis(self.expert_actions[:, 0], self.a_a)
            jsd_all_act_turn_rates = calculate_js_dis(self.expert_actions[:, 1], self.a_lc)
            print("jsd_all_act_accs: ", jsd_all_act_accs)
            print("jsd_all_act_turn_rates: ", jsd_all_act_turn_rates)
            self.writer.add_scalar('rl_training/jsd_all_act_accs', jsd_all_act_accs, epoch)
            self.writer.add_scalar('rl_training/jsd_all_act_turn_rates', jsd_all_act_turn_rates, epoch)
        else:
            jsd_all_act_accs = None
            jsd_all_act_turn_rates = None

        if len(self.speeds) > 0:
            jsd_speed = calculate_js_dis(self.data_all_speeds, self.speeds)
            jsd_all_accs = calculate_js_dis(self.data_all_accs, self.accs)
            jsd_all_turn_rates = calculate_js_dis(self.data_all_turn_rates, self.turn_rates)
            jsd_all_ittc = calculate_js_dis(self.data_all_ittc, self.ittc)
            jsd_all_lane_and_speed = calculate_js_dis(self.data_all_lane_and_speed, self.l_s)
            jsd_all_dist_acc = calculate_js_dis(self.data_all_dist_acc, self.d_a)
            jsd_all_jerks = calculate_js_dis(self.data_all_jerks, self.jerks)

            sum_of_all_jsd = jsd_speed + jsd_all_accs + jsd_all_turn_rates + \
                jsd_all_ittc + jsd_all_lane_and_speed + jsd_all_dist_acc + jsd_all_jerks

            print("jsd_speed: ", jsd_speed)
            print("jsd_all_accs: ", jsd_all_accs)
            print("jsd_all_turn_rates: ", jsd_all_turn_rates)
            print("jsd_all_ittc: ", jsd_all_ittc)
            print("jsd_all_dist_acc: ", jsd_all_dist_acc)
            print("jsd_all_lane_and_speed: ", jsd_all_lane_and_speed)
            print("jsd_all_jerks: ", jsd_all_jerks)
            print("sum_of_all_jsd: ", sum_of_all_jsd)
            self.writer.add_scalar('rl_training/jsd_speed', jsd_speed, epoch)
            self.writer.add_scalar('rl_training/jsd_all_accs', jsd_all_accs, epoch)
            self.writer.add_scalar('rl_training/jsd_all_turn_rates', jsd_all_turn_rates, epoch)
            self.writer.add_scalar('rl_training/jsd_all_ittc', jsd_all_ittc, epoch)
            self.writer.add_scalar('rl_training/jsd_all_dist_acc', jsd_all_dist_acc, epoch)
            self.writer.add_scalar('rl_training/jsd_all_lane_and_speed', jsd_all_lane_and_speed, epoch)
            self.writer.add_scalar('rl_training/jsd_all_jerks', jsd_all_jerks, epoch)
            self.writer.add_scalar('rl_training/sum_of_all_jsd', sum_of_all_jsd, epoch)

            if jsd_all_act_accs is None:
                self.write_csv(self.csv_path, epoch, total_km, self.n_lane_change, self.n_hard_brake, self.collisions,
                               jsd_speed, jsd_all_accs, jsd_all_turn_rates, jsd_all_jerks, jsd_all_ittc,
                               jsd_all_lane_and_speed, jsd_all_dist_acc, sum_of_all_jsd, sum_of_all_jsd, sum_of_all_jsd)
            else:
                self.write_csv(self.csv_path, epoch, total_km, self.n_lane_change, self.n_hard_brake, self.collisions,
                               jsd_speed, jsd_all_accs, jsd_all_turn_rates, jsd_all_jerks, jsd_all_ittc,
                               jsd_all_lane_and_speed, jsd_all_dist_acc, sum_of_all_jsd,
                               jsd_all_act_accs, jsd_all_act_turn_rates)

        if self.clear_n_epoch > 0 and epoch > 1 and epoch % self.clear_n_epoch == 0:
            self.clear_all()

    def calculate_kld(self, epoch):
        metric_file = "highD_test_kld_metrics.csv"
        csv_path = os.path.join(self.run_dir, metric_file)
        with open(csv_path, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["episode", "total_km", "n_lc", "hard_brake", "collisions",
                             "kld_speed", "kld_all_accs", "kld_all_turn_rates",
                             "kld_all_jerks", "kld_all_ittc", "kld_all_lane_and_speed", "kld_all_dist_acc",
                             "sum_of_all_kld", "kld_all_act_accs", "kld_all_act_turn_rates"])

        if self.expert_actions is not None and len(self.a_a) > 0 and len(self.speeds) > 0:
            kld_all_act_accs = calculate_kl_div(self.expert_actions[:, 0], self.a_a)
            kld_all_act_turn_rates = calculate_kl_div(self.expert_actions[:, 1], self.a_lc)
            print("kld_all_act_accs: ", kld_all_act_accs)
            print("kld_all_act_turn_rates: ", kld_all_act_turn_rates)

            kld_speed = calculate_kl_div(self.data_all_speeds, self.speeds)
            kld_all_accs = calculate_kl_div(self.data_all_accs, self.accs)
            kld_all_turn_rates = calculate_kl_div(self.data_all_turn_rates, self.turn_rates)
            kld_all_ittc = calculate_kl_div(self.data_all_ittc, self.ittc)
            kld_all_lane_and_speed = calculate_kl_div(self.data_all_lane_and_speed, self.l_s)
            kld_all_dist_acc = calculate_kl_div(self.data_all_dist_acc, self.d_a)
            kld_all_jerks = calculate_kl_div(self.data_all_jerks, self.jerks)

            sum_of_all_kld = kld_speed + kld_all_accs + kld_all_turn_rates + \
                kld_all_ittc + kld_all_lane_and_speed + kld_all_dist_acc + kld_all_jerks

            print("kld_speed: ", kld_speed)
            print("kld_all_accs: ", kld_all_accs)
            print("kld_all_turn_rates: ", kld_all_turn_rates)
            print("kld_all_ittc: ", kld_all_ittc)
            print("kld_all_dist_acc: ", kld_all_dist_acc)
            print("kld_all_lane_and_speed: ", kld_all_lane_and_speed)
            print("kld_all_jerks: ", kld_all_jerks)
            print("sum_of_all_kld: ", sum_of_all_kld)

            if self.total_driven_dist > 0:
                total_km = float(self.total_driven_dist) / 1000.0
            else:
                total_km = 0

            self.write_csv(csv_path, epoch, total_km, self.n_lane_change, self.n_hard_brake, self.collisions,
                           kld_speed, kld_all_accs, kld_all_turn_rates, kld_all_jerks, kld_all_ittc,
                           kld_all_lane_and_speed, kld_all_dist_acc, sum_of_all_kld,
                           kld_all_act_accs, kld_all_act_turn_rates)

    def write_csv(self, csv_path, episode, total_km, total_n_ls, total_N_hard_brake, all_collisions,
                  jsd_speed, jsd_all_accs, jsd_all_turn_rates, jsd_all_jerks, jsd_all_ittc,
                  jsd_all_lane_and_speed, jsd_all_dist_acc, sum_of_all_jsd,
                  jsd_all_act_accs, jsd_all_act_turn_rates):
        with open(csv_path, "a") as csvfile:
            writer = csv.writer(
                csvfile, delimiter=";", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            writer.writerow(["{:.6f}".format(episode),
                             "{:.6f}".format(total_km),
                             "{:.6f}".format(total_n_ls),
                             "{:.6f}".format(total_N_hard_brake),
                             "{:.6f}".format(all_collisions),
                             "{:.6f}".format(jsd_speed),
                             "{:.6f}".format(jsd_all_accs),
                             "{:.6f}".format(jsd_all_turn_rates),
                             "{:.6f}".format(jsd_all_jerks),
                             "{:.6f}".format(jsd_all_ittc),
                             "{:.6f}".format(jsd_all_lane_and_speed),
                             "{:.6f}".format(jsd_all_dist_acc),
                             "{:.6f}".format(sum_of_all_jsd),
                             "{:.6f}".format(jsd_all_act_accs),
                             "{:.6f}".format(jsd_all_act_turn_rates)])


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr, lr_min=1e-5):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    if lr < lr_min:
        lr = lr_min
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Necessary for KFAC implementation.
# See: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1).to(device=x.device)
        else:
            bias = self._bias.t().view(1, -1, 1, 1).to(device=x.device)

        return x + bias


class DiagStd(nn.Module):
    def __init__(self, num_outputs):
        super(DiagStd, self).__init__()
        self._bias = nn.Parameter(torch.eye(num_outputs))

    def forward(self, x):
        return x + self._bias


def calculate_kl_div(target, approx):
    min = np.min(target)
    max = np.max(target)
    eps = 1e-8

    target_hist, target_bin_edges = np.histogram(target, bins=np.arange(min, max, (max-min)/100), density=True)
    p = (target_hist+eps) * np.diff(target_bin_edges)

    approx_hist, approx_bin_edges = np.histogram(approx, bins=np.arange(min, max, (max - min) / 100), density=True)
    q = (approx_hist+eps) * np.diff(approx_bin_edges)

    kl_pq = rel_entr(p, q)
    return sum(kl_pq)


def calculate_js_dis(target, approx):
    min = np.min(target)
    max = np.max(target)
    eps = 1e-8

    target_hist, target_bin_edges = np.histogram(target, bins=np.arange(min, max, (max-min)/100), density=True)
    p = (target_hist+eps) * np.diff(target_bin_edges)

    approx_hist, approx_bin_edges = np.histogram(approx, bins=np.arange(min, max, (max - min) / 100), density=True)
    q = (approx_hist+eps) * np.diff(approx_bin_edges)

    js_pq = distance.jensenshannon(p, q)
    return js_pq


