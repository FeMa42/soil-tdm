import os
import numpy as np
import csv
import random
import gym
try:
    import gym_sumo
except:
    pass
import torch
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from common import sumo_transition_model

from common import helper
from nflib.data_generator import gen_data

def single_eval_env(env, actor_critic, action_scale=None, visualize=False, max_env_steps=2000,
                    use_additional_normal=False, state_est_model=None, state_cond_ll_model=None,
                    forward_backward_cond_ll_model_form=None, state_mod_function=None):
    '''Evaluate arbitrary OPENAI Gym Environment by collecting its reward'''
    device = actor_critic.device
    is_discrete = False
    if env.action_space.__class__.__name__ == "Discrete":
        is_discrete = True
    state = env.reset()
    if visualize:
        env.render()
    done = False
    total_reward = 0
    env_steps = 0
    actions = []
    states = []
    next_states = []
    mod_ep_reward, expert_ep_reward = 0., 0.
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = actor_critic(state)
        action_sampled = dist.sample()
        action_pol_log_prob = dist.log_prob(action_sampled)
        if is_discrete:
            action = action_sampled.squeeze(-1)
        else:
            action = action_sampled
            if use_additional_normal:
                action = action[:, 0].unsqueeze(1)
            if action_scale is not None:
                action = torch.clamp(
                    action, -action_scale, action_scale)

        action_gym = action.cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action_gym)
        if state_mod_function is not None:
            next_state = state_mod_function(next_state)
        state_np = state.clone().detach().cpu().numpy()
        states.append(state_np)
        next_states.append(np.expand_dims(next_state, axis=0))
        actions.append(action_sampled.clone().detach().cpu().numpy())

        if visualize: env.render()
        state = next_state
        total_reward += reward
        env_steps += 1
        if env_steps > max_env_steps:
            done = True

    if state_est_model is not None:
        state_vector = torch.FloatTensor(
            np.concatenate(states, axis=0)).to(device)
        next_state_vector = torch.FloatTensor(
            np.concatenate(next_states, axis=0)).to(device)
        action_vector = torch.FloatTensor(
            np.concatenate(actions, axis=0)).to(device)

        _, state_prior_logprob, state_log_det = state_est_model(
            next_state_vector, state_vector)
        expert_reward = (state_prior_logprob +
                         state_log_det).clone().detach().unsqueeze(1)
        if state_cond_ll_model is not None:
            expert_ep_reward = state_cond_ll_model.clamp_expert(expert_reward.clone())
        else:
            expert_ep_reward = expert_reward.clone()
        expert_ep_reward = expert_ep_reward.sum().detach().cpu().numpy()
        if state_cond_ll_model is not None:
            mod_reward = state_cond_ll_model.estimate_mod_reward(next_state_vector, state_vector, action_vector,
                                                                 expert_reward, action_pol_log_prob)
            mod_ep_reward = mod_reward.sum().detach().cpu().numpy()

        elif forward_backward_cond_ll_model_form is not None: 
            _, policy_state_prior_logprob, policy_state_log_det = forward_backward_cond_ll_model_form(
                next_state_vector, state_vector)
            policy_state_ll = (policy_state_prior_logprob +
                               policy_state_log_det).clone().detach().unsqueeze(1)

            mod_reward = expert_reward - policy_state_ll
            mod_ep_reward = mod_reward.sum().detach().cpu().numpy()

    return total_reward, mod_ep_reward, expert_ep_reward


def eval_env(env, actor_critic, action_scale=None, visualize=False, max_env_steps=2000, use_additional_normal=False,
             output_dir="./", epoch=1, state_est_model=None, state_cond_ll_model=None, state_mod_function=None):
    '''Evaluate arbitrary OPENAI Gym Environment by collecting its reward'''
    device = actor_critic.device
    determistic = actor_critic.has_deterministic
    is_discrete = False
    if env.action_space.__class__.__name__ == "Discrete":
        is_discrete = True
    if visualize:
        env.render()
    state = env.reset()
    if visualize:
        env.render()
    done = False
    total_reward = 0
    env_steps = 0
    actions = []
    states = []
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = actor_critic(state, deterministic=determistic)
        action = dist.sample()
        if is_discrete:
            # _, action = action.max(dim=1)
            action = action.squeeze(-1)
        else:
            if use_additional_normal:
                action = action[:, 0].unsqueeze(1)
            if action_scale is not None:
                action = torch.clamp(action, -action_scale, action_scale)
        action_gym = action.cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action_gym)
        if state_mod_function is not None:
            next_state = state_mod_function(next_state)
        states.append(next_state)
        actions.append(action_gym)
        if state_cond_ll_model is not None and state_est_model is not None:
            next_state_ll = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            mod_reward = state_cond_ll_model.estimate_mod_reward(next_state_ll, state, action, state_est_model)
            _, state_prior_logprob, state_log_det = state_est_model(next_state, state)
            expert_reward = (state_prior_logprob + state_log_det).clone().detach().unsqueeze(1)
        else:
            mod_reward, expert_reward = 0., 0.

        if visualize:
            env.render()
        state = next_state
        total_reward += reward
        env_steps +=1
        if env_steps > max_env_steps:
            done = True

    expert_states = np.asarray(states, dtype="float32")
    if expert_states.shape[1] == 2:
        plt.figure(figsize=(6, 8))
        plt.scatter(expert_states[:, 0], expert_states[:, 1], c='g', s=5)
        plt.legend(['train'])
        plt.axis('scaled')
        plt.title('states')
        plt.savefig(os.path.join(output_dir, "test_policy_state_distribution_" + str(epoch) + ".png"))
        plt.close()

    return total_reward  #, actions


def eval_policy_actions(output_dir, env_name, envs, model, gen_eval_steps=1000, max_horizon=250, data_gen=None,
                        valid_state=None, valid_action=None, is_flow_policy=False, iteration=None,
                        sumo_jsd_calc=None, test_start=0):
    # model.eval()
    n_actions = envs.action_space.shape[0]

    if iteration is not None:
        output_dir = os.path.join(output_dir, "iter_"+str(iteration))
        os.mkdir(output_dir)

    # plot_val_series(output_dir, env_name, valid_state, valid_action, model, max_horizon=max_horizon)

    if is_flow_policy and valid_state is not None and valid_action is not None:
        valid_state = np.array(valid_state[:]).reshape(-1, valid_state.shape[-1])
        valid_action = np.array(valid_action[:]).reshape(-1, valid_action.shape[-1])
        plot_policy_2dim(valid_state, valid_action, output_dir, env_name, model, data_type="_test")
        plot_policy_distirbutions(valid_state, valid_action, output_dir, env_name, model, n_actions, data_type="_test")
        eval_policy_series(output_dir, env_name, envs, model, gen_eval_steps=gen_eval_steps, max_horizon=max_horizon,
                           data_gen=data_gen, action_exp=None, sumo_jsd_calc=sumo_jsd_calc, test_start=test_start)
    elif valid_state is not None and valid_action is not None:
        plot_policy_2dim(valid_state, valid_action, output_dir,
                         env_name, model, data_type="_test", is_flow=False)
        eval_policy_series(output_dir, env_name, envs, model, gen_eval_steps=gen_eval_steps, max_horizon=max_horizon,
                           data_gen=data_gen, action_exp=None, sumo_jsd_calc=sumo_jsd_calc, test_start=test_start)


def eval_policy_series(output_dir, env_name, envs, model, gen_eval_steps=1000, max_horizon=250, data_gen=None,
                        action_exp=None, sumo_jsd_calc=None, test_start=0):
    print("Running trajectorie evaluation... ")
    model.eval()
    n_featrue = envs.observation_space.shape[0]

    if action_exp is not None:
        plot_expert_traj(action_exp, output_dir, env_name)

    if data_gen is None:
        data_gen = gen_data(gen_eval_steps, envs, model, model.device, max_horizon=max_horizon,
                            sumo_jsd_calc=sumo_jsd_calc, test_start=test_start)

    # parse states to get velocity, acceleration, distance_headway, time_headway, y_pos, lateral_change
    velocity, acceleration, distance_headway, time_headway, y_pos, lateral_change = state_parser(data_gen, n_featrue,
                                                                                                 output_dir, env_name)

    plot_2_dim_distributions(output_dir, velocity, acceleration, distance_headway, time_headway, y_pos,
                             lateral_change)

    selected_trajs = torch.cat(data_gen).cpu().numpy()
    a = selected_trajs[:, n_featrue:]
    plt.figure(figsize=(6, 8))
    plt.scatter(a[:, 0], a[:, 1], c='r', s=5, alpha=0.5)
    plt.legend(['action', 'z->a'])
    plt.axis('scaled')
    plt.title('z -> a')
    plt.savefig(os.path.join(output_dir, env_name + "_simulated_2_dim_actions" + ".png"))
    plt.close()


def plot_single_traj(selected_traj, n_featrue, output_dir, env_name, sample):
    # selected_traj_1 = data_gen[sample].cpu().numpy()
    selected_y = selected_traj[:, n_featrue:]
    s_acc = selected_y[:, 0]
    s_lc = selected_y[:, 1]
    s_lat_pos = selected_traj[:, 0]

    f, axes = plt.subplots(3, 1, sharex=True)
    plt.grid(b=True)
    sns.set_style("darkgrid")
    axes[0].set_title('Longitudinal Action (Acceleration)')
    axes[0].grid(b=True)
    axes[0].plot(s_acc)
    axes[1].set_title('Lateral Action')
    axes[1].grid(b=True)
    axes[1].plot(s_lc)
    axes[2].set_title('Lateral Position')
    axes[2].grid(b=True)
    axes[2].plot(s_lat_pos)
    # plt.show()
    plt.savefig(os.path.join(output_dir, env_name + "_actions_series_" + str(sample) + ".png"))
    plt.close()


def state_parser(data_gen, n_featrue, output_dir, env_name):
    velocity, acceleration, distance_headway, time_headway, y_pos, lateral_change = [], [], [], [], [], []
    iteration = 0
    for sampled_trajs in data_gen:
        iteration += 1
        if iteration <= 40:
            plot_single_traj(sampled_trajs.cpu().numpy(), n_featrue, output_dir, env_name, iteration)
        velocity.extend(sampled_trajs[:, 2].cpu().numpy())
        acceleration.extend(sampled_trajs[:, 3].cpu().numpy())
        distance_headway.extend(sampled_trajs[:, 13].cpu().numpy())
        y_pos.extend(sampled_trajs[:, 0].cpu().numpy())
        lateral_change.extend(sampled_trajs[:, 7].cpu().numpy())

    velocity = np.stack(velocity)
    acceleration = np.stack(acceleration)
    distance_headway = np.stack(distance_headway)
    time_headway = distance_headway / (velocity+1e-12)
    time_headway[velocity < 0.1] = 0.
    time_headway[distance_headway == 100.0] = 0.
    distance_headway[distance_headway == 100.0] = 0.
    velocity = velocity*3.6
    y_pos = np.stack(y_pos)
    lateral_change = np.stack(lateral_change)

    return velocity, acceleration, distance_headway, time_headway, y_pos, lateral_change


def plot_2_dim_distributions(output_dir, velocity, acceleration, distance_headway, time_headway, y_pos,
                             lateral_change, filename=None):
    stacked_info = np.stack(
            [velocity,
             acceleration,
             distance_headway,
             time_headway,
             y_pos,
             lateral_change], axis=1)
    stacked_dataframe = pd.DataFrame(stacked_info,
                                     columns=['velocity [km/h]', 'acceleration [m/s^2]', 'dhw [m]', 'thw [s]',
                                                  'y position [m]', 'lateral Change [m/des]'])

    plt.style.use('seaborn-darkgrid')
    sns.set(font_scale=8)
    grid = sns.PairGrid(stacked_dataframe, diag_sharey=False, height=15, aspect=1.0)
    grid.map_upper(sns.scatterplot, alpha=0.6, marker="+")
    grid.map_diag(sns.histplot, bins=50)
    grid.map_lower(sns.histplot, bins=50)
    if filename is None:
        filename = "Pairplot"
    plt.savefig(os.path.join(output_dir, filename + ".png"))
    sns.set()


def plot_expert_traj(action_exp, output_dir, env_name):
    for sample in range(40):
        fig, ax = plt.subplots()
        selected_y = action_exp[sample, :, :]
        # selected_y = selected_traj_1[:, n_featrue:]
        s_acc = selected_y[:, 0]
        s_lc = selected_y[:, 1]
        plt.scatter(s_acc, s_lc)
        plt.plot(s_acc, s_lc)
        for ts in range(s_acc.shape[0]):
            ax.annotate(ts, (s_acc[ts], s_lc[ts]))
        # plt.show()
        plt.savefig(os.path.join(output_dir, env_name + "_data_actions_" + str(sample) + ".png"))
        plt.close()

        f, axes = plt.subplots(2, 1, sharex=True)
        plt.grid(b=True)
        sns.set_style("darkgrid")
        axes[0].set_title('Acceleration')
        axes[0].grid(b=True)
        axes[0].plot(s_acc)
        axes[1].set_title('Lateral Change')
        axes[1].grid(b=True)
        axes[1].plot(s_lc)
        # plt.show()
        plt.savefig(os.path.join(output_dir, env_name + "_data_actions_series_" + str(sample) + ".png"))
        plt.close()


def plot_policy_2dim(data_state, data_action, output_dir, env_name, model, data_type="train", is_flow=True):
    print("Plot 2 dimensional scatter plot..")
    offset = 0
    num_samples = 5000 + offset
    all_data_y = data_action[offset:num_samples, :] + np.random.randn(*data_action[offset:num_samples, :].shape)*0.002
    all_data_x = data_state[offset:num_samples, :]
    data_y = torch.tensor(all_data_y, dtype=torch.float).to(
        model.device)
    data_x = torch.tensor(all_data_x, dtype=torch.float).to(
        model.device)
    y = all_data_y
    if is_flow:
        model.reset_hidden(num_samples)
        zs, log_probs = model.eval_action(data_y, data_x)
        z = zs[-1]
        # forward direction
        z = z.detach().cpu().numpy()
        # p = model.actor.sample_base(num_samples)
        p = model.actor_flow.sample_base(num_samples)
        p = p.cpu().numpy()
        plt.figure(figsize=(6, 8))
        plt.scatter(p[:, 0], p[:, 1], c='g', s=5)
        plt.scatter(z[:, 0], z[:, 1], c='r', s=5)
        plt.scatter(y[:, 0], y[:, 1], c='b', s=5)
        plt.legend(['base', 'a->z', 'action'])
        plt.axis('scaled')
        plt.title('a -> z')
        plt.savefig(os.path.join(output_dir, env_name + data_type + "_2_dim_base" + ".png"))
        plt.close()

    # zs = model.sample(128 * 8)
    model.reset_hidden(num_samples)
    dist, _ = model(data_x)
    a = dist.sample()
    a = a.detach().cpu().numpy()
    plt.figure(figsize=(6, 8))
    plt.scatter(y[:, 0], y[:, 1], c='b', s=5, alpha=0.5)
    plt.scatter(a[:, 0], a[:, 1], c='r', s=5, alpha=0.5)
    plt.legend(['action', 'z->a'])
    plt.axis('scaled')
    plt.title('z -> a')
    plt.savefig(os.path.join(output_dir, env_name + data_type + "_2_dim_actions" + ".png"))
    plt.close()


def plot_policy_distirbutions(data_state, data_action, output_dir, env_name, model, n_actions, data_type="train"):
    print("Evaluate distribution of policy outputs...")
    all_data_y = data_action[:100000, :] + np.random.randn(*data_action[:100000, :].shape)*0.002
    all_data_x = data_state[:100000, :]
    n_samples = all_data_y.shape[0]
    eval_bs = 200
    n_iter = int(n_samples / eval_bs)
    all_probs = []
    all_logprobs = []
    flow_actions = []
    for eval_iter in range(n_iter):
        model.reset_hidden(eval_bs)
        data_y = torch.tensor(all_data_y[eval_iter*eval_bs:(eval_iter+1)*eval_bs, :], dtype=torch.float).to(model.device)
        data_x = torch.tensor(all_data_x[eval_iter*eval_bs:(eval_iter+1)*eval_bs, :], dtype=torch.float).to(model.device)
        zs, log_probs = model.eval_action(data_y, data_x)
        all_probs.append(torch.clamp(torch.exp(log_probs), min=-10, max=10).detach().cpu())
        all_logprobs.append(log_probs.detach().cpu())
        model.reset_hidden(eval_bs)
        # action, _ = model.sample(data_x)
        dist, _ = model(data_x)
        action = dist.sample()

        flow_actions.append(action.detach().cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_logprobs = torch.cat(all_logprobs).numpy()
    flow_actions = torch.cat(flow_actions).numpy()

    action_name = {0: "Acceleration", 1: "LateralChange"}
    for action_it in range(n_actions):
        bin_edges_eval = np.histogram_bin_edges(all_data_y[:, action_it], bins='auto')
        try:
            bin_edges_gen = np.histogram_bin_edges(flow_actions[:, action_it], bins='auto')
        except MemoryError:
            print("Could not generate bins ")
            bin_edges_gen = None
        if bin_edges_gen is not None:
            if bin_edges_gen is not None and bin_edges_eval.shape[0] < bin_edges_gen.shape[0]:
                bin_edges = bin_edges_eval
                del bin_edges_gen
            else:
                bin_edges = bin_edges_gen
                del bin_edges_eval
        else:
            bin_edges = bin_edges_eval
        f, axes = plt.subplots(2, 1, figsize=(14, 14), sharex=True, sharey=True)
        plt.grid(b=True)
        axes[0].set_title('Data Actions')
        axes[0].grid(b=True)
        sns.distplot(all_data_y[:, action_it], bins=bin_edges, color="blue", ax=axes[0],
                     label="Data", norm_hist=True)
        axes[1].set_title('Flow Actions Data')
        axes[1].grid(b=True)
        sns.distplot(flow_actions[:, action_it], bins=bin_edges, color="olive", ax=axes[1],
                     label="Generated", norm_hist=True)
        plt.savefig(os.path.join(output_dir, env_name + data_type + "_action_" + action_name[action_it] + ".png"))
        plt.close()

    sns.distplot(all_probs[:], color="blue", label="log porb", norm_hist=True)
    plt.savefig(os.path.join(output_dir, env_name + data_type + "_probs" + ".png"))
    plt.close()
    sns.distplot(all_logprobs[:], color="blue", label="log porb", norm_hist=True)
    plt.savefig(os.path.join(output_dir, env_name + data_type + "_log_probs" + ".png"))
    plt.close()


def eval_policy(envs, env_name, model, output_dir, n_test=500):
    # env = gym.make(env_name)
    # seed_val = 2 * 49 + 1
    # env.seed(seed_val)
    test_steps = 250
    all_speeds = []
    all_accs = []
    all_actions = []
    all_collisions = 0
    n_traj = 0
    total_n_ls = 0
    total_N_hard_brake = 0
    total_driven_distance = 0
    all_turn_rates = []
    all_jerks = []
    all_ittc = []
    all_l_s = []
    all_d_a = []
    all_sflc = 0
    all_sba = 0
    for n in range(n_test):
        _, speeds, accs, collisions, actions, n_agents, n_ls, \
        n_hard_brake, turn_rates, jerks, driven_dist, ittc, l_s, d_a, sflc, sba = helper.eval_env(envs, model,
                                                                                                       test_steps=test_steps,
                                                                                                       device=model.device,
                                                                                                       state_scaler=None)
        all_speeds.extend(speeds)
        all_accs.extend(accs)
        all_actions.extend(actions)
        total_driven_distance += driven_dist
        all_collisions += collisions
        n_traj += n_agents
        total_n_ls += n_ls
        total_N_hard_brake += n_hard_brake
        all_sflc += sflc
        all_sba += sba
        all_turn_rates.extend(turn_rates)
        all_jerks.extend(jerks)
        all_ittc.extend(ittc)
        all_l_s.extend(l_s)
        all_d_a.extend(d_a)

    if len(all_actions) > 0:

        print("Tests done. Calculating metrics...")
        all_actions = np.stack(all_actions, 0)
        sns.distplot(all_actions[:, 0], kde=False, norm_hist=True)
        plt.savefig(os.path.join(output_dir, env_name + "_policy_acc" + ".png"))
        plt.close()
        sns.distplot(all_actions[:, 1], kde=False, norm_hist=True)
        plt.savefig(os.path.join(output_dir, env_name + "_policy_lc" + ".png"))
        plt.close()
        # Plot measurements
        sns.distplot(all_speeds, kde=False, norm_hist=True)
        plt.savefig(os.path.join(output_dir, env_name + "_speed" + ".png"))
        plt.close()
        sns.distplot(all_accs, kde=False, norm_hist=True)
        plt.savefig(os.path.join(output_dir, env_name + "_acc" + ".png"))
        plt.close()
        sns.distplot(all_turn_rates, kde=False, norm_hist=True)
        plt.savefig(os.path.join(output_dir, env_name + "_turn_rate" + ".png"))
        plt.close()
        sns.distplot(all_jerks, kde=False, norm_hist=True)
        plt.savefig(os.path.join(output_dir, env_name + "_jerk" + ".png"))
        plt.close()
        sns.distplot(all_ittc, kde=False, norm_hist=True)
        plt.savefig(os.path.join(output_dir, env_name + "_ittc" + ".png"))
        plt.close()
        all_l_s = np.stack(all_l_s)
        df_all_lane_and_speed = pd.DataFrame(data={"lane": all_l_s[:, 0], "speed": all_l_s[:, 1]})
        sns.violinplot(x="lane", y="speed", data=df_all_lane_and_speed)
        plt.savefig(os.path.join(output_dir, env_name + "_lane_speed" + ".png"))
        plt.close()
        all_d_a = np.stack(all_d_a)
        if all_d_a.shape[0] > 10000:
            n_samples = 10000
        else:
            n_samples = all_d_a.shape[0]-1
        df_dist_acc = pd.DataFrame(data={"distance": all_d_a[:, 0], "acceleration": all_d_a[:, 1]})
        sns.jointplot("distance", "acceleration", df_dist_acc.sample(n_samples), kind='kde')
        plt.savefig(os.path.join(output_dir, env_name + "_dist_acc" + ".png"))
        plt.close()

        total_km = float(total_driven_distance) / 1000.0
        print("Number of trajectories: ", n_traj)
        print("Rel number of lane changes: ", total_n_ls / n_traj)
        print("Hard brake rate: ", total_N_hard_brake / n_traj)
        print("collision rate: ", all_collisions / n_traj)
        print("collision km rate: ", all_collisions / total_km)
        print("Number of safety brakes: ", all_sba)
        print("Number lc violations: ", all_sflc)

        dir_feature = './data/'
        data_all_speeds = np.load(os.path.join(dir_feature, "all_speeds" + ".npy"))
        data_all_accs = np.load(os.path.join(dir_feature, "all_accs" + ".npy"))
        data_all_turn_rates = np.load(os.path.join(dir_feature, "all_turn_rates" + ".npy"))
        data_all_jerks = np.load(os.path.join(dir_feature, "all_jerks" + ".npy"))
        data_all_ittc = np.load(os.path.join(dir_feature, "all_ittc" + ".npy"))
        data_all_dist_acc = np.load(os.path.join(dir_feature, "all_dist_acc" + ".npy"))
        data_all_lane_and_speed = np.load(os.path.join(dir_feature, "all_lane_and_speed" + ".npy"))

        jsd_speed = helper.calculate_js_dis(data_all_speeds, all_speeds)
        jsd_all_accs = helper.calculate_js_dis(data_all_accs, all_accs)
        jsd_all_turn_rates = helper.calculate_js_dis(data_all_turn_rates, all_turn_rates)
        jsd_all_jerks = helper.calculate_js_dis(data_all_jerks, all_jerks)
        jsd_all_ittc = helper.calculate_js_dis(data_all_ittc, all_ittc)
        jsd_all_dist_acc = helper.calculate_js_dis(data_all_dist_acc, all_d_a)
        jsd_all_lane_and_speed = helper.calculate_js_dis(data_all_lane_and_speed, all_l_s)
        print("jsd_speed: ", jsd_speed)
        print("jsd_all_accs: ", jsd_all_accs)
        print("jsd_all_turn_rates: ", jsd_all_turn_rates)
        print("jsd_all_jerks: ", jsd_all_jerks)
        print("jsd_all_ittc: ", jsd_all_ittc)
        print("jsd_all_dist_acc: ", jsd_all_dist_acc)
        print("jsd_all_lane_and_speed: ", jsd_all_lane_and_speed)

        csv_path = os.path.join(output_dir, env_name  + "_metrics.csv")
        with open(csv_path, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["n_traj", "total_km", "n_lc", "hard_brake", "collisions",
                             "safety_brakes", "safety_lc_violantions",
                             "jsd_speed", "jsd_all_accs", "jsd_all_turn_rates",
                             "jsd_all_jerks", "jsd_all_ittc", "jsd_all_lane_and_speed", "jsd_all_dist_acc"])
            writer.writerow(["{:.6f}".format(n_traj),
                             "{:.6f}".format(total_km),
                             "{:.6f}".format(total_n_ls),
                             "{:.6f}".format(total_N_hard_brake),
                             "{:.6f}".format(all_collisions),
                             "{:.6f}".format(all_sba),
                             "{:.6f}".format(all_sflc),
                             "{:.6f}".format(jsd_speed),
                             "{:.6f}".format(jsd_all_accs),
                             "{:.6f}".format(jsd_all_turn_rates),
                             "{:.6f}".format(jsd_all_jerks),
                             "{:.6f}".format(jsd_all_ittc),
                             "{:.6f}".format(jsd_all_lane_and_speed),
                             "{:.6f}".format(jsd_all_dist_acc)])


def plot_val_series(output_dir, env_name, data_state, data_action, model, max_horizon=20):
    expert_states = data_state[:2000, :, :]
    expert_action = data_action[:2000, :, :]
    traj_length = max_horizon

    model.reset_hidden(expert_states.shape[0])
    data_x = torch.tensor(expert_states[:], dtype=torch.float).to(
        model.device)

    start_index = 5  # random.sample(range(0, data_x.shape[1] - (traj_length + 1)), 1)[0]
    policy_actions = []
    next_state = data_x[:, start_index, :]
    for time_step in range(traj_length):
        # index = start_index + time_step
        # data_next_state = data_x[:, index, :]
        # action, _ = model.sample(next_state)
        dist, _ = model(next_state)
        action = dist.sample()
        next_state = sumo_transition_model.forward_model(next_state, action, model.device)
        policy_actions.append(action.detach().cpu().unsqueeze(1))

    policy_actions = torch.cat(policy_actions, dim=1).numpy()

    # plot RMSE
    expert_action_pt = torch.tensor(expert_action[:], dtype=torch.float).cpu()
    policy_acc = policy_actions[:, :, 0]
    policy_lc = policy_actions[:, :, 1]
    expert_acc = expert_action_pt[:, 0:traj_length, 0]
    expert_lc = expert_action_pt[:, 0:traj_length, 1]
    error_acc = (expert_acc - policy_acc)**2
    error_acc = torch.mean(error_acc, dim=0)
    error_acc = torch.sqrt(error_acc)
    error_lc = torch.sqrt(torch.mean((expert_lc - policy_lc) ** 2, dim=0))
    f, axes = plt.subplots(2, 1, sharex='col')  # , figsize=(28, 14)
    plt.grid(b=True)
    sns.set_style("darkgrid")
    axes[0].set_title('Acceleration RMSE')
    axes[0].grid(b=True)
    axes[0].plot(error_acc)
    axes[1].set_title('Lateral Change RMSE')
    axes[1].grid(b=True)
    axes[1].plot(error_lc)
    # plt.show()
    plt.savefig(os.path.join(output_dir, env_name + "_policy_val_rmse_" + ".png"))
    plt.close()

    # Plot trajectory comparison
    for sample in range(40):
        s_acc = policy_actions[sample, :,  0]
        s_lc = policy_actions[sample, :,  1]
        fig, ax = plt.subplots()
        plt.scatter(s_acc, s_lc)
        plt.plot(s_acc, s_lc)
        for ts in range(s_acc.shape[0]):
            ax.annotate(ts, (s_acc[ts], s_lc[ts]))
        # plt.show()
        plt.savefig(os.path.join(output_dir, env_name + "_val_actions_" + str(sample) + ".png"))
        plt.close()
        s_acc_real = expert_action[sample, 0:max_horizon, 0]
        s_lc_real = expert_action[sample, 0:max_horizon, 1]

        f, axes = plt.subplots(4, 1, sharex='col')  #, figsize=(28, 14)
        plt.grid(b=True)
        sns.set_style("darkgrid")
        axes[0].set_title('Acceleration, Policy')
        axes[0].grid(b=True)
        axes[0].plot(s_acc)
        axes[1].set_title('Lateral Change, Policy')
        axes[1].grid(b=True)
        axes[1].plot(s_lc)
        axes[2].set_title('Acceleration, Expert')
        axes[2].grid(b=True)
        axes[2].plot(s_acc_real)
        axes[3].set_title('Lateral Change, Expert')
        axes[3].grid(b=True)
        axes[3].plot(s_lc_real)
        axes[3].set_xlabel('time steps')
        # plt.show()
        plt.savefig(os.path.join(output_dir, env_name + "_policy_val_action_series_" + str(sample) + ".png"))
        plt.close()

    # Plot distribution comparison
    actions = policy_actions.reshape((-1, policy_actions.shape[-1]))
    expert_action = expert_action.reshape((-1, expert_action.shape[-1]))
    plt.figure(figsize=(6, 8))
    plt.scatter(expert_action[:, 0], expert_action[:, 1], c='b', s=5)
    plt.scatter(actions[:, 0], actions[:, 1], c='r', s=5, alpha=0.5)
    plt.legend(['val action', 'sampled action'])
    plt.axis('scaled')
    plt.title('validation actions')
    plt.savefig(os.path.join(output_dir, env_name + "_val_2_dim_actions" + ".png"))
    plt.close()
