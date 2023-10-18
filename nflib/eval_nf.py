import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from common import helper
from mpl_toolkits import mplot3d
from common.eval_policy import plot_2_dim_distributions


def state_parser(states):
    velocity = states[:, 2]
    acceleration = states[:, 3]
    distance_headway = states[:, 13]
    y_pos = states[:, 0]
    lateral_change = states[:, 7]

    velocity = np.stack(velocity)
    acceleration = np.stack(acceleration)
    distance_headway = np.stack(distance_headway)
    time_headway = distance_headway / (velocity + 1e-12)
    time_headway[velocity == 0] = 0.
    time_headway[distance_headway == 100.0] = 0.
    distance_headway[distance_headway == 100.0] = 0.
    velocity = velocity * 3.6
    y_pos = np.stack(y_pos)
    lateral_change = np.stack(lateral_change)

    return velocity, acceleration, distance_headway, time_headway, y_pos, lateral_change


def eval_state_ll_model(test_data_gen, state_model, output_dir,
                        env_name="Sumo_multi-v0", n_eval_steps=200, eval_bs=256,
                        do_forward=True, do_backward=True, calculate_jsd=False, grid_plot=False):
    # state_model.eval()
    device = state_model.device
    n_featrue = test_data_gen.n_feature
    print("Start evaluation...")
    state_ll = 0.
    if do_forward:
        print("run forward direction ...")
        state_ll_list = []
        all_base_flows = []
        all_base_samples = []
        for it in range(n_eval_steps):
            np_state, np_next_state, _ = test_data_gen.sample(eval_bs)

            if test_data_gen.is_state_conditional:
                pt_state = torch.tensor(np_state, dtype=torch.float).to(device)
                pt_next_state = torch.tensor(np_next_state, dtype=torch.float).to(device)
            else:
                pt_state = None
                pt_next_state = torch.tensor(np_state, dtype=torch.float).to(device)

            state_zs, state_prior_logprob, state_log_det = state_model(pt_next_state, pt_state)

            state_logprob = state_prior_logprob + state_log_det
            state_ll = state_logprob
            state_ll_list.extend(state_ll.detach().cpu().numpy())
            base_flow = state_zs[-1]
            all_base_flows.append(base_flow.detach().cpu().numpy())
            all_base_samples.append(state_model.sample_base(base_flow.shape[0]).cpu().numpy())

        print("run forward direction using random noise...")
        if test_data_gen.std_values is not None:
            std_values = test_data_gen.std_values
            std_values = std_values * np.ones((eval_bs, test_data_gen.n_feature))
            mean_values = test_data_gen.mean_values * np.ones((eval_bs, test_data_gen.n_feature))
        else:
            std_values = np.ones((eval_bs, test_data_gen.n_feature))
            mean_values = np.zeros((eval_bs, test_data_gen.n_feature))
        random_state_ll_list = []
        for it in range(n_eval_steps):
            if test_data_gen.is_state_conditional:
                random_o_state = np.clip(np.random.normal(scale=std_values), -1, 1) + mean_values
                random_state = torch.tensor(random_o_state, dtype=torch.float).to(device)
                random_f_state = torch.tensor(random_state, dtype=torch.float).to(device)
            else:
                random_state = None
                random_f_state = np.clip(np.random.normal(scale=std_values), -1, 1) + mean_values
                random_f_state = torch.tensor(random_f_state, dtype=torch.float).to(device)

            _, r_state_prior_logprob, r_state_log_det = state_model(random_f_state, random_state)
            r_state_logprob = r_state_prior_logprob + r_state_log_det
            random_state_ll = r_state_logprob
            random_state_ll_list.extend(random_state_ll.detach().cpu().numpy())

        print("evaluate forward direction ...")
        state_ll = np.mean(state_ll_list)
        print("State Log Likelihood: ", state_ll)

        random_state_ll = np.mean(random_state_ll_list)
        print("Random state Log Likelihood: ", random_state_ll)

        csv_path = os.path.join(output_dir, env_name + "_ll_" + ".csv")
        with open(csv_path, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["state Log Likelihood", "{:.6f}".format(state_ll)])
            writer.writerow(["random state Log Likelihood", "{:.6f}".format(random_state_ll)])

        if grid_plot and n_featrue == 2 and not test_data_gen.is_state_conditional:
            print("run forward direction using grid data...")
            grid_state_ll_list = []

            # low=[1.2, 0.1], high=[1.2, 0.1]
            x = np.linspace(-1.3, 0.7, num=1000)
            y = np.linspace(-0.08, 0.08, num=1000)

            xx, yy = np.meshgrid(x, y)

            for it in range(xx.shape[0]):

                state_x = torch.tensor(xx[it, :], dtype=torch.float).unsqueeze(1).to(device)
                state_y = torch.tensor(yy[it, :], dtype=torch.float).unsqueeze(1).to(device)
                state = torch.cat((state_x, state_y), dim=1)

                _, r_state_prior_logprob, r_state_log_det = state_model(state)
                grid_state_ll = r_state_prior_logprob + r_state_log_det
                grid_state_ll = torch.clamp(grid_state_ll, min=-10., max=100.)
                grid_state_ll_list.extend(grid_state_ll.unsqueeze(0).detach().cpu().numpy())

            grid_state_ll = np.asarray(grid_state_ll_list, dtype="float32")

            fig = plt.figure()
            ax = plt.axes(projection='3d')

            ax.scatter(xx, yy, grid_state_ll, c=grid_state_ll, cmap='viridis', linewidth=0.5)
            ax.set_title('state Likelihood')
            ax.set_xlabel('x')
            ax.set_ylabel('v')
            ax.set_zlabel('Log Likelihood')
            ax.view_init(60, 35)

            plt.title('state log likelihoods')
            plt.savefig(os.path.join(output_dir, env_name + "grid_state_ll_distribution_3d" + ".png"))
            plt.close()

            plt.contourf(xx, yy, grid_state_ll, cmap='RdGy')
            plt.colorbar()

            plt.title('state log likelihoods')
            plt.savefig(os.path.join(output_dir, env_name + "grid_state_ll_distribution" + ".png"))
            plt.close()

            plt.imshow(grid_state_ll, cmap='hot', interpolation='nearest')
            plt.title('state log likelihoods')
            plt.savefig(os.path.join(output_dir, env_name + "grid_state_ll_distribution_hm" + ".png"))
            plt.close()

    if do_backward:
        print("run backward direction ...")
        all_gen_states = []
        all_eval_states = []
        for it in range(n_eval_steps):
            eval_start_states, eval_next_states, _ = test_data_gen.sample(eval_bs)

            pt_eval_start_states = torch.tensor(
                eval_start_states, dtype=torch.float).to(state_model.device)

            states_values = state_model.sample(eval_bs, pt_eval_start_states)
            state = states_values[-1].detach().cpu().numpy()
            if not np.isnan(state).any():
                # prev_state = process_nf_states(prev_state)
                # eval_start_states = process_nf_states(eval_start_states)
                all_gen_states.append(state)
                all_eval_states.append(eval_next_states)
            else:
                print(np.isnan(state))
                for state_it in range(n_featrue):
                    if np.isnan(state[:, state_it]).any():
                        print("NaN in " + helper.features[state_it])

        print("evaluate backward direction ...")
        all_gen_states = np.concatenate(all_gen_states, 0)
        all_eval_states = np.concatenate(all_eval_states, 0)
        if all_gen_states.shape[0] != all_eval_states.shape[0]:
            print("mismatch in evaluation sizes!!")
            print(all_gen_states.shape)
            print(all_eval_states.shape)
        # all_gen_states = np.clip(all_gen_states, -1, 1)

        velocity, acceleration, distance_headway, time_headway, y_pos, lateral_change = state_parser(all_gen_states)
        plot_2_dim_distributions(output_dir, velocity, acceleration, distance_headway, time_headway, y_pos,
                                 lateral_change, filename="expert_model_generated_pairplot")
        print("Pairplot of expert model, done!")

        velocity, acceleration, distance_headway, time_headway, y_pos, lateral_change = state_parser(all_eval_states)
        plot_2_dim_distributions(output_dir, velocity, acceleration, distance_headway, time_headway, y_pos,
                                 lateral_change, filename="eval_data_pairplot")
        print("Pairplot of expert model, done!")

        if n_featrue == 2:
            plt.figure(figsize=(6, 8))
            plt.scatter(all_gen_states[:, 0], all_gen_states[:, 1], c='g', s=5)
            # plt.scatter(all_eval_states[:, 0], all_eval_states[:, 1], c='r', s=5)
            plt.legend(['Generated'])  # , 'valid'
            plt.axis('scaled')
            plt.title('states')
            plt.savefig(os.path.join(output_dir, env_name + "generated_state_distribution" + ".png"))
            plt.close()
        else:
            for state_it in range(n_featrue):
                bin_edges_gen = np.histogram_bin_edges(all_gen_states[:, state_it], bins='auto')
                bin_edges = bin_edges_gen
                f, axes = plt.subplots(2, 1, figsize=(14, 14), sharex=True, sharey=True)
                plt.grid(b=True)
                axes[0].set_title('Data')
                axes[0].grid(b=True)
                sns.distplot(all_eval_states[:, state_it], bins=bin_edges, color="skyblue", ax=axes[0],  #
                             label="Data", norm_hist=True)
                axes[1].set_title('Generated')
                axes[1].grid(b=True)
                sns.distplot(all_gen_states[:, state_it], bins=bin_edges, color="olive", ax=axes[1],   # , bins=bin_edges
                             label="Generated", norm_hist=True)
                plt.savefig(os.path.join(output_dir, env_name + "_data_eval_" + str(state_it)
                                         + "_" + helper.features[state_it] + ".png"))
                plt.close()

        if calculate_jsd:
            state_jsd = []
            for state_it in range(n_featrue):
                jsd = helper.calculate_js_dis(all_eval_states[:, state_it],
                                       all_gen_states[:, state_it])
                if jsd >= 0.:
                    state_jsd.append(jsd)
                else:
                    state_jsd.append(1)
                    print("found wrong state metric! ", jsd)
            print("mean jsd all states: ", state_jsd)
            print("total mean jsd: ", np.mean(state_jsd))

    return state_ll_list


def eval_state_ll_model_and_policy_data(nf_test_data_gen, predeccessor_traj_generator, state_model, output_dir,
                                        env_name="Sumo_multi-v0", n_eval_steps=200, eval_bs=256,
                                        do_forward=True, iteration=0, plot_states=False):
    # state_model.eval()
    device = state_model.device
    n_featrue = nf_test_data_gen.n_feature
    print("Start evaluation...")
    if do_forward:
        print("run forward direction ...")
        state_ll_list = []
        all_data_states = []
        for it in range(n_eval_steps):
            start_state, _, _ = nf_test_data_gen.sample(eval_bs)
            start_state = torch.tensor(start_state, dtype=torch.float).to(device)
            future_state = None  

            # state model
            state_zs, state_prior_logprob, state_log_det = state_model(start_state,
                                                                       future_state)
            state_logprob = state_prior_logprob + state_log_det
            state_ll = state_logprob
            state_ll_list.extend(state_ll.detach().cpu().numpy())
            all_data_states.extend(start_state.detach().cpu().numpy())

        print("run forward direction with policy data...")
        pol_state_ll_list = []
        all_pol_states = []
        for it in range(n_eval_steps):
            start_state, _, _ = predeccessor_traj_generator.sample(eval_bs)
            start_state = torch.tensor(start_state, dtype=torch.float).to(device)
            future_state = None

            # state model
            state_zs, state_prior_logprob, state_log_det = state_model(start_state,
                                                                       future_state)
            state_logprob = state_prior_logprob + state_log_det
            state_ll = state_logprob
            pol_state_ll_list.extend(state_ll.detach().cpu().numpy())
            all_pol_states.extend(start_state.detach().cpu().numpy())

        print("evaluate forward direction ...")
        state_ll = np.mean(state_ll_list)
        print("State Log Likelihood: ", state_ll)

        pol_state_ll = np.mean(pol_state_ll_list)
        print("Policy state Log Likelihood: ", pol_state_ll)

        np.save(os.path.join(output_dir, env_name + "_data_ll_" + str(iteration) + ".npy"), np.array(state_ll_list))
        np.save(os.path.join(output_dir, env_name + "_pol_ll_" + str(iteration) + ".npy"), np.array(pol_state_ll_list))

        f, axes = plt.subplots(2, 1, figsize=(14, 14))
        plt.grid(b=True)
        axes[0].set_title('Data')
        axes[0].grid(b=True)
        sns.distplot(state_ll_list, color="blue", ax=axes[0], 
                     label="Data", norm_hist=True, kde=False)
        axes[1].set_title('policy')
        axes[1].grid(b=True)
        sns.distplot(pol_state_ll_list, color="purple", ax=axes[1],
                     label="Policy", norm_hist=True, kde=False)
        plt.savefig(os.path.join(output_dir, env_name + "_r_d_p_state_ll_" + str(iteration) + ".png"))
        plt.close()

        csv_path = os.path.join(output_dir, env_name + "_ll_" + str(iteration) + ".csv")
        with open(csv_path, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["state Log Likelihood", "{:.6f}".format(state_ll)])
            writer.writerow(["policy state Log Likelihood", "{:.6f}".format(pol_state_ll)])

        if plot_states:
            all_pol_state_ll = np.stack(pol_state_ll_list)
            all_data_states = np.stack(all_data_states)
            all_pol_states = np.stack(all_pol_states)
            all_pol_states = np.clip(all_pol_states, -1, 1)
            selected_pol_states = all_pol_states[all_pol_state_ll < -200, :]
            print("Amount of selected pol states ", selected_pol_states.shape)
            for state_it in range(n_featrue):
                bin_edges_eval = np.histogram_bin_edges(all_data_states[:, state_it], bins='auto')
                bin_edges_gen = np.histogram_bin_edges(all_pol_states[:, state_it], bins='auto')
                if bin_edges_eval.shape[0] < bin_edges_gen.shape[0]:
                    bin_edges = bin_edges_eval
                    del bin_edges_gen
                else:
                    bin_edges = bin_edges_gen
                    del bin_edges_eval
                f, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True, sharey=True)
                plt.grid(b=True)
                axes[0].set_title('Data')
                axes[0].grid(b=True)
                sns.distplot(all_data_states[:, state_it], bins=bin_edges, color="skyblue", ax=axes[0],
                             label="Data", norm_hist=True)  # , kde=False
                axes[1].set_title('Generated')
                axes[1].grid(b=True)
                sns.distplot(all_pol_states[:, state_it], bins=bin_edges, color="olive", ax=axes[1],
                             label="Generated", norm_hist=True)  # , kde=False
                axes[2].set_title('Selected')
                axes[2].grid(b=True)
                sns.distplot(selected_pol_states[:, state_it], bins=bin_edges, color="purple", ax=axes[2],
                             label="Selected", norm_hist=True)
                plt.savefig(os.path.join(output_dir, env_name + "_sel_pol_data_features_" + str(state_it)
                                         + "_" + helper.features[state_it] + ".png"))
                plt.close()

