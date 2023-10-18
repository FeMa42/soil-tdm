import os
import numpy as np
import pickle
import torch
from nflib import rollout_storage



def load_gen_data(output_dir, env_name, nf_trajectorie_generator):
    with open(os.path.join(output_dir, env_name + ".pkl"), 'rb') as fp:
        finished_trajectories = pickle.load(fp)
    try:
        nf_trajectorie_generator.push(finished_trajectories)
    except:
        raise FileNotFoundError()


def gen_data(n_data_gen_steps, envs, model, device='cpu', max_horizon=250, sumo_jsd_calc=None,
             test_start=0):
    '''
    generate training data using sumo gym
    :param n_data_gen_steps:
    :return:
    '''
    data_frames = 0
    sumo_jsd_calc.clear_all()

    state = envs.reset()
    model.reset_hidden(batch_size=state.shape[0])

    state_pt = torch.FloatTensor(state).to(device)
    finished_trajectories = []
    storages = []
    state_list = envs.unpack_vector(state_pt)
    horizon_counter = 0
    for it in range(envs.nenvs):
        storages.append(rollout_storage.RolloutStorageGPRIL(len(state_list[it])))

    while data_frames <= n_data_gen_steps:
        dist, value = model(state_pt)
        action = dist.sample()
        action_sumo = action.clone().cpu().numpy()
        # action_sumo = np.around(action_sumo, decimals=2)
        next_state, sumo_rws, dones, infos = envs.step(action_sumo)
        # log_prob = dist.log_prob(action)

        state_list = envs.unpack_vector(state_pt)
        action_list = envs.unpack_vector(action)
        done_list = envs.unpack_vector(dones)

        sumo_jsd_calc.append_infos(infos)
        sumo_jsd_calc.append_actions(action_sumo)

        if horizon_counter >= max_horizon:
            finish = True
            horizon_counter = 0
            next_state = envs.reset()
            model.reset_hidden(batch_size=next_state.shape[0])
        else:
            finish = False
            horizon_counter += 1

        for it in range(envs.nenvs):
            storages[it].append_experience(state_list[it],
                                           action_list[it],
                                           done_list[it])
            if "reset" in infos[it] or finish:
                sumo_jsd_calc.add_drive_dist(infos[it]["total_dist"])
                storages[it].finish_trajectories()
                envs.get_n_agents()
                n_agents = len(envs.unpack_vector(next_state)[it])
                storages[it].reset(n_agents)

        state_pt = torch.FloatTensor(next_state).to(device)
        data_frames += 1

        if data_frames > n_data_gen_steps:
            sumo_jsd_calc.calculate_distance(infos)

    for it in range(envs.nenvs):
        finished_trajectories.extend(storages[it].finish_trajectories())

    print("Generated trajectories: ", len(finished_trajectories))
    sumo_jsd_calc.estimate_jsd(test_start+1)

    return finished_trajectories


def safe_data(finished_trajectories, output_dir, env_name):
    filename = os.path.join(output_dir, env_name + ".pkl")
    with open(filename, 'wb') as fp:
        pickle.dump(finished_trajectories, fp)

