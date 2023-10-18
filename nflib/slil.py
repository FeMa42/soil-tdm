import os
import numpy as np
import torch
from common.helper import update_linear_schedule
from nflib.eval_nf import eval_state_ll_model
from rllib.sac import ReplayBuffer
import copy


def optimize_state_est_model(state_est_model, state_optimizer, np_state, np_next_state, clip, is_state_conditional=True):
    if is_state_conditional:
        pt_state = torch.tensor(np_state, dtype=torch.float).to(
                state_est_model.device)
        pt_next_state = torch.tensor(
                np_next_state, dtype=torch.float).to(state_est_model.device)

        state_zs, state_prior_logprob, state_log_det = state_est_model(
                pt_next_state, pt_state)
    else:
        future_state = torch.tensor(
                np_state, dtype=torch.float).to(state_est_model.device)
        state_zs, state_prior_logprob, state_log_det = state_est_model(
                future_state)

    state_logprob = state_prior_logprob + state_log_det
    state_loss = -torch.mean(state_logprob)  # NLL

    state_est_model.zero_grad()
    state_loss.backward()
    torch.nn.utils.clip_grad_norm_(state_est_model.parameters(), clip)
    state_optimizer.step()

    return state_loss, state_prior_logprob, state_log_det


def train_policy_state_est_model(state_cond_ll_model_form, replay_buffer, state_cond_ll_model_form_optimizer, 
                                 n_train=10, batch_size=256, next_state_noise=0.0005, clip=100):
    lp_ns_s_losses = []

    for k in range(n_train):
        np_state, _, _, np_next_state, _, _, _, _ = replay_buffer.sample(
            batch_size)
        np_next_state = np_next_state + \
            np.random.randn(*np_next_state.shape) * next_state_noise

        # optimize log p(s'|s)
        state_loss, _, _ = optimize_state_est_model(
            state_cond_ll_model_form, state_cond_ll_model_form_optimizer, np_state, np_next_state, clip,
            is_state_conditional=True)
        
        lp_ns_s_losses.append(state_loss.item())
    return np.mean(lp_ns_s_losses)


def train_expert_state_est_model(n_train, state_est_model, expert_train_generator, bc_batch_size, state_optimizer,
                          output_dir, env_name, clip, lr, writer=None, use_linear_lr_decay=False,
                          expert_test_generator=None, noise_value=0.002, use_noise_sched=False, 
                          final_noise_value=0.002, noise_red_factor=2., use_early_stopping=True):
    # train normalizing flow model
    state_est_model.train()
    best_test_ll = -np.inf
    noise_value_in = noise_value
    for k in range(n_train):
        if use_linear_lr_decay:
            update_linear_schedule(state_optimizer, k, n_train, lr, lr_min=5e-4)
        if use_noise_sched:
            noise_value_in = final_noise_value + \
                (1-min(1, (k*noise_red_factor)/n_train))*noise_value

        np_state, np_next_state, _ = expert_train_generator.sample(bc_batch_size,
                                                                   noise_value_y=noise_value_in,
                                                                   noise_value_x=noise_value_in)
        
        state_loss, state_prior_logprob, state_log_det = optimize_state_est_model(
            state_est_model, state_optimizer, np_state, np_next_state, clip, 
            is_state_conditional=expert_train_generator.is_state_conditional)

        if writer is not None:
            writer.add_scalar('training/state_loss', state_loss.item(), k)
            writer.add_histogram("state_prior_logprob",
                                 state_prior_logprob, global_step=k)
            writer.add_histogram("state_log_det", state_log_det, global_step=k)

        if k % 10 == 0:
            print("Step", k)
            print("Train state loss: ", state_loss.item())

        if k % 10 == 0 and k > 0:
            if expert_test_generator is not None:
                dataset_size = expert_test_generator.data_x.shape[0]
                if dataset_size > 250:
                    n_eval_steps = 100
                    eval_bs = 250
                else:
                    n_eval_steps = 20
                    eval_bs = dataset_size-2
                state_ll = eval_state_ll_model(expert_test_generator, state_est_model, output_dir,
                                               env_name=env_name, n_eval_steps=n_eval_steps, eval_bs=eval_bs,
                                               do_forward=True, do_backward=False, calculate_jsd=True)
                state_ll = np.mean(state_ll)
                if writer is not None:
                    writer.add_scalar('training/state_ll', state_ll, k)
                if best_test_ll < state_ll:
                    best_test_ll = state_ll
                    if state_est_model.freia_inn:
                        state_est_model.flow.save(os.path.join(output_dir, env_name + "_" + "best" + "_state_d.pt"))
                    else:
                        torch.save({'state_model': state_est_model.state_dict()},
                                   os.path.join(output_dir, env_name + "_" + "best" + "_state_d.pt"))
                elif use_early_stopping and state_ll < -1. and k > 200:
                    break

            if state_est_model.freia_inn:
                state_est_model.flow.save(os.path.join(output_dir, env_name + "_state_d.pt"))

    if n_train > 0:
        if state_est_model.freia_inn:
            state_est_model.flow.save(os.path.join(output_dir, env_name + "_state_d.pt"))
            state_est_model.flow.save(os.path.join(output_dir, env_name + "_state_d.pt"))
        else:
            torch.save({'state_model': state_est_model.state_dict()},
                       os.path.join(output_dir, env_name + "_state_d.pt"))


def update_policy_forward_backward_models(state_cond_ll, replay_buffer, actor_critic, target_forward_backward_cond_ll_model, 
                                          n_train=10, n_train_a_fac = 3, batch_size=256, update_a = True, 
                                          importance_alpha=0, next_state_noise=0.0005, max_nll=10.):
    lp_ns_a_s_losses = []
    lp_ns_s_losses = []
    lp_a_ns_s_losses = []
    lp_a_s_losses = []
    device = state_cond_ll.a_ns_s_model.device
    for k in range(n_train):
        state, action, _, next_state, _, _, _, _ = replay_buffer.sample_train(batch_size)
        next_state = next_state + \
            np.random.randn(*next_state.shape) * next_state_noise

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).float().to(device)

        # optimize log p(s'|a,s)
        lp_ns_a_s_losses.append(state_cond_ll.optimize_step_lp_ns_given_a_s(state, action, next_state))
    if update_a:
        if importance_alpha > 0:
            for k in range(n_train*n_train_a_fac):
                state, action, _, next_state, _, _, _ = replay_buffer.sample_train(
                    batch_size)
                next_state = next_state + \
                    np.random.randn(*next_state.shape) * next_state_noise

                state = torch.FloatTensor(state).to(device)
                next_state = torch.FloatTensor(next_state).to(device)
                action = torch.FloatTensor(action).float().to(device)#

                # optimize log p(a|s)
                lp_a_s_losses.append(state_cond_ll.optimize_step_lp_a_given_s(state, action))
        for k in range(n_train*n_train_a_fac):
            state, action, _, next_state, _, _, _, _ = replay_buffer.sample_train(
                batch_size)
            next_state = next_state + \
                np.random.randn(*next_state.shape) * next_state_noise

            state = torch.FloatTensor(state).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            action = torch.FloatTensor(action).float().to(device)
            if importance_alpha > 0:
                log_w = (actor_critic.model.forward_action(state, action)[
                         :, 0] - state_cond_ll.estimete_lp_a_given_s(state, action))*importance_alpha 
                log_w = log_w - torch.logsumexp(log_w,0)
                w = torch.exp(log_w).detach()
            else:
                w = 1 / state.shape[0]

            #optimize log p(a|s',s)
            lp_a_ns_s_losses.append(state_cond_ll.optimize_step_lp_a_given_ns_s(state, action, next_state, w = w))

    # estimate model performance on test set:
    ns_test_losses = []
    a_test_losses = []
    target_ns_test_losses = []
    target_a_test_losses = []
    with torch.no_grad():
        for k in range(n_train):
            state, action, _, next_state, _, _, _, _ = replay_buffer.sample_test(
                    batch_size)
            next_state = next_state + \
                    np.random.randn(*next_state.shape) * next_state_noise

            state = torch.FloatTensor(state).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            action = torch.FloatTensor(action).float().to(device)

            ns_given_a_s_test_loss, a_given_ns_s_test_loss = state_cond_ll.test_both_model(
                    state, action, next_state, max_nll=max_nll)
            ns_test_losses.append(ns_given_a_s_test_loss.item())
            a_test_losses.append(a_given_ns_s_test_loss.item())

            target_ns_given_a_s_test_loss, target_a_given_ns_s_test_loss = target_forward_backward_cond_ll_model.test_both_model(
                state, action, next_state, max_nll=max_nll)
            target_ns_test_losses.append(target_ns_given_a_s_test_loss.item())
            target_a_test_losses.append(target_a_given_ns_s_test_loss.item())

    mean_ns_given_a_s_test_loss = np.mean(ns_test_losses)
    mean_a_given_ns_s_test_loss = np.mean(a_test_losses)
    mean_target_ns_given_a_s_test_loss = np.mean(target_ns_test_losses)
    mean_target_a_given_ns_s_test_loss = np.mean(target_a_test_losses)

    if len(lp_ns_s_losses) == 0.:
        lp_ns_s_losses.append(0.)
    return np.mean(lp_ns_a_s_losses), np.mean(lp_a_ns_s_losses), np.mean(lp_ns_s_losses), \
           np.mean(lp_a_s_losses), mean_ns_given_a_s_test_loss, mean_a_given_ns_s_test_loss, \
           mean_target_ns_given_a_s_test_loss, mean_target_a_given_ns_s_test_loss


class RolloutStorageSAC(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.masks = []
        self.step = 0

    def insert(self, states, actions, rewards, masks, next_states):
        self.states.append(states)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.masks.append(masks)
        self.next_states.append(next_states)
        self.step += 1

    def fill_replay_buffer(self, actor_critic_optimizer, additional_buffer=None):
        if len(self.next_states) > 0:
            next_actions = self.actions[1:]
            next_actions.append(torch.zeros(self.next_actions[-1].shape))
            next_next_states = self.next_states[1:]
            next_next_states.append(torch.zeros(self.next_states[-1].shape))
            actor_critic_optimizer.add_to_replay_buffer(self.states, self.actions,
                                                        self.rewards, self.next_states,
                                                        self.masks, next_actions, next_next_states)
            if additional_buffer is not None:
                additional_buffer.add_to_buffer(self.states, self.actions,
                                                self.rewards, self.next_states,
                                                self.masks, next_actions, next_next_states)
        self.clear()

    def clear(self):
        self.states = []
        self.next_states = []
        self.rewards = []
        self.actions = []
        self.masks = []
        self.step = 0


def generate_data_sumo(state, n_data_gen_steps, actor_critic, envs, actor_critic_optimizer, sumo_jsd_calc,
                       state_est_model=None, normalize_logprob=10., max_nll=10.):
    device = actor_critic.device
    sim_steps = 0
    data_frames = 0
    env_rewards = []
    rewards = []
    state_cond_ll_buffer = ReplayBuffer(100000)

    rollout_storages = []
    for it in range(envs.nenvs):
        rollout_storages.append(RolloutStorageSAC())

    for env_it in range(n_data_gen_steps):
        dist, value = actor_critic(state)
        action = dist.sample()
        action = torch.clamp(action, -actor_critic.act_limit, actor_critic.act_limit)
        if torch.isnan(action).any():
            print("found NaN in action!")
        next_state, env_reward, done, infos = envs.step(action.cpu().numpy())

        # estimate reward
        if state_est_model is not None:
            next_state_flow = next_state + np.random.randn(*next_state.shape) * 0.002
            next_state_flow = torch.tensor(next_state_flow, dtype=torch.float).to(actor_critic.device)
            _, state_prior_logprob, state_log_det = state_est_model(next_state_flow, state)
            state_logprob = state_prior_logprob + state_log_det
            state_eval = state_logprob.detach()
            reward = torch.clamp(state_eval, min=-max_nll, max=1e9)
            reward = reward / normalize_logprob
        else:
            reward = torch.FloatTensor(env_reward).cpu()

        # reward
        env_rewards.append(torch.FloatTensor(env_reward).cpu())
        rewards.append(reward.clone().cpu())

        finished_env = False
        data_frames += next_state.shape[0]
        sim_steps += 1
        sumo_jsd_calc.append_infos(infos)
        if env_it + 1 >= n_data_gen_steps:
            sumo_jsd_calc.calculate_distance(infos)
        tmp_next_state = torch.tensor(next_state, dtype=torch.float)
        state_list = envs.unpack_vector(state.clone().cpu())
        next_state_list = envs.unpack_vector(tmp_next_state)
        masks_list = envs.unpack_vector((1 - done))
        action_list = envs.unpack_vector(action.clone().cpu())
        rewards_list = envs.unpack_vector(reward.clone().cpu())
        for it in range(envs.nenvs):
            env_done = False
            if True not in masks_list[it]:
                env_done = True
            if infos[it]["ego_collision"]:
                env_done = True
            if env_done:
                finished_env = True
                masks_list[it] = np.zeros(masks_list[it].shape)
                rollout_storages[it].insert(state_list[it], action_list[it],
                                            rewards_list[it], masks_list[it],
                                            next_states=next_state_list[it].clone())
                rollout_storages[it].fill_replay_buffer(actor_critic_optimizer, additional_buffer=state_cond_ll_buffer)
                intermediate_next_state = envs.reset_env(it)
                next_state_list[it] = torch.tensor(intermediate_next_state, dtype=torch.float)
            else:
                rollout_storages[it].insert(state_list[it], action_list[it],
                                            rewards_list[it], masks_list[it],
                                            next_states=next_state_list[it].clone())

        next_state = torch.FloatTensor(next_state).to(device)
        if finished_env:
            next_state = torch.cat(next_state_list, dim=0).to(device)

        state = next_state

    for rollout_storage in rollout_storages:
        rollout_storage.fill_replay_buffer(actor_critic_optimizer, additional_buffer=state_cond_ll_buffer)

    rewards = torch.cat(rewards).clone().cpu().numpy()
    env_rewards = torch.cat(env_rewards).clone().cpu().numpy()
    return state, data_frames, rewards, env_rewards, sim_steps, state_cond_ll_buffer



def generate_data(state, n_data_gen_steps, actor_critic, envs, actor_critic_optimizer, length, state_est_model=None,
                  normalize_logprob=10., max_nll=10., state_cond_ll_buffer=None):
    device = actor_critic.device
    sim_steps = 0
    data_frames = 0
    states = []
    next_states = []
    actions = []
    masks = []
    env_rewards = []
    rewards = []
    lengths = []

    for env_it in range(n_data_gen_steps):

        dist, value = actor_critic(state, add_noise = False)

        action = dist.sample()
        action = torch.clamp(action, -actor_critic.act_limit, actor_critic.act_limit)
        if torch.isnan(action).any():
            print("found NaN in action!")
        next_state, env_reward, done, infos = envs.step(action.cpu().numpy())

        #print("infos", infos[0])
        if state_est_model is not None:
            #next_state = next_state + np.random.randn(*next_state.shape) * 0.0005
            next_state_flow = next_state
            next_state_flow = torch.tensor(next_state_flow, dtype=torch.float).to(actor_critic.device)
            _, state_prior_logprob, state_log_det = state_est_model(next_state_flow, state)
            state_logprob = state_prior_logprob + state_log_det
            state_eval = state_logprob.detach()

            #reward = torch.clamp(state_eval, min=-max_nll, max=1e9)
            #reward = torch.where(state_eval > -max_nll, state_eval, -max_nll - torch.log(-state_eval-max_nll+1))
            reward = state_eval

            if torch.isnan(reward).any():
                print("found NaN in reward!")
            reward = reward
            rewards.append(reward.clone().cpu())
        else:
            rewards.append(torch.FloatTensor(env_reward).cpu())

        next_state = torch.FloatTensor(next_state).to(device)
        data_frames += next_state.shape[0]
        sim_steps += 1
        masks.append((1 - done))
        lengths.append(length.clone())
        states.append(state.clone().cpu())
        actions.append(action.clone().cpu())
        next_states.append(next_state.clone().cpu())
        env_rewards.append(torch.FloatTensor(env_reward).cpu())

        # if True in done:
        #     print("finished environment")

        state = next_state
        length = (length + 1) * (1 - done)

    #masks[-1] = np.zeros(masks[-1].shape)
    next_actions = actions[1:]
    next_actions.append(torch.zeros(next_actions[-1].shape))
    next_next_states = next_states[1:]
    next_next_states.append(torch.zeros(next_states[-1].shape))
    actor_critic_optimizer.add_to_replay_buffer(states, actions, rewards, next_states, masks, next_actions,
                                                next_next_states, lengths)
    if state_cond_ll_buffer is not None:
        state_cond_ll_buffer.add_to_buffer(states, actions, rewards, next_states, masks, next_actions,
                                           next_next_states, lengths)

    rewards = torch.cat(rewards).clone().cpu().numpy()
    env_rewards = torch.cat(env_rewards).clone().cpu().numpy()
    return state, data_frames, rewards, env_rewards, sim_steps, length


def find_optimal_action(state, n_sample_steps, actor_critic, expert_state_est_model, forward_backward_cond_ll_model):
    device = actor_critic.device
    dist, value = actor_critic(state, add_noise = False)
    max_reward = torch.ones((value.shape[0],), device=device) * -1e9
    final_action = None
    for it in range(n_sample_steps):
        action = dist.sample()
        action = torch.clamp(action, -actor_critic.act_limit, actor_critic.act_limit)   
        if torch.isnan(action).any():
            print("found NaN in action!")
        
        if final_action is None:
            final_action = action.clone()
        next_state_flow = forward_backward_cond_ll_model.generate_states_based_on_actions(
            state, action)

        # next_state_flow = next_state + np.random.randn(*next_state.shape) * 0.0005
        # next_state_flow = torch.tensor(next_state_flow, dtype=torch.float).to(actor_critic.device)
        _, state_prior_logprob, state_log_det = expert_state_est_model(next_state_flow, state)
        state_logprob = state_prior_logprob + state_log_det
        state_eval = state_logprob.detach()
        reward = state_eval
        if torch.isnan(reward).any():
            print("found NaN in reward!")
        reward = reward

        bigger_reward_label = max_reward < reward
        max_reward[bigger_reward_label] = reward[bigger_reward_label]
        final_action[bigger_reward_label] = action[bigger_reward_label]
    return action



def generate_data_new(state, n_data_gen_steps, actor_critic, envs, actor_critic_optimizer=None, length=None, state_est_model=None,
                      state_cond_ll_buffer=None, state_mod_function=None, forward_backward_cond_ll_model=None, 
                      use_expert_reward_action_selectio=False, horizon=250, use_horizon=False):
    device = actor_critic.device
    sim_steps = 0
    data_frames = 0
    env_rewards = []
    rewards = []
    steps = []

    for env_it in range(n_data_gen_steps):

        dist, value = actor_critic(state)
        if use_expert_reward_action_selectio: 
            action = find_optimal_action(state, 1000, actor_critic,
                                state_est_model, forward_backward_cond_ll_model)
        else: 
            action = dist.sample()
            action = torch.clamp(
                action, -actor_critic.act_limit, actor_critic.act_limit)
            if torch.isnan(action).any():
                print("found NaN in action!")
        next_state, env_reward, done, infos = envs.step(action.cpu().numpy())

        if state_mod_function is not None:
            next_state = state_mod_function(next_state)

        if state_est_model is not None:
            next_state_flow = next_state
            next_state_flow = torch.tensor(
                next_state_flow, dtype=torch.float).to(actor_critic.device)
            _, state_prior_logprob, state_log_det = state_est_model(
                next_state_flow, state)
            state_logprob = state_prior_logprob + state_log_det
            state_eval = state_logprob.detach()
            reward = state_eval

            if torch.isnan(reward).any():
                print("found NaN in reward!")
            reward = reward
            rewards.append(reward.clone().cpu())
        else:
            rewards.append(torch.FloatTensor(env_reward).cpu())

        next_state = torch.FloatTensor(next_state).to(device)
        data_frames += next_state.shape[0]
        sim_steps += 1

        buffer_next_states = next_state.clone().cpu().numpy()
        if length is None:
            length = torch.ones((next_state.shape[0],))
        buffer_length = length.clone().cpu().numpy()
        finished_env = False

        buffer_state = state.clone().cpu().numpy()
        buffer_actions = action.clone().cpu().numpy()
        buffer_rewards = copy.deepcopy(env_reward)
        buffer_dones = copy.deepcopy((done))

        if actor_critic_optimizer is not None:
            actor_critic_optimizer.push_to_replay_buffer(buffer_state, buffer_actions, buffer_rewards, buffer_next_states, buffer_dones, buffer_actions,
                                                         buffer_next_states, buffer_length)
        if state_cond_ll_buffer is not None:
            state_cond_ll_buffer.push_to_buffer(buffer_state, buffer_actions, buffer_rewards, buffer_next_states, buffer_dones, buffer_actions,
                                                        buffer_next_states, buffer_length)

        env_rewards.append(torch.FloatTensor(env_reward).cpu())

        if use_horizon and env_it % horizon == 0 and env_it>0:
            finished_env = True
            next_state = envs.reset()
            if state_mod_function is not None:
                next_state = state_mod_function(next_state)
            next_state = torch.FloatTensor(next_state).to(device)
            length = torch.ones((next_state.shape[0],))

        state = next_state
        if finished_env:
            length = torch.ones((next_state.shape[0],))
        else:
            length = (length + 1) * (1 - done)

    sim_length = np.mean(length.cpu().numpy())
    rewards = torch.cat(rewards).clone().cpu().numpy()
    env_rewards = torch.cat(env_rewards).clone().cpu().numpy()
    return state, data_frames, rewards, env_rewards, sim_steps, length, sim_length
