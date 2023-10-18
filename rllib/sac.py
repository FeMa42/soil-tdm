# This implementation is based on the Spinning UP implementation of SAC (see: https://github.com/openai/spinningup)
import random
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from nflib.straight_throug_clamp import StraightThrougClamp

def compress(x,max_nll):
    x = torch.clamp(torch.where(x > -max_nll, x, -max_nll - torch.log(torch.max(-x - max_nll + 1,x*0+1))),-100,1e9)
    x = torch.where(torch.isnan(x),torch.ones(x.shape).to(x.device)*-100,x)
    return x

def compress_grad_corr(x, max_nll):
    return torch.where(x > -max_nll, x*0+1, torch.exp(-x - max_nll))


class ReplayBuffer:
    def __init__(self, capacity, use_randomized_buffer=True):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.use_randomized_buffer = use_randomized_buffer

    def push(self, state, action, reward, next_state, done, next_action, next_next_state, time_steps):
        if self.use_randomized_buffer: 
            if len(self.buffer) < 4: 
                self.buffer.append((state, action, reward, next_state,
                                done, next_action, next_next_state, time_steps))
            if len(self.buffer) < self.capacity:
                self.buffer.insert(random.randint(0, len(self.buffer)-1), (state, action, reward, next_state,
                                                                           done, next_action, next_next_state, time_steps))
                self.position = (self.position + 1) 
            else:
                self.position = random.randint(0, len(self.buffer)-1)
                self.buffer[self.position] = (state, action, reward, next_state, done, next_action, next_next_state, time_steps)
        else:
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (state, action, reward, next_state, done, next_action, next_next_state, time_steps)
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, next_action, next_next_state, time_steps = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, next_action, next_next_state, time_steps

    def sample_train(self, batch_size):
        buffer_size = len(self.buffer)
        train_buffer_size = int(buffer_size*0.8)
        if int(buffer_size*0.2) > batch_size:
            batch = random.sample(self.buffer[:train_buffer_size], batch_size)
        else:
            batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, next_action, next_next_state, time_steps = map(
            np.stack, zip(*batch))
        return state, action, reward, next_state, done, next_action, next_next_state, time_steps
    
    def sample_test(self, batch_size):
        buffer_size = len(self.buffer)
        train_buffer_size = int(buffer_size*0.8)
        if int(buffer_size*0.2) > batch_size:
            batch = random.sample(self.buffer[train_buffer_size:], batch_size)
        else:
            batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, next_action, next_next_state, time_steps = map(
            np.stack, zip(*batch))
        return state, action, reward, next_state, done, next_action, next_next_state, time_steps

    def add_to_buffer(self, state, action, reward, next_state, masks, next_action, next_next_states, time_steps):
        state = torch.cat(state).detach().cpu().numpy()
        action = torch.cat(action).detach().cpu().numpy()
        reward = torch.cat(reward).detach().cpu().numpy()
        time_steps = torch.cat(time_steps).detach().cpu().numpy()
        next_state = torch.cat(next_state).detach().cpu().numpy()
        next_action = torch.cat(next_action).detach().cpu().numpy()
        next_next_states = torch.cat(next_next_states).detach().cpu().numpy()
        dones = 1-np.concatenate(masks)
        assert(len(state)==len(action))
        assert(len(state)==len(reward))
        assert (len(state) == len(time_steps))
        assert(len(state)==len(next_state))
        assert(len(state)==len(dones))
        assert(len(state)==len(next_action))
        assert(len(state)==len(next_next_states))
        for it in range(state.shape[0]):
            self.push(state[it, :], action[it, :], reward[it], next_state[it, :], dones[it],
                                    next_action[it,:], next_next_states[it, :], time_steps[it])

    def push_to_buffer(self, state, action, reward, next_state, dones, next_action, next_next_states, time_steps):
        assert(len(state) == len(action))
        assert(len(state) == len(reward))
        assert (len(state) == len(time_steps))
        assert(len(state) == len(next_state))
        assert(len(state) == len(dones))
        assert(len(state) == len(next_action))
        assert(len(state) == len(next_next_states))
        for it in range(state.shape[0]):
            self.push(state[it, :], action[it, :], reward[it], next_state[it, :], dones[it],
                      next_action[it, :], next_next_states[it, :], time_steps[it])

    def __len__(self):
        return len(self.buffer)

def compute_log_w(state_cond_ll, action_log_prob, state, action, next_state):
    #ns_a_s = state_cond_ll.estimete_lp_ns_given_a_s(state, action, next_state).unsqueeze(-1)
    #a_ns_s = state_cond_ll.estimete_lp_a_given_ns_s(state, action, next_state).unsqueeze(-1)
    #ns_s = state_cond_ll.estimete_lp_ns_given_s(state, next_state).unsqueeze(-1)
    a_s = state_cond_ll.estimete_lp_a_given_s(state,action).unsqueeze(1)

    #log_w = ns_a_s + action_log_prob - a_ns_s - ns_s
    log_w = action_log_prob - a_s
    return log_w

class SAC():
    def __init__(self,
                 model,
                 batch_size=256,
                 device="cpu",
                 epochs=1,
                 soft_q_lr=1e-3,
                 policy_lr=1e-3,
                 replay_buffer_size=1000000,
                 weight_clip_val=1.,
                 grad_clip_val=10.,
                 alpha=1.,
                 gamma=0.99,
                 polyak=0.995,
                 use_automatic_entropy_tuning=True,
                 target_entropy=None,
                 max_nll=4.,
                 use_fifo_replay_buffer=False):
        self.sac_update = True
        self.batch_size = batch_size
        self.device = device
        self.epochs = epochs
        self.model = model
        self.weight_clipping = False
        self.grad_clipping = True
        self.weight_clip_val = weight_clip_val
        self.grad_clip_val = grad_clip_val
        self.gamma = gamma
        self.polyak = polyak
        self.max_nll = max_nll
        self.running_fq_mean = None

        # implementation of entropy tuning see https://github.com/vitchyr/rlkit
        self.alpha = alpha
        # self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.log_alpha = torch.tensor(
            [np.log(alpha)], requires_grad=True, device=device, dtype=torch.float32)
        if target_entropy is None:
            # Use heuristic value from SAC paper
            self.target_entropy = -np.prod(
                self.model.action_shape).item()
        else:
            self.target_entropy = target_entropy
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning

        self.soft_q_criterion_1 = nn.MSELoss()
        self.soft_q_criterion_2 = nn.MSELoss()

        q_net_parameters = [p for p in self.model.soft_q_net_1.parameters() if p.requires_grad]
        q_net_parameters.extend([p for p in self.model.soft_q_net_2.parameters() if p.requires_grad])
        self.soft_q_optimizer = optim.Adam(q_net_parameters, lr=soft_q_lr)
        self.policy_lr = policy_lr
        self.soft_q_lr = soft_q_lr
        #print("Policy parameters:", self.model.get_policy_parameter())
        self.policy_optimizer = optim.Adam(self.model.get_policy_parameter(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=policy_lr)

        self.replay_buffer = ReplayBuffer(
            replay_buffer_size, use_randomized_buffer=(not use_fifo_replay_buffer))

    def update_policy_optimizer(self):
        self.policy_optimizer = optim.Adam(self.model.get_policy_parameter(), lr=self.policy_lr)

    def update_p_q_linear_schedule(self, epoch, total_num_epochs, start_epoch=0, lr_min=1e-5):
        """Decreases the learning rate linearly"""
        lr_p = self.update_p_linear_schedule(
            epoch, total_num_epochs, start_epoch, lr_min)
        epoch = epoch - start_epoch
        if epoch > 0:
            lr_q = self.soft_q_lr - \
                (self.soft_q_lr * (epoch / float(total_num_epochs)))
            if lr_q < lr_min:
                lr_q = lr_min
            for param_group in self.soft_q_optimizer.param_groups:
                param_group['lr'] = lr_q
        else:
            lr_q = self.soft_q_lr
        return lr_p, lr_q

    def update_p_linear_schedule(self, epoch, total_num_epochs, start_epoch=0, lr_min=1e-5):
        """Decreases the learning rate linearly"""
        epoch = epoch - start_epoch
        if epoch > 0:
            lr_p = self.policy_lr - \
                (self.policy_lr * (epoch / float(total_num_epochs)))
            if lr_p < lr_min:
                lr_p = lr_min
            for param_group in self.policy_optimizer.param_groups:
                param_group['lr'] = lr_p
        else:
            lr_q = self.soft_q_lr
            lr_p = self.policy_lr
        return lr_p

    def add_to_replay_buffer(self, state, action, reward, next_state, masks,
                             next_actions, next_next_states, time_steps):
        self.replay_buffer.add_to_buffer(state, action, reward, next_state, masks,
                                         next_actions, next_next_states, time_steps)
        # state = torch.cat(state).detach().cpu().numpy()
        # action = torch.cat(action).detach().cpu().numpy()
        # reward = torch.cat(reward).detach().cpu().numpy()
        # next_state = torch.cat(next_state).detach().cpu().numpy()
        # next_next_states = torch.cat(next_next_states).detach().cpu().numpy()
        # dones = 1-np.concatenate(masks)
        # for it in range(state.shape[0]):
        #     self.replay_buffer.push(state[it, :], action[it, :], reward[it], next_state[it, :], dones[it],
        #                             next_next_states[it, :])

    def push_to_replay_buffer(self, state, action, reward, next_state,
                              dones, next_action, next_next_states, time_steps):
        self.replay_buffer.push_to_buffer(state, action, reward, next_state,
                                dones, next_action, next_next_states, time_steps)

    def soft_q_update_soil_tdm(self, pol_update=True, state_cond_ll=None, discriminator=None,
                               state_est_model = None, update_q = True, method="model", modify_entropy=False,
                               alpha_fac_nonpol = 0, alpha_policy = 0, cur_gamma = 0.9, mci_ns_samples = 1,
                               done_reward=0.0, exp_survival_ext=None, max_exp_state_ll=None):
        policy_losses = []
        q_value_losses = []
        policy_base_losses = []
        log_probs = []
        alpha_losses = []
        next_lp_ns_g_s = []
        sampled_expert_rewards = []
        sampled_q = []
        training_rewards = []
        penalty_added = 0
        for epoch in range(self.epochs):
            for param in self.model.get_policy_parameter():
                param.requires_grad = True
            if state_est_model is not None:
                for param in state_est_model.parameters():
                    param.requires_grad = False
            for param in self.model.soft_q_net_1.parameters():
                param.requires_grad = False
            for param in self.model.soft_q_net_2.parameters():
                param.requires_grad = False

            if state_cond_ll is not None:
                for param in state_cond_ll.get_all_model_parameter():
                    param.requires_grad = False

            state, action, reward, next_state, done, next_action, \
            next_next_state, time_steps = self.replay_buffer.sample(self.batch_size)

            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            next_next_state = torch.FloatTensor(next_next_state).to(self.device)
            action = torch.FloatTensor(action).float().to(self.device)
            next_action = torch.FloatTensor(next_action).float().to(self.device)
            reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
            done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
            time_steps = torch.FloatTensor(time_steps).unsqueeze(1).to(self.device)

            _, state_prior_logprob, state_log_det = state_est_model(next_state, state)
            reward = (state_prior_logprob + state_log_det).clone().detach().unsqueeze(1)

            if discriminator is not None:
                raise Exception("not adapted/implemented yet")

            if max_exp_state_ll is not None:
                creward_tmp = StraightThrougClamp.apply(
                    reward, -max_exp_state_ll*1.5, max_exp_state_ll*1.5)
            else:
                creward_tmp = compress(reward, self.max_nll)

            
            if exp_survival_ext is not None:
                survival_new_reward = exp_survival_ext(time_steps, done)
                if torch.any(survival_new_reward < 0):
                    penalty_added+=1
                creward_tmp += survival_new_reward
            sampled_expert_rewards.append(creward_tmp.clone().cpu().numpy().mean())

            # alpha loss
            if self.use_automatic_entropy_tuning:
                raise Exception("not adapted/implemented yet")
            else:
                alpha_loss = 0
                alpha = self.alpha

            new_action, new_action_log_prob, z, mean, log_std = self.model.evaluate(state, add_noise=False)
            if max_exp_state_ll is not None:
                new_action_log_prob = StraightThrougClamp.apply(new_action_log_prob, -max_exp_state_ll, max_exp_state_ll)
            policy_log_probs = new_action_log_prob.mean()

            q_value_1 = self.model.soft_q_net_1(state, new_action)
            q_value_2 = self.model.soft_q_net_2(state, new_action)
            used_action_q_value = torch.min(q_value_1, q_value_2) 
            
            policy_entropy = -torch.mean(new_action_log_prob / 1)
            sampled_q.append(torch.mean(used_action_q_value).clone().detach().cpu().item())
            
            if state_cond_ll is None:
                policy_loss_base = ((alpha * new_action_log_prob) - used_action_q_value).mean()
            else:
                w = 1/state.shape[0]
                policy_loss_base = (w*(-used_action_q_value) - alpha_policy*policy_entropy/state.shape[0]).sum()

            policy_loss = policy_loss_base 

            if pol_update:
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                if self.grad_clipping:
                    norm = torch.nn.utils.clip_grad_value_(self.model.get_policy_parameter(), self.grad_clip_val)
                self.policy_optimizer.step()


                if self.weight_clipping:
                    for p in self.model.get_policy_parameter():
                        p.data.clamp_(-self.weight_clip_val, self.weight_clip_val)

            for param in self.model.soft_q_net_1.parameters():
                param.requires_grad = True
            for param in self.model.soft_q_net_2.parameters():
                param.requires_grad = True

            # q value loss
            with torch.no_grad():
                if state_cond_ll is not None:
                    if method.startswith("model"):
                        if modify_entropy:
                            raise Exception("not adapted/implemented yet")
                        else:
                            #next_mod_q = alpha_policy*new_next_action_log_prob
                            if max_exp_state_ll is not None:
                                creward = StraightThrougClamp.apply(
                                    reward, -max_exp_state_ll*1.5, max_exp_state_ll*1.5)
                                max_nll = max_exp_state_ll
                            else:
                                creward = compress(reward, self.max_nll)
                                max_nll = self.max_nll
                            if exp_survival_ext is not None:
                                survival_new_reward = exp_survival_ext(time_steps, done)
                                creward += survival_new_reward
                            mod_reward = [-state_cond_ll.estimate_state_cod_prob(state, action,
                                                                                 next_state, creward*0, alpha_fac_nonpol, max_nll) + creward]

                            for i in range(mci_ns_samples-1):
                                new_new_next_state = state_cond_ll.generate_states_based_on_actions(state,action)
                                _, new_state_prior_logprob, new_state_log_det = state_est_model(new_new_next_state,
                                                                                               state)
                                new_new_reward = (new_state_prior_logprob + new_state_log_det)[:, None]
                                # creward = compress(new_new_reward, self.max_nll)
                                if max_exp_state_ll is not None:
                                    creward = StraightThrougClamp.apply(
                                        new_new_reward, -max_exp_state_ll*1.5, max_exp_state_ll*1.5)
                                else:
                                    creward = compress(
                                        new_new_reward, self.max_nll)
                                if exp_survival_ext is not None:
                                    survival_new_reward = exp_survival_ext(time_steps, done)
                                    creward += survival_new_reward
                                mod_reward.append(-state_cond_ll.estimate_state_cod_prob(state, action,
                                                                                         new_new_next_state, creward*0, alpha_fac_nonpol, max_nll) + creward)
                            mod_reward = torch.mean(torch.cat(mod_reward, dim=1), dim=1,
                                                         keepdims=True)

                            future_q_value = 0
                            next_q_value_mean = 0
                            next_action_log_prob_mean = 0
                            for i in range(mci_ns_samples):
                                new_next_action, new_next_action_log_prob, _, _, _ = self.model.evaluate(next_state)
                                if max_exp_state_ll is not None:
                                    new_next_action_log_prob = StraightThrougClamp.apply(
                                        new_next_action_log_prob, -max_exp_state_ll, max_exp_state_ll)
                                next_q_value_1 = self.model.target_q_net_1(next_state, new_next_action)
                                next_q_value_2 = self.model.target_q_net_2(next_state, new_next_action)
                                next_q_value = 1 * torch.min(next_q_value_1,
                                                             next_q_value_2)  # + 0.3*torch.max(expected_next_q_value_1, expected_next_q_value_2)
                                next_q_value_mean = next_q_value_mean + next_q_value
                                next_action_log_prob_mean = next_action_log_prob_mean + new_next_action_log_prob
                                #future_q_value += (next_q_value - (alpha_policy * new_next_action_log_prob) / 1)
                            next_q_value_mean = next_q_value_mean / mci_ns_samples
                            next_action_log_prob_mean = next_action_log_prob_mean / mci_ns_samples
                            next_action_log_prob_mean = next_action_log_prob_mean #- torch.mean(next_action_log_prob_mean)
                            future_q_value = next_q_value_mean - alpha_policy * next_action_log_prob_mean

                            next_mod_q = 0
                        w = 1/state.shape[0]
                else:
                    raise Exception("not implemented yet")

                next_lp_ns_g_s.append(np.array([0]))

                target_q_value = mod_reward + cur_gamma * future_q_value
                mean_target_q = ((1 - done) * target_q_value).sum() / ((1 - done).sum() + 1e-6)
                if True in done:
                    test_q_value = target_q_value * 0 + 0*mean_target_q + done_reward
                target_q_value = torch.where(done > 0.5, mod_reward + done_reward, target_q_value)

                training_rewards.append(np.mean(mod_reward.cpu().detach().clone().numpy()))

                target_q_value = target_q_value #- torch.mean(target_q_value)

            current_q_value_1 = self.model.soft_q_net_1(state, action.clone().detach())
            current_q_value_2 = self.model.soft_q_net_2(state, action.clone().detach())

            w = 1/state.shape[0]

            #print(w.shape, expected_new_q_value_1.shape, target_q_value.shape)
            q_value_loss_1 = (w*(current_q_value_1 - target_q_value.clone().detach()).abs()**1.2).sum()
            q_value_loss_2 = (w*(current_q_value_2 - target_q_value.clone().detach()).abs()**1.2).sum()
            q_value_loss = q_value_loss_1 + q_value_loss_2

            if update_q:
                self.soft_q_optimizer.zero_grad()
                q_value_loss.backward()
                if self.grad_clipping:
                    q_net_parameters = [p for p in self.model.soft_q_net_1.parameters() if p.requires_grad]
                    q_net_parameters.extend([p for p in self.model.soft_q_net_2.parameters() if p.requires_grad])
                    norm = torch.nn.utils.clip_grad_norm_(q_net_parameters, self.grad_clip_val)
                self.soft_q_optimizer.step()

            for target_param, param in zip(self.model.target_q_net_1.parameters(), self.model.soft_q_net_1.parameters()):
                target_param.data.mul_(self.polyak)
                target_param.data.add_((1 - self.polyak) * param.data)

            for target_param, param in zip(self.model.target_q_net_2.parameters(), self.model.soft_q_net_2.parameters()):
                target_param.data.mul_(self.polyak)
                target_param.data.add_((1 - self.polyak) * param.data)

            if state_cond_ll is not None:
                for param in state_cond_ll.get_all_model_parameter():
                    param.requires_grad = True

            policy_losses.append(policy_loss.item())
            q_value_losses.append(q_value_loss.item())
            policy_base_losses.append(policy_loss_base.item())
            log_probs.append(policy_log_probs.clone().detach().cpu().numpy().mean())
            if self.use_automatic_entropy_tuning:
                alpha_losses.append(alpha_loss.item())
            else:
                alpha_losses.append(alpha_loss)

        print("penalties added: ", penalty_added)
        print("mean mod_rewards: ", np.mean(training_rewards))

        return np.mean(policy_losses), np.mean(q_value_losses), np.mean(policy_base_losses), np.mean(log_probs), \
               np.mean(alpha_losses), np.mean(training_rewards), np.mean(sampled_expert_rewards), np.mean(sampled_q)

    def soft_q_update_std(self, pol_update=True, expert_state_est_model=None, policy_state_est_model=None, discriminator=None):
        policy_losses = []
        q_value_losses = []
        log_probs = []
        alpha_losses = []
        next_lp_ns_g_s = []
        training_rewards = []
        sampled_q = []
        expert_sampled_rewards = []
        for _ in range(self.epochs):
            
            state, action, reward, next_state, done, _, \
                next_next_state, _ = self.replay_buffer.sample(
                    self.batch_size)

            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            next_next_state = torch.FloatTensor(
                next_next_state).to(self.device)
            action = torch.FloatTensor(action).float().to(self.device)
            reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
            done = torch.FloatTensor(np.float32(
                done)).unsqueeze(1).to(self.device)

            # get new action and action log prob
            new_action, log_prob, z, mean, log_std = self.model.evaluate(state)

            if expert_state_est_model is not None:
                _, expert_state_prior_logprob, expert_state_log_det = expert_state_est_model(
                    next_state, state)
                expert_state_ll = (expert_state_prior_logprob +
                                   expert_state_log_det).clone().detach().unsqueeze(1)
                
                expert_sampled_rewards.append(compress(
                    expert_state_ll.clone(), self.max_nll).cpu().numpy().mean())
            
                if policy_state_est_model is not None:
                    _, policy_state_prior_logprob, policy_state_log_det = policy_state_est_model(
                        next_state, state)
                    policy_state_ll = (policy_state_prior_logprob +
                                    policy_state_log_det).clone().detach().unsqueeze(1)

                    reward = expert_state_ll - policy_state_ll

                    reward = compress(reward.clone(), self.max_nll)
                else:
                    reward = compress(expert_state_ll.clone(), self.max_nll)
                
            elif discriminator is not None:
                reward = discriminator.predict_reward(state, action.clone().detach())
                reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
            
            training_rewards.append(reward.clone().cpu().numpy().mean())

            # alpha loss
            if self.use_automatic_entropy_tuning:
                lp_ns_given_s_ = log_prob.clone().detach()

                alpha_loss = -(self.log_alpha * (lp_ns_given_s_ +
                               self.target_entropy).detach()).mean()
                alpha = self.log_alpha.exp().detach()
                if pol_update:
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
            else:
                alpha_loss = 0
                alpha = self.alpha

            # policy Loss
            expected_new_q_value_1 = self.model.soft_q_net_1(state, new_action)
            expected_new_q_value_2 = self.model.soft_q_net_2(state, new_action)
            expected_new_q_value = torch.min(expected_new_q_value_1, expected_new_q_value_2)
            log_prob_target = expected_new_q_value

            sampled_q.append(torch.mean(
                expected_new_q_value).clone().detach().cpu().item())

            policy_loss = ((alpha * log_prob) -
                                    log_prob_target).mean()

            if pol_update:
                self.policy_optimizer.zero_grad()
                for param in self.model.soft_q_net_1.parameters():
                    param.requires_grad = False
                for param in self.model.soft_q_net_2.parameters():
                    param.requires_grad = False
                policy_loss.backward()
                if self.grad_clipping:
                    norm = torch.nn.utils.clip_grad_norm_(
                        self.model.get_policy_parameter(), self.grad_clip_val)
                self.policy_optimizer.step()

                for param in self.model.soft_q_net_1.parameters():
                    param.requires_grad = True
                for param in self.model.soft_q_net_2.parameters():
                    param.requires_grad = True

                if self.weight_clipping:
                    for p in self.model.get_policy_parameter():
                        p.data.clamp_(-self.weight_clip_val,
                                      self.weight_clip_val)

            # q value loss
            expected_q_value_1 = self.model.soft_q_net_1(state, action)
            expected_q_value_2 = self.model.soft_q_net_2(state, action)
            with torch.no_grad():
                next_action, next_log_prob, _, _, _ = self.model.evaluate(
                    next_state)

                next_lp_ns_g_s.append(next_log_prob.cpu().clone().numpy())

                expected_next_q_value_1 = self.model.target_q_net_1(
                    next_state, next_action)
                expected_next_q_value_2 = self.model.target_q_net_2(
                    next_state, next_action)
                expected_next_q_value = torch.min(
                    expected_next_q_value_1, expected_next_q_value_2)
                target_value = (expected_next_q_value - (next_log_prob*alpha))

                target_q_value = reward + \
                    (1 - done) * self.gamma * target_value

            q_value_loss_1 = (
                (expected_q_value_1 - target_q_value.detach())**2).mean()
            q_value_loss_2 = (
                (expected_q_value_2 - target_q_value.detach())**2).mean()
            q_value_loss = q_value_loss_1 + q_value_loss_2

            self.soft_q_optimizer.zero_grad()
            q_value_loss.backward()
            if self.grad_clipping:
                q_net_parameters = [
                    p for p in self.model.soft_q_net_1.parameters() if p.requires_grad]
                q_net_parameters.extend(
                    [p for p in self.model.soft_q_net_2.parameters() if p.requires_grad])
                norm = torch.nn.utils.clip_grad_norm_(
                    q_net_parameters, self.grad_clip_val)
            self.soft_q_optimizer.step()

            for target_param, param in zip(self.model.target_q_net_1.parameters(), self.model.soft_q_net_1.parameters()):
                target_param.data.mul_(self.polyak)
                target_param.data.add_((1 - self.polyak) * param.data)

            for target_param, param in zip(self.model.target_q_net_2.parameters(), self.model.soft_q_net_2.parameters()):
                target_param.data.mul_(self.polyak)
                target_param.data.add_((1 - self.polyak) * param.data)

            policy_losses.append(policy_loss.item())
            q_value_losses.append(q_value_loss.item())
            log_probs.append(log_prob.detach().cpu().numpy().mean())
            if self.use_automatic_entropy_tuning:
                alpha_losses.append(alpha_loss.item())
            else:
                alpha_losses.append(alpha_loss)

            if len(expert_sampled_rewards) == 0:
                expert_sampled_rewards.append(0.)
        return np.mean(policy_losses), np.mean(q_value_losses), np.mean(log_probs), \
               np.mean(alpha_losses), np.asarray(next_lp_ns_g_s), np.mean(training_rewards), \
               np.mean(expert_sampled_rewards), np.mean(sampled_q)
