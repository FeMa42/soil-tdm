import csv
import os
import time
import argparse
import json
from rllib.model_sac import FlowSAC, MLPSAC
import h5py
import numpy as np
import gym
import torch
from torch.distributions import Normal
from common.multiprocessing_env import SubprocVecEnv
from tensorboardX import SummaryWriter

from rllib.sac import SAC
from common.eval_policy import eval_env


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--env-name",
                        default="HalfCheetahBulletEnv-v0",
                        help="Test Environment name used by OpenAI Gym (e.g. BipedalWalker-v3, "
                             "HalfCheetahBulletEnv-v0, AntBulletEnv-v0, HopperBulletEnv-v0,"
                             "Walker2DBulletEnv-v0, HumanoidBulletEnv-v0, KukaBulletEnv-v0, RacecarBulletEnv-v0,"
                             "CartPoleContinuousBulletEnv-v0).")
    parser.add_argument(
        '--use-mlp', action='store_true', default=False, help='Use conditional Gaussian MLP policy instead of a conditiona normalizing flow policy.')
    parser.add_argument("--exp-dir", required=True,
                        help="Name of the experiment. A folder with this name is generated")
    parser.add_argument(
        '--max-frames', type=int, default=300000, help='Max number of collected frames in training (default 300k)')
    parser.add_argument(
        '--load-model', action='store_true', default=False, help='load previously trained models.')
    parser.add_argument(
        '--num-envs', type=int, default=16, help='Number of environments (default 37)')
    parser.add_argument(
        '--lr-p', type=float, default=3e-4, help='learning rate for policy')
    parser.add_argument(
        '--lr-q', type=float, default=5e-4, help='learning rate for q estimation model')
    parser.add_argument(
        '--polyak', type=float, default=0.99, help='polyak rate')
    parser.add_argument(
        '--alpha', type=float, default=0.2, help='entropy factor in sac optimization (higher means higher entropy)')
    parser.add_argument(
        '--replay-buffer-size', type=int, default=1000000, help='The maximum size of the replay buffer')
    parser.add_argument(
        '--hidden-size', type=int, default=512, help='hidden size of Q function')
    parser.add_argument(
        '--mini-batch-size', type=int, default=512, help='batch size in sac optimization')
    parser.add_argument(
        '--internal-update-epochs', type=int, default=200, help='amount of updates after each data collection epoch')
    parser.add_argument(
        '--num-env-steps', type=int, default=200, help='amount of data collection steps per epoch')
    parser.add_argument(
        '--exp-clamping', type=float, default=2.0, help='Policy flow exponent clamping factor')
    parser.add_argument(
        '--hidden-size-cond', type=int, default=64, help='hidden size of policy condition function')
    parser.add_argument(
        '--policy-flow-hidden', type=int, default=8, help='hidden size of policy flow function')
    parser.add_argument(
        '--n-flows-policy', type=int, default=16, help='amount of flow blocks used in policy')
    parser.add_argument(
        '--con-dim-features', type=int, default=8, help='Dimension of calculated condition feature vector')
    parser.add_argument(
        '--target-entropy', type=float, default=None, help='Policy flow exponent clamping factor')
    parser.add_argument(
    '--dont-use-automatic-entropy-tuning', action='store_true', default=False, help='Use fixed alpha value for SAC.')

    args = parser.parse_args()

    # Example (resulted in a reward > 2000):
    # python gym_exp_data.py --max-frames 300000 --exp-dir halfchetah_SAC --env-name HalfCheetahBulletEnv-v0
    # --num-envs 16 --hidden-size 256 --lr-p 1e-4 --lr-q 1e-4 --mini-batch-size 256
    # --num-env-steps 10 --internal-update-epochs 10 --polyak 0.99

    for k, v in args._get_kwargs():
        print(k, "=", v)

    seed = 4
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)

    output_dir = os.path.join("./gym_exp", args.exp_dir)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with open(os.path.join(output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))

    use_mps = False
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_num_threads(1)
    elif torch.backends.mps.is_available() and use_mps:
        device = torch.device("mps")
        torch.set_num_threads(1)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(4)

    num_envs = args.num_envs  
    env_name = args.env_name 
    test_env = False
    visualize = False

    writer = SummaryWriter(output_dir, comment=env_name)

    if "Bullet" in env_name or "Kuka" in env_name:
        import pybullet_envs
        from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
        from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv

    def make_env():
        def _thunk():
            if env_name == "KukaBulletEnv-v0":
                env = KukaGymEnv(renders=visualize, isDiscrete=False)
            elif env_name == "KukaDiverseObjectGrasping-v0":
                env = KukaDiverseObjectEnv(renders=visualize, isDiscrete=False)
            else:
                env = gym.make(env_name)
            # env.render()
            # env.reset()
            return env

        return _thunk

    # action_scale = 1.
    if env_name == "KukaBulletEnv-v0" and visualize:
        env = KukaGymEnv(renders=visualize, isDiscrete=False)
    elif env_name == "KukaDiverseObjectGrasping-v0" and visualize:
        env = KukaDiverseObjectEnv(renders=visualize, isDiscrete=False)
    else:
        env = gym.make(env_name)

    if visualize:
        env.render(mode="human")
        env.reset()


    if len(env.observation_space.shape) == 3:
        num_observations = env.observation_space.shape
    else:
        num_observations = env.observation_space.shape[0]

    if env.action_space.__class__.__name__ == "Discrete":
        num_action = env.action_space.n
        is_discrete = True
        action_scale_low = 0
        action_scale = num_action
        use_mlp = True # args.use_mlp
    else:
        is_discrete = False
        num_action = env.action_space.shape[0]
        action_scale_low = env.action_space.low
        action_scale_high = env.action_space.high
        action_scale = action_scale_high[0]
        use_mlp = args.use_mlp

    if not("Kuka" in env_name and visualize):
        if args.max_frames == 0:
            env.close()
        envs = [make_env() for i in range(num_envs)]
        envs = SubprocVecEnv(envs, use_norm=False)
    else:
        max_frames = 0
        envs = env

    # Hyper params:
    hidden_size = args.hidden_size  # 256
    mini_batch_size = args.mini_batch_size  # 100
    internal_update_epochs = args.internal_update_epochs  # 50  # TODO: 10

    use_ppo = False
    use_additional_normal = False
    if use_mlp is False and num_action < 2:
        use_additional_normal = True
        normal = Normal(0., 1.)
        num_action += 1

    # SAC Model parameter
    exp_clamping = args.exp_clamping  # 3.
    hidden_size_cond = args.hidden_size_cond # 128
    n_flows_policy = args.n_flows_policy  # 16
    policy_flow_hidden = args.policy_flow_hidden  # 128
    con_dim_features = args.con_dim_features  # 16

    # SAC Update Parameter
    gamma = 0.99
    polyak = args.polyak  # 0.995
    alpha = args.alpha  # 0.2
    if args.target_entropy is not None:
        target_entropy = -args.target_entropy  # 0.1  # None  # 2
    else:
        target_entropy = None

    soft_q_lr = args.lr_q  # 1e-3  # TODO 5e-4
    policy_lr = args.lr_p  # 1e-3  # TODO 5e-4
    replay_buffer_size = args.replay_buffer_size

    if use_ppo:
        raise(NotImplementedError)
    else:
        if use_mlp:
            actor_critic = MLPSAC(num_inputs=num_observations, num_outputs=num_action, hidden_size=hidden_size,
                                  act_limit=action_scale, device=device, is_discrete=is_discrete).to(device)
        else:
            actor_critic = FlowSAC(num_inputs=num_observations, num_outputs=num_action, hidden_size=hidden_size,
                                   n_flows=n_flows_policy, flow_hidden=policy_flow_hidden,
                                   exp_clamping=exp_clamping, hidden_size_cond=hidden_size_cond,
                                   con_dim_features=con_dim_features, act_limit=action_scale, device=device).to(device)

        actor_and_critic_optimizer = SAC(actor_critic, batch_size=mini_batch_size, device=device,
                                         epochs=internal_update_epochs,
                                         soft_q_lr=soft_q_lr, policy_lr=policy_lr,
                                         replay_buffer_size=replay_buffer_size,
                                         alpha=alpha, polyak=polyak, gamma=gamma,
                                         use_automatic_entropy_tuning=(not args.dont_use_automatic_entropy_tuning),
                                         target_entropy=target_entropy)

    if args.load_model:
        model_location = os.path.join(output_dir, env_name + "_d.pt")
        if os.path.isfile(model_location):
            actor_critic.load(model_location)

    num_steps = args.num_env_steps
    max_frames = args.max_frames
    if test_env:
        max_frames = 0
    step_idx = 0
    test_rewards = []

    state = envs.reset()
    update_iter = 0
    successes = 0
    length = torch.ones((state.shape[0],))
    global_start = time.time()
    while step_idx < max_frames:
        start = time.time()
        print_results = False
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []
        next_states = []
        max_positions = []
        lengths = []
        for _ in range(num_steps):
            state = torch.FloatTensor(state).to(device)
            dist, value = actor_critic(state)

            action = dist.sample()
            if torch.isnan(action).any():
                print("found NaN in action!")

            if is_discrete:
                _, gym_action = action.max(dim=1)  
                gym_action = gym_action.cpu().numpy()
                add_reward = 0.
            else:
                if action_scale is not None:
                    action = torch.clamp(action, -action_scale, action_scale)
                if use_additional_normal:
                    gym_action = action[:, 0].unsqueeze(1).cpu().numpy()
                    add_reward = normal.log_prob(action[:, 1].cpu()).numpy()
                else:
                    gym_action = action.cpu().numpy()
                    add_reward = 0.

            next_state, reward, done, _ = envs.step(gym_action)
            reward += np.array(add_reward).astype(reward.dtype)
            length = (length + 1) * (1 - done)

            states.append(state.clone())
            actions.append(action)
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)
            values.append(value)
            lengths.append(length.clone())
            if use_ppo:
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
            else:
                next_states.append(torch.FloatTensor(next_state).to(device))
                rewards.append(torch.FloatTensor(reward).to(device))
                masks.append(1 - done)

            state = next_state
            step_idx += 1

        frame_idx = step_idx * num_envs
        if len(max_positions) > 0:
            print("Max reached position: ", max(max_positions))
            print("Successes: ", successes)

        next_actions = actions[1:]
        next_actions.append(torch.zeros(next_actions[-1].shape).to(device))
        next_next_states = next_states[1:]
        next_next_states.append(torch.zeros(next_states[-1].shape).to(device))
        actor_and_critic_optimizer.add_to_replay_buffer(states, actions, rewards, next_states, masks, next_actions,
                                            next_next_states, lengths)
        if frame_idx >= mini_batch_size*5:
            update_iter += 1
            policy_loss, q_value_loss, log_probs, alpha_loss, _, _, _, _ = actor_and_critic_optimizer.soft_q_update_std()
        else:
            policy_loss, q_value_loss, log_probs, alpha_loss = 0, 0, 0, 0
        if update_iter % 2 == 0 and update_iter > 0:
            alpha = actor_and_critic_optimizer.log_alpha.exp().clone().detach().cpu().numpy()[0]
            rewards_mean = torch.cat(rewards).detach().clone().cpu().numpy().mean()
            length_mean = np.concatenate(lengths).mean()
            end = time.time()
            elapsed_time = end - start
            print("___________________________________________")
            print("update iteration: ", update_iter)
            print("number of Steps: ", step_idx)
            print("number of Frames: ", frame_idx)
            print("time for epoch (s): ", elapsed_time)
            print("time till finish (h): ", (float(elapsed_time) * (float(max_frames - step_idx) / num_steps)) / 3600.)
            print('rewards_mean ', rewards_mean)
            print('q_value_loss ', q_value_loss)
            print('policy_loss ', policy_loss)
            print('log_probs ', log_probs)
            print('alpha_loss ', alpha_loss)
            print('alpha ', alpha)
            print('length_mean ', length_mean)
            writer.add_scalar('training/num_frames', frame_idx, update_iter)
            writer.add_scalar('training/elapsed_time', elapsed_time, update_iter)
            writer.add_scalar('training/rewards_mean', rewards_mean, update_iter)
            writer.add_scalar('training/q_value_loss', q_value_loss, update_iter)
            writer.add_scalar('training/policy_loss', policy_loss, update_iter)
            writer.add_scalar('training/log_probs', log_probs, update_iter)
            writer.add_scalar('training/alpha_loss', alpha_loss, update_iter)
            writer.add_scalar('training/alpha', alpha, update_iter)
            writer.add_scalar('training/length_mean', length_mean, update_iter)

        if env_name == "KukaDiverseObjectGrasping-v0":
            state = envs.reset()

        if update_iter % 100 == 0 and update_iter > 0:
            test_reward = np.mean([eval_env(env, actor_critic, visualize=False, action_scale=action_scale,
                                            use_additional_normal=use_additional_normal) for _ in range(10)])
            test_rewards.append(test_reward)
            print("Test Reward: ", test_reward)
            writer.add_scalar('training/test_reward', test_reward, update_iter)
            actor_critic.save(os.path.join(output_dir, env_name + "_d.pt"))

    if max_frames > 0:
        actor_critic.save(os.path.join(output_dir, env_name + "_d.pt"))

    global_end = time.time()
    elapsed_time = global_end - global_start
    print("___________________________________________")
    print("Total time (s): ", elapsed_time)
##################################################################################################################################################
##################################################################################################################################################
######################################################### Test Model #############################################################################
##################################################################################################################################################
##################################################################################################################################################
    if test_env:
        test_rewards = []
        eval_env(env, actor_critic, visualize=visualize, action_scale=action_scale,
                 use_additional_normal=use_additional_normal)
        for _ in range(100):
            test_reward = eval_env(env, actor_critic, action_scale=action_scale,
                                   use_additional_normal=use_additional_normal)
            print("Test Reward: ", test_reward)
            test_rewards.append(test_reward)
        test_reward = np.mean(test_rewards)
        print("Mean Test Reward: ", test_reward)

    from itertools import count

    max_expert_num = 1000000000
    n_traj = 0
    max_n_traj = 200
    num_steps = 0
    expert_states = []
    expert_next_states = []
    expert_actions = []
    expert_time_steps = []

    expert_states_traj = []
    expert_next_states_traj = []
    expert_actions_traj = []
    expert_time_steps_traj = []

    if "CartPole" in args.env_name:
        traj_length = 200
        use_reward_selection = True
        min_reward = 200
        max_diff_traj_length = 0
        discard_small_traj = True
    elif "BulletEnv" in args.env_name:
        traj_length = 1000
        use_reward_selection = True
        min_reward = 2200
        max_diff_traj_length = 20
        discard_small_traj = True
    else:
        traj_length = 1000
        use_reward_selection = False
        min_reward = 100
        max_diff_traj_length = 0
        discard_small_traj = False

    max_runs = 1000
    saved_rewards = []

    if max_expert_num > 0:
        for i_episode in count():
            state = env.reset()
            done = False
            total_reward = 0
            time_step = 0

            while not done:
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                dist, _ = actor_critic(state)
                action = dist.sample()

                if is_discrete:
                    _, gym_action = action.max(dim=1)
                    gym_action = gym_action.cpu().numpy()[0]
                    action = action.cpu().numpy()[0]
                else:
                    if action_scale is not None:
                        action = torch.clamp(action, -action_scale, action_scale)
                    if use_additional_normal:
                        gym_action = action[:, 0].unsqueeze(1)
                    else:
                        gym_action = action
                    gym_action = gym_action.cpu().numpy()[0]
                    action = action.cpu().numpy()[0]

                next_state, reward, done, _ = env.step(gym_action)
                total_reward += reward
                expert_states.append(state.cpu().numpy())
                expert_next_states.append(np.expand_dims(next_state, axis=0))
                expert_actions.append(np.expand_dims(action, axis=0))
                expert_time_steps.append(np.expand_dims(np.array([time_step], dtype="float32"), axis=0))
                state = next_state
                num_steps += 1
                time_step += 1

            print("episode:", i_episode, "reward:", total_reward)
            max_action = np.max(expert_actions)
            mean_action = np.mean(expert_actions)
            print("max_action", max_action)
            print("mean_action", mean_action)

            gen_traj_len = len(expert_states)
            discard_traj = False
            if discard_small_traj and gen_traj_len < (traj_length-max_diff_traj_length):
                discard_traj = True
            if use_reward_selection and total_reward < min_reward:
                discard_traj = True
            
            if discard_traj:
                print("Generated traj. to short: ", gen_traj_len)
                expert_states = []
                expert_next_states = []
                expert_actions = []
                expert_time_steps = []
                if i_episode > max_runs:
                    print("to much runs without long traj, aborting... ")
                    break
            else:
                saved_rewards.append(total_reward)
                n_traj += 1
                if gen_traj_len < traj_length:
                    state = expert_states[-10]
                    next_state = expert_next_states[-10]
                    expert_action = expert_actions[-10]
                    expert_time_step = expert_time_steps[-10]
                    add_steps = (traj_length - len(expert_states))
                    for step in range(add_steps):
                        expert_states.append(state)
                        expert_next_states.append(next_state)
                        expert_actions.append(expert_action)
                        expert_time_steps.append(expert_time_step)

                saved_traj_len = len(expert_states)

                print("Generated traj. length: ", gen_traj_len)
                print("Saved traj. length: ", saved_traj_len)

                expert_states = np.concatenate(expert_states, axis=0)
                expert_states_traj.append(expert_states)

                expert_next_states = np.concatenate(expert_next_states, axis=0)
                expert_next_states_traj.append(expert_next_states)

                expert_actions = np.concatenate(expert_actions, axis=0)
                expert_actions_traj.append(expert_actions)

                expert_time_steps = np.concatenate(expert_time_steps, axis=0)
                expert_time_steps_traj.append(expert_time_steps)

                expert_states = []
                expert_next_states = []
                expert_actions = []
                expert_time_steps = []

                if n_traj >= max_n_traj:
                    break

        print("Expert file mean reward: ", np.mean(saved_rewards))
        csv_path = os.path.join(
            output_dir, env_name + "_" + str(int(np.mean(saved_rewards))) + "_metrics.csv")
        with open(csv_path, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                ["mean reward", "{:.6f}".format(np.mean(saved_rewards))])
            writer.writerow(saved_rewards)
        
        expert_states = np.asarray(expert_states_traj, dtype="float32")

        expert_next_states = np.asarray(expert_next_states_traj, dtype="float32")

        expert_actions = np.asarray(expert_actions_traj, dtype="float32")

        expert_time_steps = np.asarray(expert_time_steps_traj, dtype="float32")

        train_size = int(np.shape(expert_states)[0] * 0.8)   # TODO: 0.8
        train_x, valid_x = expert_states[0:train_size, :], expert_states[train_size:, :]
        train_n_x, valid_n_x = expert_next_states[0:train_size, :], expert_next_states[train_size:, :]
        train_y, valid_y = expert_actions[0:train_size, :], expert_actions[train_size:, :]
        train_t, valid_t = expert_time_steps[0:train_size, :], expert_time_steps[train_size:, :]

        train_file = os.path.join(output_dir, env_name + "expert.hdf5")
        if not os.path.isfile(train_file):
            print("generate hdf5 file")
            with h5py.File(train_file, "a") as hdf5_store:
                train_x_h = hdf5_store.create_dataset("train_x", train_x.shape)
                train_x_h[:] = train_x[:]
                train_n_x_h = hdf5_store.create_dataset("train_n_x", train_n_x.shape)
                train_n_x_h[:] = train_n_x[:]
                valid_n_x_h = hdf5_store.create_dataset("valid_n_x", valid_n_x.shape)
                valid_n_x_h[:] = valid_n_x[:]
                train_y_h = hdf5_store.create_dataset("train_y", train_y.shape)
                train_y_h[:] = train_y[:]
                valid_x_h = hdf5_store.create_dataset("valid_x", valid_x.shape)
                valid_x_h[:] = valid_x[:]
                valid_y_h = hdf5_store.create_dataset("valid_y", valid_y.shape)
                valid_y_h[:] = valid_y[:]
                train_t_h = hdf5_store.create_dataset("train_t", train_t.shape)
                train_t_h[:] = train_t[:]
                valid_t_h = hdf5_store.create_dataset("valid_t", valid_t.shape)
                valid_t_h[:] = valid_t[:]

        print("expert file generated!")
    quit()

