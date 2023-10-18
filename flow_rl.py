from functools import reduce
import sys
import os
import csv
import argparse
import json
import random
import time
import numpy as np
import gym 
import copy 
import math
import socket
import logging
import pybullet_envs
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import MultivariateNormal
from tensorboardX import SummaryWriter
from collections import deque
from matplotlib import pyplot as plt

from common import helper
from common.helper import make_env
from common.multiprocessing_env import BatchMultiVecEnv, SubprocVecEnv
from common.eval_policy import eval_policy_actions, eval_env, single_eval_env

from rllib.sac import SAC, compress
from rllib import batch_generator
from rllib.model_sac import FlowSAC, MLPSAC
from rllib.model import ActorCritic
from rllib.behavioral_cloning import BehavioralCloning
from rllib.discriminator import Discriminator, train_discriminator
from rllib.forward_backward_prob import ForwardBackwardCondLL
from rllib.survival_est import ExpSurvivalEst
from rllib.sac import ReplayBuffer

from nflib.flows import NormalizingFlowModel
from nflib.eval_nf import eval_state_ll_model
from nflib.slil import train_expert_state_est_model
from nflib.freia_c_flow import CINN
from nflib.slil import update_policy_forward_backward_models, train_policy_state_est_model, generate_data_new
from nflib.straight_throug_clamp import StraightThrougClamp


if __name__ == '__main__':
    old_stdout = sys.stdout
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--env-name",
                        default="HalfCheetahBulletEnv-v0",
                        help="Test Environment name used by OpenAI Gym (e.g.BipedalWalker-v3,"
                             "HalfCheetahBulletEnv-v0, AntBulletEnv-v0, HopperBulletEnv-v0,"
                             "Walker2DBulletEnv-v0, HumanoidBulletEnv-v0, KukaBulletEnv-v0, RacecarBulletEnv-v0,"
                             "CartPoleContinuousBulletEnv-v0).")
    parser.add_argument("--exp-name", required=True,
                        help="Name of the experiment. A folder with this name is generated")
    parser.add_argument("--output-dir",
                        default="./Experiments/",
                        help="Where to put output files, should be unique")
    parser.add_argument(
        '--il-method',
        type=str,
        default="soil", 
        help='il method, can be "gail", "soil", "form", "bc", "abl"')
    parser.add_argument('--overwrite',
                        action='store_true',
                        default=False,
                        help='Overwrite existing data in directory')
    parser.add_argument(
        '--seed',
        type=int,
        default=7,
        help='seed used for torch and numpy default: 7')
    parser.add_argument('--expert-filename', default='feature_pandas', help='name of data set file')
    parser.add_argument(
        '--n-trajectories',
        type=int,
        default=10,
        help='Number of update steps (default: 10, max: 25, all: -1, min: 1).')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=250,
        help='Number of update steps (default: 400).')
    parser.add_argument(
        '--total-epochs',
        type=int,
        default=400,
        help='Number of update steps (default: 1000).')
    parser.add_argument(
        '--n-state-est-train',
        type=int,
        default=0,
        help='Number of update steps (recommended: 40000).')
    parser.add_argument(
        '--num-bc-updates',
        type=int,
        default=0,
        help='Number of update steps (recommended between 10000 and 200000).')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size used for training of state estimation flow model (default: 256).')
    parser.add_argument(
        '--n-flows',
        type=int,
        default=16,
        help='Amount of stacked NF transformations in state estimation flow model.')
    parser.add_argument(
        '--n-flows-policy',
        type=int,
        default=16,
        help='Amount of stacked flow blocks in policy.')
    parser.add_argument(
        '--policy-flow-hidden',
        type=int,
        default=8,
        help='Hidden layer size in NF models.')
    parser.add_argument(
        '--acondsps-alpha',
        type=float,
        default=0.0,
        help='exponent of the a|s\',s weights for statecondll training, set to 0 if no importance weighting is desired')
    parser.add_argument(
        '--mci-ns-samples',
        type=int,
        default=1,
        help='Number of next-state samples for monte carlos integration for reward computation')
    parser.add_argument(
        '--alpha-nonpolicy-rampup',
        type=int,
        default=0,
        help='Number of epochs until the nonpolicy terms of statecondll are at alpha. Use 150 for halfcheetah, ant and walker2d')
    parser.add_argument(
        '--statell-epochs',
        type=int,
        default=3,
        help='Number of epochs that only the statell model is trained (SOIL-TDM)')
    parser.add_argument(
        '--q-epochs',
        type=int,
        default=2,
        help='Number of epochs that the q-function is trained (after statell training)')
    parser.add_argument(
        '--exp-clamping',
        type=float,
        default=4.,
        help='exp clamping value in policy')
    parser.add_argument(
        '--hidden-size-cond',
        type=int,
        default=64,
        help='Hidden size in codition network in policy')
    parser.add_argument(
        '--con-dim-features',
        type=int,
        default=8,
        help='Number of feature in condition vector in policy')
    parser.add_argument(
        '--lr', type=float, default=1e-4, help='learning rate for state estimation model(default: 1e-4)')
    parser.add_argument(
        '--lr-bc', type=float, default=1e-4, help='learning rate for policy using BC(default: 1e-4)')
    parser.add_argument(
        '--lr-p', type=float, default=5e-4, help='learning rate for policy using ESLIL(default: 1e-4)')
    parser.add_argument(
        '--lr-q', type=float, default=1e-3, help='learning rate for q estimation model')
    parser.add_argument(
        '--lr-d', type=float, default=7e-4, help='learning rate for q estimation model')
    parser.add_argument(
        '--polyak', type=float, default=0.995, help='polyak rate')
    parser.add_argument(
        '--gamma', type=float, default=0.7, help='gamma used in SAC for future reward')
    parser.add_argument(
        '--bc-batch-size',
        type=int,
        default=256,
        help='Batch size used in pretraining policy.')
    parser.add_argument(
        '--weight-decay', type=float, default=1e-5, help='weight decay used by adam optimizer (default: 1e-5)')
    parser.add_argument(
        '--clip', type=float, default=100, help='clip l2 norm value for training '
                                                'of state estimation model (default: 100)')
    parser.add_argument(
        '--load-model',
        action='store_true',
        default=False,
        help='load previously trained models.')
    parser.add_argument(
        '--print-to-file',
        action='store_true',
        default=False,
        help='print messages into file.')
    parser.add_argument(
        '--num-envs',
        type=int,
        default=16,
        help='Number of environments (default 37)')
    parser.add_argument(
        '--grad-clip-val',
        type=float,
        default=1.,
        help='Value used for ppo gradient clipping')
    parser.add_argument(
        '--mini-batch-size', type=int, default=2048, help='batch size in sac optimization')
    parser.add_argument(
        '--internal-update-epochs', type=int, default=500, help='amount of updates after each data collection epoch')
    parser.add_argument(
        '--alpha', type=float, default=1., help='entropy factor in sac optimization (higher means higher entropy), '
                                                 'will be adapted if automatic temperature adaptation is used.')
    parser.add_argument(
        '--alpha-sll', type=float, default=1., help='entropy factor in sac optimization (higher means higher entropy), '
                                                'weighting for state ll model estimates. (SOIL-TDM)')
    parser.add_argument(
        '--alpha-sll-a', type=float, default=1., help='entropy factor in sac optimization (higher means higher entropy), '
                                                    'weighting for state ll action model estimates (a given s ns). (SOIL-TDM)')
    parser.add_argument(
        '--automatic-entropy-tuning',
        action='store_true',
        default=False,
        help='enable automatic entropy tuning (trains alpha (entropy weighting) based on current policy)')
    parser.add_argument(
        '--target-entropy',
        type=float,
        default=None,
        help='targeted policy entropy, if None heuristic value measure is used (amount of actions)')
    parser.add_argument(
        '--replay-buffer-size', type=int, default=100000, help='batch size in sac optimization')
    parser.add_argument(
        '--load-sllm',
        action='store_true',
        default=False,
        help='load pretrained state log likelihood model saved in ./data')
    parser.add_argument(
        '--normalize-logprob',
        type=float,
        default=1.,
        help='Log Probability Normalization. Should be set that the resulting log probs are abs(lp)<10')
    parser.add_argument(
        '--hidden-size-critics', type=int, default=512, help='batch size in sac optimization')
    parser.add_argument(
        '--use-mlp',
        action='store_true',
        default=False,
        help='Use MLP instead of NF for policy')
    parser.add_argument(
        '--n-disc-train', type=int, default=20, help='batch size in sac optimization')
    parser.add_argument(
        '--exp-clamping-expert',
        type=float,
        default=4.,
        help='exp clamping value in policy (SOIL-TDM)')
    parser.add_argument(
        '--con-dim-features-ell',
        type=int,
        default=32, 
        help='Expert Model Condition Vector Size (SOIL-TDM)')
    parser.add_argument(
        '--hidden-size-cond-ell',
        type=int,
        default=64,
        help='Expert Model hidden Size (SOIL-TDM)')
    parser.add_argument(
        '--flow-state-hidden',
        type=int,
        default=64,
        help='Hidden layer size in state estimation flow model (SOIL-TDM)')
    parser.add_argument(
        '--exponent-clamping-scll',
        type=float,
        default=1.,
        help='exp clamping value in state conditional likelihood model (SOIL-TDM)')
    parser.add_argument(
        '--flow-state-hidden-scll',
        type=int,
        default=256,  # 32
        help='Hidden layer size in state conditional likelihood model (SOIL-TDM)')
    parser.add_argument(
        '--con-dim-features-scll',
        type=int,
        default=32,
        help='Number of feature in condition vector in state conditional likelihood model (SOIL-TDM)')
    parser.add_argument(
        '--hidden-size-cond-scll',
        type=int,
        default=256,
        help='Hidden size in codition network in state conditional likelihood model (SOIL-TDM)')
    parser.add_argument(
        '--exponent-clamping-scll-ns',
        type=float,
        default=1.,
        help='exp clamping value in state conditional likelihood model for the next state (SOIL-TDM)')
    parser.add_argument(
        '--flow-state-hidden-scll-ns',
        type=int,
        default=48,  # 32
        help='Hidden layer size in state conditional likelihood model for the next state (SOIL-TDM)')
    parser.add_argument(
        '--con-dim-features-scll-ns',
        type=int,
        default=32,
        help='Number of feature in condition vector in state conditional likelihood model for the next state (SOIL-TDM)')
    parser.add_argument(
        '--hidden-size-cond-scll-ns',
        type=int,
        default=48,
        help='Hidden size in codition network in state conditional likelihood model for the next state (SOIL-TDM)')
    parser.add_argument(
        '--early-term-penalty',
        type=float,
        default=0.,
        help='Penalty for early termination')
    parser.add_argument(
        '--use-state-cond-ll-buffer',
        action='store_true',
        default=False,
        help='Use separate buffer for state conditional log likelihood (SOIL-TDM)')
    parser.add_argument(
        '--alpha-sched',
        action='store_true',
        default=False,
        help='Use alpha scheduling')
    parser.add_argument(
        '--expert-noise-value',
        type=float,
        default=0.002,
        help='add noise to expert data - the initial value used at the beginning of training (SOIL-TDM or FORM)')
    parser.add_argument(
        '--final-noise-value',
        type=float,
        default=0.002,
        help='add noise to expert data - the final value used at the end of training (SOIL-TDM or FORM)')
    parser.add_argument(
        '--use-noise-sched',
        action='store_true',
        default=False,
        help='Set to true if you want to use noise schedule for exper model training for SOIL-TDM or FORM. Highly recommended to use noise schedule for exper model training!')
    parser.add_argument(
        '--split-traj',
        action='store_true',
        default=False,
        help='splitting trajectories into train and validation set instead of using separate trajectories')
    parser.add_argument(
        '--n-train-a-fac',
        type=int,
        default=3,
        help='Factor which determines how much more the action log likelihood model is trained with respect to the state log likelihodd model updates')
    parser.add_argument(
        '--start-traj',
        type=int,
        default=0,
        help='Select different trajectorie for training and validation - With this you can check home homogeneous/heterogeneous your data is')
    parser.add_argument(
        '--start-epoch-lrsched',
        type=int,
        default=0,
        help='When to start to decreases the learning rate linearly for actor critic optimizer')
    parser.add_argument(
        '--update-rl-lr',
        action='store_true',
        default=False,
        help='set to true if you want to update the learning rate of the actor critic optimizer')
    parser.add_argument(
        '--noise-red-factor',
        type=float,
        default=2.,
        help='Factor which determines how much the noise is reduced in each epoch for training of the expert model in SOIL-TDM or FORM')
    parser.add_argument(
        '--use-delta-state',
        action='store_true',
        default=False,
        help='Use next_state = next_state - state in state conditional likelihood models')
    parser.add_argument(
        '--next-state-ll-noise',
        type=float,
        default=0.0005,
        help='Amount of noise added to next state in state conditional likelihood models')
    parser.add_argument(
        '--use-fifo-replay-buffer',
        action='store_true',
        default=False,
        help='Set true if Replaybuffer for SAC optimization should be FIFO instead of random')
    parser.add_argument(
        '--reduce-n-train',
        action='store_true',
        default=False,
        help='Reduce amout of optimizations per epoch for forward backword ll models (SOIL-TDM), state ll model (FORM), discriminator (GAIL) after 250 epochs.')
    parser.add_argument(
        '--con-ll-batch-size',
        type=int,
        default=512,
        help='Batch size for training of forward backword conditional log likelihood model (SOIL-TDM), state ll model (FORM), discriminator (GAIL)')
    parser.add_argument(
        '--use-cpu',
        action='store_true',
        default=False,
        help='Avoid using GPU')
    parser.add_argument(
        '--number-of-threads',
        type=int,
        default=4,
        help='For torch cpu - torch.set_num_threads - default: 4')
    parser.add_argument(
        '--exp-data-perc',
        type=float,
        default=1.0,
        help='Percentage of the experct dataset used. Alternative to n-trajectories')
    parser.add_argument(
        '--hidden-size-policy',
        type=int,
        default=0,
        help='If 0 the hidden size if the critic will be used.')
    parser.add_argument(
        '--bc-noise-value',
        type=float,
        default=0.0,
        help='Amplitude of Gaussian noise added to the state values during BC training.')
    parser.add_argument(
        '--enforce-horizon',
        type=int,
        default=-1,
        help='Set to number of steps to enforce horizon for. Set to -1 to disable.')
    parser.add_argument(
        '--single-step-initialization',
        action='store_true',
        default=False,
        help='Start policy training with a single environment step.')
    parser.add_argument(
        '--single-step-initialization-epochs',
        type=int,
        default=25,
        help='Amount of epochs for single step initialization.')

    args = parser.parse_args()

    for k, v in args._get_kwargs():
        print(k, "=", v)

    # These parameter should stay
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if args.use_cpu: 
        device = "cpu"
        torch.set_num_threads(args.number_of_threads)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            torch.set_num_threads(1)
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            torch.set_num_threads(1)
        else:
            device = torch.device("cpu")
            torch.set_num_threads(args.number_of_threads)

    num_envs = args.num_envs
    output_dir = os.path.join(args.output_dir, args.exp_name)

    # Simulation and expert data set
    env_name = args.env_name  # "MountainCar-v0"
    expert_file_dir = "./gym_exp" 
    expert_filename = env_name + "expert"  # "MountainCar-v0expert.hdf5"
    load_pt_state_est_model = args.load_sllm

    visualize = False

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    elif len(os.listdir(output_dir)) > 0 and not args.overwrite:
        print("Directory already available. This would overwrite existing files!")
        print("Set overwrite if you want to overwrite existing files!")
        raise FileExistsError("Directory already available. Abort!")

    with open(os.path.join(output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))

    if args.print_to_file:
        log_file = open(os.path.join(output_dir, "message.log"), "w")
        sys.stdout = log_file

    writer = SummaryWriter(output_dir, comment=env_name + "_flow_rl")

    ################################
    ####### Load expert data #######
    ################################
    test_mode = True
    n_trajectories = args.n_trajectories
    exp_data_perc = args.exp_data_perc
    train_state, train_action, valid_state, valid_action, \
    train_next_state, valid_next_state = helper.load_hdf5_data_wns(expert_file_dir, expert_filename,
                                                                       n_trajectories,
                                                                       start_traj=args.start_traj, test_mode=test_mode, 
                                                                       load_perc=exp_data_perc)

    ################################
    #### Preprocess expert data ####
    ################################
    if args.split_traj:
        train_size = int(np.shape(train_state)[0] * 0.8)
        train_state, valid_state = train_state[:train_size, :], train_state[train_size:, :]
        train_action, valid_action = train_action[:train_size, :], train_action[train_size:, :]
        if train_next_state is not None:
            train_next_state, valid_next_state = train_next_state[:train_size, :], train_next_state[train_size:, :]

    ################################
    ##### Get Scale information ####
    ######## from the data #########
    ################################
    train_action_scale = np.max(np.abs(train_action))
    test_action_scale = np.max(np.abs(valid_action))
    action_scale_ = max(train_action_scale, test_action_scale)
    use_norm = False
    env = gym.make(env_name)
    if visualize:
        env.render()
        env.reset()
        env.render()

    ################################
    #### Build Gym Environments ####
    ################################
    def make_env_simple():
            def _thunk():
                env = gym.make(env_name)
                return env
            return _thunk
    envs = [make_env_simple() for i in range(num_envs)]
    envs = SubprocVecEnv(envs, use_norm=False)
    test_envs = env
        

    ################################
    # Get Environment Information ##
    ################################
    use_additional_normal = False
    num_observations = envs.observation_space.shape[0]
    num_observations = train_state.shape[1]
    if envs.action_space.__class__.__name__ == "Discrete":
        raise NotImplementedError()
    else:
        num_action = envs.action_space.shape[0]
        action_scale_low = env.action_space.low
        action_scale_high = env.action_space.high
        action_scale = action_scale_high[0]
        print("action scale from data: ", action_scale_)
        print("action scale from environment: ", action_scale_high[0])
        print("action scale used: ", action_scale)
        # action_scale = 10.
        if not args.use_mlp and num_action == 1:
            print("one dimensional action, we add one action to use flow")
            use_additional_normal = True
            num_action += 1

    ################################
    ########## Build Policy ########
    ################################
    use_std_cond_normal = False
    max_nll = 12
    use_stacked_policy_blocks = False
    use_mlp = args.use_mlp        
    if args.hidden_size_policy > 0:
        hidden_size_policy = args.hidden_size_policy
    else:
        hidden_size_policy = None 
    if use_mlp:
        if use_std_cond_normal:
            # Standard Conditional Normal without Tanh (for SAC, hence don't use with SAC)
            actor_critic = ActorCritic(num_inputs=num_observations, num_outputs=num_action,
                                       hidden_size=args.hidden_size_critics, device=device, act_limit=action_scale)
        else:
            # Standard Conditional Normal Tanh (for SAC)
            actor_critic = MLPSAC(num_inputs=num_observations, num_outputs=num_action, hidden_size=args.hidden_size_critics,
                                  act_limit=action_scale, device=device, hidden_size_policy=hidden_size_policy).to(device)

    else:
        # Conditional Normalizing Flow Policy with Tanh (for SAC)
        actor_critic = FlowSAC(num_inputs=num_observations, num_outputs=num_action, hidden_size=args.hidden_size_critics,
                               n_flows=args.n_flows_policy, flow_hidden=args.policy_flow_hidden,
                               exp_clamping=args.exp_clamping, hidden_size_cond=args.hidden_size_cond,
                               act_limit=action_scale, con_dim_features=args.con_dim_features,
                               use_stacked_blocks=use_stacked_policy_blocks, device=device).to(device)


    ################################
    ########## Build Agent #########
    ################################
    if args.target_entropy is not None:
        target_entropy = -args.target_entropy  # 0.1  # None  # 2
    else:
        target_entropy = None
    actor_and_critic_optimizer = SAC(actor_critic, batch_size=args.mini_batch_size, device=device,
                                     epochs=args.internal_update_epochs,
                                     soft_q_lr=args.lr_q, policy_lr=args.lr_p,
                                     replay_buffer_size=args.replay_buffer_size,
                                     alpha=args.alpha, polyak=args.polyak, gamma=args.gamma,
                                     grad_clip_val=args.grad_clip_val,
                                     use_automatic_entropy_tuning=args.automatic_entropy_tuning,
                                     target_entropy=target_entropy, max_nll=max_nll, use_fifo_replay_buffer=args.use_fifo_replay_buffer)

    ####################################
    # define which method we are using #
    ####################################
    use_state_cond_ll_form = False
    use_state_cond_ll = False
    use_gail = False
    eval_expert_model_mode = False
    if "soil" in args.il_method:
        use_state_cond_ll = True
    elif "form" in args.il_method:
        use_state_cond_ll_form = True
    elif "abl" in args.il_method:
        use_only_expert_model = True
    elif "gail" in args.il_method or "dac" in args.il_method:
        use_gail = True
    elif "eval-expert" in args.il_method:
        eval_expert_model_mode = True
    elif "bc" in args.il_method:
        if args.load_sllm:
            use_state_cond_ll = True


    ################################
    ###### Build Expert Model ######
    ##### to estimate reward #######
    ################################
    exponent_clamping = args.exp_clamping_expert
    # for SOIL-TDM Model based estimation
    def build_state_est_model():
        state_est_obs = num_observations

        state_prior = MultivariateNormal(torch.zeros(state_est_obs).to(device),
                                         torch.eye(state_est_obs).to(device))

        state_flows = CINN(n_data=state_est_obs, n_cond=state_est_obs, n_blocks=args.n_flows,
                           internal_width=args.flow_state_hidden, hidden_size_cond=args.hidden_size_cond_ell,
                           exponent_clamping=args.exp_clamping_expert, y_dim_features=args.con_dim_features_ell,
                           use_snn=False, model_device=device).to(device)

        state_est_model = NormalizingFlowModel(state_prior, state_flows, device=device,
                                               prep_con=False, is_freia=True).to(device)
        state_est_model.train()
        return state_est_model

    if use_gail:
        # Build Discriminator for GAIL 
        discrim_hidden_size = 64
        use_spec_norm = False
        # lr_d = 7e-4
        use_std_reward = False
        discriminator = Discriminator(num_observations + num_action, discrim_hidden_size,
                                      use_spec_norm=use_spec_norm, device=device,
                                      use_std_reward=use_std_reward).to(device)

        discrim_criterion = nn.BCELoss()

        optimizer_discrim = optim.Adam(discriminator.parameters(), lr=args.lr_d)
        expert_state_est_model = None
    elif "bc" in args.il_method:
        # we don't need a discriminator or expert model
        discriminator = None
        expert_state_est_model = None
    else:  
        # Build Expert Model for SOIL-TDM, FORM, ablation-study or expert evaluation 
        discriminator = None
        expert_state_est_model = build_state_est_model()

        # optimizer
        state_optimizer = optim.Adam(expert_state_est_model.parameters(), lr=args.lr, amsgrad=True, betas=(0.9, 0.998))
        print("number of params in state model: ", sum(p.numel() for p in expert_state_est_model.parameters()))

    # load state estimation model
    state_model_loaded = False
    if load_pt_state_est_model and expert_state_est_model is not None:
        dir_feature = './data/'
        state_model_name = env_name + "_state_" + str(args.n_trajectories) + "_d.pt"
        state_model_location = os.path.join(dir_feature, state_model_name)
        if os.path.isfile(state_model_location):
            expert_state_est_model.flow.load(state_model_location)
            state_model_loaded = True


    forward_backward_cond_ll_model = None
    forward_backward_cond_ll_model_form = None
    if use_state_cond_ll:
        # Build SOIL-TDM forward and Backward Model
        forward_backward_cond_ll_model = ForwardBackwardCondLL(num_observations, num_action, n_flows=args.n_flows,
                                                               flow_state_hidden=args.flow_state_hidden_scll,
                                                               hidden_size_cond=args.hidden_size_cond_scll, exp_clamping=args.exponent_clamping_scll,
                                                               y_dim_features=args.con_dim_features_scll,
                                                               flow_state_hidden_ns=args.flow_state_hidden_scll_ns,
                                                               hidden_size_cond_ns=args.hidden_size_cond_scll_ns,
                                                               y_dim_features_ns=args.con_dim_features_scll_ns,
                                                               exp_clamping_ns=args.exponent_clamping_scll_ns,
                                                               max_nll=max_nll, alpha_pol=args.alpha,
                                                               alpha_a=args.alpha_sll_a, alpha_ns=args.alpha_sll, grad_clip_val=args.grad_clip_val,
                                                               act_limit=action_scale, use_delta_state=args.use_delta_state,
                                                               device=device)
    elif use_state_cond_ll_form:
        # Build FORM based policy Model
        forward_backward_cond_ll_model_form = build_state_est_model()
        # optimizer
        state_cond_ll_model_form_optimizer = optim.Adam(forward_backward_cond_ll_model_form.parameters(
        ), lr=args.lr, amsgrad=True, betas=(0.9, 0.998))
        print("number of params in state model: ", sum(p.numel()
              for p in forward_backward_cond_ll_model_form.parameters()))

    # Build penalty model for early termination
    use_survival_est = False
    if use_survival_est:
        env_steps = 1000
        exp_survival_ext = ExpSurvivalEst(time_steps_expert=env_steps, penalty=-args.early_term_penalty)
    else:
        exp_survival_ext = None  # No early termination penalty

    ################################
    #### Load pretrained Models ####
    ################################
    if args.load_model:
        model_location = os.path.join(output_dir, env_name + "_d.pt")
        if os.path.isfile(model_location):
            actor_critic.load(model_location)
        else:
            print("No policy model available at " + model_location)
        if not state_model_loaded:
            state_model_name = env_name + "_best_state_d.pt"
            state_model_location = os.path.join(output_dir, state_model_name)
            state_model_loaded = False
            if os.path.isfile(state_model_location):
                expert_state_est_model.flow.load(state_model_location)
                state_model_loaded = True
        if forward_backward_cond_ll_model is not None:
            state_location = os.path.join(output_dir, env_name + "_state_model.pt")
            if os.path.isfile(state_location):
                forward_backward_cond_ll_model.load(state_location)
        if forward_backward_cond_ll_model_form is not None:
            state_location = os.path.join(
                output_dir, env_name + "_state_model_form.pt")
            if os.path.isfile(state_location):
                forward_backward_cond_ll_model_form.load(state_location)
        if discriminator is not None:
            discr_location = os.path.join(output_dir, env_name + "_discriminator.pt")
            if os.path.isfile(discr_location):
                discriminator.load(discr_location)

    ############################################
    #### Train expert state estimation model ###
    ############################################
    start_exptrain = time.time()
    test_gen_models = False
    if args.n_state_est_train > 0 and expert_state_est_model is not None:
        expert_test_generator = batch_generator.SimpleBatchGenerator(valid_state, valid_next_state,
                                                                     is_state_conditional=True,
                                                                     noise_value=args.expert_noise_value
                                                                     )

        expert_train_generator = batch_generator.SimpleBatchGenerator(train_state, train_next_state,
                                                                      is_state_conditional=True,
                                                                      noise_value=args.expert_noise_value)

        train_expert_state_est_model(args.n_state_est_train, expert_state_est_model, expert_train_generator, args.bc_batch_size,
                                     state_optimizer, output_dir, env_name, args.clip, args.lr, writer=writer,
                                     use_linear_lr_decay=True, expert_test_generator=expert_test_generator,
                                     noise_value=args.expert_noise_value, use_noise_sched=args.use_noise_sched, 
                                     final_noise_value=args.final_noise_value, noise_red_factor=args.noise_red_factor,
                                     use_early_stopping=True)

        best_model_name = os.path.join(output_dir, env_name + "_" + "best" + "_state_d.pt")
        if os.path.isfile(best_model_name):
            expert_state_est_model = build_state_est_model()
            expert_state_est_model.flow.load(best_model_name)

        if test_gen_models:
            ## Test State Flow Model:
            eval_state_ll_model(expert_test_generator, expert_state_est_model, output_dir,
                                env_name=env_name, n_eval_steps=2000, eval_bs=256,
                                do_forward=True, do_backward=False, calculate_jsd=False, grid_plot=True)
    ############################################
    #### Test expert state estimation model ####
    ############################################
    exp_state_ll = 0
    max_exp_state_ll = max_nll
    if expert_state_est_model is not None:
        expert_test_generator = batch_generator.SimpleBatchGenerator(valid_state, valid_next_state,
                                                                     is_state_conditional=True,
                                                                     noise_value=args.expert_noise_value)

        dataset_size = expert_test_generator.data_x.shape[0]
        do_backward = False

        eval_bs = args.bc_batch_size
        if dataset_size > eval_bs:
            n_eval_steps = int((dataset_size/eval_bs)*2)  # 1000
        else:
            n_eval_steps = 10
            eval_bs = dataset_size - 2
            if eval_expert_model_mode:
                n_eval_steps = n_eval_steps*4
        if dataset_size > 500000 and not eval_expert_model_mode:
            eval_bs = args.bc_batch_size
            n_eval_steps = 500
        
        exp_state_ll_list = eval_state_ll_model(expert_test_generator, expert_state_est_model, output_dir,
                                                env_name=env_name, n_eval_steps=n_eval_steps, eval_bs=eval_bs,
                                                do_forward=True, do_backward=do_backward, calculate_jsd=True)

        max_exp_state_ll = np.max(exp_state_ll_list)
        exp_state_ll = np.mean(exp_state_ll_list)
        print("Maximum Expert State log Likelihood: ", max_exp_state_ll)
        print("Mean Expert State log Likelihood: ", exp_state_ll)

        for param in expert_state_est_model.parameters():
            param.requires_grad = False

        if eval_expert_model_mode:
            end_exptrain = time.time()
            elapsed_time_exptrain = end_exptrain - start_exptrain
            print("time for expert model training and testing (s): ", elapsed_time_exptrain)
            
            try: 
                envs.close()
            except:
                print("Closing envs did not work...")

            try:
                test_envs.close()
            except:
                print("Closing test_envs did not work...")

            if args.print_to_file:
                sys.stdout = old_stdout
                log_file.close()

            print("done!")
            quit()

    if forward_backward_cond_ll_model is not None:
        forward_backward_cond_ll_model.use_straight_throug_clamp
        forward_backward_cond_ll_model.max_nll = max_exp_state_ll

    end_exptrain = time.time()
    elapsed_time_exptrain = end_exptrain - start_exptrain
    print("time for expert model training and testing (s): ", elapsed_time_exptrain)

    ################################################################
    #### pretrain policy on expert data using Behaviroal Cloning ###
    ################################################################
    helper.pretrain_bc(args, train_state, train_action, valid_state, valid_action,
                       BehavioralCloning, actor_critic, output_dir, env_name,
                       use_mlp, writer, device, num_bc_updates=args.num_bc_updates,
                       noise_value_x=args.bc_noise_value)

    # batch generator for discriminator training
    if discriminator is not None:
        expert_train_generator = batch_generator.SimpleBatchGenerator(train_state, train_action,
                                                                      is_state_conditional=True,
                                                                      noise_value=args.expert_noise_value)


    ########################################
    #### Start of RL main algorithm ####
    ########################################
    epoch = 0
    update_epoch = 0
    total_sim_steps = 0
    if args.total_epochs > 0:
        # reset envs
        device = actor_critic.device
        state = envs.reset()
        state = torch.FloatTensor(state).to(device)
        time_steps = np.zeros(state.shape[0])

        # train with horizon
        use_horizon_reset = False
        horizon = 1000  # set to -1 to deactivate external resetting
        if args.enforce_horizon > 0:
            use_horizon_reset = True
            horizon = args.enforce_horizon
        standard_horizon = horizon
        normalize_logprob = args.normalize_logprob
        length = torch.ones((state.shape[0],))
        lengths = []
        if args.use_state_cond_ll_buffer:
            state_cond_ll_buffer = ReplayBuffer(100000, use_randomized_buffer=(not args.use_fifo_replay_buffer))
        else:
            state_cond_ll_buffer = None
        target_forward_backward_cond_ll_model = copy.deepcopy(
            forward_backward_cond_ll_model)
        best_episode_reward = -np.inf
        best_mod_reward = -np.inf
        best_expert_reward = -np.inf
        best_episode_losse = -np.inf
        best_train_est_rewards = -np.inf 
        best_test10_mod_reward = -np.inf
        best_test10_expert_reward = -np.inf
        best_test10_env_reward = -np.inf
        total_episode_est_rewards = deque(maxlen=10)
        total_episode_mod_rewards = deque(maxlen=10)
        total_episode_losses = deque(maxlen=10)
        total_train_est_rewards = deque(maxlen=10)
        prev_best_ns_given_a_s_test_loss = np.inf
        prev_best_a_given_ns_s_test_loss = np.inf
        target_ns_updated = 0 
        target_a_updated = 0
        first_run = True

    for epoch in range(args.total_epochs):
        print("Start epoch: ", epoch)
        start = time.time()

        num_env_steps = args.num_env_steps
        if args.single_step_initialization and epoch < args.single_step_initialization_epochs:
            horizon = 1
        else: 
            horizon = standard_horizon

        # first interact with the environment and collect data
        state, data_frames, rewards, env_rewards, \
            sim_steps, length, sim_length = generate_data_new(state, num_env_steps,
                                                              actor_critic, envs,
                                                              actor_and_critic_optimizer,
                                                              length,
                                                              expert_state_est_model,
                                                              state_cond_ll_buffer=state_cond_ll_buffer,
                                                              horizon=horizon, use_horizon=use_horizon_reset)

        env_rewards = compress(torch.tensor(env_rewards), max_nll).cpu().numpy()
        rewards = StraightThrougClamp.apply(torch.tensor(
                rewards), -max_exp_state_ll, max_exp_state_ll).cpu().numpy()

        total_sim_steps += sim_steps
        use_horizon_reset_classic = False
        if use_horizon_reset_classic and use_horizon_reset and horizon > 0 and total_sim_steps >= horizon:
            state = envs.reset()
            state = torch.FloatTensor(state).to(device)
            total_sim_steps = 0
            length = torch.ones((state.shape[0],))

        # optimize additional models on rollout data
        lp_ns_losses, lp_a_losses = 0, 0
        lp_ns_a_s_losses, lp_a_ns_s_losses, lp_ns_s_losses, lp_a_s_losses = 0, 0, 0, 0
        mean_ns_given_a_s_test_loss, mean_a_given_ns_s_test_loss, mean_target_ns_given_a_s_test_loss, mean_target_a_given_ns_s_test_loss = 0, 0, 0, 0
        discriminator_loss = 0
        sample_batch_size = args.con_ll_batch_size
        if state_cond_ll_buffer is not None:
            replay_buffer = state_cond_ll_buffer
        else:
            replay_buffer = actor_and_critic_optimizer.replay_buffer
        if len(replay_buffer.buffer) > sample_batch_size:
            n_train = args.n_disc_train
            
            # Optimize forward and backward models of SOIL-TDM
            if forward_backward_cond_ll_model is not None:
                if args.reduce_n_train and epoch > 250: 
                    n_train = int(args.n_disc_train/4)
                lp_ns_a_s_losses, lp_a_ns_s_losses, lp_ns_s_losses, \
                lp_a_s_losses, mean_ns_given_a_s_test_loss, \
                mean_a_given_ns_s_test_loss, \
                mean_target_ns_given_a_s_test_loss, \
                mean_target_a_given_ns_s_test_loss = update_policy_forward_backward_models(forward_backward_cond_ll_model,
                                                                                           replay_buffer,
                                                                                           actor_critic = actor_and_critic_optimizer,
                                                                                           target_forward_backward_cond_ll_model=target_forward_backward_cond_ll_model,
                                                                                           n_train=n_train,
                                                                                           n_train_a_fac=args.n_train_a_fac,
                                                                                           batch_size=sample_batch_size,
                                                                                           importance_alpha=args.acondsps_alpha, 
                                                                                           next_state_noise=args.next_state_ll_noise,
                                                                                           max_nll=max_exp_state_ll) 
                use_target_forward_backward_cond_ll_model = True
                if use_target_forward_backward_cond_ll_model:
                    if first_run: 
                        target_forward_backward_cond_ll_model = copy.deepcopy(forward_backward_cond_ll_model)
                        ns_a_s_optimizer_sd = forward_backward_cond_ll_model.ns_a_s_optimizer.state_dict()
                        target_forward_backward_cond_ll_model.redefine_ns_a_s_optimizer()
                        target_forward_backward_cond_ll_model.ns_a_s_optimizer.load_state_dict(
                            ns_a_s_optimizer_sd)
                        a_ns_s_optimizer_sd = forward_backward_cond_ll_model.a_ns_s_optimizer.state_dict()
                        target_forward_backward_cond_ll_model.redefine_a_ns_s_optimizer()
                        target_forward_backward_cond_ll_model.a_ns_s_optimizer.load_state_dict(
                            a_ns_s_optimizer_sd)
                        target_ns_updated += 1
                        target_a_updated += 1
                        first_run = False
                    else:    
                        if mean_ns_given_a_s_test_loss < mean_target_ns_given_a_s_test_loss:
                            prev_best_ns_given_a_s_test_loss = mean_ns_given_a_s_test_loss
                            target_forward_backward_cond_ll_model.ns_a_s_model = copy.deepcopy(
                                forward_backward_cond_ll_model.ns_a_s_model)
                            ns_a_s_optimizer_sd = forward_backward_cond_ll_model.ns_a_s_optimizer.state_dict()
                            target_forward_backward_cond_ll_model.redefine_ns_a_s_optimizer()
                            target_forward_backward_cond_ll_model.ns_a_s_optimizer.load_state_dict(
                                ns_a_s_optimizer_sd)
                            target_ns_updated += 1
                        else:
                            print("reset training model....")
                            forward_backward_cond_ll_model.ns_a_s_model = copy.deepcopy(
                                target_forward_backward_cond_ll_model.ns_a_s_model)
                            ns_a_s_optimizer_sd = target_forward_backward_cond_ll_model.ns_a_s_optimizer.state_dict()
                            forward_backward_cond_ll_model.redefine_ns_a_s_optimizer()
                            forward_backward_cond_ll_model.ns_a_s_optimizer.load_state_dict(
                                ns_a_s_optimizer_sd)
                            
                        if mean_a_given_ns_s_test_loss < mean_target_a_given_ns_s_test_loss:
                            prev_best_a_given_ns_s_test_loss = mean_a_given_ns_s_test_loss
                            target_forward_backward_cond_ll_model.a_ns_s_model = copy.deepcopy(
                                forward_backward_cond_ll_model.a_ns_s_model)
                            a_ns_s_optimizer_sd = forward_backward_cond_ll_model.a_ns_s_optimizer.state_dict()
                            target_forward_backward_cond_ll_model.redefine_a_ns_s_optimizer()
                            target_forward_backward_cond_ll_model.a_ns_s_optimizer.load_state_dict(
                                a_ns_s_optimizer_sd)
                            target_a_updated += 1
                        else:
                            print("reset training model....")
                            forward_backward_cond_ll_model.a_ns_s_model = copy.deepcopy(
                                target_forward_backward_cond_ll_model.a_ns_s_model)
                            a_ns_s_optimizer_sd = target_forward_backward_cond_ll_model.a_ns_s_optimizer.state_dict()
                            forward_backward_cond_ll_model.redefine_a_ns_s_optimizer()
                            forward_backward_cond_ll_model.a_ns_s_optimizer.load_state_dict(
                                a_ns_s_optimizer_sd)
                else:
                    target_forward_backward_cond_ll_model = forward_backward_cond_ll_model

            # Optmimize policy state estimation model of Form
            if forward_backward_cond_ll_model_form is not None: 
                lp_ns_s_losses = train_policy_state_est_model(forward_backward_cond_ll_model_form, replay_buffer, 
                                                              state_cond_ll_model_form_optimizer, n_train=n_train,
                                                              # next_state_noise=args.next_state_ll_noise,
                                                              batch_size=sample_batch_size, next_state_noise=args.final_noise_value,
                                                              clip=args.grad_clip_val)

            # optimize Discriminator
            if discriminator is not None:
                discriminator_loss = train_discriminator(discriminator,
                                                         actor_and_critic_optimizer.replay_buffer,
                                                         expert_train_generator, sample_batch_size,
                                                         optimizer_discrim=optimizer_discrim,
                                                         discrim_criterion=discrim_criterion,
                                                         discriminator_update_iterations=n_train)

        # policy optimization using SAC
        lr_p = actor_and_critic_optimizer.policy_lr
        lr_q = actor_and_critic_optimizer.soft_q_lr
        if len(actor_and_critic_optimizer.replay_buffer.buffer) > actor_and_critic_optimizer.batch_size:
            update_epoch += 1
            aplha_pol = args.alpha

            if args.alpha_nonpolicy_rampup <= 1:  # 20, 70, 100, 150
                alpha_fac_nonpol = 1
            else:
                alpha_fac_nonpol = max(0,min(1.0,(epoch+1-args.statell_epochs)/args.alpha_nonpolicy_rampup))
            if args.update_rl_lr: 
                lr_p = actor_and_critic_optimizer.update_p_linear_schedule(
                    epoch, args.total_epochs, start_epoch=args.start_epoch_lrsched, lr_min=1e-5)
            use_soft_q_update_soil_tdm = True
            if args.alpha_sll_a <= 0 and args.alpha_sll <= 0:
                use_soft_q_update_soil_tdm = False
            if use_soft_q_update_soil_tdm and forward_backward_cond_ll_model is not None:
                if args.alpha_sched and epoch >= 300:
                    target_forward_backward_cond_ll_model.set_alpha(
                        alpha_pol=0.1, alpha_a=0.1, alpha_ns=0.1)
                    aplha_pol = 0.1
                policy_loss, q_value_loss, policy_base_losses, policy_log_probs, \
                alpha_loss, training_rewards, \
                    sampled_expert_rewards, sampled_q = actor_and_critic_optimizer.soft_q_update_soil_tdm(pol_update = epoch > (args.statell_epochs+args.q_epochs),
                                                                                                          state_cond_ll = target_forward_backward_cond_ll_model,
                                                                                                          discriminator = discriminator,
                                                                                                          state_est_model = expert_state_est_model,
                                                                                                          update_q = epoch > args.statell_epochs,
                                                                                                          method = "model",
                                                                                                          modify_entropy = False,
                                                                                                          alpha_fac_nonpol = alpha_fac_nonpol,
                                                                                                          alpha_policy=aplha_pol,
                                                                                                          cur_gamma = args.gamma,
                                                                                                          mci_ns_samples = args.mci_ns_samples,
                                                                                                          exp_survival_ext = exp_survival_ext,
                                                                                                          max_exp_state_ll = max_exp_state_ll)
            else:
                policy_base_losses = 0.
                policy_loss, q_value_loss, policy_log_probs,  alpha_loss, _, training_rewards, \
                sampled_expert_rewards, sampled_q = actor_and_critic_optimizer.soft_q_update_std(pol_update=epoch > (args.statell_epochs+args.q_epochs), 
                                                                                          expert_state_est_model=expert_state_est_model,
                                                                                          policy_state_est_model=forward_backward_cond_ll_model_form,
                                                                                          discriminator=discriminator)

        else:
            policy_loss, q_value_loss, policy_base_losses, policy_log_probs, alpha_loss = 0, 0, 0, 0, 0
            training_rewards, sampled_expert_rewards, sampled_q = [0], 0, 0

        # log losses:
        alpha = actor_and_critic_optimizer.log_alpha.exp().clone().detach().cpu().numpy()[0]
        mean_rewards = np.mean(rewards)
        mean_env_rewards = np.mean(env_rewards)
        buffer_sampled_expert_rewards = np.mean(sampled_expert_rewards)
        mean_q = np.mean(sampled_q)
        writer.add_scalar('rl_training/policy_loss', policy_loss, epoch)
        writer.add_scalar('rl_training/q_value_loss', q_value_loss, epoch)
        writer.add_scalar('rl_training/buffer_sampled_expert_rewards',
                          buffer_sampled_expert_rewards, epoch)
        writer.add_scalar('rl_training/env_rewards', mean_env_rewards, epoch)
        writer.add_scalar('rl_training/current_policy_expert_rewards', mean_rewards, epoch)
        writer.add_scalar('rl_training/current_policy_q', mean_q, epoch)
        writer.add_scalar('rl_training/policy_base_losses', policy_base_losses, epoch)
        writer.add_scalar('rl_training/policy_log_probs', policy_log_probs, epoch)
        writer.add_scalar('rl_training/alpha_loss', alpha_loss, epoch)
        writer.add_scalar('rl_training/alpha', alpha, epoch)
        writer.add_scalar('rl_training/lp_ns_a_s_losses', lp_ns_a_s_losses, epoch)
        writer.add_scalar('rl_training/lp_ns_given_a_s_test_loss',
                          mean_ns_given_a_s_test_loss, epoch)
        writer.add_scalar('rl_training/lp_a_given_ns_s_test_loss',
                          mean_a_given_ns_s_test_loss, epoch)
        writer.add_scalar('rl_training/lp_ns_given_a_s_target_test_loss',
                          mean_target_ns_given_a_s_test_loss, epoch)
        writer.add_scalar('rl_training/lp_a_given_ns_s_target_test_loss',
                          mean_target_a_given_ns_s_test_loss, epoch)
        writer.add_scalar('rl_training/lp_ns_s_losses', lp_ns_s_losses, epoch)
        writer.add_scalar('rl_training/lp_a_ns_s_losses', lp_a_ns_s_losses, epoch)
        writer.add_scalar('rl_training/mean_training_rewards', training_rewards, epoch)
        writer.add_scalar('rl_training/discriminator_loss', discriminator_loss, epoch)
        writer.add_scalar('rl_training/mean_length', sim_length, epoch)
        writer.add_scalar('rl_training/lr_p', lr_p, epoch)
        writer.add_scalar('rl_training/lr_q', lr_q, epoch)
        writer.add_scalar('rl_training/target_ns_updated',
                          target_ns_updated, epoch)
        writer.add_scalar('rl_training/target_a_updated',
                          target_a_updated, epoch)

        if exp_state_ll > 0.:
            writer.add_scalar('rl_training/rel_est_rewards', mean_rewards/exp_state_ll, epoch)


        print("_____________________________________")
        print("at iteration: ", epoch)
        print("collected experience ", data_frames)
        print('buffer_sampled_expert_rewards ', buffer_sampled_expert_rewards)
        if exp_state_ll > 0.:
            print('rel_est_rewards ', mean_rewards/exp_state_ll)
        print('env_rewards ', mean_env_rewards)
        print('current_policy_expert_rewards ', mean_rewards)
        print('current_policy_q ', mean_q)
        print('lp_ns_a_s_losses ', lp_ns_a_s_losses)
        print('mean_ns_given_a_s_test_loss ', mean_ns_given_a_s_test_loss)
        print('mean_target_ns_given_a_s_test_loss ', mean_target_ns_given_a_s_test_loss)
        print('lp_a_ns_s_losses ', lp_a_ns_s_losses)
        print('mean_a_given_ns_s_test_loss ', mean_a_given_ns_s_test_loss)
        print('mean_target_a_given_ns_s_test_loss ', mean_target_a_given_ns_s_test_loss)
        print('policy_loss ', policy_loss)
        print('q_value_loss ', q_value_loss)
        print('policy_base_losses ', policy_base_losses)
        print('policy_log_probs ', policy_log_probs)
        print('alpha_loss ', alpha_loss)
        print('alpha ', alpha)
        print('mean training_rewards ', training_rewards)
        print("discriminator_loss: ", discriminator_loss)
        print('mean_length ', sim_length)
        print("target_ns_updated: ", target_ns_updated)
        print("target_a_updated: ", target_a_updated)

        end = time.time()
        elapsed_time = end - start
        print("time for epoch (s): ", elapsed_time)
        print("time till finish (h): ", (float(elapsed_time) * float(args.total_epochs - epoch)) / 3600.)

        if epoch % 20 == 0 and epoch > 0:
            actor_critic.save(os.path.join(output_dir, env_name + "_d.pt"))
            if forward_backward_cond_ll_model is not None:
                forward_backward_cond_ll_model.save(
                    os.path.join(output_dir, env_name + "_state_model.pt"))
            if discriminator is not None:
                discriminator.save(os.path.join(output_dir, env_name + "_discriminator.pt"))

        # eval policy every N epochs
        if epoch % 1 == 0:
            test_reward, mod_ep_reward, est_ep_reward = single_eval_env(env, actor_critic,
                                                                        use_additional_normal=use_additional_normal,
                                                                        state_est_model=expert_state_est_model,
                                                                        state_cond_ll_model=target_forward_backward_cond_ll_model,
                                                                        forward_backward_cond_ll_model_form=forward_backward_cond_ll_model_form)
            if best_episode_reward < test_reward:
                best_episode_reward = test_reward
            print("Single Test Reward: ", test_reward)
            print("Single Best Reward: ", best_episode_reward)
            writer.add_scalar('rl_training/benchmark_reward', test_reward, epoch)
            writer.add_scalar('rl_training/best_benchmark_reward', best_episode_reward, epoch)
            csv_path = os.path.join(output_dir, env_name + "bechmark_metrics.csv")
            with open(csv_path, 'a+') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([test_reward])

            total_episode_est_rewards.append(est_ep_reward)
            total_episode_mod_rewards.append(mod_ep_reward)
            total_episode_losses.append(policy_loss)
            total_train_est_rewards.append(mean_rewards)
            
            mean_est_ep_rewards = np.mean(total_episode_est_rewards)
            mean_episode_losses = np.mean(total_episode_losses)
            mean_mod_ep_rewards = np.mean(total_episode_mod_rewards)
            mean_train_est_rewards = np.mean(total_train_est_rewards)
            eval_env_already_run = False
            test_rewards = None

            if best_mod_reward < mean_mod_ep_rewards:
                best_mod_reward = mean_mod_ep_rewards

                actor_critic.save(os.path.join(output_dir, env_name + "_best_mod_d.pt"))
                test_rewards = [eval_env(env, actor_critic, use_additional_normal=use_additional_normal,
                                         output_dir=output_dir, epoch=args.total_epochs + 1) for _ in range(10)]
                print("Best mod reward - Test Reward: ", np.mean(test_rewards))
                writer.add_scalar('rl_training/mod_test_reward',
                                  np.mean(test_rewards), epoch)

                csv_path = os.path.join(output_dir, env_name + "best_mod_metrics.csv")
                with open(csv_path, 'a+') as csvfile:
                    csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(test_rewards)

            if best_expert_reward < mean_est_ep_rewards:
                best_expert_reward = mean_est_ep_rewards
                actor_critic.save(os.path.join(output_dir, env_name + "_best_est_d.pt"))
                if test_rewards is None:
                    test_rewards = [eval_env(env, actor_critic, use_additional_normal=use_additional_normal,
                                             output_dir=output_dir, epoch=args.total_epochs + 1) for _ in range(10)]
                print("Best est reward - Test Reward: ", np.mean(test_rewards))
                writer.add_scalar('rl_training/est_test_reward',
                                  np.mean(test_rewards), epoch)

                csv_path = os.path.join(output_dir, env_name + "best_est_metrics.csv")
                with open(csv_path, 'a+') as csvfile:
                    csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(test_rewards)

            if best_episode_losse < mean_episode_losses:
                best_episode_losse = mean_episode_losses
                actor_critic.save(os.path.join(
                    output_dir, env_name + "_best_loss_d.pt"))
                if test_rewards is None:
                    test_rewards = [eval_env(env, actor_critic, use_additional_normal=use_additional_normal,
                                             output_dir=output_dir, epoch=args.total_epochs + 1) for _ in range(10)]
                print("Best loss - Test Reward: ", np.mean(test_rewards))
                writer.add_scalar('rl_training/loss_test_reward',
                                  np.mean(test_rewards), epoch)

                csv_path = os.path.join(
                    output_dir, env_name + "best_loss_metrics.csv")
                with open(csv_path, 'a+') as csvfile:
                    csv_writer = csv.writer(
                        csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(test_rewards)
            
            if best_train_est_rewards < mean_train_est_rewards:
                best_train_est_rewards = mean_train_est_rewards
                actor_critic.save(os.path.join(
                    output_dir, env_name + "_best_est_train_d.pt"))
                if test_rewards is None:
                    test_rewards = [eval_env(env, actor_critic, use_additional_normal=use_additional_normal,
                                             output_dir=output_dir, epoch=args.total_epochs + 1) for _ in range(10)]
                print("Best est Reward from Training - Test Reward: ", np.mean(test_rewards))
                writer.add_scalar('rl_training/est_train_test_reward',
                                  np.mean(test_rewards), epoch)

                csv_path = os.path.join(
                    output_dir, env_name + "best_est_train_metrics.csv")
                with open(csv_path, 'a+') as csvfile:
                    csv_writer = csv.writer(
                        csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(test_rewards)

    if args.total_epochs > 0:
        actor_critic.save(os.path.join(output_dir, env_name + "_d.pt"))
        if forward_backward_cond_ll_model is not None:
            forward_backward_cond_ll_model.save(
                os.path.join(output_dir, env_name + "_state_model.pt"))
        if discriminator is not None:
            discriminator.save(os.path.join(output_dir, env_name + "_discriminator.pt"))

    # eval policy after training
    print("Start policy evaluation... ")
    actor_critic.test_mode = True
    if visualize:
        eval_env(env, actor_critic, visualize=visualize, use_additional_normal=use_additional_normal)
    test_reward = [eval_env(env, actor_critic, visualize=visualize, use_additional_normal=use_additional_normal,
                                            output_dir=output_dir, epoch=args.total_epochs+1) for _ in range(10)]
    print("Test Reward: ", np.mean(test_reward))
    csv_path = os.path.join(output_dir, env_name + "_" + str(int(np.mean(test_reward))) + "_metrics.csv")
    with open(csv_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["mean reward", "{:.6f}".format(np.mean(test_reward))])
        writer.writerow(test_reward)

    test_envs.close()
    envs.close()

    if args.print_to_file:
        sys.stdout = old_stdout
        log_file.close()

    print("done!")
    quit()
