# Single- and Multi-Agent Imitation Learning Framework

This repository contains the code for the paper "Imitation learning by state-only distribution matching" by Damian Boborzi, Christoph-Nikolas Straehle, Jens S. Buchner, and Lars Mikelsons.

## installation 
 conda environment config is in ./conda-spec-file.txt
 pip installations are in ./requirements.txt
 
 Following repositories are important: 
 Adapted FrEIA: https://github.com/FeMa42/FrEIA-NF-policy/tree/NF-Policy
 
 To install pybullet use: ```pip install pybullet```

 > Note that SOIL-TDM, FORM and the Normalizing Flow policy all use the FrEIA repository for the implementation of the Normalizing Flows. The FrEIA repository is therefor neccesary for these methods to work. 
 > Also note that (at this moment) the Normalizing Flow implementations are only compatible with CUDA-GPU and CPU versions of torch (tested under Linux Ubuntu 20.04). The Metal-GPU (for Apple M1 and M2 GPUs) implementation of torch is not yet compatible with the Normalizing Flows based on the FrEIA repository. As a fallback for Mac systems you can use the the cpu version of torch (this is slow). Using A MLP as policy with DAC and BC is also possible with the metal-GPU version of torch.

## Implementation of BC, DAC, FORM, SOIL-TDM

Contains Implementation for DAC, FORM and SOIL-TDM all using SAC as the RL-Optmizer. 

Training and Pretraining with behavioral cloning is also possible. It is possible to use a Normalizing Flow policy with DAC and BC instead of a conditional Gaussian policy. 

### Starting Expert Policy Training

Start file: gym_exp_data.py

Examples for training Expert policies using SAC:

```
python gym_exp_data.py --max-frames 100000 --exp-dir pendulum_SAC_MLP --env-name Pendulum-v1 --num-envs 16 --use-mlp --hidden-size 128 --num-env-steps 10 --internal-update-epochs 10 --replay-buffer-size 100000
python gym_exp_data.py --max-frames 400000 --exp-dir cheetah_SAC_MLP --env-name HalfCheetahBulletEnv-v0 --num-envs 16 --hidden-size 512 --lr-p 3e-4 --lr-q 5e-4 --mini-batch-size 512 --num-env-steps 10 --internal-update-epochs 10 --polyak 0.99 --use-mlp
python gym_exp_data.py --max-frames 400000 --exp-dir hopper_SAC_MLP --env-name HopperBulletEnv-v0 --num-envs 16 --hidden-size 512 --lr-p 3e-4 --lr-q 5e-4 --mini-batch-size 512 --num-env-steps 10 --internal-update-epochs 10 --polyak 0.99 --use-mlp
python gym_exp_data.py --max-frames 600000 --exp-dir humanoid_SAC_MLP --env-name HumanoidBulletEnv-v0 --num-envs 16 --hidden-size 512 --lr-p 3e-4 --lr-q 5e-4 --mini-batch-size 512 --num-env-steps 10 --internal-update-epochs 10 --polyak 0.99 --use-mlp
python gym_exp_data.py --max-frames 600000 --exp-dir walker2D_SAC_MLP --env-name Walker2DBulletEnv-v0 --num-envs 16 --hidden-size 512 --lr-p 3e-4 --lr-q 5e-4 --mini-batch-size 512 --num-env-steps 10 --internal-update-epochs 10 --polyak 0.99 --use-mlp
python gym_exp_data.py --max-frames 600000 --exp-dir ant_SAC_MLP --env-name AntBulletEnv-v0 --num-envs 16 --hidden-size 512 --lr-p 3e-4 --lr-q 5e-4 --mini-batch-size 512 --num-env-steps 10 --internal-update-epochs 10 --polyak 0.99 --use-mlp
python gym_exp_data.py --max-frames 100000 --exp-dir cartpole_SAC_MLP --env-name CartPoleContinuousBulletEnv-v0 --num-envs 16 --hidden-size 512 --lr-p 3e-4 --lr-q 5e-4 --mini-batch-size 512 --num-env-steps 10 --internal-update-epochs 10 --polyak 0.99 --use-mlp
```

### Starting Imitation Learning Routines

Start file: flow_rl.py.

It saves the parameters and logs as well as tensorboard in the output directory under "Experiments/..." which has to be provided.

Example for Imitation Learning using SOIL-TDM:

```
python flow_rl.py --exp-name cheetah_soil_tdm_n4 --env-name HalfCheetahBulletEnv-v0 --n-trajectories 4 --overwrite --use-mlp --total-epochs 400 --use-noise-sched --noise-red-factor 1.0 --expert-noise-value 0.05 --final-noise-value 0.02 --n-state-est-train 1000 --num-envs 16 --alpha-nonpolicy-rampup 150
```

Example for Imitation Learning using Form:
```
python flow_rl.py --exp-name hopper_form_n4 --env-name HopperBulletEnv-v0 --il-method "form" --n-trajectories 4 --overwrite --use-mlp --total-epochs 400 --use-noise-sched --noise-red-factor 1.0 --expert-noise-value 0.05 --final-noise-value 0.02 --n-state-est-train 1000 --num-envs 16 --automatic-entropy-tuning
```

Example for Imitation Learning using DAC:
```
python flow_rl.py --exp-name cheetah_dac_n4 --il-method dac --env-name HalfCheetahBulletEnv-v0 --num-envs 16 --n-trajectories 4 --overwrite --use-mlp --total-epochs 400
```

Example for Imitation Learning using BC:
```
python flow_rl.py --exp-name cheetah_bc_n4 --il-method bc --env-name HalfCheetahBulletEnv-v0 --num-envs 2 --n-trajectories 4 --overwrite --use-mlp --total-epochs 0 --num-bc-updates 10000
```


## Normalizing Flow Policy

Example for Imitation Learning with Normalizing Flow policy using BC:
```
python flow_rl.py --exp-name cheetah_bc_nfpolicy_n10 --il-method bc --env-name HalfCheetahBulletEnv-v0 --num-envs 2 --n-trajectories 10 --overwrite --total-epochs 0 --num-bc-updates 10000
```

Example for Imitation Learning with Normalizing Flow policy using DAC:
```
python flow_rl.py --exp-name cheetah_dac_nfpolicy_n10 --il-method dac --env-name HalfCheetahBulletEnv-v0 --num-envs 16 --n-trajectories 10 --overwrite --use-mlp --total-epochs 400
```

You can also train an expert model with a Normalizing Flow policy:
```
python gym_exp_data.py --max-frames 400000 --exp-dir cheetah_SAC_nfpolicy --env-name HalfCheetahBulletEnv-v0 --num-envs 16 --hidden-size 512 --lr-p 3e-4 --lr-q 5e-4 --mini-batch-size 512 --num-env-steps 10 --internal-update-epochs 10 --polyak 0.99
````
``




