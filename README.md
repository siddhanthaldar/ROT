# Watch and Match: Supercharging Imitation with Regularized Optimal Transport

This is a repository containing the code for the paper "Watch and Match: Supercharging Imitation with Regularized Optimal Transport".

![github_intro](https://user-images.githubusercontent.com/25313941/175857612-3cde39eb-b4ea-4231-bded-76157bb5754b.png)

## Download expert demonstrations, weights and environment libraries [[link]](https://osf.io/vyu7q/?view_only=040ed766b96847b4aadaba8acd6ab3dd)
The link contains the following:
- The expert demonstrations for all tasks in the paper.
- The weight files for the expert (DrQ-v2) and behavior cloning (BC).
- The supporting libraries for environments (Gym-Robotics, metaworld) in the paper.
- Extract the files provided in the link
  - set the `path/to/dir` portion of the `root_dir` path variable in `cfgs/config.yaml` to the path of the `ROT` repository.
  - place the `expert_demos` and `weights` folders in `${root_dir}/ROT`.


## Instructions
- Install [Mujoco](http://www.mujoco.org/) based on the instructions given [here](https://github.com/facebookresearch/drqv2).
- Install the following libraries:
```
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```
- Install dependencies
  - Set up Environment
  ```
  conda env create -f conda_env.yml
  conda activate rot
  ```
  - Install Gym-Robotics
  ```
  pip install -e /path/to/dir/Gym-Robotics
  ```
  - Install Meta-World
  ```
  pip install -e /path/to/dir/metaworld
  ```
  - Install particle environment (for experiment in Fig. 2 in the paper)
  ```
  pip install -e /path/to/dir/gym-envs
  ```

- Train BC agent - We provide three different commands for running the code on the DeepMind Control Suite, OpenAI Robotics Suite and the Meta-World Benchmark
  - For pixel-based input
  ```
  python train.py agent=bc suite=dmc obs_type=pixels suite/dmc_task=walker_run num_demos=10
  ```
  ```
  python train.py agent=bc suite=openaigym obs_type=pixels suite/openaigym_task=fetch_reach num_demos=50
  ```
  ```
  python train.py agent=bc suite=metaworld obs_type=pixels suite/metaworld_task=hammer num_demos=1
  ```
  ```
  python train_robot.py agent=bc suite=robot_gym obs_type=pixels suite/robotgym_task=reach num_demos=1
  ```
  - For state-based input
  ```
  python train.py agent=bc suite=dmc obs_type=features suite/dmc_task=walker_run num_demos=10
  ```
  ```
  python train.py agent=bc suite=openaigym obs_type=features suite/openaigym_task=fetch_reach num_demos=50
  ```
  ```
  python train.py agent=bc suite=metaworld obs_type=features suite/metaworld_task=hammer num_demos=1
  ```

- Train ROT - We provide three different commands for running the code on the DeepMind Control Suite, OpenAI Robotics Suite and the Meta-World Benchmark
  - For pixel-based input
  ```
  python train.py agent=potil suite=dmc obs_type=pixels suite/dmc_task=walker_run load_bc=true bc_regularize=true num_demos=10
  ```
  ```
  python train.py agent=potil suite=openaigym obs_type=pixels suite/openaigym_task=fetch_reach load_bc=true bc_regularize=true num_demos=50
  ```
  ```
  python train.py agent=potil suite=metaworld obs_type=pixels suite/metaworld_task=hammer load_bc=true bc_regularize=true num_demos=1
  ```
  ```
  python train_robot.py agent=potil suite=robotgym obs_type=pixels suite/robotgym_task=reach load_bc=true bc_regularize=true num_demos=1
  ```
  - For state-based input
  ```
  python train.py agent=potil suite=dmc obs_type=features suite/dmc_task=walker_run load_bc=true bc_regularize=true num_demos=10
  ```
  ```
  python train.py agent=potil suite=openaigym obs_type=features suite/openaigym_task=fetch_reach load_bc=true bc_regularize=true num_demos=50
  ```
  ```
  python train.py agent=potil suite=metaworld obs_type=features suite/metaworld_task=hammer load_bc=true bc_regularize=true num_demos=1
  ```
- Monitor results
```
tensorboard --logdir exp_local
```
