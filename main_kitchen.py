# pylint: disable=protected-access

"""
Trains agents on a static, offline dataset and
evaluates their performance periodically.
"""

import yaml
import torch
from argparse import ArgumentParser
import datetime
from pathlib import Path

from agents.workspaces import OfflineKitchenWorkspace
from agents.fb.agent import FB
from agents.cfb.agent import CFB
from agents.breeze.agent import BREEZE
from agents.fb.replay_buffer import FBReplayBuffer
from rewards.kitchen_utils import make_env
from utils import set_seed_everywhere

parser = ArgumentParser()
parser.add_argument("algorithm", type=str)
parser.add_argument("domain_name", type=str, default='kitchen-mixed-v0')
parser.add_argument("dataset_path", type=str, default='~/.d4rl/datasets/kitchen_microwave_kettle_bottomburner_light-v0.hdf5')
parser.add_argument("--wandb_logging", type=str, default="True")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--alpha", type=float, default=0.01)
parser.add_argument("--discount", type=float, default=0.98)
parser.add_argument("--z_dimension", type=int, default=50)
parser.add_argument("--weighted_cml", type=bool, default=False)
parser.add_argument("--total_action_samples", type=int, default=12)
parser.add_argument("--ood_action_weight", type=float, default=0.5)
parser.add_argument("--learning_steps", type=int, default=1000000)
parser.add_argument("--z_inference_steps", type=int, default=10000)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--model_name", type=str, default=None)
parser.add_argument("--lagrange", type=str, default="True")
parser.add_argument("--target_conservative_penalty", type=float, default=50.0)
parser.add_argument("--action_condition_index", type=int)
parser.add_argument("--action_condition_value", type=float)
parser.add_argument("--cql_alpha", type=float, default=0.01)
args = parser.parse_args()

if args.wandb_logging == "True":
    args.wandb_logging = True
elif args.wandb_logging == "False":
    args.wandb_logging = False
else:
    raise ValueError("wandb_logging must be either True or False")

if args.algorithm in ("vcfb"):
    args.vcfb = True
    args.mcfb = False
elif args.algorithm in ("mcfb"):
    args.vcfb = False
    args.mcfb = True

if args.lagrange == "True":
    args.lagrange = True
elif args.lagrange == "False":
    args.lagrange = False

# action condition for subsampling dataset
if args.action_condition_index is not None:
    args.action_condition = {args.action_condition_index: args.action_condition_value}
else:
    args.action_condition = None

working_dir = Path.cwd()
if args.algorithm in ("vcfb", "mcfb"):
    algo_dir = "cfb"
    config_path = working_dir / "agents" / algo_dir / "config.yaml"
    model_dir = working_dir / "agents" / algo_dir / "saved_models"
else:
    config_path = working_dir / "agents" / args.algorithm / "config.yaml"
    model_dir = working_dir / "agents" / args.algorithm / "saved_models"

time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

with open(config_path, "rb") as f:
    config = yaml.safe_load(f)

config.update(vars(args))
config["device"] = torch.device(
    "cuda" if torch.cuda.is_available() else ("mps" if torch.has_mps else "cpu")
)

set_seed_everywhere(config["seed"])

kitchen_env = make_env(config["domain_name"])
observation_length = int(kitchen_env.observation_space.shape[0] / 2)
action_length = kitchen_env.action_space.shape[0]
action_range = [-1.0, 1.0]

if config["algorithm"] == "fb":

    if config["domain_name"] == "point_mass_maze":
        config["discount"] = 0.99
        config["z_dimension"] = 100

    agent = FB(
        observation_length=observation_length,
        action_length=action_length,
        preprocessor_hidden_dimension=config["preprocessor_hidden_dimension"],
        preprocessor_output_dimension=config["preprocessor_output_dimension"],
        preprocessor_hidden_layers=config["preprocessor_hidden_layers"],
        forward_hidden_dimension=config["forward_hidden_dimension"],
        forward_hidden_layers=config["forward_hidden_layers"],
        forward_number_of_features=config["forward_number_of_features"],
        backward_hidden_dimension=config["backward_hidden_dimension"],
        backward_hidden_layers=config["backward_hidden_layers"],
        actor_hidden_dimension=config["actor_hidden_dimension"],
        actor_hidden_layers=config["actor_hidden_layers"],
        preprocessor_activation=config["preprocessor_activation"],
        forward_activation=config["forward_activation"],
        backward_activation=config["backward_activation"],
        actor_activation=config["actor_activation"],
        z_dimension=config["z_dimension"],
        critic_learning_rate=config["critic_learning_rate"],
        actor_learning_rate=config["actor_learning_rate"],
        learning_rate_coefficient=config["learning_rate_coefficient"],
        orthonormalisation_coefficient=config["orthonormalisation_coefficient"],
        discount=config["discount"],
        batch_size=config["batch_size"],
        z_mix_ratio=config["z_mix_ratio"],
        gaussian_actor=config["gaussian_actor"],
        std_dev_clip=config["std_dev_clip"],
        std_dev_schedule=config["std_dev_schedule"],
        tau=config["tau"],
        device=config["device"],
        name=config["name"],
    )

    replay_buffer = FBReplayBuffer(
        reward_constructor=None,
        transitions=None,
        dataset_path=args.dataset_path,
        device=config["device"],
        discount=config["discount"],
        kitchen=True
    )

    z_inference_steps = config["z_inference_steps"]
    train_std = config["std_dev_schedule"]
    eval_std = config["std_dev_eval"]

elif config["algorithm"] == "breeze":

    agent = BREEZE(
        observation_length=observation_length,
        action_length=action_length,
        forward_config=config["forward_config"],
        backward_config=config["backward_config"],
        v_config=config['v_config'],
        actor_config=config["actor_config"],
        code_options_parameters=config["code_option_params"],
        z_dimension=config["z_dimension"],
        z_mix_ratio=config["z_mix_ratio"],
        orthonormalisation_coefficient=config["orthonormalisation_coefficient"],
        batch_size=config["batch_size"],
        tau=config["tau"],
        discount=config["discount"],
        device=config["device"],
        name=config["name"]
    )

    replay_buffer = FBReplayBuffer(
        reward_constructor=None,
        transitions=None,
        dataset_path=args.dataset_path,
        device=config["device"],
        discount=config["discount"],
        kitchen=True,
        normalize=True
    )
    agent.s_mean, agent.s_std = replay_buffer.s_mean, replay_buffer.s_std
    agent.a_mean, agent.a_std = replay_buffer.a_mean, replay_buffer.a_std
    
    z_inference_steps = config["z_inference_steps"]
    train_std = config["actor_config"]["std_dev_schedule"]
    eval_std = config["actor_config"]["std_dev_eval"]
    
elif config["algorithm"] in ("vcfb", "mcfb"):
    if config["domain_name"] == "point_mass_maze":
        config["discount"] = 0.99
        config["z_dimension"] = 100

    agent = CFB(
        observation_length=observation_length,
        action_length=action_length,
        preprocessor_hidden_dimension=config["preprocessor_hidden_dimension"],
        preprocessor_output_dimension=config["preprocessor_output_dimension"],
        preprocessor_hidden_layers=config["preprocessor_hidden_layers"],
        forward_hidden_dimension=config["forward_hidden_dimension"],
        forward_hidden_layers=config["forward_hidden_layers"],
        forward_number_of_features=config["forward_number_of_features"],
        backward_hidden_dimension=config["backward_hidden_dimension"],
        backward_hidden_layers=config["backward_hidden_layers"],
        actor_hidden_dimension=config["actor_hidden_dimension"],
        actor_hidden_layers=config["actor_hidden_layers"],
        preprocessor_activation=config["preprocessor_activation"],
        forward_activation=config["forward_activation"],
        backward_activation=config["backward_activation"],
        actor_activation=config["actor_activation"],
        z_dimension=config["z_dimension"],
        actor_learning_rate=config["actor_learning_rate"],
        critic_learning_rate=config["critic_learning_rate"],
        learning_rate_coefficient=config["learning_rate_coefficient"],
        orthonormalisation_coefficient=config["orthonormalisation_coefficient"],
        discount=config["discount"],
        batch_size=config["batch_size"],
        z_mix_ratio=config["z_mix_ratio"],
        gaussian_actor=config["gaussian_actor"],
        std_dev_clip=config["std_dev_clip"],
        std_dev_schedule=config["std_dev_schedule"],
        tau=config["tau"],
        device=config["device"],
        vcfb=config["vcfb"],
        mcfb=config["mcfb"],
        total_action_samples=config["total_action_samples"],
        ood_action_weight=config["ood_action_weight"],
        alpha=config["alpha"],
        target_conservative_penalty=config["target_conservative_penalty"],
        lagrange=config["lagrange"],
    )

    replay_buffer = FBReplayBuffer(
        reward_constructor=None,
        transitions=None,
        dataset_path=args.dataset_path,
        device=config["device"],
        discount=config["discount"],
        kitchen=True
    )

    z_inference_steps = config["z_inference_steps"]
    train_std = config["std_dev_schedule"]
    eval_std = config["std_dev_eval"]
else:
    raise NotImplementedError(f"Algorithm {config['algorithm']} not implemented")

workspace = OfflineKitchenWorkspace(
    env=kitchen_env,
    learning_steps=config["learning_steps"],
    model_dir=model_dir,
    eval_frequency=config["eval_frequency"],
    eval_rollouts=config["eval_rollouts"],
    train_std=train_std,
    eval_std=eval_std,
    wandb_logging=config["wandb_logging"],
    device=config["device"],
    save=config["save"],
)

if __name__ == "__main__":

    workspace.train(
        agent=agent,
        agent_config=config,
        replay_buffer=replay_buffer,
    )