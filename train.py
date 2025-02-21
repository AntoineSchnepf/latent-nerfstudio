# Copyright 2025, authored by Antoine Schnepf and Karim Kassab. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os,random, torch
import numpy as np
from pathlib import Path
from typing import Optional
import wandb
from nerfstudio.utils import comms, profiler
from dataclasses import asdict

# custom imports
from latent_nerfstudio.trainer import Trainer, TrainerConfig
from latent_nerfstudio.vae import VAEConfig
from latent_nerfstudio.method_configs import get_latent_method_configs
import tyro

from latent_nerfstudio.method_configs import get_latent_method_configs
from latent_nerfstudio.vae_configs import get_vae_configs
from typing import Tuple

def construct_dict_from_dataclass(dataclass):
    return asdict(dataclass)

def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_loop(trainer_config: TrainerConfig, vae_config: VAEConfig, local_rank: int = 0):
    """Main training function that sets up and runs the trainer per process

    Args:
        config: config file specifying training regimen
    """
    trainer = trainer_config.setup(
        vae_config=vae_config, 
        local_rank=local_rank
    )
    trainer.setup()
    trainer.train()
    profiler.flush_profiler(trainer_config.logging)


def main(trainer_config: TrainerConfig, vae_config: VAEConfig):
    trainer_config.pipeline.datamanager.data = trainer_config.data

    trainer_config.set_timestamp()
    trainer_config.print_to_terminal()
    trainer_config.save_config()

    wandb.init(
        project=trainer_config.project_name,
        config=construct_dict_from_dataclass(trainer_config),
        name=trainer_config.experiment_name,
    )

    # Launch training
    _set_random_seed(seed=1234)
    train_loop(trainer_config=trainer_config, vae_config=vae_config)

def entrypoint():
    # Fetching default configurations for latent_methods and VAEs
    latent_method_configs = get_latent_method_configs()
    vae_configs = get_vae_configs()

    # Getting chosen configurations from CLI
    AnnotatedBaseConfigUnion = tyro.conf.OmitSubcommandPrefixes[
        Tuple[
                tyro.extras.subcommand_type_from_defaults(defaults=latent_method_configs),
                tyro.extras.subcommand_type_from_defaults(defaults=vae_configs),
        ]
    ]
    trainer_config, vae_config = tyro.cli(AnnotatedBaseConfigUnion)
    main(trainer_config, vae_config)

if __name__ == '__main__' :
    entrypoint()
