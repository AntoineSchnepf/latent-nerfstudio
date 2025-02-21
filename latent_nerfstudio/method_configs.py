# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

# This file is a modified version of nerfstudio/configs/method_configs.py
# Modifications done by Antoine Schnepf and Karim Kassab

"""
Put all the method implementations in one location.
"""

from __future__ import annotations

from typing import Dict, Union

import tyro

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.external_methods import ExternalMethodDummyTrainerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.tensorf import TensoRFModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from kplanes_nerfstudio.kplanes.kplanes_configs import  kplanes_method
from kplanes_nerfstudio.kplanes.kplanes import KPlanesModelConfig

# our imports
from latent_nerfstudio.pipeline import CustomPipelineConfig
from latent_nerfstudio.trainer import TrainerConfig
from latent_nerfstudio.datamanager import CustomFullImageDatamanagerConfig

def get_latent_method_configs(): 
    "Configuration for the different methods in a NeRFStudio style"

    lr_factor = {
        "nerfacto": 0.1,
        "vanilla-nerf": 1.,
        "tensorf": 0.1,
        "kplanes": 0.1,
        "instant-ngp": 0.1,
    }
    near_plane = 0.2
    far_plane = 2.4

    dataparser_config = NerfstudioDataParserConfig(
                train_split_fraction=0.9,
                downscale_factor=1.0
            )
    
    # we use the same datamanager for all methods
    datamanager_config = CustomFullImageDatamanagerConfig(
                dataparser=dataparser_config,
            )
    custom_method_configs: Dict[str, Union[TrainerConfig, ExternalMethodDummyTrainerConfig]] = {}

    # nerfacto
    method_name = "nerfacto"
    custom_method_configs[method_name] = TrainerConfig(
        method_name=method_name,
        mixed_precision=True, 
        pipeline=CustomPipelineConfig(
            datamanager=datamanager_config,
            model=NerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                average_init_density=0.01,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
                near_plane=near_plane,
                far_plane=far_plane,
                use_appearance_embedding=False,
                use_average_appearance_embedding=True,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2*lr_factor[method_name], eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001*lr_factor[method_name], max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2*lr_factor[method_name], eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001*lr_factor[method_name], max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3*lr_factor[method_name], eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4*lr_factor[method_name], max_steps=5000),
            },
        },
    )

    # vanilla nerf
    method_name = "vanilla-nerf"
    custom_method_configs[method_name] = TrainerConfig(
        method_name=method_name,
        pipeline=CustomPipelineConfig(
            datamanager=datamanager_config,
            model=VanillaModelConfig(
                _target=NeRFModel,
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4*lr_factor[method_name], eps=1e-08),
                "scheduler": None,
            },
            "temporal_distortion": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4*lr_factor[method_name], eps=1e-08),
                "scheduler": None,
            },
        },
    )

    method_name = "tensorf"
    custom_method_configs[method_name] = TrainerConfig(
        method_name=method_name,
        mixed_precision=False,
        pipeline=CustomPipelineConfig(
            datamanager=datamanager_config,
            model=TensoRFModelConfig(
                regularization="tv",
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=0.001*lr_factor[method_name]),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001*lr_factor[method_name], max_steps=30000),
            },
            "encodings": {
                "optimizer": AdamOptimizerConfig(lr=0.02*lr_factor[method_name]),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.002*lr_factor[method_name], max_steps=30000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4*lr_factor[method_name], eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=5000),
            },
        },
    )

    method_name = "kplanes"
    custom_method_configs[method_name] = TrainerConfig(
        method_name=method_name,
        mixed_precision=True,
        pipeline=CustomPipelineConfig(
            datamanager=datamanager_config,
            model=KPlanesModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                grid_base_resolution=[128, 128, 128],
                grid_feature_dim=32,
                multiscale_res=[1, 2, 4],
                proposal_net_args_list=[
                    {"num_output_coords": 8, "resolution": [128, 128, 128]},
                    {"num_output_coords": 8, "resolution": [256, 256, 256]}
                ],
                loss_coefficients={
                    "interlevel": 1.0,
                    "distortion": 0.01,
                    "plane_tv": 0.01,
                    "plane_tv_proposal_net": 0.0001,
                },
                near_plane = near_plane,
                far_plane = far_plane,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2*lr_factor[method_name], eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2*lr_factor[method_name], eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
        },
    )

    method_name = "instant-ngp"
    custom_method_configs[method_name] = TrainerConfig(
        method_name=method_name,
        mixed_precision=True,
        pipeline=CustomPipelineConfig(
            datamanager=datamanager_config,
            model=InstantNGPModelConfig(
                eval_num_rays_per_chunk=8192,
                use_appearance_embedding=False,
                far_plane=4.0,
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2*lr_factor[method_name], eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001*lr_factor[method_name], max_steps=200000),
            }
        },
    )

    return custom_method_configs






