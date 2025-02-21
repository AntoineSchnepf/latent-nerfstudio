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

from latent_nerfstudio.vae import VAEConfig

def get_vae_configs():
    vae_configs = {}

    vae_configs['stable-diffusion'] = VAEConfig(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        n_latent_channels=4,
        downsample_factor=8,
        bg_color=(0.6176, 0.5711, 0.5046, 0.4422),
        subfolder='vae',
        revision='main',
        checkpoint=None
    )

    vae_configs['ostris'] = VAEConfig(
        pretrained_model_name_or_path="ostris/vae-kl-f8-d16",
        n_latent_channels=16,
        downsample_factor=8,
        bg_color=(0.4090, 0.5550, 0.4015, 0.5846, 0.4492, 0.5380, 0.5314, 0.4108, 0.5015, 0.6012, 0.4245, 0.5039, 0.4920, 0.4955, 0.5296, 0.4999),
        subfolder=None,
        revision=None,
        checkpoint=None
    )

    return vae_configs
