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

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
from nerfstudio.configs.base_config import PrintableConfig

from pathlib import Path

@dataclass
class VAEConfig(PrintableConfig):
    pretrained_model_name_or_path: str = "ostris/vae-kl-f8-d16" 
    """Pretrained model name or path. Supported models: 'ostris/vae-kl-f8-d16' and 'runwayml/stable-diffusion-v1-5' """
    n_latent_channels: int = 16
    downsample_factor: int = 8
    bg_color: tuple =  (0.4090, 0.5550, 0.4015, 0.5846, 0.4492, 0.5380, 0.5314, 0.4108, 0.5015, 0.6012, 0.4245, 0.5039, 0.4920, 0.4955, 0.5296, 0.4999) 
    subfolder: Optional[str] = None,
    revision: Optional[str] = None,
    checkpoint: Optional[Path] = None 
    """Path to VAE checkpoint."""


class NormalizerConfig(PrintableConfig):
    tanh_scale: float = 0.02 
    """scale value for tanh normalizer"""
    eps: float = 1.e-6  
    """epsilon value for numerical stability"""
