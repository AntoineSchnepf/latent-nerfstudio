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

import torch
import matplotlib.pyplot as plt
from pathlib import Path
from prodict import Prodict
from diffusers import AutoencoderKL
from functools import partial
import torchvision
import wandb

from typing import List, Type, Literal, Optional
from dataclasses import dataclass, field
import torch

from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# nerfstudio imports
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.models.base_model import ModelConfig
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.configs.base_config import InstantiateConfig
from typing import Any, Dict, List, Literal, Optional, Tuple, Type
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
import torchvision.utils as vutils
from torch.nn import Parameter
from time import time

# our imports 
from latent_nerfstudio.utils import TanhNormalizer, get_rgb_img_key, step_in_range, interactive_plot
from latent_nerfstudio.datamanager import CustomFullImageDatamanager

to_pil_transform = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(lambda x : x.clamp(0, 1)),
    torchvision.transforms.ToPILImage()
])

latent_to_pil_transform = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(lambda x : x.clamp(0, 1)),
    torchvision.transforms.Lambda(lambda x : x[:3]),
    torchvision.transforms.ToPILImage()
])

def cat_and_downsample(image, downsample_factor):
    image = torch.cat([image,  image], dim=-1) 
    tr = torchvision.transforms.Resize((128//downsample_factor, 128//downsample_factor), antialias=True)
    image = tr(image.permute(2,0,1)).permute(1,2,0)

def downsample_camera(camera, downsample_factor):
    camera.cx = camera.cx // downsample_factor
    camera.cy = camera.cy // downsample_factor
    camera.fx = camera.fx // downsample_factor
    camera.fy = camera.fy // downsample_factor
    camera.height = camera.height // downsample_factor
    camera.width = camera.width // downsample_factor

    return camera

@dataclass
class CustomPipelineConfig(InstantiateConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: CustomPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=DataManagerConfig)
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=ModelConfig)
    """specifies the model config"""
    loss_d_factor: float = 0.01
    """factor for the decoder loss"""
    loss_ae_factor: float = 1.0
    """factor for the autoencoder loss"""


class CustomPipeline(Pipeline): #CustomPipeline

    def __init__(
        self, 
        config: CustomPipelineConfig, 
        vae: AutoencoderKL,
        normalizer: TanhNormalizer,
        eval_funcs: Dict[str, torch.nn.Module],
        loss_e_range: tuple,
        loss_d_range: tuple,
        loss_ae_range: tuple,
        num_channels: int,
        downsample_factor: int,
        latent_bg_color: tuple,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,

    ):
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager = config.datamanager.setup(
            vae=vae, 
            normalizer=normalizer,
            device=device, 
            test_mode=test_mode, 
            world_size=1, 
            local_rank=local_rank,
        )

        # The following is still obscure 
        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]
            seed_pts = (pts, pts_rgb)

        self.datamanager.to(device)
        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self.latent_bg_color = latent_bg_color

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            seed_points=seed_pts,
            out_dim=num_channels,
            latent_or_rgb="latent",
            latent_bg_color=self.latent_bg_color,
        )
        
        # Verify that the scene_box is in accordance with our conventions
        assert torch.all(torch.isclose(self._model.scene_box.aabb, torch.tensor([[-1., -1., -1.],[ 1.,  1.,  1.]])))
        
        self.world_size = 1
        
        # -- our custom code --- 
        self.normalizer = normalizer
        self.vae = vae
        self.loss_e_range = loss_e_range
        self.loss_d_range = loss_d_range
        self.loss_ae_range = loss_ae_range
        self.num_channels = num_channels
        self.downsample_factor = downsample_factor

        self.psnr_func = eval_funcs['psnr_func']
        self.ssim_func = eval_funcs['ssim_func']
        self.lpips_func = eval_funcs['lpips_func']
        self.mse_func = eval_funcs['mse_func']
        
    def get_rgb_quality_metrics(self, img_gt, img, mode="HWC", space='rgb'): 
        "expects img_gt and img to be in [0,1] range"
        assert mode in ["HWC", "CHW"]
        
        if mode == "HWC":
            img_gt = img_gt.permute(2,0,1)
            img = img.permute(2,0,1)

        metrics = dict(
                psnr = self.psnr_func(img_gt, img),
                ssim = self.ssim_func(img_gt.unsqueeze(0), img.unsqueeze(0)),
                # mse = self.mse_func(img_gt, img),
            )
        if space == "latent":
            return metrics

        metrics['lpips'] = self.lpips_func(img_gt.unsqueeze(0), img.unsqueeze(0))

        return metrics

    def decode_latent(self, latent_image):
        "expects latent_image in format HWC and in [0,1] range. Img cannot be batched"
        latent_img_unorm = self.normalizer.deapply_norm(latent_image) 
        decoded_latent_img = self.vae.decode(latent_img_unorm.permute(2,0,1).unsqueeze(0)).sample.squeeze(0).permute(1,2,0)
        decoded_latent_img = 0.5 * (decoded_latent_img + 1)
        return  decoded_latent_img

    def encode_latent(self, img):
        "expects img in format HWC and in [0,1] range and returns its encoded version in the same range and HWC format. Img cannot be batched"
        img = 2 * img - 1
        encoded_img = self.vae.encode(img.permute(2,0,1).unsqueeze(0)).latent_dist.sample().squeeze(0).permute(1,2,0)
        return self.normalizer.apply_norm(encoded_img)
        
    def _process_datapoint(self, camera, data):
        """Process a datapoint from the datamanager according to the current nerf mode.
        Returns the processed camera and batch
        """
        batch = {
            'image_idx': data['image_idx'],
            'image_rgb': data['image'][..., :3], # gt_rgb
        }

        if 'latent_image' in data:
            encoded_gt_image = data['latent_image']
        else:
            encoded_gt_image = self.encode_latent(data['image'][..., :3])

        batch['image'] = encoded_gt_image
        camera = downsample_camera(camera, downsample_factor=self.downsample_factor)
        return camera, batch
    
    def _next_datapoint(self, step, split):

        if split == 'train':
            camera, data = self.datamanager.next_train(step=step)
        elif split == 'eval':
            camera, data = self.datamanager.next_eval(step=step)
        else:
            raise Exception(f"split {split} not supported.")
        
        rgb_height, rgb_width = camera.height, camera.width
        camera, batch = self._process_datapoint(camera, data)
        latent_height, latent_width = camera.height, camera.width
        camera_ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True, obb_box=None)
        ray_bundle = camera_ray_bundle.flatten()

        # Computing additionnal metrics and model specific losses
        feature_dim = batch['image'].shape[-1]
        batch['image'] = batch['image'].reshape(-1, feature_dim)
        return ray_bundle, batch, rgb_height, rgb_width, latent_height, latent_width

    def _get_loss_dict(self, step: int, split: str, mode:str):
        assert split in ['train', 'eval']
        assert mode in['expensive', 'cheap']
        ray_bundle, batch, rgb_height, rgb_width, latent_height, latent_width = self._next_datapoint(step, split=split)
        model_outputs =self._model(ray_bundle) 

        rgb_key = get_rgb_img_key(model_outputs, self.num_channels)

        gt_rgb = batch['image_rgb'].reshape(rgb_height, rgb_width, -1).to(self.device)
        encoded_gt_img = batch['image'].reshape(latent_height, latent_width, -1)
        render_img = model_outputs[rgb_key].reshape(latent_height, latent_width, -1)

        loss_dict = {}
        metrics_dict = {}
        
        if mode == "expensive": 
            with torch.no_grad():
                decoded_render_img = self.decode_latent(render_img) 
                decoded_encoded_gt_image = self.decode_latent(encoded_gt_img)
                metrics_dict.update(self.model.get_metrics_dict(model_outputs, batch))
                loss_dict.update(self.model.get_loss_dict(model_outputs, batch, metrics_dict))           
                loss_dict['loss_d'] = self.config.loss_d_factor * torch.nn.functional.mse_loss(decoded_render_img, gt_rgb)
                loss_dict['loss_ae'] = self.config.loss_ae_factor * torch.nn.functional.mse_loss(gt_rgb, decoded_encoded_gt_image)
                metrics_dict['metrics_e'] = metrics_dict.copy()
                metrics_dict['metrics_d'] = self.get_rgb_quality_metrics(img_gt=gt_rgb, img=decoded_render_img.clamp(0,1), mode='HWC')
                metrics_dict['metrics_ae'] = self.get_rgb_quality_metrics(img_gt=gt_rgb, img=decoded_encoded_gt_image.clamp(0,1), mode='HWC')
            return model_outputs, loss_dict, metrics_dict

        # loss E
        if step_in_range(step, self.loss_e_range): 
            metrics_dict.update(self.model.get_metrics_dict(model_outputs, batch))
            metrics_dict['metrics_e'] = self.get_rgb_quality_metrics(img_gt=encoded_gt_img, img=render_img, mode='HWC', space='latent')
            # metrics_dict['metrics_e'] = metrics_dict.copy()
            loss_dict.update(self.model.get_loss_dict(model_outputs, batch, metrics_dict))

        # loss D
        if step_in_range(step, self.loss_d_range):
            decoded_render_img = self.decode_latent(render_img) 
            loss_dict['loss_d'] = self.config.loss_d_factor * torch.nn.functional.mse_loss(decoded_render_img, gt_rgb)
            metrics_dict['metrics_d'] = self.get_rgb_quality_metrics(img_gt=gt_rgb, img=decoded_render_img.clamp(0,1), mode='HWC', space='rgb')

        # loss AE
        if step_in_range(step, self.loss_ae_range):
            decoded_encoded_gt_image = self.decode_latent(encoded_gt_img)
            loss_dict['loss_ae'] = self.config.loss_ae_factor * torch.nn.functional.mse_loss(gt_rgb, decoded_encoded_gt_image)
            metrics_dict['metrics_ae'] = self.get_rgb_quality_metrics(img_gt=gt_rgb, img=decoded_encoded_gt_image.clamp(0,1), mode='HWC', space='rgb')


        return model_outputs, loss_dict, metrics_dict

    def get_train_loss_dict(self, step: int):
        model_outputs, loss_dict, metrics_dict = self._get_loss_dict(step, split='train', mode='cheap')
        return model_outputs, loss_dict, metrics_dict
    
    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    @torch.no_grad()
    def get_eval_metrics(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        self.eval()
        self.vae.eval()
        model_outputs, loss_dict, metrics_dict = self._get_loss_dict(step, split='eval', mode='expensive')
        self.train()
        self.vae.train()
        return model_outputs, loss_dict, metrics_dict

    @torch.no_grad()
    def get_images(self, step, split, detailed_return=False):
        self.eval()
        self.vae.eval()

        ray_bundle, batch, rgb_height, rgb_width, latent_height, latent_width = self._next_datapoint(step=step, split=split)
        model_outputs =self._model(ray_bundle) 

        rgb_key = get_rgb_img_key(model_outputs, self.num_channels)

        gt_rgb = batch['image_rgb'].reshape(rgb_height, rgb_width, -1).to(self.device)
        encoded_gt_img = batch['image'].reshape(latent_height, latent_width, -1)
        render_img = model_outputs[rgb_key].reshape(latent_height, latent_width, -1)

        decoded_render_img = self.decode_latent(render_img)
        decoded_encoded_gt_image = self.decode_latent(encoded_gt_img)

        if detailed_return:
            return dict(
            step=step,
            render_img=render_img, 
            encoded_gt_img=encoded_gt_img, 
            decoded_render_img=to_pil_transform(decoded_render_img.permute(2,0,1)), 
            gt_rgb=to_pil_transform(gt_rgb.permute(2,0,1)),
            decoded_encoded_gt_img=to_pil_transform(decoded_encoded_gt_image.permute(2,0,1)),
        )

        fig = interactive_plot(
            step=step,
            render_img=render_img, 
            encoded_gt_img=encoded_gt_img, 
            decoded_render_img=to_pil_transform(decoded_render_img.permute(2,0,1)), 
            gt_rgb=to_pil_transform(gt_rgb.permute(2,0,1)),
            decoded_encoded_gt_img=to_pil_transform(decoded_encoded_gt_image.permute(2,0,1)),
        )

        self.train()
        self.vae.train()
        return fig
    
    @torch.no_grad()
    def get_all_eval_images(self):
            """Iterate over all the images in the eval dataset and get the average.

            Args:
                step: current training step
                output_path: optional path to save rendered images to
                get_std: Set True if you want to return std with the mean metric.

            Returns:
                metrics_dict: dictionary of metrics
            """
            self.eval()
            self.vae.eval()
            assert isinstance(self.datamanager, (FullImageDatamanager, CustomFullImageDatamanager))
            num_images = len(self.datamanager.fixed_indices_eval_dataloader)

            all_eval_images={}
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                MofNCompleteColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
                idx = 0
                for camera, batch in self.datamanager.fixed_indices_eval_dataloader:

                    # Encode camera and batch
                    camera, batch = self._process_datapoint(camera, batch)

                    # Forward pass through nerf
                    outputs = self.model.get_outputs_for_camera(camera=camera)
                    metrics_dict, image_dict = self.model.get_image_metrics_and_images(outputs, batch)


                    # Decode image rendering and encoded gt image
                    rgb_key = get_rgb_img_key(outputs, self.num_channels)
                    gt_image = batch['image_rgb']
                    rendered_image=outputs[rgb_key]
                    decoded_rendered_image = self.decode_latent(rendered_image)
                    encoded_gt_image = batch['image']
                    decoded_encoded_gt_image = self.decode_latent(encoded_gt_image)

                    all_eval_images[idx] = dict(
                        gt_image=gt_image,
                        decoded_rendered_image=decoded_rendered_image,
                        rendered_image=rendered_image,
                        encoded_gt_image=encoded_gt_image,
                        decoded_encoded_gt_image=decoded_encoded_gt_image,
                    )
    
                    progress.advance(task)
                    idx = idx + 1


            self.vae.train()
            self.train()

            return all_eval_images


    @torch.no_grad()
    def get_average_eval_metrics(
            self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
        ):
            """Iterate over all the images in the eval dataset and get the average.

            Args:
                step: current training step
                output_path: optional path to save rendered images to
                get_std: Set True if you want to return std with the mean metric.

            Returns:
                metrics_dict: dictionary of metrics
            """
            self.eval()
            self.vae.eval()
            metrics_dict_list = []
            metrics_ae_dict_list = []
            metrics_d_dict_list = []
            metrics_e_dict_list = []
            metrics_d_ae_dict_list = []
            assert isinstance(self.datamanager, (FullImageDatamanager, CustomFullImageDatamanager))
            num_images = len(self.datamanager.fixed_indices_eval_dataloader)
            if output_path is not None:
                output_path.mkdir(exist_ok=True, parents=True)
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                MofNCompleteColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
                idx = 0
                for camera, batch in self.datamanager.fixed_indices_eval_dataloader:

                    # Encode camera and batch
                    camera, batch = self._process_datapoint(camera, batch)

                    # Forward pass through nerf
                    outputs = self.model.get_outputs_for_camera(camera=camera)
                    metrics_dict, image_dict = self.model.get_image_metrics_and_images(outputs, batch)


                    # Decode image rendering and encoded gt image
                    rgb_key = get_rgb_img_key(outputs, self.num_channels)
                    gt_image = batch['image_rgb']
                    rendered_image=outputs[rgb_key]
                    decoded_rendered_image = self.decode_latent(rendered_image)
                    encoded_gt_image = batch['image']
                    decoded_encoded_gt_image = self.decode_latent(encoded_gt_image)

                    # Calculating metrics
                    e_metrics = self.get_rgb_quality_metrics(encoded_gt_image, rendered_image, mode='HWC', space='latent')
                    metrics_e_dict_list.append(e_metrics)
                    ae_metrics = self.get_rgb_quality_metrics(gt_image, decoded_encoded_gt_image.clamp(0,1), mode='HWC', space='rgb')
                    metrics_ae_dict_list.append(ae_metrics)
                    d_metrics = self.get_rgb_quality_metrics(gt_image, decoded_rendered_image.clamp(0,1), mode='HWC', space='rgb')
                    metrics_d_dict_list.append(d_metrics)
                    d_ae_metrics = self.get_rgb_quality_metrics(decoded_rendered_image.clamp(0,1), decoded_encoded_gt_image.clamp(0,1), mode='HWC', space='rgb')
                    metrics_d_ae_dict_list.append(d_ae_metrics)

                    if output_path is not None:
                        gt_image = to_pil_transform(gt_image.permute(2,0,1))
                        gt_image.save(output_path / f"gt_{idx}.png")
                        decoded_rendered_image = to_pil_transform(decoded_rendered_image.permute(2,0,1))
                        decoded_rendered_image.save(output_path / f"decoded_rendered_{idx}.png")
                        rendered_image = latent_to_pil_transform(rendered_image.permute(2,0,1))
                        rendered_image.save(output_path / f"rendered_{idx}.png")
                        encoded_gt_image = latent_to_pil_transform(encoded_gt_image.permute(2,0,1))
                        encoded_gt_image.save(output_path / f"encoded_gt_{idx}.png")
                        decoded_encoded_gt_image = to_pil_transform(decoded_encoded_gt_image.permute(2,0,1))
                        decoded_encoded_gt_image.save(output_path / f"decoded_encoded_gt_{idx}.png")
                        

                    # assert "num_rays_per_sec" not in metrics_dict
                    # metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                    # fps_str = "fps"
                    # assert fps_str not in metrics_dict
                    # metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                    metrics_dict_list.append(metrics_dict)
                    progress.advance(task)
                    idx = idx + 1

            # average the metrics list
            #metrics_dict = self.get_avg_metrics(metrics_dict_list, get_std)
            metrics_e_dict = self.get_avg_metrics(metrics_e_dict_list, get_std)
            metrics_ae_dict = self.get_avg_metrics(metrics_ae_dict_list, get_std)
            metrics_d_dict = self.get_avg_metrics(metrics_d_dict_list, get_std)
            metrics_d_ae_dict = self.get_avg_metrics(metrics_d_ae_dict_list, get_std)
            self.vae.train()
            self.train()
            
            # TO CLEAN, you don't need metrics_dict anymore ()
            # The only metrics we really care about are e, ae, d
            return metrics_e_dict, metrics_ae_dict, metrics_d_dict, metrics_d_ae_dict

    def get_avg_metrics(self, metrics_dict_list, get_std):
        if len(metrics_dict_list) == 0:
            return {}
        
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])))
        return metrics_dict

    def load_state_dict(self, state_dict, strict=False):
        is_ddp_model_state = True
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("_model."):
                # remove the "_model." prefix from key
                model_state[key[len("_model.") :]] = value
                # make sure that the "module." prefix comes from DDP,
                # rather than an attribute of the model named "module"
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False
        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = {key[len("module.") :]: value for key, value in model_state.items()}

        pipeline_state = {key: value for key, value in state_dict.items() if not key.startswith("_model.")}

        try:
            self.model.load_state_dict(model_state, strict=True)
        except RuntimeError:
            if not strict:
                self.model.load_state_dict(model_state, strict=False)
            else:
                raise

        super().load_state_dict(pipeline_state, strict=False)
    
    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)
        self.load_state_dict(state)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks
        return callbacks
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        return {**datamanager_params, **model_params}


