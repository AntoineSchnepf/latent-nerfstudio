# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""
Data manager that outputs cameras / images instead of raybundles

Good for things like gaussian splatting which require full cameras instead of the standard ray
paradigm
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, ForwardRef, Generic, List, Literal, Optional, Tuple, Type, Union, cast, get_args, get_origin
from diffusers import AutoencoderKL
from latent_nerfstudio.utils import TanhNormalizer

import cv2
import numpy as np
import torch
from rich.progress import track
from torch.nn import Parameter
from typing_extensions import assert_never

from nerfstudio.cameras.camera_utils import fisheye624_project, fisheye624_unproject_helper
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, TDataset
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.utils.misc import get_orig_class
from nerfstudio.utils.rich_utils import CONSOLE
    

@dataclass
class CustomFullImageDatamanagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: CustomFullImageDatamanager)
    dataparser: AnnotatedDataParserUnion = field(default_factory=NerfstudioDataParserConfig)
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    eval_num_images_to_sample_from: int = -1
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = -1
    """When not evaluating on all images, number of iterations before picking
    new images. If -1, never pick new images."""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the image indices to use during eval; if None, uses all."""
    cache_images: Literal["cpu", "gpu"] = "cpu"
    """Whether to cache images in memory. If "cpu", caches on cpu. If "gpu", caches on device."""
    cache_latents: bool = True 
    "Wheter to perform latent caching with the encoder"
    cache_images_type: Literal["uint8", "float32"] = "float32"
    """The image type returned from manager, caching images in uint8 saves memory"""
    max_thread_workers: Optional[int] = None
    """The maximum number of threads to use for caching images. If None, uses all available threads."""


class CustomFullImageDatamanager(DataManager, Generic[TDataset]):
    """
    A datamanager that outputs full images and cameras instead of raybundles. This makes the
    datamanager more lightweight since we don't have to do generate rays. Useful for full-image
    training e.g. rasterization pipelines
    """

    config: CustomFullImageDatamanagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset

    def __init__(
        self,
        config: CustomFullImageDatamanagerConfig,
        vae: AutoencoderKL,
        normalizer: TanhNormalizer,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time

        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")
        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        if len(self.train_dataset) > 500 and self.config.cache_images == "gpu":
            CONSOLE.print(
                "Train dataset has over 500 images, overriding cache_images to cpu",
                style="bold yellow",
            )
            self.config.cache_images = "cpu"
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        if self.config.masks_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("mask")
        if self.config.images_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("image")

        # Some logic to make sure we sample every camera in equal amounts
        self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]
        self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        assert len(self.train_unseen_cameras) > 0, "No data found in dataset"

        super().__init__()

        self.vae = vae
        self.normalizer = normalizer
        
    @cached_property
    def cached_train(self) -> List[Dict[str, torch.Tensor]]:
        """Get the training images. Will load and undistort the images the
        first time this (cached) property is accessed."""
        return self._load_images("train", cache_images_device=self.config.cache_images)

    @cached_property
    def cached_eval(self) -> List[Dict[str, torch.Tensor]]:
        """Get the eval images. Will load and undistort the images the
        first time this (cached) property is accessed."""
        return self._load_images("eval", cache_images_device=self.config.cache_images)

    @torch.no_grad()
    def encode_latent(self, img):
        initial_device=img.device
        compute_device=self.device
        "expects img in format HWC and in [0,1] range and returns its encoded version in the same range and HWC format. Img cannot be batched"
        img=img.to(compute_device)
        img = 2 * img - 1
        encoded_img = self.vae.encode(img.permute(2,0,1).unsqueeze(0)).latent_dist.sample().squeeze(0).permute(1,2,0)
        latent_image = self.normalizer.apply_norm(encoded_img)
        latent_image = latent_image.to(initial_device)
        return latent_image

    def _load_images(
        self, split: Literal["train", "eval"], cache_images_device: Literal["cpu", "gpu"]
    ) -> List[Dict[str, torch.Tensor]]:
        undistorted_images: List[Dict[str, torch.Tensor]] = []

        # Which dataset?
        if split == "train":
            dataset = self.train_dataset
        elif split == "eval":
            dataset = self.eval_dataset
        else:
            assert_never(split)

        def undistort_idx(idx: int) -> Dict[str, torch.Tensor]:
            data = dataset.get_data(idx, image_type=self.config.cache_images_type)
            camera = dataset.cameras[idx].reshape(())
            K = camera.get_intrinsics_matrices().numpy()
            if camera.distortion_params is None:
                return data
            distortion_params = camera.distortion_params.numpy()
            image = data["image"].numpy()

            K, image, mask = _undistort_image(camera, distortion_params, data, image, K)
            data["image"] = torch.from_numpy(image)
            if self.config.cache_latents:
                data['latent_image'] = self.encode_latent(data['image'][..., :3])
            if mask is not None:
                data["mask"] = mask

            dataset.cameras.fx[idx] = float(K[0, 0])
            dataset.cameras.fy[idx] = float(K[1, 1])
            dataset.cameras.cx[idx] = float(K[0, 2])
            dataset.cameras.cy[idx] = float(K[1, 2])
            dataset.cameras.width[idx] = image.shape[1]
            dataset.cameras.height[idx] = image.shape[0]
            return data

        CONSOLE.log(f"Caching / undistorting {split} images")
        with ThreadPoolExecutor(max_workers=2) as executor:
            undistorted_images = list(
                track(
                    executor.map(
                        undistort_idx,
                        range(len(dataset)),
                    ),
                    description=f"Caching / undistorting {split} images",
                    transient=True,
                    total=len(dataset),
                )
            )

        # Move to device.
        if cache_images_device == "gpu":
            for cache in undistorted_images:
                cache["image"] = cache["image"].to(self.device)
                if "mask" in cache:
                    cache["mask"] = cache["mask"].to(self.device)
                if "latent_image" in cache:
                    cache["latent_image"] = cache["latent_image"].to(self.device)
        elif cache_images_device == "cpu":
            for cache in undistorted_images:
                cache["image"] = cache["image"].pin_memory()
                if "mask" in cache:
                    cache["mask"] = cache["mask"].pin_memory()
                if "latent_image" in cache:
                    cache["latent_image"] = cache["latent_image"].pin_memory()
        else:
            assert_never(cache_images_device)

        return undistorted_images

    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training"""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation"""
        return self.dataset_type(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[CustomFullImageDatamanager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is CustomFullImageDatamanager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is CustomFullImageDatamanager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is CustomFullImageDatamanager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def setup_train(self):
        """Sets up the data loaders for training"""

    def setup_eval(self):
        """Sets up the data loader for evaluation"""

    @property
    def fixed_indices_eval_dataloader(self) -> List[Tuple[Cameras, Dict]]:
        """
        Pretends to be the dataloader for evaluation, it returns a list of (camera, data) tuples
        """
        image_indices = [i for i in range(len(self.eval_dataset))]
        data = deepcopy(self.cached_eval)
        _cameras = deepcopy(self.eval_dataset.cameras).to(self.device)
        cameras = []
        for i in image_indices:
            data[i]["image"] = data[i]["image"].to(self.device)
            if "latent_image" in data[i].keys():
                data[i]["latent_image"] = data[i]["latent_image"].to(self.device)
            cameras.append(_cameras[i : i + 1])
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        return list(zip(cameras, data))

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}

    def get_train_rays_per_batch(self):
        # TODO: fix this to be the resolution of the last image rendered
        return 800 * 800

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        image_idx = self.train_unseen_cameras.pop(random.randint(0, len(self.train_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]

        data = deepcopy(self.cached_train[image_idx])
        data["image"] = data["image"].to(self.device)
        if "latent_image" in data.keys():
            data["latent_image"] = data["latent_image"].to(self.device)

        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle"""
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = deepcopy(self.cached_eval[image_idx])
        data["image"] = data["image"].to(self.device)
        if "latent_image" in data.keys():
            data["latent_image"] = data["latent_image"].to(self.device)

        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        return camera, data

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle

        TODO: Make sure this logic is consistent with the vanilladatamanager"""
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = deepcopy(self.cached_eval[image_idx])
        data["image"] = data["image"].to(self.device)
        if "latent_image" in data.keys():
            data["latent_image"] = data["latent_image"].to(self.device)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        return camera, data


def _undistort_image(
    camera: Cameras, distortion_params: np.ndarray, data: dict, image: np.ndarray, K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Optional[torch.Tensor]]:
    mask = None
    if camera.camera_type.item() == CameraType.PERSPECTIVE.value:
        distortion_params = np.array(
            [
                distortion_params[0],
                distortion_params[1],
                distortion_params[4],
                distortion_params[5],
                distortion_params[2],
                distortion_params[3],
                0,
                0,
            ]
        )
        # because OpenCV expects the pixel coord to be top-left, we need to shift the principal point by 0.5
        # see https://github.com/nerfstudio-project/nerfstudio/issues/3048
        K[0, 2] = K[0, 2] - 0.5
        K[1, 2] = K[1, 2] - 0.5
        if np.any(distortion_params):
            newK, roi = cv2.getOptimalNewCameraMatrix(K, distortion_params, (image.shape[1], image.shape[0]), 0)
            image = cv2.undistort(image, K, distortion_params, None, newK)  # type: ignore
        else:
            newK = K
            roi = 0, 0, image.shape[1], image.shape[0]
        # crop the image and update the intrinsics accordingly
        x, y, w, h = roi
        image = image[y : y + h, x : x + w]
        if "depth_image" in data:
            data["depth_image"] = data["depth_image"][y : y + h, x : x + w]
        if "mask" in data:
            mask = data["mask"].numpy()
            mask = mask.astype(np.uint8) * 255
            if np.any(distortion_params):
                mask = cv2.undistort(mask, K, distortion_params, None, newK)  # type: ignore
            mask = mask[y : y + h, x : x + w]
            mask = torch.from_numpy(mask).bool()
            if len(mask.shape) == 2:
                mask = mask[:, :, None]
        newK[0, 2] = newK[0, 2] + 0.5
        newK[1, 2] = newK[1, 2] + 0.5
        K = newK

    elif camera.camera_type.item() == CameraType.FISHEYE.value:
        K[0, 2] = K[0, 2] - 0.5
        K[1, 2] = K[1, 2] - 0.5
        distortion_params = np.array(
            [distortion_params[0], distortion_params[1], distortion_params[2], distortion_params[3]]
        )
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, distortion_params, (image.shape[1], image.shape[0]), np.eye(3), balance=0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, distortion_params, np.eye(3), newK, (image.shape[1], image.shape[0]), cv2.CV_32FC1
        )
        # and then remap:
        image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
        if "mask" in data:
            mask = data["mask"].numpy()
            mask = mask.astype(np.uint8) * 255
            mask = cv2.fisheye.undistortImage(mask, K, distortion_params, None, newK)
            mask = torch.from_numpy(mask).bool()
            if len(mask.shape) == 2:
                mask = mask[:, :, None]
        newK[0, 2] = newK[0, 2] + 0.5
        newK[1, 2] = newK[1, 2] + 0.5
        K = newK
    elif camera.camera_type.item() == CameraType.FISHEYE624.value:
        fisheye624_params = torch.cat(
            [camera.fx, camera.fy, camera.cx, camera.cy, torch.from_numpy(distortion_params)], dim=0
        )
        assert fisheye624_params.shape == (16,)
        assert (
            "mask" not in data
            and camera.metadata is not None
            and "fisheye_crop_radius" in camera.metadata
            and isinstance(camera.metadata["fisheye_crop_radius"], float)
        )
        fisheye_crop_radius = camera.metadata["fisheye_crop_radius"]

        # Approximate the FOV of the unmasked region of the camera.
        upper, lower, left, right = fisheye624_unproject_helper(
            torch.tensor(
                [
                    [camera.cx, camera.cy - fisheye_crop_radius],
                    [camera.cx, camera.cy + fisheye_crop_radius],
                    [camera.cx - fisheye_crop_radius, camera.cy],
                    [camera.cx + fisheye_crop_radius, camera.cy],
                ],
                dtype=torch.float32,
            )[None],
            params=fisheye624_params[None],
        ).squeeze(dim=0)
        fov_radians = torch.max(
            torch.acos(torch.sum(upper * lower / torch.linalg.norm(upper) / torch.linalg.norm(lower))),
            torch.acos(torch.sum(left * right / torch.linalg.norm(left) / torch.linalg.norm(right))),
        )

        # Heuristics to determine parameters of an undistorted image.
        undist_h = int(fisheye_crop_radius * 2)
        undist_w = int(fisheye_crop_radius * 2)
        undistort_focal = undist_h / (2 * torch.tan(fov_radians / 2.0))
        undist_K = torch.eye(3)
        undist_K[0, 0] = undistort_focal  # fx
        undist_K[1, 1] = undistort_focal  # fy
        undist_K[0, 2] = (undist_w - 1) / 2.0  # cx; for a 1x1 image, center should be at (0, 0).
        undist_K[1, 2] = (undist_h - 1) / 2.0  # cy

        # Undistorted 2D coordinates -> rays -> reproject to distorted UV coordinates.
        undist_uv_homog = torch.stack(
            [
                *torch.meshgrid(
                    torch.arange(undist_w, dtype=torch.float32),
                    torch.arange(undist_h, dtype=torch.float32),
                ),
                torch.ones((undist_w, undist_h), dtype=torch.float32),
            ],
            dim=-1,
        )
        assert undist_uv_homog.shape == (undist_w, undist_h, 3)
        dist_uv = (
            fisheye624_project(
                xyz=(
                    torch.einsum(
                        "ij,bj->bi",
                        torch.linalg.inv(undist_K),
                        undist_uv_homog.reshape((undist_w * undist_h, 3)),
                    )[None]
                ),
                params=fisheye624_params[None, :],
            )
            .reshape((undist_w, undist_h, 2))
            .numpy()
        )
        map1 = dist_uv[..., 1]
        map2 = dist_uv[..., 0]

        # Use correspondence to undistort image.
        image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

        # Compute undistorted mask as well.
        dist_h = camera.height.item()
        dist_w = camera.width.item()
        mask = np.mgrid[:dist_h, :dist_w]
        mask[0, ...] -= dist_h // 2
        mask[1, ...] -= dist_w // 2
        mask = np.linalg.norm(mask, axis=0) < fisheye_crop_radius
        mask = torch.from_numpy(
            cv2.remap(
                mask.astype(np.uint8) * 255,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            / 255.0
        ).bool()[..., None]
        if len(mask.shape) == 2:
            mask = mask[:, :, None]
        assert mask.shape == (undist_h, undist_w, 1)
        K = undist_K.numpy()
    else:
        raise NotImplementedError("Only perspective and fisheye cameras are supported")

    return K, image, mask
