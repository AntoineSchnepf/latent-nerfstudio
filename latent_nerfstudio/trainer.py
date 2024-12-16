from __future__ import annotations

import math
import dataclasses
import functools
import os
import time
import yaml
import tqdm
import random
from datetime import datetime
from diffusers import AutoencoderKL
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Tuple, Type, cast
import torch
from rich import box, style
from rich.panel import Panel
from rich.table import Table
from torch.cuda.amp.grad_scaler import GradScaler
from prodict import Prodict
import matplotlib.pyplot as plt
import wandb
from functools import partial
from tqdm import trange

import numpy as np
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# nerfstudio imports
from nerfstudio.configs.base_config import InstantiateConfig, LoggingConfig, MachineConfig, ViewerConfig, PrintableConfig
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import OptimizerConfig, Optimizers
from nerfstudio.engine.schedulers import SchedulerConfig, Scheduler
from nerfstudio.utils import profiler
from nerfstudio.utils.decorators import check_eval_enabled, check_main_thread, check_viewer_enabled
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.configs.config_utils import to_immutable_dict
# our imports
from latent_nerfstudio.pipeline import CustomPipeline, CustomPipelineConfig
from latent_nerfstudio.vae import VAEConfig, NormalizerConfig
from latent_nerfstudio.utils import TanhNormalizer, get_rgb_img_key, step_in_range, interactive_plot
from latent_nerfstudio.utils import set_requires_grad, optimizer_to

TRAIN_INTERATION_OUTPUT = Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
TORCH_DEVICE = str



class custom_dict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 1

    def __missing__(self, key):
        self[key] = custom_dict()
        return self[key]

    def add(self, cd):
        assert set(self.keys()) == set(cd.keys())
        for key in cd.keys():
            self[key] += cd[key]
        self.counter += 1

    def get_avg(self):
        self.scale(1/self.counter)
        return self

    def scale(self, scalar):
        for key in self.keys():
            if isinstance(self[key], torch.Tensor):
                self[key] = self[key].float()
            self[key] *= scalar
        return self

def extract_metrics_to_log(metrics_dict, to_log, metrics_to_log, mode='train'):
    
    for metric_type in metrics_to_log:
        to_log[f"{mode}_metrics"][metric_type] = metrics_dict[metric_type]

    return to_log

def recursive_transform_loss_dict(loss_dict):
    logged_dict = {}
    for key, value in loss_dict.items():
        if isinstance(value, dict):
            logged_dict[key] = recursive_transform_loss_dict(value)
        else:
            if 'rgb' in key:
                logged_dict[key.replace('rgb', 'latent')] = value
            else:
                logged_dict[key] = value
    return logged_dict

def free_dict(input_dict: dict):
    "recursive detach of tensors inside input_dict and moves them to cpu"
    res = {}
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            res[k] = v.detach().cpu()
        elif isinstance(v, dict):
            res[k] = free_dict(v)
    return res


@dataclass
class EncoderConfig(PrintableConfig):
    lr: float = 1.e-4
    gamma_scheduler: float = 0.99969

@dataclass
class DecoderConfig(PrintableConfig):
    lr: float = 1.e-4
    gamma_scheduler: float = 0.99969

@dataclass
class ExperimentConfig(InstantiateConfig):
    """Full config contents for running an experiment. Any experiment types (like training) will be
    subclassed from this, and must have their _target field defined accordingly."""
    output_dir: Path = Path("outputs")
    """relative or absolute output directory to save all checkpoints and logging"""
    method_name: Optional[str] = None
    """Method name. Required to set in python or via cli"""
    experiment_name: Optional[str] = "latent_nerfstudio_experiment"
    """Experiment name. If None, will automatically be set to dataset name"""
    project_name: Optional[str] = "latent-nerfstudio"
    """Project name."""
    timestamp: str = "{timestamp}"
    """Experiment timestamp."""
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    """Logging configuration"""
    optimizers: Dict[str, Any] = to_immutable_dict(
            {
                "fields": {
                    "optimizer": OptimizerConfig(),
                    "scheduler": SchedulerConfig(),
                }
            }
        )
    """Dictionary of optimizer groups and their schedulers"""
    pipeline: CustomPipelineConfig = field(default_factory=CustomPipeline)
    """Pipeline configuration"""
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    """Decoder configuration"""
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    """Encoder configuration"""
    data: Optional[Path] = None
    """Alias for --pipeline.datamanager.data"""
    vis: Literal["wandb"] = "wandb"
    """Which visualizer to use."""
    relative_model_dir: Path = Path("nerfstudio_models/")
    """Relative path to save all checkpoints."""

    def set_timestamp(self) -> None:
        """Dynamically set the experiment timestamp"""
        if self.timestamp == "{timestamp}":
            self.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    def set_experiment_name(self) -> None:
        """Dynamically set the experiment name"""
        if self.experiment_name is None:
            datapath = self.pipeline.datamanager.data
            if datapath is not None:
                datapath = datapath.parent if datapath.is_file() else datapath
                self.experiment_name = str(datapath.stem)
            else:
                self.experiment_name = "unnamed"

    def get_base_dir(self) -> Path:
        """Retrieve the base directory to set relative paths"""
        # check the experiment and method names
        assert self.method_name is not None, "Please set method name in config or via the cli"
        self.set_experiment_name()
        return Path(f"{self.output_dir}/{self.experiment_name}/{self.method_name}/{self.timestamp}")

    def get_checkpoint_dir(self) -> Path:
        """Retrieve the checkpoint directory"""
        return Path(self.get_base_dir() / self.relative_model_dir)

    def print_to_terminal(self) -> None:
        """Helper to pretty print config to terminal"""
        CONSOLE.rule("Config")
        CONSOLE.print(self)
        CONSOLE.rule("")

    def save_config(self) -> None:
        """Save config to base directory"""
        base_dir = self.get_base_dir()
        assert base_dir is not None
        base_dir.mkdir(parents=True, exist_ok=True)
        config_yaml_path = base_dir / "config.yml"
        CONSOLE.log(f"Saving config to: {config_yaml_path}")
        config_yaml_path.write_text(yaml.dump(self), "utf8")


@dataclass
class TrainerConfig(ExperimentConfig):
    _target: Type = field(default_factory=lambda: Trainer)
    """Target for trainer"""
    # vae: VAEConfig = field(default_factory=VAEConfig)
    # """VAE Configuration."""
    normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)
    """Normalizer Configuration."""
    steps_per_save: int = 1000
    """Number of steps between saves."""
    steps_per_eval_image: int = 500
    """Number of steps between single eval images."""
    steps_per_eval_all_images: int = 1000
    """Number of steps between eval all images."""    
    max_num_iterations: int = 25000
    """Maximum number of iterations to run."""
    train_nerf_range: tuple = (0, 25000) 
    """Iteration range for activating the training of the nerf."""
    train_encoder_range: tuple = (0, 0)
    """Iteration range for activating the training of the decoder."""
    train_decoder_range: tuple = (10000, 25000)
    """Iteration range for activating the training of the encoder."""
    loss_d_range: tuple = (10000, 25000)
    """Iteration range for activating the the RGB alignment loss."""
    loss_ae_range: tuple = (0, 0)
    """Iteration range for activating the the autoencoding loss."""
    loss_e_range: tuple = (0, 10000)
    """Iteration range for activating the the latent supervision loss."""
    mixed_precision: bool = False
    """Whether or not to use mixed precision for training."""
    use_grad_scaler: bool = False
    """Use gradient scaler even if the automatic mixed precision is disabled."""
    save_only_latest_checkpoint: bool = True
    """Whether to only save the latest checkpoint or all checkpoints."""
    # optional parameters if we want to resume training
    load_dir: Optional[Path] = None
    """Optionally specify a pre-trained model directory to load from."""
    load_step: Optional[int] = None
    """Optionally specify model step to load from; if none, will find most recent model in load_dir."""
    load_config: Optional[Path] = None
    """Path to config YAML file."""
    load_checkpoint: Optional[Path] = None
    """Path to checkpoint file."""
    log_gradients: bool = False
    """Optionally log gradients during training"""
    gradient_accumulation_steps: Dict[str, int] = field(default_factory=lambda: {})
    """Number of steps to accumulate gradients over. Contains a mapping of {param_group:num}"""
    device: TORCH_DEVICE = "cuda"
    """Device to run training on."""

class Trainer:

    pipeline: CustomPipeline
    optimizers: Optimizers
    callbacks: List[TrainingCallback]

    def __init__(self, config: TrainerConfig, vae_config: VAEConfig, local_rank: int = 0) -> None:
        self.train_lock = Lock()
        self.config = config
        self.vae_config = vae_config
        self.local_rank = local_rank
        self.world_size = 1
        self.device: TORCH_DEVICE = config.device

        if self.device == "cuda":
            self.device += f":{local_rank}"
        self.mixed_precision: bool = self.config.mixed_precision
        self.use_grad_scaler: bool = self.mixed_precision or self.config.use_grad_scaler
        self.gradient_accumulation_steps: DefaultDict = defaultdict(lambda: 1)
        self.gradient_accumulation_steps.update(self.config.gradient_accumulation_steps)

        if self.device == "cpu":
            self.mixed_precision = False
            CONSOLE.print("Mixed precision is disabled for CPU training.")
        self._start_step: int = 0
        self.grad_scaler = GradScaler(enabled=self.use_grad_scaler)
        self.base_dir: Path = config.get_base_dir()
        self.checkpoint_dir: Path = config.get_checkpoint_dir()

        CONSOLE.log(f"Saving checkpoints to: {self.checkpoint_dir}")
        self.viewer_state = None
        

        
    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datasets into memory
                'inference': does not load any dataset into memory
        """
        # setup normalizer
        self.normalizer = TanhNormalizer(scale=self.config.normalizer.tanh_scale, eps=self.config.normalizer.eps)

        # setup vae
        self.vae = AutoencoderKL.from_pretrained(
            self.vae_config.pretrained_model_name_or_path, 
            revision=self.vae_config.revision,
            subfolder=self.vae_config.subfolder
        )

        if self.vae_config.checkpoint:
            checkpoint = torch.load(self.vae_config.checkpoint, map_location=torch.device('cpu'))
            self.vae.load_state_dict(checkpoint['vae'])
        self.vae.to(self.device)
        # setup evaluation functions
        psnr_func = PeakSignalNoiseRatio(data_range=(0.0, 1.0))
        ssim_func = partial(structural_similarity_index_measure, data_range=(0.0, 1.0))
        lpips_func = LearnedPerceptualImagePatchSimilarity(normalize=True)
        mse_func = torch.nn.functional.mse_loss
        
        lpips_func.to(self.device)
        psnr_func.to(self.device)
        self.eval_funcs = {
            'psnr_func': psnr_func,
            'ssim_func': ssim_func,
            'lpips_func': lpips_func,
            'mse_func': mse_func
        }

        # setup pipeline
        self.pipeline = self.config.pipeline.setup(
            vae=self.vae,
            normalizer=self.normalizer,
            eval_funcs=self.eval_funcs,
            loss_e_range=self.config.loss_e_range,
            loss_d_range=self.config.loss_d_range,
            loss_ae_range=self.config.loss_ae_range,
            num_channels=self.vae_config.n_latent_channels,
            downsample_factor=self.vae_config.downsample_factor,
            latent_bg_color=self.vae_config.bg_color,
            device=self.device,
            test_mode=test_mode,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
        )


        # Setup optimizers
        self.setup_encoder_optim()
        self.setup_decoder_optim()
        self.setup_nerf_optim() 
        self.callbacks = self.pipeline.get_training_callbacks(
        TrainingCallbackAttributes(
                optimizers=self.optimizers, grad_scaler=self.grad_scaler, pipeline=self.pipeline, trainer=self
            )
        )

        # Disable optimizers at start
        self.disable_encoder_training()
        self.disable_decoder_training()
        self.disable_nerf_training() 

        self.pipeline.model.to(self.device)
        self._load_checkpoint()

    def setup_encoder_optim(self):
        self.optimizer_encoder = torch.optim.Adam([{
            'params' : self.vae.encoder.parameters(), 
            'lr' : self.config.encoder.lr
        }])
        self.scheduler_encoder = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer_encoder, 
            gamma=self.config.encoder.gamma_scheduler
        )
    def setup_decoder_optim(self):
        self.optimizer_decoder = torch.optim.Adam([{
            'params' : self.vae.decoder.parameters(), 
            'lr' : self.config.decoder.lr 
        }])
        self.scheduler_decoder = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer_decoder, 
            gamma= self.config.decoder.gamma_scheduler 
        )

        
    def enable_encoder_training(self): 
        self.train_encoder = True
        set_requires_grad(self.vae.encoder, requires_grad=True)

    def disable_encoder_training(self):
        self.train_encoder = False
        #self.optimizer_encoder = None
        #self.scheduler_encoder = None
        set_requires_grad(self.vae.encoder, requires_grad=False)

    def enable_decoder_training(self): 
        self.train_decoder = True
        set_requires_grad(self.vae.decoder, requires_grad=True)

    def disable_decoder_training(self):
        self.train_decoder = False
        #self.optimizer_decoder = None
        #self.scheduler_decoder = None
        set_requires_grad(self.vae.decoder, requires_grad=False)

    def setup_nerf_optim(self, print_n_params=False):
        optimizer_config = self.config.optimizers.copy()
        param_groups = self.pipeline.get_param_groups()
        self.optimizers = Optimizers(optimizer_config, param_groups)
        self.optimizers_initial_state = {
            'optimizers' : {
                key: optimizer.state_dict() for key, optimizer in self.optimizers.optimizers.items()
            },
            'schedulers': {
                key: scheduler.state_dict() for key, scheduler in self.optimizers.schedulers.items()
            }
        }

        if print_n_params:
            self.n_nerf_param = 0
            for param_group in param_groups.values():
                for param in param_group:
                    self.n_nerf_param += param.numel()
            print(f"METHOD_NAME: {self.config.method_name} - Number of nerf parameters: {self.n_nerf_param}")

    def reset_nerf_optimizers_states(self):
        for key, optimizer in self.optimizers.optimizers.items():
            optimizer.load_state_dict(self.optimizers_initial_state['optimizers'][key])
        for key, scheduler in self.optimizers.schedulers.items():
            scheduler.load_state_dict(self.optimizers_initial_state['schedulers'][key])

    def enable_nerf_training(self) -> None:
        self.train_nerf = True        
        param_groups = self.pipeline.get_param_groups()
        for param_group in param_groups.values():
            for param in param_group:
                param.requires_grad_(True)

    def disable_nerf_training(self) -> None: 
        self.train_nerf = False
        param_groups = self.pipeline.get_param_groups()
        for param_group in param_groups.values():
            for param in param_group:
                param.requires_grad_(False)

    def toogle_modules_in_range(self, step:int) ->None: 
        "Enable / disable training of encoder / decoder / nerf modules based on the current step."
        if step_in_range(step, self.config.train_encoder_range):
            if not self.train_encoder: 
                self.enable_encoder_training()
                if self.train_nerf:
                    self.reset_nerf_optimizers_states()
        else:
            if self.train_encoder:
                self.disable_encoder_training()

        if step_in_range(step, self.config.train_decoder_range):
            if not self.train_decoder:
                self.enable_decoder_training()
                if self.train_nerf:
                    self.reset_nerf_optimizers_states()
        else:
            if self.train_decoder:
                self.disable_decoder_training()
                
        if step_in_range(step, self.config.train_nerf_range):
            if not self.train_nerf:
                self.reset_nerf_optimizers_states()
                self.enable_nerf_training()
        else:
            if self.train_nerf:
                self.disable_nerf_training()

    def get_metric_types_to_log(self, step):
        loss_e_status = step_in_range(step,  self.config.loss_e_range)
        loss_d_status = step_in_range(step,  self.config.loss_d_range)
        loss_ae_status = step_in_range(step,  self.config.loss_ae_range)
        metrics_to_log = []
        if loss_e_status:
            metrics_to_log.append('metrics_e')
        if loss_d_status:
            metrics_to_log.append('metrics_d')
        if loss_ae_status:
            metrics_to_log.append('metrics_ae')
        return metrics_to_log

    def train(self) -> None:
        """Train the model."""
        
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"
        self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(
            self.base_dir / self.config.method_name / "dataparser_transforms.json"
        )

        max_num_iterations = self.config.max_num_iterations

        torch.cuda.empty_cache()
        #for step_start in trange(self._start_step, self._start_step + max_num_iterations, self.config.next_scene_batch_every, desc="Training"):
        for step in trange(max_num_iterations, desc="Training"):

            # eval iteration
            self.eval_iteration(step)

            # train iteration
            self.pipeline.train()
            self.toogle_modules_in_range(step=step)
            total_loss, scene_loss_dict, metrics_dict = self.train_iteration(step)

            if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                loss_e_status = step_in_range(step,  self.config.loss_e_range)
                loss_d_status = step_in_range(step,  self.config.loss_d_range)
                loss_ae_status = step_in_range(step,  self.config.loss_ae_range)

                to_log = custom_dict({ 
                        'global_information/training_step': step,
                        'global_information/GPU memory (MB)': torch.cuda.max_memory_allocated() / (1024**2),
                        'train_losses/total_loss': total_loss,
                        'train_losses/' : scene_loss_dict,
                        'training_status/train_nerf': int(self.train_nerf),
                        'training_status/train_encoder': int(self.train_encoder),
                        'training_status/train_decoder': int(self.train_decoder),
                        'training_status/loss_e': int(loss_e_status),
                        'training_status/loss_d': int(loss_d_status),
                        'training_status/loss_ae': int(loss_ae_status),
                    })
                

                # Computation of single-view train metrics 
                for metric_type in self.get_metric_types_to_log(step):
                    to_log[f"train_metrics/"][metric_type] = metrics_dict[metric_type] 

                wandb.log(to_log)

            if step_check(step, self.config.steps_per_save):
                self.save_checkpoint(step)
                

        # save checkpoint at the end of training
        self.save_checkpoint(step)

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("Config File", str(self.config.get_base_dir() / "config.yml"))
        table.add_row("Checkpoint Directory", str(self.checkpoint_dir))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Training Finished :tada:[/bold]", expand=False))

        # after train end callbacks
        for callback in self.callbacks:
            callback.run_callback_at_location(step=step, location=TrainingCallbackLocation.AFTER_TRAIN)

    def callback_before(self, step):
        for callback in self.callbacks:
            callback.run_callback_at_location(
                step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
            )

    def callback_after(self, step):
        for callback in self.callbacks:
            callback.run_callback_at_location(
                step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
            )

    def forward_pass(self, step):
        # forward pass
        cpu_or_cuda_str: str = self.device.split(":")[0]
        cpu_or_cuda_str = "cpu" if cpu_or_cuda_str == "mps" else cpu_or_cuda_str

        if self.train_nerf:
            needs_zero = [
                group for group in self.optimizers.parameters.keys() if step % self.gradient_accumulation_steps[group] == 0
            ]
            self.optimizers.zero_grad_some(needs_zero)

        # ---- backward here ----
        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
            assert len(loss_dict) != 0, "You should optimize at least one loss"
            loss = functools.reduce(torch.add, loss_dict.values())

        return loss, loss_dict, metrics_dict
    
    def optimizer_scheduler_step(self, step):
        if self.train_nerf:
            needs_step = [
                group
                for group in self.optimizers.parameters.keys()
                if step % self.gradient_accumulation_steps[group] == self.gradient_accumulation_steps[group] - 1
            ]
            self.optimizers.optimizer_scaler_step_some(self.grad_scaler, needs_step)
            scale = self.grad_scaler.get_scale()
            self.grad_scaler.update()
            # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
            if scale <= self.grad_scaler.get_scale():
                self.optimizers.scheduler_step_all(step)                

    def train_iteration(self, step):
        """Makes one forward pass on each scene  of the batch then the backward and optimization on the total loss"""
        loss_dict = custom_dict()
        metrics_dict = custom_dict()

            
        self.callback_before(step)
        loss, loss_dict, metrics_dict = self.forward_pass(step)
        self.grad_scaler.scale(loss).backward()  
        self.optimizer_scheduler_step(step) 
        self.callback_after(step)

        # for logging need to detach the tensors in loss_dict and metrics_dict
        loss_dict = free_dict(loss_dict)
        metrics_dict = free_dict(metrics_dict)
        

        # steps on encoder and decoder optimizer / scheduler
        if self.train_encoder:
            self.optimizer_encoder.step()
            self.scheduler_encoder.step()
            self.optimizer_encoder.zero_grad()
        if self.train_decoder:
            self.optimizer_decoder.step()
            self.scheduler_decoder.step()
            self.optimizer_decoder.zero_grad()

        for param_group in self.pipeline.model.get_param_groups().values():
            for param in param_group:
                param.grad = None

        return loss.detach().cpu(), loss_dict, metrics_dict

    
    @torch.no_grad()
    def eval_iteration(self, step : int) -> None:
        """Run one iteration with different batch/image/all image evaluations depending on step size.

        Args:
            step: Current training step.
        """
        # loggging images
        if step_check(step, self.config.steps_per_eval_image):
            to_log = custom_dict({"global_information/training_step": step})
            to_log["figures/fig_eval"] = self.pipeline.get_images(step=step, split='eval')
            to_log["figures/fig_train"] = self.pipeline.get_images(step=step, split='train')

            plt.close("all")

            wandb.log(to_log)

        # logging psnr across all test images
        if step_check(step, self.config.steps_per_eval_all_images):
            """
            This logs the metrics (PSNR, SSIM, LPIPS) for test views:
                - metrics_dict_e: comparison between rendered and encoded ground truth views
                - metrics_d_dict: comparison between decoded renderings and ground truth views
                - metrics_ae_dict: comparison between autoencoded and ground truth views
                - metrics_d_ae_dict: comparison between decoded renderings and autoencoded views
            """

            pipeline = self.pipeline
            metrics_dict_e, metrics_ae_dict, metrics_d_dict, metrics_d_ae_dict = pipeline.get_average_eval_metrics(step=step)
            
            to_log = custom_dict({
                "global_information/training_step": step,
                "eval_metrics/metrics_e" : metrics_dict_e,
                "eval_metrics/metrics_ae" : metrics_ae_dict,
                "eval_metrics/metrics_d" : metrics_d_dict,
                "eval_metrics/metrics_d-ae" : metrics_d_ae_dict
            })   

            wandb.log(to_log)

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.load_dir
        load_checkpoint = self.config.load_checkpoint
        if load_dir is not None:
            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest Nerfstudio checkpoint from load_dir...")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")

            CONSOLE.print(f"Loading Nerfstudio checkpoint from {load_path}")

        elif load_checkpoint is not None:
            assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
            loaded_state = torch.load(load_checkpoint, map_location="cpu")

            CONSOLE.print(f"Loading Nerfstudio checkpoint from {load_checkpoint}")
        else:
            loaded_state = None
            CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")
        
        if loaded_state:
            self._start_step = loaded_state["step"] + 1
            self.vae.load_state_dict(loaded_state['vae'])
            self.optimizer_encoder.load_state_dict(loaded_state["optimizer_encoder"])
            self.scheduler_encoder.load_state_dict(loaded_state["scheduler_encoder"])
            self.optimizer_decoder.load_state_dict(loaded_state["optimizer_decoder"])
            self.scheduler_decoder.load_state_dict(loaded_state["scheduler_decoder"])

            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state['step'])

            # load the checkpoints for pipeline, optimizers, and gradient scalar

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers

        Args:
            step: number of steps in training for given checkpoint
        """

        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path: Path = self.checkpoint_dir / f"step-{step:09d}.ckpt"

        # Save the VAE, the optimizers, and the pipeline
        state = {
            "step": step,
            'vae' : self.vae.state_dict(),
            'optimizer_encoder' : self.optimizer_encoder.state_dict(),
            'scheduler_encoder' : self.scheduler_encoder.state_dict(),
            'optimizer_decoder' : self.optimizer_decoder.state_dict(),
            'scheduler_decoder' : self.scheduler_decoder.state_dict(),
            "pipeline": self.pipeline.module.state_dict()  # type: ignore
            if hasattr(self.pipeline, "module")
            else self.pipeline.state_dict(),
            "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
            "schedulers": {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()},
            "scalers": self.grad_scaler.state_dict(),
        }
        
        torch.save(
            state,
            ckpt_path,
        )

        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("*"):
                if f != ckpt_path:
                    f.unlink()



