import PIL.Image
import os, yaml
import collections.abc
import torch
import matplotlib.pyplot as plt
import PIL

# --- logging utils ---
class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# --- config utils ---
def yaml_load(cfg_name, load_dir):
    config_path = os.path.join(load_dir, cfg_name)
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def deep_update(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        assert key in source.keys(), f"key {key} not in source"
        if isinstance(value, collections.abc.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]

    return source

def load_config(cfg_name, load_dir, from_default=False, default_cfg_name='default.yaml') :
    """Load a configuration file. If from_default is True, load 
    the default config and update it with the config file"""
    
    config = yaml_load(cfg_name, load_dir)

    if from_default :
        default_config = yaml_load(default_cfg_name, load_dir)
        config = deep_update(default_config, config)

    return config

# --- normalization utils ---
class TanhNormalizer():

    def __init__(self, scale=0.02, eps=1e-6):
        self.scale = scale
        self.eps = eps
        self.tanh = torch.nn.Tanh()
    
    def apply_norm(self, x):
        "Expect input in the latent space range and maps it in [0,1]"
        return renormalize_img(self.tanh(self.scale * x))

    def deapply_norm(self, x):
        "Expect input in the [0, 1] range and maps in in the latent space range"
        x = denormalize_img(x)
        x = x.clamp(-1 + self.eps, 1 - self.eps)
        return (1/self.scale) * torch.atanh(x)

def renormalize_img(x):
    "expect image in [-1,1] range and returns it in [0,1] range"
    return 0.5 * (x + 1)

def denormalize_img(x):
    "expect image in [0,1] range and returns it in [-1,1] range"
    return 2 * x - 1

# --- torch utils ---- 
def set_requires_grad(m, requires_grad):
    for param in m.parameters():
        param.requires_grad_(requires_grad)

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
                        

# --- custom pipeline utils ---

def get_rgb_img_key(model_outputs, num_channels):
    if 'rgb' in model_outputs.keys():
        rgb_img_key = 'rgb'
    elif 'rgb_fine' in model_outputs.keys():
        rgb_img_key = 'rgb_fine'
    # safety check
    assert model_outputs[rgb_img_key].size(-1) == num_channels
    return rgb_img_key

def interactive_plot(step:int, render_img:torch.Tensor, encoded_gt_img:torch.Tensor, decoded_render_img:PIL.Image, gt_rgb:PIL.Image, decoded_encoded_gt_img: PIL.Image):
    num_channels_pred = render_img.size(-1)
    num_channel_encoded = encoded_gt_img.size(-1)
    
    fig, axes = plt.subplots(
        nrows=2, 
        ncols=max(num_channels_pred, num_channel_encoded) + 2, 
        figsize=(2*max(num_channels_pred, num_channel_encoded), 4)
    )
    fig.suptitle(f"Training step {step}")

    for i in range(num_channels_pred):
        axes[0, i].imshow(render_img[..., i].detach().cpu(), vmin=0.0, vmax=1.0)
        axes[0, i].set_title(f"Render c{i}")
        axes[0, i].axis('off')
    axes[0, i+1].imshow(decoded_render_img)
    axes[0, i+1].set_title("Render dec")
    axes[0, i+1].axis('off')
    axes[0, i+2].axis('off')

    for i in range(num_channel_encoded):
        axes[1, i].imshow(encoded_gt_img[..., i].detach().cpu(), vmin=0.0, vmax=1.0)
        axes[1, i].set_title(f"GT enc c{i}")
        axes[1, i].axis('off')

    axes[1, i+1].imshow(gt_rgb)
    axes[1, i+1].set_title("GT")
    axes[1, i+1].axis('off')

    axes[1, i+2].imshow(decoded_encoded_gt_img)
    axes[1, i+2].set_title("AutoEnc")
    axes[1, i+2].axis('off')

    return fig

def step_in_range(step, train_range):
    if train_range[1] == 'inf':
        return train_range[0] <= step
    elif train_range[0] == 'inf':
        return False
    return train_range[0] <= step < train_range[1]


    
if __name__ == '__main__': 
    norm = TanhNormalizer()
    t = torch.randn(1, 3, 256, 256) * 3
    t_norm = norm.apply_norm(t)
    t_ = norm.deapply_norm(t_norm)

    threshold = 1e-5
    assert ((t - t_).abs() < threshold).all()