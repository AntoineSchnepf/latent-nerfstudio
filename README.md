TODO:
- test environement installation


# Implementation of latent Nerfstudio extension

> Antoine Schnepf*, Karim Kassab*, Jean-Yves Franceschi, Laurent Caraffa, Flavian Vasile, Jeremie Mary, Andrew Comport, Valérie Gouet-Brunet (* indicates equal contribution)<br>
| [Project Page](https://ig-ae.github.io) | [Full Paper](https://arxiv.org/abs/2410.22936) |<br>

This repo modifies [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) to support training NeRFs in a VAE latent space.
We use it to train latent NeRF architectures in our paper **Bringing NeRFs to the Latent Space: Inverse Graphics Autoencoder**.

![LatentNeRFTrainingPipeline](assets/latent_nerf_training_pipeline.svg)

### Environment 
Our code has been tested on:
- Linux (Debian)
- Python 3.8.19
- Pytorch 2.1.2
- CUDA 11.8
- `L4` and `A100` NVIDIA GPUs

You can use Anaconda to create the environment:
```
conda create --name latent-nerfstudio -y python=3.8.19
conda activate latent-nerfstudio
```

Then install pytorch
```
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c “nvidia/label/cuda-11.8.0” cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

To install the remaining requirements, execute:
```
pip install -r requirements.txt
```

### Usage
Our implementation adopts the same standards as Nerfstudio and utilizes [Tyro](https://github.com/brentyi/tyro) for configuration management.

To train a latent NeRF:
```
python train.py \
<method_name> --data <path_to_scene_data> nerfstudio-data \
<vae_name> --checkpoint <path_to_vae_checkpoint>
```
where 
- method_name $\in$ {nerfacto, vanilla-nerf, tensorf, kplanes, instant-ngp},
- vae_name $\in$ {ostris, stable-diffusion}.

Note that our code only supports the [Nerfstudio dataparser](https://docs.nerf.studio/reference/api/data/dataparsers.html#nerfstudio), and hence the data has to be formatted accordingly.


## Evaluation
We visualize and evaluate our latent NeRFs using [wandb](https://wandb.ai/site). You can find a quickstart guide [here](https://docs.wandb.ai/quickstart).
During its training, a latent NeRF can be visualized in the RGB space and latent space via the logged dashboard figure:

![LatentNeRFTrainingPipeline](assets/metrics.svg)

On this figure, the top row illustrates the channels of latent NeRF renderings (`Render c*`.) and the corresponding decoded image (`Render dec`).
The bottom row illustrate the ground truth image (`GT`), and corresponding latent encoding (`GT enc c*`), which the latent NeRF aims at reproducing. Additionally, it displays the auto-encoded reconstruction of the GT (`AutoEnc`).

Adittionaly to the above visual evaluation, we compute 3 types of metrics to quantitatively evaluate latent NeRF performances:
- `metrics_e` is the error between the encoded image (`GT enc c*`) and the image rendered by the latent NeRF (`Render c*`). 
- `metrics_d` is the error between the decoded rendering of the latent NeRF (`Render dec`) and the ground truth image (`GT`). It measures the quality of the latent NeRF in the RGB space. 
- `metrics_ae` is the error between the ground truth image (`GT`) and its auto-encoded reconstruction (`AutoEnc`). It measures the performance of the auto-encoder. 

Note that, as our scenes are learned in the latent space, the default Nerfstudio [viewer interface](https://docs.nerf.studio/quickstart/viewer_quickstart.html) is not supported in our code.

## License
This code is open-source. It is shared under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
It modifes the code of Nerfstudio which is also licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0) to support training in the latent space.
Major changes are annotated with the comment ``changes related to latent_nerfstudio``.

## Citation

If you find this research project useful, please consider citing our work:
```
@article{ig-ae,
      title={{Bringing NeRFs to the Latent Space: Inverse Graphics Autoencoder}}, 
      author={Antoine Schnepf and Karim Kassab and Jean-Yves Franceschi and Laurent Caraffa and Flavian Vasile and Jeremie Mary and Andrew Comport and Valérie Gouet-Brunet},
      journal={arXiv preprint arXiv:2410.22936},
      year={2024}
}
```