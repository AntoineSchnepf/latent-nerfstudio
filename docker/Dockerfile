FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app/docker/
ARG CUDA_ARCHITECTURES=89

# Install basics
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    g++ \
    git \
    libx264-dev \
    emacs \
    tmux \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Install Miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
    
ENV PATH=$CONDA_DIR/bin:$PATH

# Create conda environment and install packages
RUN conda create --name latent-nerfstudio -y python=3.8.19

SHELL ["conda", "run", "-n", "latent-nerfstudio", "/bin/bash", "-c"]

RUN /opt/conda/envs/latent-nerfstudio/bin/python3.8 -m pip install --no-cache-dir --upgrade pip && \
    /opt/conda/envs/latent-nerfstudio/bin/python3.8 -m pip install --no-cache-dir torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
    
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate latent-nerfstudio && \
    conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

RUN  TCNN_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" /opt/conda/envs/latent-nerfstudio/bin/python3.8 -m pip  install --no-cache-dir "git+https://github.com/NVlabs/tiny-cuda-nn.git@b3473c81396fe927293bdfd5a6be32df8769927c#subdirectory=bindings/torch"

RUN git clone https://github.com/AntoineSchnepf/latent-nerfstudio.git /app/docker/latent-nerfstudio/
WORKDIR /app/docker/latent-nerfstudio/
RUN  /opt/conda/envs/latent-nerfstudio/bin/python3.8 -m pip install --no-cache-dir -r requirements.txt
RUN  /opt/conda/envs/latent-nerfstudio/bin/python3.8 -m pip install nerfacc

# Setup entrypoint
COPY docker/entrypoint.sh ./
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate latent-nerfstudio" >> ~/.bashrc



ENTRYPOINT ["/app/docker/latent-nerfstudio/entrypoint.sh"]
CMD ["/bin/bash"]