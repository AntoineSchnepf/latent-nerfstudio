#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda activate latent-nerfstudio

# This will exec the CMD from your Dockerfile, i.e. "npm start"
exec "$@"
