#!/bin/bash

# Download the monocular depth estimation model
# We use DPT by default but the code can work with any monocular depth estimator

git submodule update --init --recursive

# Download  weights
mkdir ./DPT/weights -p
wget 'https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt' -O ./DPT/weights/dpt_hybrid-midas-501f0c75.pt -q --show-progress
