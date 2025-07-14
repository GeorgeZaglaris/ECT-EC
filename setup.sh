#!/bin/bash
# Install Git LFS
apt-get update
apt-get install -y git-lfs

# Initialize Git LFS
git lfs install

# Pull the LFS files
git lfs pull
