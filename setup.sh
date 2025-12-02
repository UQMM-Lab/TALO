#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# 2. Clone and install Salad
# echo "Cloning and installing Salad..."
# git clone https://github.com/Dominic101/salad.git
pip install -e ./salad

# 4. Clone and install VGGT
# echo "Cloning and installing VGGT..."
# git clone https://github.com/facebookresearch/vggt.git
pip install -e ./vggt


echo "Installation Complete"
