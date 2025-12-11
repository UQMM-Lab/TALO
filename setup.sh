#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# 1. Install Python dependencies
echo "Installing base requirements..."
pip3 install -r requirements.txt

# 2. Clone and install Salad
echo "Cloning and installing Salad..."
git clone https://github.com/Dominic101/salad.git
pip install -e ./salad

# 4. Clone and install VGGT
echo "Cloning and installing Compressed VGGT..."
git clone git@github.com:Xian-Bei/vggt-compressed.git
pip install -e ./vggt-compressed

# 5. Install current repo in editable mode
echo "Installing current repo..."
pip install -e .

echo "Installation Complete"