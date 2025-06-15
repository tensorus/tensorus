#!/usr/bin/env bash
# Setup script for Tensorus development environment
set -e

# Install Python dependencies
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
pip install -r requirements-test.txt

if [[ "$INSTALL_MODELS" == "1" ]]; then
  pip install -e .[models]
fi
