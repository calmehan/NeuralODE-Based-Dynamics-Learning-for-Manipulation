#!/usr/bin/env bash
set -euo pipefail


# Install remaining Python dependencies
python3 -m pip install \
    torchdiffeq \
    numpngw \
    gym \
    pandas \
    tqdm \
    pillow

echo "All dependencies installed!"
