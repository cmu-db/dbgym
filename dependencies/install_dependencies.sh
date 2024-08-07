#!/bin/bash
# You may want to create a conda environment before doing this
pip install -r dependencies/requirements.txt
cat dependencies/apt_requirements.txt | xargs sudo apt-get install -y
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y