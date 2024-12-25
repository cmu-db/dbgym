#!/bin/bash
# "sysdeps" stands for "system dependencies".
# These are dependencies unrelated to Python that the dbgym needs.
cat scripts/configs/apt_requirements.txt | xargs sudo apt-get install -y
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
