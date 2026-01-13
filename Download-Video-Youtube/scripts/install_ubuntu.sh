#!/usr/bin/env bash
set -e

sudo apt-get -y update
sudo apt-get -y install ffmpeg curl

python3 -m pip install -U pip
python3 -m pip install -r requirements.txt

# Install deno
curl -fsSL https://deno.land/install.sh | sh

echo ""
echo "Add deno to PATH (bash):"
echo '  echo '\''export PATH="$HOME/.deno/bin:$PATH"'\'' >> ~/.bashrc'
echo '  source ~/.bashrc'
echo ""
echo "Check:"
echo "  deno --version"
echo "  python3 -m ytdlp_batch --version"
