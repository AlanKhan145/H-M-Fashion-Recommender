# Install ffmpeg (winget) + deno + python packages
winget install -e --id Gyan.FFmpeg
winget install -e --id DenoLand.Deno

py -m pip install -U pip
py -m pip install -r requirements.txt

deno --version
py -m ytdlp_batch --version
