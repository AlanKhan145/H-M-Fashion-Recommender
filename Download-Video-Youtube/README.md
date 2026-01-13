
cd youtube_downloader
python3 -m venv .venv
source .venv/bin/activate

bash scripts/install_ubuntu.sh
source ~/.bashrc   # để PATH có deno

cp config.example.txt config.txt
cp urls.example.txt urls.txt
# đặt cookies.txt (nếu cần)

python -m ytdlp_batch --config config.txt
Setup (Windows)
Cài Python 3.10+ (khuyến nghị 3.11)

Mở PowerShell Admin:

powershell
Copy code
cd youtube_downloader
py -m venv .venv
.\.venv\Scripts\Activate.ps1

.\scripts\install_windows.ps1

copy config.example.txt config.txt
copy urls.example.txt urls.txt

py -m ytdlp_batch --config config.txt