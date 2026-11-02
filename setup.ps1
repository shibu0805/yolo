# 一鍵建立環境：在 yolov7-test 資料夾裡執行 .\setup.ps1（或直接點兩下 setup.bat）
# 1. 建立 venv（Python 3.10）  2. 安裝 requirements.txt  3. 下載官方模型到 yolov7\
# 已經有 venv、模型時會跳過，只補缺少的部分，所以重複執行也沒關係
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Test-Path venv\Scripts\python.exe)) {
    # 要用 Python 3.10（yolov7 需要 numpy 1.23，不支援太新的 Python）
    $py = Get-Command py -ErrorAction SilentlyContinue
    if (-not $py) { throw '找不到 py 指令，請先從 https://www.python.org 安裝 Python（安裝時勾選 py launcher）' }
    py -V:3.10 --version 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host '找不到 Python 3.10，嘗試安裝...'
        py install 3.10
    }
    Write-Host '建立 venv...'
    py -V:3.10 -m venv venv
}

Write-Host '安裝套件（第一次要下載約 3GB，請耐心等候）...'
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw '套件安裝失敗，請看上面的錯誤訊息' }

# 官方模型（檔案太大沒有放在 GitHub 上，從 yolov7 官方的 release 下載）
$ProgressPreference = 'SilentlyContinue'   # 關掉進度條，下載會快很多
foreach ($w in 'yolov7.pt', 'yolov7_training.pt', 'yolov7-tiny.pt') {
    $dst = "yolov7\$w"
    if (-not (Test-Path $dst)) {
        Write-Host "下載 $w ..."
        Invoke-WebRequest "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/$w" -OutFile $dst
    }
}

Write-Host ''
.\venv\Scripts\python.exe -c "import torch, gradio; print('torch', torch.__version__, '| CUDA 可用:', torch.cuda.is_available(), '| gradio', gradio.__version__)"
Write-Host '完成！執行網站：點兩下 start.bat，或執行 .\venv\Scripts\python.exe app.py'
