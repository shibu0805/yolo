@echo off
chcp 65001 >nul
rem 點兩下啟動網站（http://127.0.0.1:7861，會自動開瀏覽器）。關掉這個黑色視窗 = 關掉網站
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
    echo 還沒安裝，請先點兩下 setup.bat
    pause
    exit /b 1
)
venv\Scripts\python.exe app.py
pause
