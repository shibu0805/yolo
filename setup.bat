@echo off
chcp 65001 >nul
rem 點兩下就能安裝：Windows 預設不允許直接執行 .ps1，所以用這個檔案代為執行 setup.ps1
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File setup.ps1
pause
