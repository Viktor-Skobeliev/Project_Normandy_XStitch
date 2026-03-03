@echo off
echo ========================================
echo  XStitch Pattern Generator - Build
echo ========================================

set PYTHON="D:\Program files\Work\Python\python.exe"
set PIP="D:\Program files\Work\Python\Scripts\pip.exe"

echo [1/3] Installing PyInstaller...
%PYTHON% -m pip install pyinstaller --quiet

echo [2/3] Building executable...
%PYTHON% -m PyInstaller build.spec --clean --noconfirm

echo [3/3] Done!
echo.
echo Output: dist\XStitch\XStitch.exe
echo.
pause
