@echo off
setlocal
cd /d "%~dp0"

echo Building portable GeoFinder Desktop for Windows...
echo This may take 10-30 minutes and needs several GB of disk space.
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" tools\windows_desktop\build.py %*
) else (
    python tools\windows_desktop\build.py %*
)

if errorlevel 1 (
    echo.
    echo Build FAILED.
    pause
    exit /b 1
)

echo.
echo Done.
echo Folder: dist\GeoFinder\
echo Zip:    dist\GeoFinder-Windows-portable.zip
echo Copy the folder or zip to another PC and run GeoFinder.exe
echo.
pause
