@echo off
REM Starts the Disc Pad Pull Monitor web app. Double-click this file, then
REM open http://localhost:5050 in a browser. Close this window to stop the app.
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python -m webapp.app %*
pause
