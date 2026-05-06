@echo off
setlocal
REM Create a shortcut to this file on the desktop if desired.
REM This launcher is location-independent and does not store user-specific paths.

python "%~dp0ocr_workflow.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
pause
exit /b %EXIT_CODE%
