@echo off
REM Starts a SEPARATE Chrome instance with a CDP port and its own profile — for
REM utils/deepseek_verdict.py (the bot's live path) and agents/deepseek_historical_verdict.py
REM (one-off historical run). Does not touch an already open regular Chrome.
REM After it starts, log in to chat.deepseek.com in this window manually.
set DS_PORT=9222
if not "%DS_DEBUG_PORT%"=="" set DS_PORT=%DS_DEBUG_PORT%

start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" ^
  --remote-debugging-port=%DS_PORT% ^
  --remote-allow-origins=* ^
  --user-data-dir=%TEMP%\chrome_debug_profile ^
  --no-first-run ^
  --no-default-browser-check ^
  https://chat.deepseek.com/
