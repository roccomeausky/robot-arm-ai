@echo off
REM MIRA laptop launcher
cd /d "%~dp0\..\python"

if not defined ROBOT_SERIAL_PORT set ROBOT_SERIAL_PORT=COM3
if not defined ROBOT_TOF_PORT set ROBOT_TOF_PORT=COM7
if not defined ROBOT_LISTEN_SECONDS set ROBOT_LISTEN_SECONDS=8
if not defined ROBOT_WHISPER_MODEL set ROBOT_WHISPER_MODEL=base.en
if not defined ROBOT_WAKE_PHRASE set ROBOT_WAKE_PHRASE=hey mira
if not defined ROBOT_TTS_VOICE set ROBOT_TTS_VOICE=en-GB-SoniaNeural

echo [mira] serial=%ROBOT_SERIAL_PORT%  tof=%ROBOT_TOF_PORT%  whisper=%ROBOT_WHISPER_MODEL%
python robot_llm_voice.py %*
pause
