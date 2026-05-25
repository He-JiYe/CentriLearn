@echo off
REM Network Dismantling Task Training
REM usage: nd_train.bat [config_file] [options]

setlocal enabledelayedexpansion
set TRAIN_SCRIPT=tools/train.py
set DEFAULT_ARGS=--use_logging --verbose --use_tensorboard

set CONFIG_FILE=configs/network_dismantling/%~1
set CMD=python %TRAIN_SCRIPT% %CONFIG_FILE% %DEFAULT_ARGS%
shift

:parse_args
if "%~1"=="" goto run_command

set CMD=!CMD! %~1
shift
goto parse_args

:run_command
%CMD%

goto :eof

