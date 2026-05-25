@echo off
setlocal enabledelayedexpansion

set TARGET=%~1
set EXCLUDE= 
set EXTRA_ARGS=

if "%TARGET%"=="" (
    echo Usage: nd_batch.bat [config_file_or_folder] [--exclude file1,file2,...] [extra_args...]
    exit /b 1
)

set CONFIG_PATH=configs/network_dismantling/%TARGET%
set TRAIN_SCRIPT=tools/train.py
set TEST_SCRIPT=tools/test.py
set TRAIN_ARGS=--use_logging --verbose --use_tensorboard
set TEST_ARGS=--use_logging --verbose

if exist "%CONFIG_PATH%\" (
    for %%f in ("%CONFIG_PATH%\*.yaml") do (
        set SKIP=0
        if defined EXCLUDE (
            set FILENAME=%%~nxf
            for %%e in (!EXCLUDE!) do (
                if /i "!FILENAME!"=="%%e" set SKIP=1
            )
        )
        if !SKIP! equ 0 (
            echo [%%~nxf] train
            python %TRAIN_SCRIPT% "%%f" %TRAIN_ARGS% %EXTRA_ARGS%
            if !errorlevel! equ 0 (
                echo [%%~nxf] test
                python %TEST_SCRIPT% "%%f" %TEST_ARGS% %EXTRA_ARGS%
            )
        )
    )
) else (
    echo [%TARGET%] train
    python %TRAIN_SCRIPT% %CONFIG_PATH% %TRAIN_ARGS% %EXTRA_ARGS%
    if !errorlevel! equ 0 (
        echo [%TARGET%] test
        python %TEST_SCRIPT% %CONFIG_PATH% %TEST_ARGS% %EXTRA_ARGS%
    )
)
