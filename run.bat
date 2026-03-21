@echo off
REM ==================================================
REM  IRRA 模型训练启动脚本（Windows .bat）
REM  作者：光执
REM  用途：启动 CUHK-PEDES 数据集上的 IRRA 训练
REM  兼容：PyCharm 右键 Run / 双击执行 / 命令行调用
REM ==================================================

REM ========== 用户配置区 ==========
set CONDA_ENV_NAME=irra_env        REM ← ← ← 请修改为你的 conda 环境名！
set MODEL_NAME=irra
set BATCH_SIZE=32
set LOSS_NAMES=sdm+mlm+id
set DATASET_NAME=CUHK-PEDES
set ROOT_DIR=C:\Guangz\Projects\IRRA-main\data\CUHK-PEDES
set NUM_EPOCH=60
REM ==============================

REM 创建 logs 目录（若不存在）
if not exist "logs" mkdir "logs"

REM 生成带时间戳的日志文件名：logs/train_YYYYMMDD_HHMMSS.log
for /f "tokens=1-4 delims=/:. " %%a in ('wmic os get LocalDateTime ^| find "."') do set datetime=%%a
set LOG_FILE=logs\train_%datetime:~0,8%_%datetime:~8,6%.log

echo [INFO] 正在激活 conda 环境: %CONDA_ENV_NAME%
call conda activate %CONDA_ENV_NAME% 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] conda 环境激活失败！请检查环境名是否正确：'%CONDA_ENV_NAME%'
    echo        可通过 `conda env list` 查看现有环境
    pause
    exit /b 1
)

echo [INFO] 检查数据路径: %ROOT_DIR%
if not exist "%ROOT_DIR%" (
    echo [ERROR] 数据目录不存在！请检查路径：
    echo         %ROOT_DIR%
    pause
    exit /b 1
)

echo [INFO] 启动训练，日志将保存至：%LOG_FILE%
echo ================================================== >> "%LOG_FILE%"
echo [START] %date% %time% - Training IRRA on %DATASET_NAME% >> "%LOG_FILE%"
echo -------------------------------------------------- >> "%LOG_FILE%"

REM 启动训练（所有参数严格加引号防空格/特殊字符）
python train.py ^
    --name "%MODEL_NAME%" ^
    --img_aug ^
    --batch_size %BATCH_SIZE% ^
    --MLM ^
    --loss_names "%LOSS_NAMES%" ^
    --dataset_name "%DATASET_NAME%" ^
    --root_dir "%ROOT_DIR%" ^
    --num_epoch %NUM_EPOCH% ^
    >> "%LOG_FILE%" 2>&1

REM 检查训练是否成功
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] 训练完成！日志已保存至：%LOG_FILE%
    echo [END] %date% %time% - Success >> "%LOG_FILE%"
) else (
    echo [FAILED] 训练失败！请查看日志：%LOG_FILE%
    echo [END] %date% %time% - Failed >> "%LOG_FILE%"
    echo.
    type "%LOG_FILE%" | findstr /C:"Traceback" /C:"Error" /C:"Exception"
    echo.
    echo 按任意键退出...
    pause >nul
    exit /b 1
)

echo 按任意键关闭窗口...
pause >nul