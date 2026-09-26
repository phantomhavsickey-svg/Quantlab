@echo off
chcp 65001 >nul
title QuantLab 量化研究与交易系统
cd /d "%~dp0"

:: 本机 Python 3.14（绝对路径启动，不依赖 PATH）
set "PY=C:\Users\blessing\AppData\Local\Programs\Python\Python314\python.exe"
if not exist "%PY%" set "PY=python"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "LOGURU_LEVEL=ERROR"

:menu
cls
echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║       QuantLab 量化研究与交易系统 v1.1       ║
echo  ╠══════════════════════════════════════════════╣
echo  ║   1. 增量下载数据   日线追加到 data\cache     ║
echo  ║   2. 计算因子       factor_panel.parquet      ║
echo  ║   3. 训练模型       Walk-Forward 滚动 8 折    ║
echo  ║   4. 运行回测       读 predictions.parquet    ║
echo  ║   5. 模拟盘         逐日撮合，不动真钱         ║
echo  ║   6. 每日流程       live\run_daily.py 出指令  ║
echo  ║   7. 一键全流程     下载→因子→训练→回测       ║
echo  ║   8. 查看报告       打开最新 HTML 报告         ║
echo  ║   9. 自检           pytest + 前视审计         ║
echo  ║   0. 退出                                    ║
echo  ║                                              ║
echo  ║  股票池/日期/仓位规则全部读 config.yaml，     ║
echo  ║  这里不再传 --universe 与硬编码日期。         ║
echo  ╚══════════════════════════════════════════════╝
echo.

set /p choice=  请选择 [0-9]:

if "%choice%"=="1" goto download
if "%choice%"=="2" goto factors
if "%choice%"=="3" goto train
if "%choice%"=="4" goto backtest
if "%choice%"=="5" goto paper
if "%choice%"=="6" goto daily
if "%choice%"=="7" goto pipeline
if "%choice%"=="8" goto report
if "%choice%"=="9" goto selfcheck
if "%choice%"=="0" goto exit
goto menu

:download
echo.
echo  增量下载（已缓存的只补新的），耗时取决于网络...
"%PY%" main.py download
goto done

:factors
echo.
echo  计算因子 + 因子评估（几分钟）...
"%PY%" main.py factors
goto done

:train
echo.
echo  Walk-Forward 滚动训练，产出 data\cache\predictions.parquet（最慢的一步）...
"%PY%" main.py train
goto done

:backtest
echo.
echo  回测：读现有预测，改仓位参数只需重跑这一步...
"%PY%" main.py backtest
echo  报告已生成到 reports\ 目录
goto done

:paper
echo.
echo  模拟盘（纸面撮合，不会下真实委托）...
"%PY%" main.py paper-trade
goto done

:daily
echo.
echo  每日流程：增量更新 → 信号 → 调仓指令文件（live\output\）。
echo  默认不带任何下单开关，只产出文件。
echo.
set /p sim="  用模拟券商撮合? (Y/N, 默认N): "
if /i "%sim%"=="Y" (
    "%PY%" live\run_daily.py --simulate
) else (
    "%PY%" live\run_daily.py
)
echo  ⚠ 真实柜台下单请手动执行 "%PY%" live\run_daily.py --qmt --confirm
echo    （需先完成券商程序化交易报备；本菜单不提供该组合）
goto done

:pipeline
echo.
echo  全流程：下载 → 因子 → 训练 → 回测（十几分钟起）...
"%PY%" main.py pipeline
goto done

:report
echo.
for /f "delims=" %%f in ('dir /b /o-d reports\*.html 2^>nul') do (
    start "" "reports\%%f"
    goto menu
)
echo  未找到报告文件，请先运行回测
goto done

:selfcheck
echo.
echo  [1/2] 单元测试（应为 101 passed）...
"%PY%" -m pytest -q
echo.
echo  [2/2] 前视审计（几分钟，破坏切点前的净值/流水即报错）...
"%PY%" research\no_lookahead.py
goto done

:done
echo.
echo  按任意键返回菜单...
pause >nul
goto menu

:exit
echo.
echo  再见!
exit
