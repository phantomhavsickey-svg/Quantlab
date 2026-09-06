@echo off
chcp 65001 >nul
title QuantLab 量化研究系统
cd /d D:\QuantLab

:: 自动设置Python路径
set "PATH=C:\Users\blessing\AppData\Local\Programs\Python\Python311;%PATH%"

:menu
cls
echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║       QuantLab 量化研究与交易系统 v1.0       ║
echo  ╠══════════════════════════════════════════════╣
echo  ║                                              ║
echo  ║   1. 下载数据  - 更新股票日线数据             ║
echo  ║   2. 计算因子  - 技术面+基本面因子            ║
echo  ║   3. 训练模型  - LightGBM 机器学习            ║
echo  ║   4. 运行回测  - 历史回测+绩效报告            ║
echo  ║   5. 模拟交易  - Walk-Forward 模拟盘          ║
echo  ║   6. 实时行情  - 接新浪实时数据               ║
echo  ║   7. 一键全流程 - 下载→因子→训练→回测        ║
echo  ║   8. 查看报告  - 打开HTML回测报告              ║
echo  ║   9. 退出                                     ║
echo  ║                                              ║
echo  ╚══════════════════════════════════════════════╝
echo.
echo  当前缓存: 已下载数据, 可直接回测
echo.

set /p choice=  请选择 [1-9]:

if "%choice%"=="1" goto download
if "%choice%"=="2" goto factors
if "%choice%"=="3" goto train
if "%choice%"=="4" goto backtest
if "%choice%"=="5" goto paper
if "%choice%"=="6" goto live
if "%choice%"=="7" goto pipeline
if "%choice%"=="8" goto report
if "%choice%"=="9" goto exit
goto menu

:download
echo.
echo  [1/4] 下载股票数据...
echo  默认: 沪深300, 2022-2025
set /p start="  起始日期 (回车=2022-01-01): "
if "%start%"=="" set start=2022-01-01
set /p end="  结束日期 (回车=2025-12-31): "
if "%end%"=="" set end=2025-12-31
python main.py download --universe 000300 --start %start% --end %end%
echo.
echo  按任意键返回...
pause >nul
goto menu

:factors
echo.
echo  [2/4] 计算因子...
python main.py factors --start 2022-01-01 --end 2025-12-31
echo.
echo  按任意键返回...
pause >nul
goto menu

:train
echo.
echo  [3/4] 训练 LightGBM 模型...
python main.py train --train-end 2023-12-31 --test-start 2024-01-01
echo.
echo  按任意键返回...
pause >nul
goto menu

:backtest
echo.
echo  [4/4] 运行回测...
echo.
python main.py backtest --capital 1000000
echo.
echo  报告已生成到 reports\ 目录
echo  按任意键返回...
pause >nul
goto menu

:paper
echo.
echo  [模拟盘] Walk-Forward 逐日仿真交易...
echo.
python main.py paper-trade --capital 1000000
echo.
echo  按任意键返回...
pause >nul
goto menu

:live
echo.
echo  [实时行情] 接新浪财经实时数据...
echo  注意: 仅在交易时段(工作日9:30-15:00)有实时数据
echo  非交易时段请用[演示模式]
echo.
set /p demo="  演示模式? (Y/N, 默认Y): "
if /i "%demo%"=="N" (
    python live/live_trading.py --capital 1000000
) else (
    python live/live_trading.py --demo
)
echo.
echo  按任意键返回...
pause >nul
goto menu

:pipeline
echo.
echo  [一键全流程] 下载 → 因子 → 训练 → 回测
echo  预计耗时: 10-20分钟 (取决于网络)
echo.
python main.py pipeline --universe 000300 --start 2022-01-01 --end 2025-12-31 --capital 1000000
echo.
echo  全流程完成! 按任意键返回...
pause >nul
goto menu

:report
echo.
echo  打开最新回测报告...
for /f "delims=" %%f in ('dir /b /o-d reports\*.html 2^>nul') do (
    start "" "reports\%%f"
    goto menu
)
echo  未找到报告文件, 请先运行回测
echo  按任意键返回...
pause >nul
goto menu

:exit
echo.
echo  再见!
exit
