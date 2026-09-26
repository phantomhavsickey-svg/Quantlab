#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QuantLab 菜单启动器 —— 旧版交互菜单（与 启动.bat 功能重复，已被其取代）。

保留原因：8~11 四项指向 legacy/ 那条 Top-K 等权链路，旧持仓状态还要靠它看。
现行每日流程请直接用 python live/run_daily.py（不带 --qmt/--confirm 只出文件）。

    python legacy/menu.py
"""

import os
import sys
import subprocess
from datetime import datetime

PYTHON = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根
os.chdir(ROOT)


def run(cmd: str):
    """运行命令并等待完成。"""
    print(f"\n{'='*55}")
    print(f">>> {cmd}")
    print(f"{'='*55}\n")
    return subprocess.run(cmd, shell=True, cwd=ROOT)


def check_data() -> str:
    """检查数据状态。"""
    cache_dir = os.path.join(ROOT, "data", "cache", "daily")
    if os.path.exists(cache_dir):
        count = len([f for f in os.listdir(cache_dir) if f.endswith(".parquet")])
        models_dir = os.path.join(ROOT, "models", "saved")
        models = []
        if os.path.exists(models_dir):
            models = [f for f in os.listdir(models_dir) if f.endswith(".txt")]
        return f"缓存: {count}只股票 | 模型: {len(models)}个"
    return "缓存: 无数据，请先下载"


def main():
    while True:
        os.system("cls" if os.name == "nt" else "clear")

        print(f"""
  +======================================================+
  |       QuantLab 量化研究与交易系统 v2.0               |
  +======================================================+
  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}    {check_data():<30} |
  +======================================================+
  |                                                      |
  |   1. 下载数据   更新股票日线 (股票池取 config.yaml)               |
  |   2. 计算因子   技术面+基本面因子                    |
  |   3. 训练模型   LightGBM 机器学习                    |
  |   4. 运行回测   历史回测 + 绩效报告                  |
  |   5. 实时行情   接新浪财经实时数据                    |
  |   6. 一键全流程 下载→因子→训练→回测                  |
  |   7. 查看报告   打开最新HTML报告                      |
  |  --- 持仓管理 ---                                    |
  |   8. 查看持仓   持仓+实时盈亏                        |
  |   9. 自动调仓   按最新信号调仓(先预览再执行)         |
  |  10. 每日盈亏   查看盈亏历史 + 曲线图                 |
  |  11. 每日流程   收盘后: 更新数据+出信号+调仓         |
  |   0. 退出                                            |
  |                                                      |
  +======================================================+
        """)

        choice = input("  请选择 [0-11]: ").strip()

        if choice == "0":
            print("\n  再见!\n")
            break
        elif choice == "1":
            today = datetime.now().strftime("%Y-%m-%d")
            start = input(f"  起始日期 [2023-01-01]: ").strip() or "2023-01-01"
            end = input(f"  结束日期 [{today}]: ").strip() or today
            run(f'"{PYTHON}" main.py download --start {start} --end {end}')
        elif choice == "2":
            run(f'"{PYTHON}" main.py factors --start 2023-01-01 --end '
                f'{datetime.now().strftime("%Y-%m-%d")}')
        elif choice == "3":
            run(f'"{PYTHON}" main.py train')
        elif choice == "4":
            run(f'"{PYTHON}" main.py backtest --capital 1000000')
        elif choice == "5":
            demo = input("  实时行情演示? [Y/n]: ").strip().lower()
            if demo == "n":
                run(f'"{PYTHON}" live/live_trading.py --capital 1000000')
            else:
                run(f'"{PYTHON}" live/live_trading.py --demo')
        elif choice == "6":
            print("\n  一键全流程预计耗时 10-20 分钟...\n")
            run(f'"{PYTHON}" main.py pipeline --capital 1000000')
        elif choice == "7":
            reports = sorted(
                [f for f in os.listdir("reports") if f.endswith(".html")]
                if os.path.exists("reports") else [],
                reverse=True
            )
            if reports:
                path = os.path.join("reports", reports[0])
                os.startfile(os.path.abspath(path))
                print(f"\n  已打开: {path}\n")
            else:
                print("\n  未找到报告，请先运行回测\n")
        elif choice == "8":
            run(f'"{PYTHON}" legacy/portfolio_manager.py status')
        elif choice == "9":
            mode = input("  预览(Y)还是直接执行(N)? [Y/n]: ").strip().lower()
            if mode == "n":
                run(f'"{PYTHON}" legacy/portfolio_manager.py rebalance')
            else:
                run(f'"{PYTHON}" legacy/portfolio_manager.py rebalance --dry-run')
        elif choice == "10":
            print("\n  生成盈亏曲线图...\n")
            run(f'"{PYTHON}" legacy/portfolio_manager.py pnl --plot')
        elif choice == "11":
            run(f'"{PYTHON}" legacy/daily_runner.py')
        else:
            print("\n  无效选择，请重试\n")

        if choice in ("1","2","3","4","6","9","11"):
            input("\n  按回车返回菜单...")


if __name__ == "__main__":
    main()
