# -*- coding: utf-8 -*-
"""把 backtest 读的两份 parquet 重定向到 backup_20260930，其余链路一字不改。

    python research/replay_backup_bt.py
"""
import os
import runpy
import sys
import webbrowser

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

BK = "data/cache/backup_20260930"
MAP = {
    "data/cache/factor_panel.parquet": BK + "/factor_panel.parquet",
    "data/cache/predictions.parquet": BK + "/predictions.parquet",
}
_orig = pd.read_parquet


def _patched(path, *args, **kwargs):
    return _orig(MAP.get(str(path), path), *args, **kwargs)


pd.read_parquet = _patched
webbrowser.open = lambda *a, **k: False

print("重定向: " + " | ".join("%s -> %s" % (k, v) for k, v in MAP.items()))
sys.argv = ["main.py", "backtest"]
runpy.run_path("main.py", run_name="__main__")
