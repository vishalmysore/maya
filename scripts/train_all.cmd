@echo off
rem Train Maya and the no-consistency ablation one after the other (detached-friendly).
cd /d %~dp0..
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
.venv\Scripts\python scripts\train.py --out checkpoints\maya > logs\train_maya.log 2>&1
.venv\Scripts\python scripts\train.py --out checkpoints\maya-noconsist --consistency 0 > logs\train_noconsist.log 2>&1
echo done > logs\train_all.done
