@echo off
rem Maya v0.2 large candidate: smaller batches than the base run (4 texts per step) to stay within memory.
cd /d %~dp0..
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
.venv\Scripts\python scripts\train.py --base MoritzLaurer/deberta-v3-large-zeroshot-v2.0 --data data\train_v2 --epochs 0.35 --lr 1e-5 --texts-per-step 4 --boolq-per-step 1 --mnli-per-step 1 --max-length 128 --freeze-embeddings --select dev --eval-every 117 --val-texts 60 --out checkpoints\maya-v2-large > logs\train_v2_large.log 2>&1
echo done > logs\train_v2_large.done
