@echo off
rem Maya v0.2 candidates: DeBERTa-v3-base and DeBERTa-v3-large zero-shot NLI, fine-tuned on data/train_v2
rem with BoolQ + MNLI anchors; checkpoints chosen on the hand-labeled dev set. Run detached.
cd /d %~dp0..
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
:wait
if not exist logs\baselines_v3.log goto start
findstr /c:"DONE" logs\baselines_v3.log >nul || (timeout /t 30 >nul & goto wait)
:start
.venv\Scripts\python scripts\train.py --base MoritzLaurer/deberta-v3-base-zeroshot-v2.0 --data data\train_v2 --boolq-per-step 2 --mnli-per-step 2 --freeze-embeddings --select dev --eval-every 150 --val-texts 100 --out checkpoints\maya-v2-base > logs\train_v2_base.log 2>&1
.venv\Scripts\python scripts\train.py --base MoritzLaurer/deberta-v3-large-zeroshot-v2.0 --data data\train_v2 --epochs 0.6 --lr 1e-5 --boolq-per-step 2 --mnli-per-step 2 --freeze-embeddings --select dev --eval-every 130 --val-texts 80 --out checkpoints\maya-v2-large > logs\train_v2_large.log 2>&1
echo done > logs\train_v2.done
