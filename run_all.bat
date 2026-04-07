@echo off
REM ============================================================
REM  run_all.bat  - Windows batch script
REM  Runs ALL experiments: 4 datasets x (Partial / UW / FDW / Improved)
REM
REM  Usage:
REM    1. First TRAIN (once per dataset):
REM         run_all.bat train
REM    2. Then EVAL (all methods):
REM         run_all.bat eval
REM    3. Train a single dataset:
REM         run_all.bat train_ecg
REM         run_all.bat train_solar
REM         run_all.bat train_metrla
REM         run_all.bat train_traffic
REM ============================================================

SET PYTHON=D:\Anaconda\envs\pyt\python.exe
SET DEVICE=cuda:0
SET EPOCHS=50
SET BATCH=64
SET RUNS=3
SET SPLIT_RUNS=20
SET SUBSET=15

IF "%1"=="train"        GOTO TRAIN
IF "%1"=="eval"         GOTO EVAL
IF "%1"=="train_ecg"    GOTO TRAIN_ECG
IF "%1"=="train_solar"  GOTO TRAIN_SOLAR
IF "%1"=="train_metrla" GOTO TRAIN_METRLA
IF "%1"=="train_traffic" GOTO TRAIN_TRAFFIC
ECHO Usage: run_all.bat [train^|eval^|train_ecg^|train_solar^|train_metrla^|train_traffic]
EXIT /B 1

REM ─────────────────────────── TRAINING ────────────────────────────────────
:TRAIN
ECHO ===== TRAINING ALL DATASETS =====

:TRAIN_SOLAR
ECHO [1/4] SOLAR  (epochs=%EPOCHS%, runs=%RUNS%)
%PYTHON% run_improved.py --data ./data/SOLAR --model_name mtgnn --device %DEVICE% ^
    --expid 1 --epochs %EPOCHS% --batch_size %BATCH% --runs %RUNS% ^
    --step_size1 2500 --random_node_idx_split_runs 1 ^
    --lower_limit_random_node_selections 100 ^
    --upper_limit_random_node_selections 100
IF NOT "%1"=="train" EXIT /B 0

:TRAIN_ECG
ECHO [2/4] ECG  (epochs=%EPOCHS%, runs=%RUNS%)
%PYTHON% run_improved.py --data ./data/ECG --model_name mtgnn --device %DEVICE% ^
    --expid 1 --epochs %EPOCHS% --batch_size %BATCH% --runs %RUNS% ^
    --step_size1 400 --random_node_idx_split_runs 1 ^
    --lower_limit_random_node_selections 100 ^
    --upper_limit_random_node_selections 100
IF NOT "%1"=="train" EXIT /B 0

:TRAIN_METRLA
ECHO [3/4] METR-LA  (epochs=%EPOCHS%, runs=%RUNS%)
%PYTHON% run_improved.py --data ./data/METR-LA --model_name mtgnn --device %DEVICE% ^
    --expid 1 --epochs %EPOCHS% --batch_size %BATCH% --runs %RUNS% ^
    --step_size1 2500 --random_node_idx_split_runs 1 ^
    --lower_limit_random_node_selections 100 ^
    --upper_limit_random_node_selections 100
IF NOT "%1"=="train" EXIT /B 0

:TRAIN_TRAFFIC
ECHO [4/4] TRAFFIC  (epochs=%EPOCHS%, runs=%RUNS%)
%PYTHON% run_improved.py --data ./data/TRAFFIC --model_name mtgnn --device %DEVICE% ^
    --expid 1 --epochs %EPOCHS% --batch_size 32 --runs %RUNS% ^
    --step_size1 1000 --random_node_idx_split_runs 1 ^
    --lower_limit_random_node_selections 100 ^
    --upper_limit_random_node_selections 100

ECHO Training complete.
EXIT /B 0

REM ─────────────────────────── EVALUATION ──────────────────────────────────
:EVAL
ECHO ===== EVALUATING ALL METHODS ON ALL DATASETS =====

FOR %%D IN (SOLAR ECG METR-LA) DO (
    SET STEP=2500
    IF "%%D"=="ECG"     SET STEP=400
    IF "%%D"=="TRAFFIC" SET STEP=1000

    ECHO.
    ECHO ========== Dataset: %%D ==========

    ECHO --- Baseline: Partial (zero-fill) ---
    %PYTHON% run_improved.py --data ./data/%%D --model_name mtgnn --device %DEVICE% ^
        --expid 1 --epochs 0 --batch_size %BATCH% --runs %RUNS% ^
        --random_node_idx_split_runs %SPLIT_RUNS% ^
        --lower_limit_random_node_selections %SUBSET% ^
        --upper_limit_random_node_selections %SUBSET% ^
        --mask_remaining True

    ECHO --- Baseline: UW (uniform weighting) ---
    %PYTHON% run_improved.py --data ./data/%%D --model_name mtgnn --device %DEVICE% ^
        --expid 1 --epochs 0 --batch_size %BATCH% --runs %RUNS% ^
        --random_node_idx_split_runs %SPLIT_RUNS% ^
        --lower_limit_random_node_selections %SUBSET% ^
        --upper_limit_random_node_selections %SUBSET% ^
        --borrow_from_train_data True --num_neighbors_borrow 5 ^
        --dist_exp_value 0.5 --use_ewp False

    ECHO --- Baseline: FDW (original paper method) ---
    %PYTHON% run_improved.py --data ./data/%%D --model_name mtgnn --device %DEVICE% ^
        --expid 1 --epochs 0 --batch_size %BATCH% --runs %RUNS% ^
        --random_node_idx_split_runs %SPLIT_RUNS% ^
        --lower_limit_random_node_selections %SUBSET% ^
        --upper_limit_random_node_selections %SUBSET% ^
        --borrow_from_train_data True --num_neighbors_borrow 5 ^
        --dist_exp_value 0.5 --neighbor_temp 0.1 --use_ewp True

    ECHO --- Ours: Improved (cosine + adaptive-temp + joint weighting) ---
    %PYTHON% run_improved.py --data ./data/%%D --model_name mtgnn --device %DEVICE% ^
        --expid 1 --epochs 0 --batch_size %BATCH% --runs %RUNS% ^
        --random_node_idx_split_runs %SPLIT_RUNS% ^
        --lower_limit_random_node_selections %SUBSET% ^
        --upper_limit_random_node_selections %SUBSET% ^
        --use_improved_wrapper True --num_neighbors_borrow 5 --joint_alpha 0.3
)

REM TRAFFIC needs smaller batch size
ECHO.
ECHO ========== Dataset: TRAFFIC ==========

ECHO --- Baseline: Partial ---
%PYTHON% run_improved.py --data ./data/TRAFFIC --model_name mtgnn --device %DEVICE% ^
    --expid 1 --epochs 0 --batch_size 32 --runs %RUNS% ^
    --random_node_idx_split_runs %SPLIT_RUNS% ^
    --lower_limit_random_node_selections %SUBSET% ^
    --upper_limit_random_node_selections %SUBSET% ^
    --mask_remaining True

ECHO --- Baseline: UW ---
%PYTHON% run_improved.py --data ./data/TRAFFIC --model_name mtgnn --device %DEVICE% ^
    --expid 1 --epochs 0 --batch_size 32 --runs %RUNS% ^
    --random_node_idx_split_runs %SPLIT_RUNS% ^
    --lower_limit_random_node_selections %SUBSET% ^
    --upper_limit_random_node_selections %SUBSET% ^
    --borrow_from_train_data True --num_neighbors_borrow 5 ^
    --dist_exp_value 0.5 --use_ewp False

ECHO --- Baseline: FDW ---
%PYTHON% run_improved.py --data ./data/TRAFFIC --model_name mtgnn --device %DEVICE% ^
    --expid 1 --epochs 0 --batch_size 32 --runs %RUNS% ^
    --random_node_idx_split_runs %SPLIT_RUNS% ^
    --lower_limit_random_node_selections %SUBSET% ^
    --upper_limit_random_node_selections %SUBSET% ^
    --borrow_from_train_data True --num_neighbors_borrow 5 ^
    --dist_exp_value 0.5 --neighbor_temp 0.1 --use_ewp True

ECHO --- Ours: Improved ---
%PYTHON% run_improved.py --data ./data/TRAFFIC --model_name mtgnn --device %DEVICE% ^
    --expid 1 --epochs 0 --batch_size 32 --runs %RUNS% ^
    --random_node_idx_split_runs %SPLIT_RUNS% ^
    --lower_limit_random_node_selections %SUBSET% ^
    --upper_limit_random_node_selections %SUBSET% ^
    --use_improved_wrapper True --num_neighbors_borrow 5 --joint_alpha 0.3

ECHO All evaluations complete.
EXIT /B 0