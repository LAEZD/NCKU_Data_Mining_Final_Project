"""
bayes_opt.py
------------------------------------------
用 Optuna (Bayesian / TPE) 搜索 lr、weight_decay、epochs、label_smoothing
執行 10 trials，並把每個 trial 的超參數 + Log-loss + Accuracy
寫到 bayes_search_history.csv
"""
import optuna, fine_tuning
import torch, numpy as np, pandas as pd, random
import json, shutil, pathlib

# －－－ (可選) 固定亂數，保證可重現－－－
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def objective(trial):
    cfg = fine_tuning.Config()
    cfg.OPTUNA_TRIAL_ID = trial.number 

    #── 搜參 ──
    cfg.LEARNING_RATE   = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    cfg.WEIGHT_DECAY    = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    cfg.LABEL_SMOOTHING = trial.suggest_float("label_smoothing", 0.0, 0.2)
    cfg.SAVE_BEST_MODEL = True
    # ⬇ 新增學習率排程與 warm-up
    cfg.LR_SCHEDULER_TYPE = trial.suggest_categorical(
        "lr_sched", ["linear", "cosine", "cosine_with_restarts"]
    )
    warm_ratio = trial.suggest_float("warmup_ratio", 0.02, 0.10)
    cfg.WARMUP_RATIO = warm_ratio 

    #── 每個 trial 獨立輸出資料夾 ──
    cfg.OUTPUT_DIR  = f"./results/trial_{trial.number}"
    cfg.LOGGING_DIR = f"./logs/trial_{trial.number}"

    # 保證資料夾存在
    import os, pathlib; pathlib.Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(cfg.LOGGING_DIR).mkdir(parents=True, exist_ok=True)

    #── Quick test (如需) ──
    cfg.QUICK_TEST = True
    cfg.QUICK_TEST_SIZE = 5000

    #── 執行一次訓練 ──
    logloss, acc = fine_tuning.run_once_with_config(cfg)

    #── 把該 trial 最佳 checkpoint 路徑記起來 ──
    from pathlib import Path
    trial.set_user_attr("best_ckpt_dir",
                        Path(cfg.OUTPUT_DIR).joinpath("checkpoint-"+cfg.SAVE_STEPS.__str__()).parent)  # 或 trainer.state.best_model_checkpoint

    trial.set_user_attr("val_accuracy", acc)
    return logloss

if __name__ == "__main__":
    study = optuna.create_study(
        study_name   ="llm_hpo",
        direction    ="minimize",
        sampler      = optuna.samplers.TPESampler(seed=SEED)
    )

    # 這裡 n_trials=10　=> 只跑 10 次。n_jobs=1：單機單 GPU。
    study.optimize(objective, n_trials=20, n_jobs=1)

    best_trial     = study.best_trial
    best_params    = best_trial.params
    best_value     = study.best_value          # 最低 log-loss
    best_ckpt_dir  = getattr(fine_tuning.run_once_with_config, "best_ckpt_dir", None)

    # 1) 超參數 ➜ JSON
    with open("best_hpo_params.json", "w") as f:
        json.dump({
            "log_loss"     : best_value,
            "trial_number" : best_trial.number,
            **best_params
        }, f, indent=2)

    print("\n✓ 已寫出 best_hpo_params.json")

    # 2) 最佳模型 ➜ 複製到 ./best_hpo_model
    if best_ckpt_dir and pathlib.Path(best_ckpt_dir).exists():
        dst = pathlib.Path("best_hpo_model")
        shutil.rmtree(dst, ignore_errors=True)          # 覆寫舊的
        shutil.copytree(best_ckpt_dir, dst)
        print(f"✓ 已將最佳 checkpoint 內容複製到 {dst}/")
    else:
        print("⚠️  沒找到 best_model_checkpoint，請檢查 run_once_with_config 是否正確設定。")

    # 3) 仍保留完整試驗紀錄
    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    df.rename(columns={"value": "val_logloss"}, inplace=True)
    df.to_csv("bayes_search_history.csv", index=False)
    print("✓ bayes_search_history.csv 更新完成")
