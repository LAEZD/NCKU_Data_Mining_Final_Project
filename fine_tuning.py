# LLM Classification Fine-Tuning - ENHANCED UNIFIED PIPELINE
# ============================================================
# Version: 2.0 (Enhanced)
# Description:
# An enhanced unified pipeline with advanced data preprocessing capabilities:
# - Data augmentation through response swapping to reduce position bias
# - Dynamic budget allocation for optimal token usage
# - Metadata feature extraction for improved model performance
# - Modular design with clean separation of concerns
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, log_loss
from dual_encoder import DualTowerPairClassifier
from dual_dataset import DualTowerPairDataset
import json, shutil, pathlib, datetime
import safetensors.torch as st  # 在檔頭順便 import

# Import enhanced preprocessing modules
from preprocessing.enhanced_preprocessing import EnhancedPipelineModules, EnhancedTestDataset

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------
# 1. Centralized Configuration
# --------------------------------------------------------------------------
class Config:
    """
    Manages all settings, paths, and hyperparameters in a centralized location.
    Automatically detects the execution environment and sets paths accordingly.
    """
    def __init__(self):        # --- Basic Settings ---
        self.MODEL_NAME = 'distilbert-base-uncased'
        self.QUICK_TEST = False # Set to True for a quick run with a subset of data
        self.QUICK_TEST_SIZE = 2000
        self.RANDOM_STATE = 42
          # --- Enhanced Features Settings ---
        self.APPLY_AUGMENTATION = False   # Enable data augmentation 
        self.EXTRACT_METADATA = True   # Enable metadata feature extraction
        self.METADATA_TYPE = 'core'    # 'core' or 'all'
        self.LR_SCHEDULER_TYPE = "linear"             # "linear" | "cosine" | "cosine_with_restarts"
        
        # --- Training Hyperparameters ---
        self.EPOCHS = 4
        self.LEARNING_RATE = 1.329291894316217e-05
        self.TRAIN_BATCH_SIZE = 8
        self.EVAL_BATCH_SIZE = 8
        self.WEIGHT_DECAY = 0.07114476009343425
        self.WARMUP_RATIO =  0.02232334448672798
        self.LOGGING_STEPS = 50
        self.EVAL_STEPS = 600
        self.SAVE_STEPS = 600
        self.SAVE_TOTAL_LIMIT = 2
        self.LABEL_SMOOTHING = 0.21959818254342153
        self.VALIDATION_SIZE = 0.15
        self.MODEL_ARCH = 'dual'   # 'cross' 或 'dual'

        # ===== 續訓開關 =====
        self.CONTINUE_FROM_INIT_WEIGHTS = False      # ← True ➜ 讀舊權重；False ➜ 從頭
        self.INIT_CHECKPOINT_DIR = "./model_A"   # None 表示從 HF 重新抓

        self.GLOBAL_BEST_DIR   = "./global_best_model"
        self.GLOBAL_METRIC_JSON = "./global_best_model/metrics.json"

        # --- Environment Detection and Path Configuration ---
        self.IS_KAGGLE = os.path.exists('/kaggle/input')
        
        if self.IS_KAGGLE:
            print("INFO: Running in Kaggle environment (Offline Mode)")
            # Kaggle input paths
            self.DATA_DIR = "/kaggle/input/llm-classification-finetuning"
            self.TRAIN_PATH = os.path.join(self.DATA_DIR, "train.csv")
            self.TEST_PATH = os.path.join(self.DATA_DIR, "test.csv")
            
            # Kaggle model input path (assuming model is uploaded as a dataset)
            self.KAGGLE_MODEL_PATH = "/kaggle/input/distilbert_model/transformers/default/1/distilbert_model"
            
            # Kaggle output paths
            self.OUTPUT_DIR = "/kaggle/working/results"
            self.LOGGING_DIR = "/kaggle/working/logs"
            self.SUBMISSION_PATH = "/kaggle/working/submission.csv"
            self.VALIDATION_RESULTS_PATH = "/kaggle/working/validation_results.csv"
        else:
            print("INFO: Running in Local environment (Online Mode)")
            # Local data paths
            self.DATA_DIR = "."
            self.TRAIN_PATH = os.path.join(self.DATA_DIR, "train.csv")
            self.TEST_PATH = os.path.join(self.DATA_DIR, "test.csv")
            
            # Local model path (will download from Hugging Face)
            self.KAGGLE_MODEL_PATH = None # Not used in local mode
            
            # Local output paths
            self.OUTPUT_DIR = "./results"
            self.LOGGING_DIR = "./logs"
            self.SUBMISSION_PATH = os.path.join(self.OUTPUT_DIR, "submission.csv")
            self.VALIDATION_RESULTS_PATH = os.path.join(self.OUTPUT_DIR, "validation_results.csv")
            
            # Ensure local output directories exist
            os.makedirs(self.OUTPUT_DIR, exist_ok=True)
            os.makedirs(self.LOGGING_DIR, exist_ok=True)
            
        # --- Device Configuration ---
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"INFO: Using device: {self.DEVICE}")

# --------------------------------------------------------------------------
# 2. Streamlined Pipeline (Using Enhanced Modules)
# --------------------------------------------------------------------------
class PipelineModules:
    """
    Streamlined pipeline modules that delegate preprocessing to enhanced modules
    while maintaining the core training and inference logic.
    """
    
    @staticmethod
    def load_and_preprocess_data(config: Config):
        """Loads and preprocesses data using enhanced modules."""
        print("\n[Module 1/5] Loading and preprocessing data with enhancements...")
        
        # Delegate to enhanced preprocessing
        df, df_test = EnhancedPipelineModules.load_and_preprocess_data(config)
        
        print(f"  - Training data shape: {df.shape}")
        if df_test is not None:
            print(f"  - Test data shape: {df_test.shape}")
        
        return df, df_test

    @staticmethod
    def load_model_and_tokenizer(config: Config):
        """Loads the tokenizer & model; 支援從 INIT_CHECKPOINT_DIR 續訓 .safetensors。"""
        print("\n[Module 2/5] Loading model and tokenizer...")

        try:
            # ────────────────────────────
            # ❶ 若指定現成權重目錄 (model_A)
            # ────────────────────────────
            if (
                config.CONTINUE_FROM_INIT_WEIGHTS        # ← 新開關
                and config.INIT_CHECKPOINT_DIR
                and os.path.exists(config.INIT_CHECKPOINT_DIR)
            ):
                ckpt_dir = config.INIT_CHECKPOINT_DIR
                print(f"  - 🔄  Loading from existing checkpoint: {ckpt_dir}")

                tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)

                base_name = config.MODEL_NAME
                if config.MODEL_ARCH == "dual":
                    model = DualTowerPairClassifier(base_model=base_name)
                else:  # 'cross'
                    model = AutoModelForSequenceClassification.from_pretrained(
                        base_name, num_labels=3
                    )

                weight_path = os.path.join(ckpt_dir, "model.safetensors")
                if not os.path.exists(weight_path):
                    raise FileNotFoundError(f"Cannot find {weight_path}")

                # 先嘗試用當前 device 讀；失敗 fallback CPU
                try:
                    state_dict = st.load_file(weight_path, device=str(config.DEVICE))
                except Exception as err:
                    print(f"    ↪ CUDA load failed ({err}); falling back to CPU ...")
                    state_dict = st.load_file(weight_path, device="cpu")

                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print(f"    ↳ Weights loaded  (missing={len(missing)}, unexpected={len(unexpected)})")
                model_path_info = ckpt_dir

            # ────────────────────────────
            # ❷ Kaggle 離線模式
            # ────────────────────────────
            elif config.IS_KAGGLE:
                model_path = config.KAGGLE_MODEL_PATH
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Offline model not found at {model_path}")

                print(f"  - Loading model and tokenizer from offline path: {model_path}")
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, num_labels=3, local_files_only=True
                )
                model_path_info = model_path

            # ────────────────────────────
            # ❸ 本地線上下載
            # ────────────────────────────
            else:
                model_path = config.MODEL_NAME
                print(f"  - Loading model and tokenizer from Hugging Face: {model_path}")
                tokenizer = AutoTokenizer.from_pretrained(model_path)

                if config.MODEL_ARCH == "dual":
                    model = DualTowerPairClassifier(base_model=model_path)
                else:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_path, num_labels=3
                    )
                model_path_info = model_path

            # pad token 處理
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model.to(config.DEVICE)
            print("  - Model and tokenizer loaded successfully.")
            return tokenizer, model, model_path_info

        except Exception as e:
            print(f"FATAL ERROR during model loading: {e}")
            raise

    
    @staticmethod
    def create_datasets(df, tokenizer, config: Config):
        print("\n[Module 3/5] Creating datasets...")
        
        if config.MODEL_ARCH == 'dual':
            from sklearn.model_selection import train_test_split
            from sklearn.utils.class_weight import compute_class_weight
            
            train_df, val_df = train_test_split(
                df, test_size=config.VALIDATION_SIZE,
                random_state=config.RANDOM_STATE,
                stratify=df['label']
            )
            train_dataset = DualTowerPairDataset(train_df, tokenizer, max_len=512)
            val_dataset   = DualTowerPairDataset(val_df,   tokenizer, max_len=512)

            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(train_df['label']),
                y=train_df['label']
            )
            class_weights_dict = {i: w for i, w in enumerate(class_weights)}
            # val_labels / indices 給 downstream 評估用
            val_labels  = val_df['label'].tolist()
            val_indices = val_df.index.tolist()
            return train_dataset, val_dataset, val_labels, val_indices, class_weights_dict
        else:
            # ✂️ 原來的 EnhancedPipelineModules 路徑保持不動
            return EnhancedPipelineModules.create_enhanced_datasets(df, tokenizer, config)
        
    @staticmethod
    def setup_trainer(model, train_dataset, val_dataset, class_weights_dict, config: Config):
        """Configures and returns a Trainer instance."""
        print("\n[Module 4/5] Setting up trainer...")
        total_steps  = (len(train_dataset) // config.TRAIN_BATCH_SIZE) * config.EPOCHS
        warmup_steps = int(total_steps * config.WARMUP_RATIO)

        training_args = TrainingArguments(
            output_dir=config.OUTPUT_DIR,
            num_train_epochs=config.EPOCHS,
            learning_rate=config.LEARNING_RATE,
            per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
            weight_decay=config.WEIGHT_DECAY,
            warmup_steps=warmup_steps,
            logging_dir=config.LOGGING_DIR,
            logging_steps=config.LOGGING_STEPS,
            eval_strategy="steps",
            eval_steps=config.EVAL_STEPS,
            save_strategy="steps",
            save_steps=config.SAVE_STEPS,
            load_best_model_at_end=True,
            metric_for_best_model="eval_log_loss",
            greater_is_better=False,
            save_total_limit=config.SAVE_TOTAL_LIMIT,
            fp16=torch.cuda.is_available(), # Enable FP16 if CUDA is available
            dataloader_pin_memory=False,
            report_to="none",
            seed=config.RANDOM_STATE,
            lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        )
        
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            probabilities = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1).numpy()
            preds = np.argmax(logits, axis=-1)
            accuracy = accuracy_score(labels, preds)
            probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
            logloss = log_loss(labels, probabilities)
            pred_dist = np.bincount(preds, minlength=3)
            print(f"Distribution: {pred_dist}, LogLoss: {logloss:.6f}")
            return {"accuracy": accuracy, "log_loss": logloss}
 
        
        # Reinitialize classifier layer weights
        if hasattr(model, 'classifier'):
            print("  - Reinitializing classifier weights...")
            model.classifier.apply(lambda module: module.reset_parameters() if hasattr(module, 'reset_parameters') else None)
        
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                
                weights = torch.tensor(list(class_weights_dict.values()), device=logits.device, dtype=torch.float)
                loss_fct = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=config.LABEL_SMOOTHING)
                loss = loss_fct(logits, labels)
                
                return (loss, outputs) if return_outputs else loss

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
        
        print("  - Trainer setup complete.")
        return trainer
    
    @staticmethod
    def run_inference_and_save(trainer, df_test, tokenizer, config: Config):
        """Runs inference on the test set and saves the submission file."""
        print("\n[Module 5/5] Running inference and creating submission...")
        
        if df_test is None:
            if not os.path.exists(config.TEST_PATH):
                print("  - Test file not found, skipping inference.")                # For Kaggle code competitions, an empty test set in the first stage might require a dummy submission.
                if config.IS_KAGGLE:
                     pd.DataFrame({'id': [], 'winner_model_a': [], 'winner_model_b': [], 'winner_tie': []}).to_csv(config.SUBMISSION_PATH, index=False)
                     print("  - Empty submission.csv created for Kaggle environment.")                
                     return

        def process_test_batch(test_df_chunk, tokenizer):
            """Processes a single batch of test data using enhanced preprocessing."""
            # 為批次數據提取元數據特徵
            if config.EXTRACT_METADATA:
                from preprocessing.metadata_features import MetadataFeatures
                test_df_chunk = MetadataFeatures.add_metadata_features_to_dataframe(
                    test_df_chunk, feature_type=config.METADATA_TYPE
                )
            
            # 使用 EnhancedTestDataset 確保測試階段也使用統一輸入構建策略
            test_dataset = EnhancedTestDataset(
                dataframe=test_df_chunk,
                tokenizer=tokenizer,
                include_metadata=config.EXTRACT_METADATA,
                metadata_type=config.METADATA_TYPE
            )
            preds = trainer.predict(test_dataset)
            probs = torch.nn.functional.softmax(torch.from_numpy(preds.predictions), dim=-1).numpy()
            return probs

        try:
            # Decide whether to process in batches based on file size
            file_size_mb = os.path.getsize(config.TEST_PATH) / (1024**2) if os.path.exists(config.TEST_PATH) else 0
            
            if file_size_mb > 100:
                print(f"  - Processing large test file in batches...")
                all_ids, all_probs = [], []
                chunk_iter = pd.read_csv(config.TEST_PATH, chunksize=1000)
                
                for i, chunk in enumerate(chunk_iter):
                    print(f"    - Processing batch {i+1}...")
                    # 直接使用原始數據框，不需要創建 text 列
                    probs = process_test_batch(chunk, tokenizer)
                    all_ids.extend(chunk["id"].tolist())
                    all_probs.append(probs)
                    torch.cuda.empty_cache()
                
                probabilities = np.vstack(all_probs)
                test_ids = all_ids
            else:
                print("  - Processing test file in a single batch...")
                test_ids = df_test["id"].tolist()
                probabilities = process_test_batch(df_test, tokenizer)
            
            submission = pd.DataFrame({"id": test_ids})
            submission["winner_model_a"] = probabilities[:, 0]
            submission["winner_model_b"] = probabilities[:, 1]
            submission["winner_tie"] = probabilities[:, 2]

            # Normalize probabilities to ensure they sum to 1
            row_sums = submission[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1)
            submission["winner_model_a"] /= row_sums
            submission["winner_model_b"] /= row_sums
            submission["winner_tie"] /= row_sums

            submission.to_csv(config.SUBMISSION_PATH, index=False)
            print(f"  - Submission file saved to: {config.SUBMISSION_PATH}")
            
            # Display prediction distribution
            final_preds = np.argmax(probabilities, axis=-1)
            print("\n  - Test prediction distribution:")
            print(f"    Model A wins: {np.sum(final_preds == 0)}")
            print(f"    Model B wins: {np.sum(final_preds == 1)}")  
            print(f"    Ties:         {np.sum(final_preds == 2)}")

        except Exception as e:
            print(f"ERROR during test inference: {e}")
            print("  - Creating a dummy submission file to prevent failure.")
            dummy_sub = pd.DataFrame({"id": [0], "winner_model_a": [0.33], "winner_model_b": [0.33], "winner_tie": [0.34]})
            dummy_sub.to_csv(config.SUBMISSION_PATH, index=False)


def maybe_save_global_best(val_loss, val_acc, trainer, tokenizer, config, extra_info=None):
    """
    比較目前 val_loss 與歷史最佳；若更好就覆蓋 global_best_model/ 並寫 metrics.json
    """
    best_dir  = pathlib.Path(config.GLOBAL_BEST_DIR)
    best_json = best_dir / "metrics.json"

    # 讀取舊紀錄
    old_loss = None
    if best_json.exists():
        try:
            with open(best_json, "r") as f:
                old_loss = json.load(f)["log_loss"]
        except Exception:
            pass

    if (old_loss is None) or (val_loss < old_loss - 1e-6):
        print(f"🎉  New global best!  LogLoss {val_loss:.6f}" + (f"  < {old_loss:.6f}" if old_loss else ""))
        # 覆蓋模型目錄
        if best_dir.exists():
            shutil.rmtree(best_dir)
        trainer.save_model(best_dir)
        tokenizer.save_pretrained(best_dir)

        # 存新指標
        metrics = {
            "timestamp" : datetime.datetime.now().isoformat(timespec="seconds"),
            "log_loss"  : float(val_loss),
            "accuracy"  : float(val_acc),
            "model_arch": config.MODEL_ARCH,
            "hyperparams": {
                "learning_rate" : config.LEARNING_RATE,
                "weight_decay"  : config.WEIGHT_DECAY,
                "label_smoothing": config.LABEL_SMOOTHING,
                "epochs"        : config.EPOCHS,
                "batch_size"    : config.TRAIN_BATCH_SIZE,
                "max_len"       : 512 if config.MODEL_ARCH=="dual" else 512,
                "warmup_ratio"  : getattr(config, "WARMUP_RATIO", None),
                "lr_scheduler"  : trainer.args.lr_scheduler_type,
            }
        }
        if extra_info:
            metrics.update(extra_info)
        best_dir.mkdir(parents=True, exist_ok=True)
        with open(best_json, "w") as f:
            json.dump(metrics, f, indent=2)

# --------------------------------------------------------------------------
# 3. Main Execution Flow
# --------------------------------------------------------------------------
def main():
    """
    Executes the complete machine learning pipeline.
    """
    # === Step 1: Initialize Configuration ===
    config = Config()

    # === Step 2: Load and Preprocess Data ===
    df, df_test = PipelineModules.load_and_preprocess_data(config)

    # === Step 3: Load Model and Tokenizer ===
    tokenizer, model, model_path_info = PipelineModules.load_model_and_tokenizer(config)

    # === Step 4: Create Datasets ===
    train_dataset, val_dataset, val_labels, val_indices, class_weights = PipelineModules.create_datasets(
        df, tokenizer, config
    )

    # === Step 5: Setup and Run Training ===
    trainer = PipelineModules.setup_trainer(model, train_dataset, val_dataset, class_weights, config)
    
    print("\nStarting model training...")
    trainer.train()
    print("Training complete.")

    # === Step 6: Final Validation and Evaluation ===
    print("\nRunning final validation...")
    val_preds = trainer.predict(val_dataset)
    val_probs = torch.nn.functional.softmax(torch.from_numpy(val_preds.predictions), dim=-1).numpy()
    val_probabilities = torch.nn.functional.softmax(torch.from_numpy(val_preds.predictions), dim=-1).numpy()
    val_probabilities = np.clip(val_probabilities, 1e-7, 1 - 1e-7)
    val_final_loss = log_loss(val_labels, val_probabilities)
    val_final_acc = accuracy_score(val_labels, np.argmax(val_probs, axis=1))

    print(f"\n{'='*60}")
    print(f"FINAL VALIDATION RESULTS")
    print(f"  - Model Used: {model_path_info}")
    print(f"  - Log Loss:   {val_final_loss:.6f}")
    print(f"  - Accuracy:   {val_final_acc:.4f}")
    print(f"{'='*60}\n")

    maybe_save_global_best(val_final_loss, val_final_acc, trainer, tokenizer, config)

    # Save validation results for analysis
    val_results = pd.DataFrame({
        "id": df.loc[val_indices, "id"],
        "gt_label": val_labels,
        "pred_prob_a": val_probs[:, 0],
        "pred_prob_b": val_probs[:, 1],
        "pred_prob_tie": val_probs[:, 2]
    })
    val_results.to_csv(config.VALIDATION_RESULTS_PATH, index=False)
    print(f"  - Validation results saved to: {config.VALIDATION_RESULTS_PATH}")

    # === Step 7: Test Set Inference and Submission ===
    PipelineModules.run_inference_and_save(trainer, df_test, tokenizer, config)

    print("\nPipeline finished successfully!")

# --------------------------------------------------------------------------
# 4. 讓 Optuna 可以單次呼叫的包裝器
#    －－放在 fine_tuning.py 最後面即可－－
# --------------------------------------------------------------------------
def run_once_with_config(config):
    """
    執行一次完整 pipeline，並回傳 (logloss, accuracy) 供 Optuna 評估。
    參數
    ----
    config : Config
        已填好超參數的設定物件（同 bayes_opt 內自行修改後的 cfg）
    回傳
    ----
    (logloss : float, accuracy : float)
    """
    # === Step 1: Load & preprocess data ===
    df, df_test = PipelineModules.load_and_preprocess_data(config)

    # ▶ 新增：如果這次是 Optuna trial，動態改存放資料夾
    if hasattr(config, "OPTUNA_TRIAL_ID"):
        trial_id = config.OPTUNA_TRIAL_ID
        config.GLOBAL_BEST_DIR = f"./global_best_model_{trial_id}"
        config.GLOBAL_METRIC_JSON = f"{config.GLOBAL_BEST_DIR}/metrics.json"

    # === Step 2: Load model & tokenizer ===
    tokenizer, model, _ = PipelineModules.load_model_and_tokenizer(config)

    # === Step 3: Create datasets ===
    train_ds, val_ds, val_labels, _, class_wts = PipelineModules.create_datasets(
        df, tokenizer, config
    )

    # === Step 4: Trainer ===
    trainer = PipelineModules.setup_trainer(
        model, train_ds, val_ds, class_wts, config
    )

    # === Step 5: Train ===
    trainer.train()

    # === Step 6: Eval on validation set ===
    val_pred = trainer.predict(val_ds)
    val_probs = torch.nn.functional.softmax(
        torch.from_numpy(val_pred.predictions), dim=-1
    ).numpy()
    val_probs = np.clip(val_probs, 1e-7, 1 - 1e-7)

    logloss = log_loss(val_labels, val_probs)
    acc     = accuracy_score(val_labels, np.argmax(val_probs, axis=1))

    # 讓外部（Optuna）可以拿到最佳 checkpoint 路徑
    trainer.save_model(config.OUTPUT_DIR)              # 確保有存
    run_once_with_config.best_ckpt_dir = trainer.state.best_model_checkpoint

    maybe_save_global_best(logloss, acc, trainer, tokenizer, config,
                       extra_info={"trial_id": getattr(config, "OPTUNA_TRIAL_ID", None)})

    return logloss, acc

if __name__ == "__main__":
    main()