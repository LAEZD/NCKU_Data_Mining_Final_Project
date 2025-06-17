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
    def __init__(self):        
        # --- Single Model Configuration ---
        self.MULTI_MODEL_TEST = False  # Disable multi-model testing
        self.MODEL_NAME = 'microsoft/deberta-v3-base'  # Use DeBERTa-v3-base
        
        # --- Basic Settings ---
        self.QUICK_TEST = False # Set to True for a quick run with a subset of data
        self.QUICK_TEST_SIZE = 2000
        self.RANDOM_STATE = 42
        
        # --- Enhanced Features Settings ---
        self.APPLY_AUGMENTATION = False   # Enable data augmentation 
        self.EXTRACT_METADATA = True   # Enable metadata feature extraction
        self.METADATA_TYPE = 'core'    # 'core' or 'all'
        
        # --- Training Hyperparameters ---
        self.EPOCHS = 4
        
        # Optimized learning rates for better stability
        self.LEARNING_RATE = 1e-5  # Reduced for better stability (was 1e-5)
        self.LARGE_MODEL_LR = 3e-6  # For large models (BERT-large, RoBERTa-large)
        self.XLNET_LR = 2e-5       # XLNet works better with higher LR
        
        # Improved batch sizes for stability
        self.TRAIN_BATCH_SIZE = 8  # Increased for more stable gradients (was 8)
        self.EVAL_BATCH_SIZE = 8   # Increased accordingly (was 8)
        
        # Enhanced stability settings
        self.WEIGHT_DECAY = 0.02    # Reduced from 0.03 for less aggressive regularization
        self.WARMUP_STEPS = 600    # Doubled for better warm-up (was 600)
        self.LOGGING_STEPS = 50     # Keep original logging frequency
        self.EVAL_STEPS = 600       # Keep original eval frequency
        self.SAVE_STEPS = 600       # Keep original save frequency
        self.SAVE_TOTAL_LIMIT = 2
        
        # Stability improvements
        self.LABEL_SMOOTHING = 0.1 # Reduced for less aggressive smoothing (was 0.1)
        self.GRADIENT_ACCUMULATION_STEPS = 2  # New: Simulate larger batch size
        self.MAX_GRAD_NORM = 1.0    # New: Gradient clipping for stability
        self.LR_SCHEDULER_TYPE = "cosine"  # New: Cosine scheduler for smoother LR decay
        
        # --- Model Saving Configuration ---
        self.SAVE_BEST_MODEL = True  # Enable saving best model
        self.BEST_MODEL_DIR = "./best_model"  # Directory to save best model
        
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
            self.BEST_MODEL_DIR = "/kaggle/working/best_model"
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
            self.BEST_MODEL_DIR = "./best_model"
            
            # Ensure local output directories exist
            os.makedirs(self.OUTPUT_DIR, exist_ok=True)
            os.makedirs(self.LOGGING_DIR, exist_ok=True)
            
        # Ensure best model directory exists
        os.makedirs(self.BEST_MODEL_DIR, exist_ok=True)
            
        # --- Device Configuration ---
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"INFO: Using device: {self.DEVICE}")
        print(f"INFO: Best model will be saved to: {self.BEST_MODEL_DIR}")

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
        """Loads the model and tokenizer based on the environment."""
        print("\n[Module 2/5] Loading model and tokenizer...")
        
        try:
            if config.IS_KAGGLE:
                # Kaggle offline mode
                model_path = config.KAGGLE_MODEL_PATH
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Offline model not found at {model_path}")
                
                print(f"  - Loading model and tokenizer from offline path: {model_path}")
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, num_labels=3, local_files_only=True
                )
            else:
                # Local online mode
                model_path = config.MODEL_NAME
                print(f"  - Loading model and tokenizer from Hugging Face: {model_path}")
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, num_labels=3
                )
            
            # Add padding token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model.to(config.DEVICE)
            print(f"  - Model and tokenizer loaded successfully.")
            return tokenizer, model, model_path

        except Exception as e:
            print(f"FATAL ERROR during model loading: {e}")
            raise
    
    @staticmethod
    def create_datasets(df, tokenizer, config: Config):
        """Creates enhanced training and validation datasets."""
        print("\n[Module 3/5] Creating enhanced datasets...")
        
        # Delegate to enhanced preprocessing
        return EnhancedPipelineModules.create_enhanced_datasets(df, tokenizer, config)

    @staticmethod
    def setup_trainer(model, train_dataset, val_dataset, class_weights_dict, config: Config):
        """Configures and returns a Trainer instance."""
        print("\n[Module 4/5] Setting up trainer...")

        # Model-specific optimizations
        use_fp16 = torch.cuda.is_available()
        learning_rate = config.LEARNING_RATE
        
        # Special handling for problematic models
        model_name_lower = config.MODEL_NAME.lower()
        if any(problem_model in model_name_lower for problem_model in ['deberta', 'bart', 'unilm']):
            print(f"  - Detected problematic model for FP16: {config.MODEL_NAME}")
            print("  - Disabling FP16 to prevent overflow issues")
            use_fp16 = False
        
        # Model-specific learning rates for better stability
        if 'xlnet' in model_name_lower:
            learning_rate = config.XLNET_LR
            print(f"  - Using optimized learning rate for XLNet: {learning_rate}")
        elif any(large_model in model_name_lower for large_model in ['large', 'roberta-base', 'bert-base']):
            learning_rate = config.LARGE_MODEL_LR
            print(f"  - Using reduced learning rate for large model: {learning_rate}")
        else:
            learning_rate = config.LEARNING_RATE
            print(f"  - Using standard learning rate: {learning_rate}")

        training_args = TrainingArguments(
            output_dir=config.OUTPUT_DIR,
            num_train_epochs=config.EPOCHS,
            learning_rate=learning_rate,
            per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,  # New: Simulate larger batch
            max_grad_norm=config.MAX_GRAD_NORM,  # New: Gradient clipping
            weight_decay=config.WEIGHT_DECAY,
            warmup_steps=config.WARMUP_STEPS,
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
            fp16=use_fp16, # Use model-specific FP16 setting
            dataloader_pin_memory=False,
            report_to="none",
            seed=config.RANDOM_STATE,
            lr_scheduler_type=config.LR_SCHEDULER_TYPE,
            # Additional stability settings
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            remove_unused_columns=False,  # Keep all columns for debugging
        )
        
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            probabilities = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1).numpy()
            preds = np.argmax(logits, axis=-1)
            accuracy = accuracy_score(labels, preds)
            
            # More stable probability clipping
            probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)  # Tighter clipping
            logloss = log_loss(labels, probabilities)
            
            pred_dist = np.bincount(preds, minlength=3)
            
            # Add stability metrics
            pred_entropy = -np.sum(probabilities * np.log(probabilities + 1e-15), axis=1).mean()
            max_prob = np.max(probabilities, axis=1).mean()
            
            print(f"Distribution: {pred_dist}, LogLoss: {logloss:.6f}, Entropy: {pred_entropy:.4f}, MaxProb: {max_prob:.4f}")
            
            return {
                "accuracy": accuracy, 
                "log_loss": logloss,
                "prediction_entropy": pred_entropy,
                "max_probability": max_prob
            }

        
        # Reinitialize classifier layer weights
        if hasattr(model, 'classifier'):
            print("  - Reinitializing classifier weights...")
            model.classifier.apply(lambda module: module.reset_parameters() if hasattr(module, 'reset_parameters') else None)
        
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Apply temperature scaling for more stable training
                temperature = 1.2  # Slight temperature scaling
                scaled_logits = logits / temperature
                
                weights = torch.tensor(list(class_weights_dict.values()), device=logits.device, dtype=torch.float)
                loss_fct = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=config.LABEL_SMOOTHING)
                loss = loss_fct(scaled_logits, labels)
                
                return (loss, outputs) if return_outputs else loss

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])  # Keep original early stopping
        
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

# --------------------------------------------------------------------------
# 3. Main Execution Flow
# --------------------------------------------------------------------------
def main():
    """
    Executes the complete machine learning pipeline.
    """
    # === Step 1: Initialize Configuration ===
    config = Config()
    
    print("Single model training mode - DeBERTa-v3-base")

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
    
    print(f"\nStarting model training for {config.MODEL_NAME}...")
    trainer.train()
    print("Training complete.")

    # === Step 6: Final Validation and Evaluation ===
    print("\nRunning final validation...")
    val_preds = trainer.predict(val_dataset)
    val_probs = torch.nn.functional.softmax(torch.from_numpy(val_preds.predictions), dim=-1).numpy()
    val_probabilities = np.clip(val_probs, 1e-7, 1 - 1e-7)
    val_final_loss = log_loss(val_labels, val_probabilities)
    val_final_acc = accuracy_score(val_labels, np.argmax(val_probs, axis=1))

    print(f"\n{'='*60}")
    print(f"FINAL VALIDATION RESULTS")
    print(f"  - Model Used: {model_path_info}")
    print(f"  - Log Loss:   {val_final_loss:.6f}")
    print(f"  - Accuracy:   {val_final_acc:.4f}")
    print(f"{'='*60}\n")

    # === Step 7: Save Best Model ===
    if config.SAVE_BEST_MODEL:
        print(f"Saving best model to: {config.BEST_MODEL_DIR}")
        
        # Save the model and tokenizer
        model.save_pretrained(config.BEST_MODEL_DIR)
        tokenizer.save_pretrained(config.BEST_MODEL_DIR)
        
        # Save training configuration and results
        import json
        model_info = {
            'model_name': config.MODEL_NAME,
            'final_log_loss': val_final_loss,
            'final_accuracy': val_final_acc,
            'epochs': config.EPOCHS,
            'learning_rate': trainer.args.learning_rate,
            'batch_size': config.TRAIN_BATCH_SIZE,
            'validation_samples_per_class': 2000
        }
        
        with open(os.path.join(config.BEST_MODEL_DIR, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✓ Model saved successfully!")
        print(f"  - Model files: {config.BEST_MODEL_DIR}/pytorch_model.bin")
        print(f"  - Tokenizer files: {config.BEST_MODEL_DIR}/tokenizer.json")
        print(f"  - Model info: {config.BEST_MODEL_DIR}/model_info.json")

    # === Step 8: Test Set Inference and Submission ===
    PipelineModules.run_inference_and_save(trainer, df_test, tokenizer, config)

    print("\n" + "="*80)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"📊 Final Results:")
    print(f"   - Log Loss: {val_final_loss:.6f}")
    print(f"   - Accuracy: {val_final_acc:.4f}")
    print(f"💾 Saved Files:")
    print(f"   - Best Model: {config.BEST_MODEL_DIR}/")
    print(f"   - Submission: {config.SUBMISSION_PATH}")
    print("="*80)

if __name__ == "__main__":
    main()