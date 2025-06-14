# LLM Classification Fine-Tuning - UNIFIED & MODULAR PIPELINE
# ===============================================================
# Version: 1.0
# Description:
# A unified and modular pipeline for seamless execution in both local (online) 
# and Kaggle (fully offline) environments.
# - Auto-detects the environment (Local vs. Kaggle).
# - Centralized configuration management via a Config class.
# - Modular workflow encapsulated in a PipelineModules class for maintainability.
# ===============================================================

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils.class_weight import compute_class_weight

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
        # --- Basic Settings ---
        self.MODEL_NAME = 'distilbert-base-uncased'
        self.QUICK_TEST = True  # Set to True for a quick run with a subset of data
        self.QUICK_TEST_SIZE = 2000
        self.RANDOM_STATE = 42
        
        # --- Training Hyperparameters ---
        self.EPOCHS = 4
        self.LEARNING_RATE = 1e-5
        self.TRAIN_BATCH_SIZE = 8
        self.EVAL_BATCH_SIZE = 8
        self.WEIGHT_DECAY = 0.03
        self.WARMUP_STEPS = 600
        self.LOGGING_STEPS = 50
        self.EVAL_STEPS = 600
        self.SAVE_STEPS = 600
        self.SAVE_TOTAL_LIMIT = 2
        self.LABEL_SMOOTHING = 0.1
        self.VALIDATION_SIZE = 0.15

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
# 2. Modular Pipeline
# --------------------------------------------------------------------------
class PipelineModules:
    """
    Encapsulates the core functional modules of the pipeline.
    All methods are static for direct invocation.
    """
    
    @staticmethod
    def load_and_preprocess_data(config: Config):
        """Loads training and test data, then applies preprocessing steps."""
        print("\n[Module 1/5] Loading and preprocessing data...")
        
        # Load training data
        try:
            df = pd.read_csv(config.TRAIN_PATH)
            print(f"  - Training data loaded. Shape: {df.shape}")
        except FileNotFoundError:
            print(f"ERROR: Training file not found at {config.TRAIN_PATH}")
            raise
            
        # Quick test mode
        if config.QUICK_TEST and len(df) > config.QUICK_TEST_SIZE:
            df = df.sample(n=config.QUICK_TEST_SIZE, random_state=config.RANDOM_STATE).reset_index(drop=True)
            print(f"  - Quick test mode enabled. Sampled to {df.shape}")

        # Create labels
        def get_label(row):
            if row["winner_model_a"] == 1: return 0
            if row["winner_model_b"] == 1: return 1
            return 2
        df["label"] = df.apply(get_label, axis=1)
        
        # Text preprocessing
        def create_optimized_input(row):
            prompt = str(row['prompt']).strip()
            response_a = str(row['response_a']).strip()
            response_b = str(row['response_b']).strip()
            
            len_a, len_b = len(response_a), len(response_b)
            text = f"Compare responses to: {prompt} [SEP] Option A ({len_a} chars): {response_a} [SEP] Option B ({len_b} chars): {response_b}"
            
            if len(text) > 480:
                max_prompt = min(100, len(prompt))
                max_resp = min(140, len(response_a), len(response_b))
                text = f"Compare: {prompt[:max_prompt]} [SEP] A: {response_a[:max_resp]} [SEP] B: {response_b[:max_resp]}"
            return text

        df["text"] = df.apply(create_optimized_input, axis=1)
        print("  - Text inputs and labels have been processed.")
        
        # Load test data
        df_test = None
        if os.path.exists(config.TEST_PATH):
            file_size_mb = os.path.getsize(config.TEST_PATH) / (1024**2)
            if file_size_mb > 100:
                print(f"  - Large test file ({file_size_mb:.1f} MB) detected. Will process in batches.")
            else:
                df_test = pd.read_csv(config.TEST_PATH)
                df_test["text"] = df_test.apply(create_optimized_input, axis=1)
                print(f"  - Test data loaded and processed. Shape: {df_test.shape}")
        else:
            print("  - WARNING: Test file not found. Inference will be skipped.")
            
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
        """Creates training and validation datasets."""
        print("\n[Module 3/5] Creating datasets...")
        
        train_indices, val_indices, train_labels, val_labels = train_test_split(
            df.index, df["label"].tolist(), 
            test_size=config.VALIDATION_SIZE, 
            random_state=config.RANDOM_STATE, 
            stratify=df["label"]
        )
        
        train_texts = df.loc[train_indices, "text"].tolist()
        val_texts = df.loc[val_indices, "text"].tolist()
        
        print(f"  - Training samples: {len(train_texts)}")
        print(f"  - Validation samples: {len(val_texts)}")

        class LLMDataset(Dataset):
            def __init__(self, texts, labels=None):
                self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
                self.labels = labels
            def __len__(self):
                return len(self.encodings["input_ids"])
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                if self.labels is not None:
                    item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item

        train_dataset = LLMDataset(train_texts, train_labels)
        val_dataset = LLMDataset(val_texts, val_labels)
        
        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
        print(f"  - Class weights calculated: {class_weights_dict}")

        return train_dataset, val_dataset, val_labels, val_indices, class_weights_dict

    @staticmethod
    def setup_trainer(model, train_dataset, val_dataset, class_weights_dict, config: Config):
        """Configures and returns a Trainer instance."""
        print("\n[Module 4/5] Setting up trainer...")

        training_args = TrainingArguments(
            output_dir=config.OUTPUT_DIR,
            num_train_epochs=config.EPOCHS,
            learning_rate=config.LEARNING_RATE,
            per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
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
            fp16=torch.cuda.is_available(), # Enable FP16 if CUDA is available
            dataloader_pin_memory=False,
            report_to="none",
            seed=config.RANDOM_STATE,
            lr_scheduler_type="linear",
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
                logits = outputs.logits
                
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
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        print("  - Trainer setup complete.")
        return trainer

    @staticmethod
    def run_inference_and_save(trainer, df_test, tokenizer, config: Config):
        """Runs inference on the test set and saves the submission file."""
        print("\n[Module 5/5] Running inference and creating submission...")
        
        if df_test is None:
            if not os.path.exists(config.TEST_PATH):
                print("  - Test file not found, skipping inference.")
                # For Kaggle code competitions, an empty test set in the first stage might require a dummy submission.
                if config.IS_KAGGLE:
                     pd.DataFrame({'id': [], 'winner_model_a': [], 'winner_model_b': [], 'winner_tie': []}).to_csv(config.SUBMISSION_PATH, index=False)
                     print("  - Empty submission.csv created for Kaggle environment.")
                return

        def process_test_batch(test_df_chunk, tokenizer):
            """Processes a single batch of test data."""
            class TestDataset(Dataset):
                def __init__(self, texts):
                    self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
                def __len__(self): return len(self.encodings["input_ids"])
                def __getitem__(self, idx): return {key: val[idx] for key, val in self.encodings.items()}
            
            test_dataset = TestDataset(test_df_chunk["text"].tolist())
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
                    chunk["text"] = chunk.apply(lambda row: PipelineModules.load_and_preprocess_data.create_optimized_input(row), axis=1)
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

if __name__ == "__main__":
    main()