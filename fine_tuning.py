import os
import requests
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# 1. 網路狀態偵測
def check_internet_connection():
    try:
        requests.get('https://google.com', timeout=3)
        return True
    except requests.ConnectionError:
        return False

ONLINE = check_internet_connection()
print(f"ONLINE: {ONLINE}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_name = 'distilbert-base-uncased'

# 2. 路徑與模型/分詞器載入
if ONLINE:
    print("Internet detected: Using online mode with local paths")
    train_path = "train.csv"
    test_path = "test.csv"
    output_dir = "./results"
    logging_dir = "./logs"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model_path = model_name
else:
    print("No internet detected: Using offline Kaggle mode")
    train_path = "/kaggle/input/llm-classification-finetuning/train.csv"
    test_path = "/kaggle/input/llm-classification-finetuning/test.csv"
    output_dir = "/kaggle/working/results"
    logging_dir = "/kaggle/working/logs"
    def load_offline_model_kaggle():
        possible_model_paths = [
            "/kaggle/input/distilbert_model/transformers/default/1/distilbert_model"
        ]
        for model_path in possible_model_paths:
            if os.path.exists(model_path):
                required_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
                model_files = ['pytorch_model.bin', 'model.safetensors']
                files_exist = []
                model_file_found = False
                for file in required_files:
                    if os.path.exists(os.path.join(model_path, file)):
                        files_exist.append(file)
                for model_file in model_files:
                    if os.path.exists(os.path.join(model_path, model_file)):
                        files_exist.append(model_file)
                        model_file_found = True
                        break
                if len(files_exist) >= 3 and model_file_found:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3, local_files_only=True)
                    return tokenizer, model, model_path
        raise Exception("Offline model not found in expected Kaggle paths")
    tokenizer, model, model_path = load_offline_model_kaggle()

model.to(device)

# 3. 資料載入
print("Loading training data...")
df = pd.read_csv(train_path)
print(f"Training data shape: {df.shape}")

if os.path.exists(test_path):
    file_size = os.path.getsize(test_path) / (1024**2)
    if file_size > 100:
        df_test = None
        print("Large test file detected, will process in batches")
    else:
        df_test = pd.read_csv(test_path)
        print(f"Test data shape: {df_test.shape}")
else:
    df_test = None

# 4. 標籤處理
def get_label(row):
    if row["winner_model_a"] == 1:
        return 0
    elif row["winner_model_b"] == 1:
        return 1
    else:
        return 2

if "label" not in df.columns:
    df["label"] = df.apply(get_label, axis=1)

# 5. 文本預處理
def create_optimized_input(row):
    prompt = str(row['prompt']).strip()
    response_a = str(row['response_a']).strip()
    response_b = str(row['response_b']).strip()
    len_a, len_b = len(response_a), len(response_b)
    text = f"Compare responses to: {prompt} [SEP] Option A ({len_a} chars): {response_a} [SEP] Option B ({len_b} chars): {response_b}"
    if len(text) > 480:
        max_prompt = min(100, len(prompt))
        max_resp = min(140, len(response_a), len(response_b))
        prompt = prompt[:max_prompt]
        response_a = response_a[:max_resp]
        response_b = response_b[:max_resp]
        text = f"Compare: {prompt} [SEP] A: {response_a} [SEP] B: {response_b}"
    return text

print("Processing text inputs...")
df["text"] = df.apply(create_optimized_input, axis=1)

# 6. 資料集類
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

# 7. 分割資料
validation_size = 0.15
train_indices, val_indices, train_labels, val_labels = train_test_split(
    df.index.tolist(), df["label"].tolist(), test_size=validation_size, random_state=42, stratify=df["label"])
train_texts = df.loc[train_indices, "text"].tolist()
val_texts = df.loc[val_indices, "text"].tolist()

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

# 8. 類別權重
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"Class weights: {class_weights_dict}")

# 9. 創建資料集
train_dataset = LLMDataset(train_texts, train_labels)
val_dataset = LLMDataset(val_texts, val_labels)

# 10. 訓練參數
num_epochs = 4
learning_rate = 1e-5

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="steps",
    eval_steps=600,
    save_strategy="steps",
    save_steps=600,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=num_epochs,
    learning_rate=learning_rate,
    weight_decay=0.03,
    warmup_steps=600,
    logging_dir=logging_dir,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_log_loss",
    greater_is_better=False,
    save_total_limit=2,
    fp16=False,
    dataloader_pin_memory=False,
    report_to="none",
    seed=42,
    gradient_accumulation_steps=1,
    lr_scheduler_type="linear"
)

# 11. 評估指標
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

from transformers import EarlyStoppingCallback

def reinit_model_weights(model):
    if hasattr(model, 'classifier'):
        torch.nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(model.classifier.bias)
    elif hasattr(model, 'score'):
        torch.nn.init.normal_(model.score.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(model.score.bias)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        device = logits.device
        weights = torch.tensor([class_weights_dict[i] for i in range(3)], dtype=torch.float32, device=device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
        loss = loss_fct(logits.view(-1, 3), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

reinit_model_weights(model)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("Starting training...")
trainer.train()

print("Evaluating on validation set...")
val_predictions = trainer.predict(val_dataset)
val_probabilities = torch.nn.functional.softmax(torch.from_numpy(val_predictions.predictions), dim=-1).numpy()
val_probabilities = np.clip(val_probabilities, 1e-7, 1 - 1e-7)
val_final_loss = log_loss(val_labels, val_probabilities)

print(f"Validation Log Loss: {val_final_loss:.6f}")

# 儲存驗證結果
val_results = pd.DataFrame()
val_results["id"] = [f"val_{i}" for i in val_indices]
val_results["gt_winner_model_a"] = (np.array(val_labels) == 0).astype(int)
val_results["gt_winner_model_b"] = (np.array(val_labels) == 1).astype(int)
val_results["gt_winner_tie"] = (np.array(val_labels) == 2).astype(int)
val_results["pred_winner_model_a"] = val_probabilities[:, 0]
val_results["pred_winner_model_b"] = val_probabilities[:, 1]
val_results["pred_winner_tie"] = val_probabilities[:, 2]
row_sums = val_results[["pred_winner_model_a", "pred_winner_model_b", "pred_winner_tie"]].sum(axis=1)
val_results["pred_winner_model_a"] /= row_sums
val_results["pred_winner_model_b"] /= row_sums
val_results["pred_winner_tie"] /= row_sums
val_results.to_csv(os.path.join(output_dir, "validation_results.csv"), index=False)
print("Validation results saved.")

# 測試集推理
print("Starting test inference...")

def process_test_batch(test_path, batch_size=1000):
    total_lines = sum(1 for line in open(test_path, 'r', encoding='utf-8')) - 1
    all_predictions = []
    all_ids = []
    chunk_iter = pd.read_csv(test_path, chunksize=batch_size)
    for i, chunk in enumerate(chunk_iter):
        print(f"Processing batch {i+1}...")
        chunk["text"] = chunk.apply(create_optimized_input, axis=1)
        test_dataset_batch = LLMDataset(chunk["text"].tolist())
        preds = trainer.predict(test_dataset_batch)
        probabilities = torch.nn.functional.softmax(torch.from_numpy(preds.predictions), dim=-1).numpy()
        probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
        all_predictions.append(probabilities)
        all_ids.extend(chunk["id"].tolist())
        del test_dataset_batch
        torch.cuda.empty_cache()
    return all_ids, np.vstack(all_predictions)

try:
    if os.path.exists(test_path):
        file_size = os.path.getsize(test_path) / (1024**2)
        if file_size > 100:
            print(f"Large test file ({file_size:.1f} MB) detected, processing in batches...")
            test_ids, probabilities = process_test_batch(test_path, batch_size=1000)
        else:
            if df_test is None:
                df_test = pd.read_csv(test_path)
            if "text" not in df_test.columns:
                df_test["text"] = df_test.apply(create_optimized_input, axis=1)
            test_dataset = LLMDataset(df_test["text"].tolist())
            preds = trainer.predict(test_dataset)
            probabilities = torch.nn.functional.softmax(torch.from_numpy(preds.predictions), dim=-1).numpy()
            probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
            test_ids = df_test["id"].tolist()
    else:
        if df_test is None:
            print("Test file not found, creating dummy submission...")
            submission = pd.DataFrame({"id": [0, 1, 2]})
            submission["winner_model_a"] = [0.33, 0.33, 0.33]
            submission["winner_model_b"] = [0.33, 0.34, 0.33]
            submission["winner_tie"] = [0.34, 0.33, 0.34]
            submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
            print("Dummy submission saved.")
        else:
            if "text" not in df_test.columns:
                df_test["text"] = df_test.apply(create_optimized_input, axis=1)
            test_dataset = LLMDataset(df_test["text"].tolist())
            preds = trainer.predict(test_dataset)
            probabilities = torch.nn.functional.softmax(torch.from_numpy(preds.predictions), dim=-1).numpy()
            probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
            test_ids = df_test["id"].tolist()
            submission = pd.DataFrame({"id": test_ids})
            submission["winner_model_a"] = probabilities[:, 0]
            submission["winner_model_b"] = probabilities[:, 1]
            submission["winner_tie"] = probabilities[:, 2]
            row_sums = submission[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1)
            submission["winner_model_a"] /= row_sums
            submission["winner_model_b"] /= row_sums
            submission["winner_tie"] /= row_sums
            submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
            print("Submission saved.")
            final_preds = np.argmax(probabilities, axis=-1)
            print(f"Test prediction distribution:")
            print(f"Model A wins: {sum(final_preds == 0)}")
            print(f"Model B wins: {sum(final_preds == 1)}")
            print(f"Ties: {sum(final_preds == 2)}")
except Exception as e:
    print(f"Test inference error: {e}")
    print("Creating dummy submission...")
    submission = pd.DataFrame({"id": [0, 1, 2]})
    submission["winner_model_a"] = [0.33, 0.33, 0.33]
    submission["winner_model_b"] = [0.33, 0.34, 0.33]
    submission["winner_tie"] = [0.34, 0.33, 0.34]
    submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)

print("Final validation results summary:")
val_final_preds = np.argmax(val_probabilities, axis=-1)
print(f"Validation Log Loss: {val_final_loss:.6f}")
print(f"Validation Accuracy: {accuracy_score(val_labels, val_final_preds):.3f}")
print(f"Class Distribution: {np.bincount(val_final_preds, minlength=3)}")
if ONLINE:
    print(f"Model used: {model_name} (online)")
else:
    print(f"Model used: {model_path} (offline)")
print("="*60)
