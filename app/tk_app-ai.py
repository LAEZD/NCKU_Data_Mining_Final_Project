# tk_app-ai.py  ─────────────────────────────────────────────
# tk_app-ai.py ── Gemini-Flash × Dual-Encoder　Response-Chooser ──
"""
‣ 請先：pip install google-generativeai transformers safetensors torch
‣ 放置：
      model.safetensors
      tokenizer.json / vocab.txt / config.json
‣ 執行：python tk_app.py
"""
import os, threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from google.generativeai import GenerativeModel, configure
from dual_encoder import DualTowerPairClassifier   # ⬅ 你專案裡的類別

# ────────── 路徑 & 裝置 ──────────
ROOT          = Path(__file__).parent
WEIGHT_FILE   = ROOT / "model.safetensors"
TOKENIZER_DIR = ROOT
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────── 載入 tokenizer / Dual-Encoder (一次) ──────────
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

model = DualTowerPairClassifier(base_model="distilbert-base-uncased", include_prompt=True)   # 若需關閉prompt請改為False
state = load_file(WEIGHT_FILE, device="cpu")     # ← 固定載到 CPU
model.load_state_dict(state, strict=False)
model.to(DEVICE)                                # 再搬去 GPU/CPU 執行裝置
model.eval()

@torch.inference_mode()
def predict(prompt: str, resp_a: str, resp_b: str) -> int:
    """
    回傳：0 ➜ A 較佳、1 ➜ B 較佳、2 ➜ Tie
    ─ 每段各自 token ize，可避免維度出錯，也較好裁切長度。"""
    tok_p = tokenizer(prompt,  truncation=True, padding="max_length",
                      max_length=512, return_tensors="pt").to(DEVICE)
    tok_a = tokenizer(resp_a,  truncation=True, padding="max_length",
                      max_length=512, return_tensors="pt").to(DEVICE)
    tok_b = tokenizer(resp_b,  truncation=True, padding="max_length",
                      max_length=512, return_tensors="pt").to(DEVICE)

    out = model(
        p_input_ids      = tok_p["input_ids"],
        p_attention_mask = tok_p["attention_mask"],
        a_input_ids      = tok_a["input_ids"],
        a_attention_mask = tok_a["attention_mask"],
        b_input_ids      = tok_b["input_ids"],
        b_attention_mask = tok_b["attention_mask"],
    )
    return out["logits"].argmax(-1).item()

# ────────── UI 佈局 ──────────
root = tk.Tk()
root.title("Gemini ↔ Dual-Encoder  Response Chooser")
root.configure(bg="#1e1e1e")

style = ttk.Style(root)
style.theme_use("clam")
style.configure(".", background="#1e1e1e", foreground="#d4d4d4")
style.configure("TButton", background="#0e639c", foreground="white")
style.map("TButton", background=[("active", "#1177bb")])

def add_lbl(text, r, c, cs=1):
    ttk.Label(root, text=text).grid(row=r, column=c, columnspan=cs,
                                    sticky="w", padx=4, pady=2)

add_lbl("Gemini API-Key:", 0,0)
api_entry = ttk.Entry(root, width=50, show="*"); api_entry.grid(row=0, column=1, columnspan=3, sticky="we", padx=4)

add_lbl("Prompt:", 1,0)
prompt_box = scrolledtext.ScrolledText(root, height=4, width=80, bg="#252526", fg="#d4d4d4")
prompt_box.grid(row=1, column=1, columnspan=3, padx=4, pady=2)

add_lbl("Response A  (gemini-2.5-flash):", 2,0)
respA_box = scrolledtext.ScrolledText(root, height=6, width=80, bg="#252526", fg="#d4d4d4")
respA_box.grid(row=2, column=1, columnspan=3, padx=4, pady=2)

add_lbl("Response B  (gemini-2.0-flash):", 3,0)
respB_box = scrolledtext.ScrolledText(root, height=6, width=80, bg="#252526", fg="#d4d4d4")
respB_box.grid(row=3, column=1, columnspan=3, padx=4, pady=2)

status_var = tk.StringVar(value="➜  輸入 Prompt 與 API-Key，然後按「產生回應」")
ttk.Label(root, textvariable=status_var, foreground="#c586c0").grid(
    row=4, column=0, columnspan=4, sticky="w", padx=4, pady=4)

# ────────── Gemini 生成 ──────────
def generate_responses():
    key    = api_entry.get().strip()
    prompt = prompt_box.get("1.0", tk.END).strip()
    if not key or not prompt:
        messagebox.showwarning("缺參數", "請填 API-Key 與 Prompt"); return

    status_var.set("🔄 Gemini 生成中…")
    respA_box.delete("1.0", tk.END); respB_box.delete("1.0", tk.END)

    def _worker():
        try:
            configure(api_key=key)
            mdlA = GenerativeModel("gemini-2.5-flash")
            mdlB = GenerativeModel("gemini-2.0-flash")

            rA = mdlA.generate_content(prompt).text
            rB = mdlB.generate_content(prompt).text

            root.after(0, lambda: respA_box.insert(tk.END, rA))
            root.after(0, lambda: respB_box.insert(tk.END, rB))
            root.after(0, lambda: status_var.set("✓  請在心裡選較佳回應，再按「模型判斷」"))
        except Exception as e:
            root.after(0, lambda: status_var.set(f"⚠️  生成失敗: {e}"))
    threading.Thread(target=_worker, daemon=True).start()

# ────────── Dual-Encoder 推理 ──────────
def run_model():
    prompt = prompt_box.get("1.0", tk.END).strip()
    a      = respA_box.get("1.0", tk.END).strip()
    b      = respB_box.get("1.0", tk.END).strip()
    if not (prompt and a and b):
        messagebox.showwarning("缺回應", "請先產生回應"); return
    status_var.set("🤖 模型推理中…")

    def _work():
        try:
            pred = predict(prompt, a, b)
            msg  = ["模型判定：A 較佳 🏆",
                    "模型判定：B 較佳 🏆",
                    "模型判定：打平 🤝"][pred]
            root.after(0, lambda: status_var.set(msg))
        except Exception as e:
            root.after(0, lambda: status_var.set(f"⚠️ 推理失敗: {e}"))
    threading.Thread(target=_work, daemon=True).start()

# ────────── 按鈕 ──────────
gen_btn  = ttk.Button(root, text="產生回應", command=generate_responses)
pred_btn = ttk.Button(root, text="模型判斷", command=run_model)
gen_btn.grid(row=5, column=1, sticky="we", pady=6)
pred_btn.grid(row=5, column=2, sticky="we", pady=6)

root.columnconfigure(1, weight=1)   # 讓輸入框隨視窗放大
root.mainloop()
