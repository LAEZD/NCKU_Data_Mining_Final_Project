# tk_app.py  ── VS-Code 暗色小視窗，用 Dual-Encoder 推理 --------------------------------
import tkinter as tk
from tkinter import scrolledtext, messagebox
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from dual_encoder import DualTowerPairClassifier

# ─────── 準備模型 ───────
ROOT         = Path(__file__).parent           # app/ 資料夾
WEIGHT_FILE  = ROOT / "model.safetensors"      # 與 tk_app.py 同層
TOKENIZERDIR = ROOT                            # tokenizer.json / vocab.txt / config.json 都放這

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZERDIR)
model     = DualTowerPairClassifier(base_model="distilbert-base-uncased", include_prompt=True).to(DEVICE)  # 若需關閉prompt請改為False
state     = load_file(WEIGHT_FILE)
model.load_state_dict(state, strict=False)
model.eval()

@torch.inference_mode()
def predict(prompt: str, resp_a: str, resp_b: str) -> str:
    # 每段文字各自做 tokenizer；都 Pad / Truncate 到 512
    tok_p = tokenizer(prompt,  return_tensors="pt",
                      truncation=True, padding="max_length", max_length=512).to(DEVICE)
    tok_a = tokenizer(resp_a,  return_tensors="pt",
                      truncation=True, padding="max_length", max_length=512).to(DEVICE)
    tok_b = tokenizer(resp_b,  return_tensors="pt",
                      truncation=True, padding="max_length", max_length=512).to(DEVICE)

    out = model(
        p_input_ids      = tok_p["input_ids"],
        p_attention_mask = tok_p["attention_mask"],
        a_input_ids      = tok_a["input_ids"],
        a_attention_mask = tok_a["attention_mask"],
        b_input_ids      = tok_b["input_ids"],
        b_attention_mask = tok_b["attention_mask"],
    )
    idx = out["logits"].argmax(-1).item()
    return {0: "A", 1: "B", 2: "Tie"}[idx]

# ─────── Tkinter UI ───────
root = tk.Tk()
root.title("LLM Preference Guess  (Dual-encoder)")
root.configure(bg="#1e1e1e")

def dark(widget):                              # 全域暗色設定
    widget.configure(bg="#1e1e1e", fg="#d4d4d4", insertbackground="#d4d4d4")

tk.Label(root, text="Prompt", bg="#1e1e1e", fg="#d4d4d4").pack(anchor="w")
box_p = scrolledtext.ScrolledText(root, width=90, height=4);  box_p.pack(); dark(box_p)

tk.Label(root, text="Response  A", bg="#1e1e1e", fg="#d4d4d4").pack(anchor="w")
box_a = scrolledtext.ScrolledText(root, width=90, height=6);  box_a.pack(); dark(box_a)

tk.Label(root, text="Response  B", bg="#1e1e1e", fg="#d4d4d4").pack(anchor="w")
box_b = scrolledtext.ScrolledText(root, width=90, height=6);  box_b.pack(); dark(box_b)

result_var = tk.StringVar(value="︙  Think of your choice, then press the button")
tk.Label(root, textvariable=result_var, bg="#1e1e1e", fg="#00d8ff").pack(pady=5)

def on_click():
    p = box_p.get("1.0", "end").strip()
    a = box_a.get("1.0", "end").strip()
    b = box_b.get("1.0", "end").strip()
    if not (p and a and b):
        messagebox.showerror("Oops", "Please fill all three boxes."); return
    try:
        winner = predict(p, a, b)
        result_var.set(f"🔮  Model guesses you chose ➜  {winner}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

btn = tk.Button(root, text="✨ Let the model guess!",
                command=on_click,
                bg="#0e639c", fg="white",
                activebackground="#1177bb",
                padx=20, pady=8)
btn.pack(pady=10)

root.mainloop()
