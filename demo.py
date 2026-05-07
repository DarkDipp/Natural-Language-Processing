"""
Demo Gradio - He thong hoi dap luat giao thong Viet Nam
4 cau hinh: A (Base), B (Base+RAG), C (FT), D (FT+RAG)
Chay: python demo.py
"""
import os, re, random
os.environ["PYTHONUNBUFFERED"] = "1"
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import faiss

# ── Config ───────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_LLM    = os.getenv("BASE_LLM", "Qwen/Qwen2.5-0.5B-Instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
MAX_NEW_TOKENS = 128
TOP_K = 5

WORK_DIR   = Path("./trafficlaw_rag")
CACHE_DIR  = WORK_DIR / "cache"
OUT_DIR    = WORK_DIR / "outputs"
DATA_DIR   = WORK_DIR / "data" / "processed"
ADAPTER_DIR = OUT_DIR / "qlora_adapter_final"

# ── Tien ich ─────────────────────────────────────────────────────────────────
def normalize_text(text) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Prompt ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp luật giao thông Việt Nam. "
    "Trả lời chính xác, trích dẫn điều khoản nếu có. "
    "Nếu không chắc chắn, hãy nói rõ."
)

def build_messages(question: str, contexts=None, use_rag=False):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    user_text = ""
    if use_rag and contexts:
        user_text += "Ngữ cảnh:\n"
        for c in contexts:
            user_text += f"[Nguồn {c['rank']} | {c['chunk_id']}]\n{c['text']}\n\n"
    user_text += f"Câu hỏi: {normalize_text(question)}"
    msgs.append({"role": "user", "content": user_text})
    return msgs

# ── Load embeddings + FAISS ──────────────────────────────────────────────────
print("=" * 60)
print("  LOADING RESOURCES")
print("=" * 60)

print("[1/3] Loading FAISS index + embedding model ...")
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer(EMBED_MODEL)
faiss_index = faiss.read_index(str(CACHE_DIR / "faiss.index"))

# Load chunks
chunks_pkl = CACHE_DIR / "chunks.pkl"
if chunks_pkl.exists():
    import pickle
    with open(chunks_pkl, "rb") as f:
        all_chunks = pickle.load(f)
    print(f"      Loaded {len(all_chunks)} chunks from chunks.pkl")
else:
    df_chunks = pd.read_csv(DATA_DIR / "corpus_chunks.csv")
    all_chunks = df_chunks.to_dict(orient="records")
    print(f"      Loaded {len(all_chunks)} chunks from corpus_chunks.csv")

def retrieve_top_k(query: str, k: int = TOP_K) -> List[Dict]:
    q_emb = embed_model.encode([f"query: {query}"], normalize_embeddings=True)
    scores, indices = faiss_index.search(q_emb.astype("float32"), k)
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        if idx < 0 or idx >= len(all_chunks):
            continue
        chunk = all_chunks[idx]
        results.append({
            "rank": rank,
            "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
            "text": chunk.get("text", ""),
            "score": float(score),
        })
    return results

# ── Load LLM ─────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

def get_model_device(model):
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_base_model(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto",
        quantization_config=bnb_config, trust_remote_code=True,
    )
    model.eval()
    return tok, model

def load_finetuned_model(model_name: str, adapter_dir: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto",
        quantization_config=bnb_config, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tok, model

print("[2/3] Loading base model ...")
base_tok, base_model = load_base_model(BASE_LLM)
print("      [OK] Base model loaded")

print("[3/3] Loading fine-tuned model ...")
ft_tok, ft_model = load_finetuned_model(BASE_LLM, str(ADAPTER_DIR))
print("      [OK] Fine-tuned model loaded")

print("=" * 60)
print("  ALL MODELS READY!")
print("=" * 60)

# ── Inference ────────────────────────────────────────────────────────────────
def build_prompt(tokenizer, question, contexts=None, use_rag=False):
    messages = build_messages(question, contexts=contexts, use_rag=use_rag)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = SYSTEM_PROMPT + "\n\n"
    if use_rag and contexts:
        prompt += "Ngữ cảnh:\n"
        for c in contexts:
            prompt += f"[Nguồn {c['rank']} | {c['chunk_id']}]\n{c['text']}\n\n"
    prompt += f"Câu hỏi: {normalize_text(question)}\n\nTrả lời:"
    return prompt

def generate_answer(tokenizer, model, question, use_rag=False, k=TOP_K):
    contexts = retrieve_top_k(question, k=k) if use_rag else []
    prompt = build_prompt(tokenizer, question, contexts=contexts, use_rag=use_rag)
    device = get_model_device(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen_cfg = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        temperature=0.0, top_p=1.0, pad_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg)
    gen_tokens = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    answer = text
    for marker in ["### Trả lời:", "Trả lời:", "Answer:"]:
        if marker in answer:
            answer = answer.split(marker, 1)[-1].strip()
    return answer.strip(), contexts

# ── 4 Config map ─────────────────────────────────────────────────────────────
CONFIG_MAP = {
    "A - Base (Không RAG, Không Fine-tune)": (base_tok, base_model, False),
    "B - Base + RAG":                         (base_tok, base_model, True),
    "C - Fine-tuned (Không RAG)":             (ft_tok,   ft_model,   False),
    "D - Fine-tuned + RAG (Tốt nhất)":        (ft_tok,   ft_model,   True),
}

def answer_single(question, config_name):
    if not question or not question.strip():
        return "Vui lòng nhập câu hỏi.", ""
    tok, mdl, use_rag = CONFIG_MAP[config_name]
    answer, contexts = generate_answer(tok, mdl, question.strip(), use_rag=use_rag)
    ctx_text = ""
    if contexts:
        for c in contexts:
            ctx_text += f"**[{c['chunk_id']}]** (score: {c['score']:.4f})\n"
            ctx_text += f"> {c['text'][:300]}...\n\n"
    else:
        ctx_text = "*Cấu hình này không sử dụng RAG.*"
    return answer, ctx_text

def compare_all(question):
    if not question or not question.strip():
        return "Vui lòng nhập câu hỏi.", "", "", ""
    results = []
    for name, (tok, mdl, use_rag) in CONFIG_MAP.items():
        ans, _ = generate_answer(tok, mdl, question.strip(), use_rag=use_rag)
        results.append(ans)
    return results[0], results[1], results[2], results[3]

EXAMPLES = [
    "Độ tuổi tối thiểu để được lái xe trên 50cc là bao nhiêu?",
    "Mức phạt khi vượt đèn đỏ là bao nhiêu?",
    "Khi nào được phép vượt xe phía trước?",
    "Giấy phép lái xe hạng B2 được điều khiển loại xe nào?",
    "Người điều khiển xe máy có bắt buộc đội mũ bảo hiểm không?",
]

# ── CSS ──────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto;
    font-family: 'Segoe UI', 'Roboto', sans-serif;
}
h1 {
    text-align: center;
    background: linear-gradient(135deg, #1e3a5f 0%, #2d8cf0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2em;
    margin-bottom: 0;
}
.subtitle {
    text-align: center;
    color: #666;
    font-size: 1.1em;
    margin-bottom: 20px;
}
.footer {
    text-align: center;
    color: #999;
    font-size: 0.85em;
    margin-top: 20px;
    padding: 10px;
    border-top: 1px solid #eee;
}
"""

# ── Gradio App ───────────────────────────────────────────────────────────────
import gradio as gr

with gr.Blocks(title="Hỏi đáp Luật Giao thông Việt Nam") as demo:

    gr.HTML("<h1>Hệ thống Hỏi đáp Luật Giao thông Việt Nam</h1>")
    gr.HTML('<p class="subtitle">RAG + Fine-tuned LLM | Qwen2.5-0.5B + FAISS + QLoRA</p>')

    with gr.Tabs():

        # ── Tab 1: Hỏi đáp ──────────────────────────────────────────────
        with gr.Tab("Hỏi đáp", id="tab_qa"):
            with gr.Row():
                with gr.Column(scale=1):
                    config_dropdown = gr.Dropdown(
                        choices=list(CONFIG_MAP.keys()),
                        value="D - Fine-tuned + RAG (Tốt nhất)",
                        label="Chọn cấu hình",
                        info="D là cấu hình tốt nhất (Fine-tuned + RAG)",
                    )
                    question_input = gr.Textbox(
                        label="Câu hỏi",
                        placeholder="Nhập câu hỏi về luật giao thông...",
                        lines=3,
                    )
                    submit_btn = gr.Button("Gửi câu hỏi", variant="primary", size="lg")
                    gr.Examples(examples=EXAMPLES, inputs=question_input,
                                label="Câu hỏi mẫu (click để thử)")

                with gr.Column(scale=1):
                    answer_output = gr.Textbox(label="Câu trả lời", lines=10, interactive=False)
                    context_output = gr.Markdown(
                        label="Nguồn tham khảo (RAG)",
                        value="*Kết quả sẽ hiển thị ở đây...*",
                    )

            submit_btn.click(
                fn=answer_single,
                inputs=[question_input, config_dropdown],
                outputs=[answer_output, context_output],
            )

        # ── Tab 2: So sánh ───────────────────────────────────────────────
        with gr.Tab("So sánh 4 cấu hình", id="tab_compare"):
            gr.Markdown("### So sánh output của 4 cấu hình A/B/C/D cùng lúc")
            question_compare = gr.Textbox(
                label="Câu hỏi",
                placeholder="Nhập câu hỏi để so sánh kết quả 4 cấu hình...",
                lines=2,
            )
            compare_btn = gr.Button("So sánh tất cả", variant="primary", size="lg")
            with gr.Row():
                out_a = gr.Textbox(label="A - Base", lines=8, interactive=False)
                out_b = gr.Textbox(label="B - Base + RAG", lines=8, interactive=False)
            with gr.Row():
                out_c = gr.Textbox(label="C - Fine-tuned", lines=8, interactive=False)
                out_d = gr.Textbox(label="D - FT + RAG", lines=8, interactive=False)
            compare_btn.click(
                fn=compare_all,
                inputs=[question_compare],
                outputs=[out_a, out_b, out_c, out_d],
            )

        # ── Tab 3: Giới thiệu ───────────────────────────────────────────
        with gr.Tab("Giới thiệu", id="tab_about"):
            gr.Markdown("""
### Thông tin dự án

| Thành phần | Chi tiết |
|-----------|----------|
| **Mô hình ngôn ngữ** | Qwen/Qwen2.5-0.5B-Instruct |
| **Embedding** | intfloat/multilingual-e5-small |
| **Vector DB** | FAISS (IndexFlatIP) |
| **Fine-tuning** | QLoRA (r=8, alpha=16, 1 epoch) |
| **Dữ liệu** | 6,820 cặp QA luật giao thông Việt Nam |
| **Knowledge base** | 20,998 chunks |

### 4 cấu hình so sánh

| Config | RAG | Fine-tune | Mô tả |
|--------|-----|-----------|--------|
| **A** | --- | --- | Base model gốc |
| **B** | Co  | --- | Base + ngữ cảnh RAG |
| **C** | --- | Co  | Fine-tuned, không RAG |
| **D** | Co  | Co  | **Tốt nhất** - FT + RAG |

### Nhóm thực hiện
- 523H0131
- 523H0147
- 523H0181

*Đồ án cuối kỳ - Nhập môn Xử lý ngôn ngữ tự nhiên*
            """)

    gr.HTML('<p class="footer">Đồ án cuối kỳ - Nhập môn Xử lý ngôn ngữ tự nhiên | Qwen2.5 + FAISS + QLoRA</p>')

demo.launch(share=False, debug=False, css=CUSTOM_CSS, theme=gr.themes.Soft())
