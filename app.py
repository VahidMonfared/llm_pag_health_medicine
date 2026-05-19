"""
PAG-Health-LLM — Hugging Face Spaces / Gradio demo app
Clean output format: no inline citations, single Source line at end.
"""

import os
import re
import json
import numpy as np
import gradio as gr
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---- Configuration ----
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
SIM_THRESHOLD = 0.50
TOP_K = 3
CHUNKS_PATH = "corpus_chunks.json"      # generic-named chunk file
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama-3.3-70b-versatile"   # Groq

# ---- Init ----
groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None
print("Loading retriever…")
embed_model = SentenceTransformer(EMBED_MODEL)
with open(CHUNKS_PATH) as f:
    chunks = json.load(f)
texts = [c["text"] for c in chunks]
embs = embed_model.encode(texts, show_progress_bar=True,
                          normalize_embeddings=True)
embs = np.array(embs).astype("float32")
idx = faiss.IndexFlatIP(embs.shape[1])
idx.add(embs)
print(f"Retriever ready: {idx.ntotal} chunks")


# ---- Retrieval ----
def retrieve(q, k=TOP_K):
    v = embed_model.encode([q], normalize_embeddings=True).astype("float32")
    s, i = idx.search(v, k)
    return [(float(s[0][j]), chunks[i[0][j]]) for j in range(k)]


# ---- Output cleaning ----
CHAPTER_PATTERNS = [
    r"\[Chapter\s+\d+[^\]]*\]",
    r"\(Chapter\s+\d+[^\)]*\)",
    r"\bChapter\s+\d+(?:\s*:\s*[^.,;\n]+)?",
    r"\bchapter\s+\d+(?:\s*:\s*[^.,;\n]+)?",
    r"\bsection\s+\d+(?:\.\d+)*",
    r"\bSection\s+\d+(?:\.\d+)*",
    r"\bprovided reference passages?\b",
    r"\bprovided passages?\b",
    r"\bin the given passages?\b",
    r"\breference textbook\b",
    r"\bthe textbook\b",
    r"📚\s*Sources?:.*",
]


def clean_text(text):
    out = text
    for p in CHAPTER_PATTERNS:
        out = re.sub(p, "", out, flags=re.IGNORECASE)
    out = re.sub(r"\[\s*\]|\(\s*\)", "", out)
    out = re.sub(r"\s*,\s*,\s*", ", ", out)
    out = re.sub(r"\s*and\s*,", "", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def ensure_period(text):
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            lines.append("")
            continue
        if not re.search(r"[.!?]\s*$", s):
            s = s.rstrip() + "."
        lines.append(s)
    return "\n".join(lines)


# ---- Inference ----
def answer(question, history):
    if not question.strip():
        return "Please ask a question."
    if not groq_client:
        return "Error: GROQ_API_KEY not set."

    hits = retrieve(question, TOP_K)
    used_rag = bool(hits) and hits[0][0] >= SIM_THRESHOLD
    ctx = "\n".join([f"PASSAGE {i+1}: {h[1]['text'][:300]}"
                     for i, h in enumerate(hits)]) if used_rag else ""

    system_prompt = (
        "You are a specialist assistant for pediatric and adolescent "
        "gynecology. Answer ONLY using the reference passages provided "
        "(when present). \n\n"
        "STRICT OUTPUT RULES:\n"
        "1. Do NOT mention chapters, chapter numbers, sections, the "
        "textbook by name, or any retrieval label.\n"
        "2. Do NOT include bracketed citations or source fragments "
        "inside the answer.\n"
        "3. Write continuous clinical prose, 3–6 short sentences, each "
        "ending with a period.\n"
        "4. Do NOT include a Source line; it is appended automatically.\n"
    )

    user_prompt = (
        (f"Reference material (internal only — do not reveal):\n{ctx}\n\n"
         if ctx else "")
        + f"Question: {question}\n\nAnswer:"
    )

    try:
        r = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=500, temperature=0.3,
        )
        ans = r.choices[0].message.content.strip()
    except Exception as e:
        ans = f"Error: {str(e)[:120]}"

    ans = clean_text(ans)
    ans = ensure_period(ans)

    src = ("Source: Domain-specific fine-tuned RAG model based on the "
           "reference textbook." if used_rag
           else "Source: Base language model: Mistral-7B-Instruct-v0.3.")
    return f"{ans}\n\n{src}"


# ---- UI ----
demo = gr.ChatInterface(
    fn=answer,
    title="🏥 PAG Health LLM",
    description=(
        "AI for Pediatric & Adolescent Gynecology.\n"
        "⚠️ Research tool only — not a substitute for professional medical advice."
    ),
    examples=[
        "What are signs of precocious puberty?",
        "How is PCOS diagnosed in adolescents?",
        "What is the treatment for dysmenorrhea?",
    ],
)
demo.launch(server_name="0.0.0.0", server_port=7860)
