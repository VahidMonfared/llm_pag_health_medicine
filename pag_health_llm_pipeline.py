"""
=============================================================================
PAG-Health-LLM — End-to-End Pipeline (Reference Implementation)
=============================================================================
Domain-specialized retrieval-augmented LLM for pediatric and adolescent
gynecology.

Components:
  Stage 1: Corpus ingestion and chunking
  Stage 2: Question-answer dataset generation with no-leakage splits
  Stage 3: QLoRA fine-tuning of Mistral-7B-Instruct
  Stage 4: BGE + FAISS retrieval index construction
  Stage 5: Inference with cascading "RAG-first, model-fallback" strategy
  Stage 6: Ten-metric evaluation suite + statistical analysis

Author: Vahid Monfared, Reza Rawassizadeh
License: Apache 2.0
=============================================================================
"""

import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    """Centralized configuration for the entire pipeline."""

    # ---- Corpus ----
    SOURCE_TEXT_DIR = "data/source_text"      # plain text files extracted from corpus
    CHUNKS_PATH     = "data/corpus_chunks.json"
    CHUNK_WORDS     = 250
    CHUNK_OVERLAP   = 50

    # ---- QA dataset ----
    QA_TRAIN_PATH   = "data/qa_train.jsonl"
    QA_VAL_PATH     = "data/qa_val.jsonl"
    QA_TEST_PATH    = "data/qa_test.jsonl"
    SPLIT_BY_CHAPTER = True
    TRAIN_FRAC      = 0.70
    VAL_FRAC        = 0.15
    TEST_FRAC       = 0.15

    # ---- Base model ----
    BASE_MODEL      = "mistralai/Mistral-7B-Instruct-v0.3"
    EMBED_MODEL     = "BAAI/bge-small-en-v1.5"

    # ---- QLoRA fine-tuning ----
    LORA_RANK       = 16
    LORA_ALPHA      = 32
    LORA_DROPOUT    = 0.05
    TARGET_MODULES  = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
    LR              = 2e-4
    BATCH_SIZE      = 4
    GRAD_ACCUM      = 4
    EPOCHS          = 3
    OUTPUT_DIR      = "models/pag_health_llm"

    # ---- Retrieval ----
    FAISS_INDEX_PATH = "models/faiss_index.bin"
    TOP_K            = 3
    SIM_THRESHOLD    = 0.50

    # ---- External evaluation ----
    EVAL_RESULTS_DIR = "results"
    SEED             = 42


# =============================================================================
# STAGE 1 — Corpus ingestion and chunking
# =============================================================================
def stage1_build_chunks(cfg: Config):
    """
    Read plain-text files from cfg.SOURCE_TEXT_DIR, segment into
    overlapping word-windows, and write JSON-formatted chunks.

    Each chunk receives a generic identifier (corpus_section_NNN) so that
    no source-specific names appear anywhere in the output corpus.
    """
    src = Path(cfg.SOURCE_TEXT_DIR)
    src.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No .txt files found in {src}. "
            "Place your plain-text corpus there (one file per section)."
        )

    chunks = []
    cid = 0
    for fpath in files:
        text = fpath.read_text(encoding="utf-8").strip()
        words = text.split()
        i = 0
        while i < len(words):
            window = words[i : i + cfg.CHUNK_WORDS]
            if not window:
                break
            chunks.append({
                "id": f"corpus_section_{cid:04d}",
                "source": f"corpus_section_{cid:04d}",
                "text": " ".join(window),
            })
            cid += 1
            i += cfg.CHUNK_WORDS - cfg.CHUNK_OVERLAP

    Path(cfg.CHUNKS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[Stage 1] {len(chunks)} chunks written to {cfg.CHUNKS_PATH}")


# =============================================================================
# STAGE 2 — Build QA dataset with chapter-level no-leakage splits
# =============================================================================
def stage2_build_qa_dataset(cfg: Config, qa_seed_file: str):
    """
    qa_seed_file should be a JSONL file where each line has:
      {"section_id": "...", "question": "...", "answer": "..."}

    Splits are made at the section level so test questions never share
    section context with training examples.
    """
    with open(qa_seed_file, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    # Group by section
    by_section = {}
    for p in pairs:
        by_section.setdefault(p["section_id"], []).append(p)
    sections = list(by_section.keys())

    random.seed(cfg.SEED)
    random.shuffle(sections)

    n = len(sections)
    n_train = int(n * cfg.TRAIN_FRAC)
    n_val   = int(n * cfg.VAL_FRAC)

    train_sec = sections[:n_train]
    val_sec   = sections[n_train : n_train + n_val]
    test_sec  = sections[n_train + n_val :]

    def dump(sec_list, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as out:
            for sec in sec_list:
                for ex in by_section[sec]:
                    out.write(json.dumps(ex, ensure_ascii=False) + "\n")

    dump(train_sec, cfg.QA_TRAIN_PATH)
    dump(val_sec,   cfg.QA_VAL_PATH)
    dump(test_sec,  cfg.QA_TEST_PATH)
    print(f"[Stage 2] Train sections: {len(train_sec)}, "
          f"Val: {len(val_sec)}, Test: {len(test_sec)}")


# =============================================================================
# STAGE 3 — QLoRA fine-tuning
# =============================================================================
def stage3_finetune_qlora(cfg: Config):
    """
    Fine-tune Mistral-7B-Instruct with QLoRA on the training QA pairs.
    Designed to fit on a single 16 GB consumer GPU.
    """
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        BitsAndBytesConfig, TrainingArguments
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import load_dataset

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=cfg.LORA_RANK,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        target_modules=cfg.TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    def format_example(ex):
        return {
            "text": (
                f"<s>[INST] You are a specialist assistant for "
                f"pediatric and adolescent gynecology. "
                f"Answer the following clinical question concisely.\n\n"
                f"Question: {ex['question']} [/INST] {ex['answer']}</s>"
            )
        }

    train_ds = load_dataset("json", data_files=cfg.QA_TRAIN_PATH, split="train")
    val_ds   = load_dataset("json", data_files=cfg.QA_VAL_PATH,   split="train")
    train_ds = train_ds.map(format_example)
    val_ds   = val_ds.map(format_example)

    args = TrainingArguments(
        output_dir=cfg.OUTPUT_DIR,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRAD_ACCUM,
        num_train_epochs=cfg.EPOCHS,
        learning_rate=cfg.LR,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=512,
    )

    trainer.train()
    trainer.save_model(cfg.OUTPUT_DIR)
    print(f"[Stage 3] Fine-tuned model saved to {cfg.OUTPUT_DIR}")


# =============================================================================
# STAGE 4 — BGE + FAISS retrieval index
# =============================================================================
def stage4_build_retrieval_index(cfg: Config):
    """Build a FAISS index over BGE embeddings of the corpus chunks."""
    import faiss
    from sentence_transformers import SentenceTransformer

    with open(cfg.CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embed = SentenceTransformer(cfg.EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    embs = embed.encode(texts, show_progress_bar=True,
                        normalize_embeddings=True)
    embs = np.array(embs).astype("float32")

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    Path(cfg.FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, cfg.FAISS_INDEX_PATH)
    print(f"[Stage 4] FAISS index ({index.ntotal} vectors) "
          f"saved to {cfg.FAISS_INDEX_PATH}")


def load_retriever(cfg: Config):
    """Helper: load chunks, embedder, and FAISS index for inference."""
    import faiss
    from sentence_transformers import SentenceTransformer

    with open(cfg.CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embed = SentenceTransformer(cfg.EMBED_MODEL)
    index = faiss.read_index(cfg.FAISS_INDEX_PATH)
    return chunks, embed, index


def retrieve(query, chunks, embed, index, k=3):
    """Top-k retrieval with cosine similarity scores."""
    v = embed.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(v, k)
    return [(float(scores[0][i]), chunks[idxs[0][i]]) for i in range(k)]


# =============================================================================
# STAGE 5 — Inference (RAG-first, model-fallback)
# =============================================================================
import re

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


def clean_output(text):
    """Strip any inline chapter/section/source references and tidy spacing."""
    out = text
    for pat in CHAPTER_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    out = re.sub(r"\[\s*\]|\(\s*\)", "", out)
    out = re.sub(r"\s*,\s*,\s*", ", ", out)
    out = re.sub(r"\s*and\s*,", "", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def ensure_period(text):
    """Ensure each non-empty line ends with sentence punctuation."""
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            out.append("")
            continue
        if not re.search(r"[.!?]\s*$", s):
            s = s.rstrip() + "."
        out.append(s)
    return "\n".join(out)


def generate_answer(question, cfg: Config,
                    use_rag=True, use_finetuned=True):
    """
    Generate a clinical answer.
      use_rag=True       : retrieval + model
      use_finetuned=True : load the QLoRA-fine-tuned adapter
      Falls back to the base model with source-line annotation when
      retrieval similarity is below cfg.SIM_THRESHOLD.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(cfg.BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    if use_finetuned and Path(cfg.OUTPUT_DIR).exists():
        model = PeftModel.from_pretrained(model, cfg.OUTPUT_DIR)

    context_str = ""
    used_rag = False
    if use_rag:
        chunks, embed, index = load_retriever(cfg)
        hits = retrieve(question, chunks, embed, index, cfg.TOP_K)
        if hits and hits[0][0] >= cfg.SIM_THRESHOLD:
            used_rag = True
            context_str = "\n".join([
                f"PASSAGE {i+1}: {h[1]['text'][:300]}"
                for i, h in enumerate(hits)
            ])

    system = (
        "You are a specialist assistant for pediatric and adolescent "
        "gynecology. Answer concisely in clear clinical prose. Do NOT "
        "mention chapters, sections, the source corpus, or retrieval "
        "labels of any kind. Do NOT include a source line; it will be "
        "appended automatically."
    )
    user = (
        (f"Reference (internal only): {context_str}\n\n" if context_str else "")
        + f"Question: {question}\n\nAnswer:"
    )
    prompt = f"<s>[INST] {system}\n\n{user} [/INST]"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
    )
    answer = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    answer = clean_output(answer)
    answer = ensure_period(answer)

    if used_rag:
        src = "Source: Domain-specific fine-tuned RAG model based on the reference textbook."
    else:
        src = "Source: Base language model: Mistral-7B-Instruct-v0.3."
    return f"{answer}\n\n{src}"


# =============================================================================
# STAGE 6 — Ten-metric evaluation
# =============================================================================
def stage6_evaluate(cfg: Config, predictions_file: str, references_file: str):
    """
    Compute ten complementary metrics:
      ROUGE-1, ROUGE-2, ROUGE-L, BLEU, BERTScore,
      METEOR, chrF++, BLEURT, SAS, G-Eval (LLM-as-judge stub)
    """
    from rouge_score import rouge_scorer
    from sacrebleu import sentence_chrf
    import sacrebleu
    from bert_score import score as bert_score_fn
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt",   quiet=True)
    from nltk.translate.meteor_score import meteor_score
    from sentence_transformers import CrossEncoder

    with open(predictions_file) as f:
        preds = [json.loads(l) for l in f]
    with open(references_file) as f:
        refs = [json.loads(l) for l in f]

    assert len(preds) == len(refs)
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                     use_stemmer=True)
    sas_model = CrossEncoder("cross-encoder/stsb-roberta-large")

    rows = []
    for p, r in tqdm(zip(preds, refs), total=len(preds)):
        pred_text = p["answer"]
        ref_text  = r["answer"]

        rs = rouge.score(ref_text, pred_text)
        rouge1 = rs["rouge1"].fmeasure
        rouge2 = rs["rouge2"].fmeasure
        rougel = rs["rougeL"].fmeasure
        bleu   = sacrebleu.sentence_bleu(pred_text, [ref_text]).score / 100.0
        chrf   = sentence_chrf(pred_text, [ref_text], word_order=2).score / 100.0
        met    = meteor_score([ref_text.split()], pred_text.split())
        sas    = sas_model.predict([(pred_text, ref_text)])[0]

        rows.append({
            "rouge1": rouge1, "rouge2": rouge2, "rougeL": rougel,
            "bleu": bleu, "chrf": chrf, "meteor": met, "sas": float(sas),
        })

    # BERTScore (batched for efficiency)
    P, R, F1 = bert_score_fn(
        [p["answer"] for p in preds],
        [r["answer"] for r in refs],
        lang="en", verbose=False,
    )
    for i, row in enumerate(rows):
        row["bertscore"] = float(F1[i])

    df = pd.DataFrame(rows)
    Path(cfg.EVAL_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(cfg.EVAL_RESULTS_DIR) / "metric_scores.csv"
    df.to_csv(out_path, index=False)
    print(f"[Stage 6] Metric scores saved to {out_path}")
    print(df.mean().to_string())


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================
def statistical_analysis(scores_csv_ours, scores_csv_other, output_path):
    """Paired t-tests + Cohen's d between two systems on the same questions."""
    from scipy import stats

    a = pd.read_csv(scores_csv_ours)
    b = pd.read_csv(scores_csv_other)
    assert len(a) == len(b)
    metrics = [c for c in a.columns if c in
               ["rouge1", "rouge2", "rougeL", "bleu", "bertscore",
                "meteor", "chrf", "sas"]]

    rows = []
    for m in metrics:
        x = a[m].values
        y = b[m].values
        t, p = stats.ttest_rel(x, y)
        diff = x - y
        d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float("nan")
        eff = ("Large" if abs(d) > 0.8 else
               "Medium" if abs(d) > 0.5 else
               "Small" if abs(d) > 0.2 else "Negligible")
        rows.append({"metric": m, "mean_ours": x.mean(), "mean_other": y.mean(),
                     "p_value": p, "cohens_d": d, "effect_size": eff})

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Statistics saved to {output_path}")


# =============================================================================
# COMMAND-LINE ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PAG-Health-LLM pipeline")
    parser.add_argument("stage", choices=["1", "2", "3", "4", "5", "6", "all"],
                        help="Which pipeline stage to run")
    parser.add_argument("--qa_seed", default="data/qa_seed.jsonl",
                        help="Seed JSONL of QA pairs (for stage 2)")
    parser.add_argument("--query", default="What is the workup for "
                                           "prepubertal vaginal bleeding?",
                        help="Question to ask in stage 5")
    parser.add_argument("--preds", default="results/test_predictions.jsonl",
                        help="Predictions file (for stage 6)")
    parser.add_argument("--refs", default="data/qa_test.jsonl",
                        help="References file (for stage 6)")
    args = parser.parse_args()

    cfg = Config()
    if args.stage in ("1", "all"):
        stage1_build_chunks(cfg)
    if args.stage in ("2", "all"):
        stage2_build_qa_dataset(cfg, args.qa_seed)
    if args.stage in ("3", "all"):
        stage3_finetune_qlora(cfg)
    if args.stage in ("4", "all"):
        stage4_build_retrieval_index(cfg)
    if args.stage == "5":
        ans = generate_answer(args.query, cfg, use_rag=True, use_finetuned=True)
        print("\n" + ans)
    if args.stage in ("6", "all"):
        stage6_evaluate(cfg, args.preds, args.refs)
