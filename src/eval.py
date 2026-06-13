import re

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .model import format_prompt, is_refusal
from .sae import topk_binarize
from .utils import TOP_K_FEATURES, get_logger, log_exception

logger = get_logger("eval")


def _encode_prompt_to_pc_vector(extractor, sae, prompt, layer_idx, feature_index_map):
    formatted = format_prompt(prompt, extractor.tokenizer)
    act = extractor.get_last_token_activation(formatted, layer_idx)
    z_dense = sae.encode_np(act)
    if z_dense.ndim == 1:
        z_dense = z_dense[None, :]
    z_bin = topk_binarize(z_dense, TOP_K_FEATURES)
    if feature_index_map is not None:
        z_bin = z_bin[:, feature_index_map]
    return z_bin.astype(np.float32)


def evaluate_attack_success(extractor, prompts, monitor=None, max_new_tokens=128, judge_fn=None):
    if judge_fn is None:
        judge_fn = is_refusal
    n_blocked = 0
    n_gen_refused = 0
    n_gen_complied = 0
    records = []
    for p in tqdm(prompts, desc="ASR", leave=False):
        try:
            blocked = False
            if monitor is not None:
                z = _encode_prompt_to_pc_vector(
                    extractor, monitor.sae, p, monitor.layer_idx, monitor.feature_index_map
                )
                pred = monitor.predict(z)
                blocked = bool(pred["is_unsafe"][0]) if pred["is_unsafe"] is not None else False
            if blocked:
                n_blocked += 1
                records.append({"prompt": p, "blocked": True, "response": None, "refused": True})
                continue
            formatted = format_prompt(p, extractor.tokenizer)
            out = extractor.generate(formatted, max_new_tokens=max_new_tokens)
            refused = judge_fn(out)
            if refused:
                n_gen_refused += 1
            else:
                n_gen_complied += 1
            records.append({"prompt": p, "blocked": False, "response": out, "refused": refused})
        except Exception:
            log_exception(logger, f"failed on prompt: {p[:60]!r}")
            records.append({"prompt": p, "blocked": False, "response": None,
                            "refused": False, "error": True})
    N = len(prompts)
    return {
        "n_prompts": N,
        "n_blocked_by_monitor": n_blocked,
        "n_refused_by_model": n_gen_refused,
        "n_attacks_successful": n_gen_complied,
        "attack_success_rate": n_gen_complied / N if N > 0 else 0.0,
        "monitor_block_rate": n_blocked / N if N > 0 else 0.0,
        "overall_refusal_rate": (n_blocked + n_gen_refused) / N if N > 0 else 0.0,
        "breakdown": {
            "blocked": n_blocked,
            "generated_refused": n_gen_refused,
            "generated_complied": n_gen_complied,
        },
        "records": records,
    }


def evaluate_over_refusal(extractor, benign_prompts, monitor=None, max_new_tokens=128, judge_fn=None):
    if judge_fn is None:
        judge_fn = is_refusal
    n_blocked = 0
    n_gen_refused = 0
    n_complied = 0
    records = []
    for p in tqdm(benign_prompts, desc="overR", leave=False):
        try:
            blocked = False
            if monitor is not None:
                z = _encode_prompt_to_pc_vector(
                    extractor, monitor.sae, p, monitor.layer_idx, monitor.feature_index_map
                )
                pred = monitor.predict(z)
                blocked = bool(pred["is_unsafe"][0]) if pred["is_unsafe"] is not None else False
            if blocked:
                n_blocked += 1
                records.append({"prompt": p, "blocked": True, "response": None, "refused": True})
                continue
            formatted = format_prompt(p, extractor.tokenizer)
            out = extractor.generate(formatted, max_new_tokens=max_new_tokens)
            refused = judge_fn(out)
            if refused:
                n_gen_refused += 1
            else:
                n_complied += 1
            records.append({"prompt": p, "blocked": False, "response": out, "refused": refused})
        except Exception:
            log_exception(logger, f"failed on prompt: {p[:60]!r}")
            records.append({"prompt": p, "blocked": False, "response": None,
                            "refused": False, "error": True})
    N = len(benign_prompts)
    n_overrefused = n_blocked + n_gen_refused
    return {
        "n_prompts": N,
        "over_refusal_rate": n_overrefused / N if N > 0 else 0.0,
        "compliance_rate": n_complied / N if N > 0 else 0.0,
        "breakdown": {
            "blocked": n_blocked,
            "generated_refused": n_gen_refused,
            "complied": n_complied,
        },
        "records": records,
    }


def evaluate_mmlu_subset(extractor, mmlu_examples, max_new_tokens=8):
    correct = 0
    records = []
    letters = ["A", "B", "C", "D"]
    for ex in tqdm(mmlu_examples, desc="MMLU", leave=False):
        try:
            q = ex.get("question", "")
            choices = ex.get("choices", ex.get("options", []))
            if len(choices) != 4:
                continue
            answer_idx = ex.get("answer", -1)
            formatted_q = (q + "\n" +
                           "\n".join([f"{letters[i]}. {choices[i]}" for i in range(4)]) +
                           "\nAnswer with a single letter (A, B, C, or D).")
            prompt = format_prompt(formatted_q, extractor.tokenizer)
            out = extractor.generate(prompt, max_new_tokens=max_new_tokens).strip()
            pred_letter = next((c for c in out if c in letters), None)
            is_correct = (pred_letter == letters[answer_idx]
                          if 0 <= answer_idx < 4 and pred_letter else False)
            if is_correct:
                correct += 1
            records.append({"question": q, "predicted": pred_letter,
                            "answer": letters[answer_idx] if 0 <= answer_idx < 4 else None,
                            "correct": is_correct})
        except Exception:
            log_exception(logger, f"MMLU failed on: {ex.get('question', '')[:60]!r}")
    N = len(records)
    return {"n_questions": N, "accuracy": correct / N if N > 0 else 0.0, "records": records}


def evaluate_gsm8k_subset(extractor, gsm_examples, max_new_tokens=256):
    correct = 0
    records = []
    for ex in tqdm(gsm_examples, desc="GSM8K", leave=False):
        try:
            q = ex.get("question", "")
            gold = ex.get("answer", "")
            gold_value = extract_final_number(gold)
            prompt = format_prompt(
                q + "\nThink step by step and put the final answer after '####'.",
                extractor.tokenizer,
            )
            out = extractor.generate(prompt, max_new_tokens=max_new_tokens)
            pred_value = extract_final_number(out)
            is_correct = (pred_value is not None and gold_value is not None
                          and abs(pred_value - gold_value) < 1e-3)
            if is_correct:
                correct += 1
            records.append({"question": q, "predicted": pred_value,
                            "gold": gold_value, "correct": is_correct})
        except Exception:
            log_exception(logger, f"GSM8K failed on: {ex.get('question', '')[:60]!r}")
    N = len(records)
    return {"n_questions": N, "accuracy": correct / N if N > 0 else 0.0, "records": records}


def extract_final_number(text):
    if not text:
        return None
    if "####" in text:
        text = text.split("####")[-1]
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def evaluate_density_auroc(monitor, X_safe, X_unsafe):
    s_safe = monitor.score(X_safe)["score"]
    s_unsafe = monitor.score(X_unsafe)["score"]
    labels = np.concatenate([np.zeros(len(s_safe)), np.ones(len(s_unsafe))])
    scores = np.concatenate([s_safe, s_unsafe])
    auroc = float(roc_auc_score(labels, scores))
    return {
        "auroc": auroc,
        "mean_score_safe": float(s_safe.mean()),
        "mean_score_unsafe": float(s_unsafe.mean()),
        "std_score_safe": float(s_safe.std()),
        "std_score_unsafe": float(s_unsafe.std()),
    }


def baseline_refusal_direction(activations_harmful, activations_benign):
    mu_h = activations_harmful.mean(axis=0)
    mu_b = activations_benign.mean(axis=0)
    direction = mu_h - mu_b
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    return direction


def baseline_score_with_direction(activations, direction):
    return activations @ direction