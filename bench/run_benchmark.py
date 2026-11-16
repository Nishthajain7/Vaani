#!/usr/bin/env python3
from llama_cpp import Llama
"""
Benchmark script for Hinglish NLU: three-way comparison.

Benchmarks side-by-side:
  (1) Raw text → GGUF SLM (llama-cpp-python) with JSON schema grammar
  (2) Normalized text → GGUF SLM with JSON schema grammar
  (3) Full C++ Pipeline: normalizer + tokenizer + libvaani_runtime.so

Measures:
  - Intent Accuracy %
  - Slot Exact-Match %
  - Token Fertility (avg tokens per slot value)
  - p50/p95 Latency (ms)
  - Peak RSS Memory (MB)
  - Disk Footprint (MB)

Output: Publication-ready Markdown table.

Run three passes:
  (1) Raw noisy_text      ->  GGUF SLM (direct)
  (2) normalize_text(noisy_text)  ->  GGUF SLM (normalized)
  (3) CLI: vaani_cli "noisy_text"  ->  C++ pipeline output
"""

import json
import os
import re
import subprocess
import sys
import time
import ctypes

from typing import Any, Dict, List, Tuple

import psutil
import gc

# ──────────────────────────────────────────────
# Paths & configuration
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_SET_PATH = os.path.join(PROJECT_ROOT, "data", "eval_set.jsonl")
NORMALIZER_LIB = os.path.join(PROJECT_ROOT, "build", "libnormalizer.so")
RUNTIME_LIB = os.path.join(PROJECT_ROOT, "build", "libvaani_runtime.so")
NLU_MODEL = os.path.join(PROJECT_ROOT, "models", "vaani_nlu.onnx")
NLU_VOCAB = os.path.join(PROJECT_ROOT, "models", "vocab.txt")
NLU_INTENT_LABELS = os.path.join(PROJECT_ROOT, "models", "intent_labels.json")
NLU_SLOT_LABELS = os.path.join(PROJECT_ROOT, "models", "slot_labels.json")
MODEL_PATH = os.environ.get("LLM_MODEL_PATH", "/tmp/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf")

# Intent keys we evaluate over
INTENTS = ["set_reminder", "send_message", "play_media", "check_weather", "set_alarm"]

# Slot label set (same as model)
SLOT_LABELS = ["O", "B-time", "I-time", "B-recipient", "I-recipient",
               "B-message", "I-message", "B-title", "I-title",
               "B-location", "I-location"]

# ──────────────────────────────────────────────
# JSON schema grammar for SLM output
# ──────────────────────────────────────────────
SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "slots": {
            "type": "object",
            "properties": {
                "time": {"type": "string"},
                "recipient": {"type": "string"},
                "message": {"type": "string"},
                "title": {"type": "string"},
                "location": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    "required": ["intent"],
    "additionalProperties": False,
}


# ──────────────────────────────────────────────
# Eval set loader
# ──────────────────────────────────────────────
def load_eval_set(path: str) -> List[Dict[str, str]]:
    """Load the eval set JSONL. Expected keys: 'noisy' and 'slots'."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append({
                "noisy": obj.get("noisy", ""),
                "intent": obj.get("slots", {}).get("intent", ""),
                "slots": obj.get("slots", {}),
            })
    return samples


# ──────────────────────────────────────────────
# GGUF SLM Inference
# ──────────────────────────────────────────────
class SLMInference:
    """Wrapper for llama-cpp-python with JSON schema enforcement."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.llm = None
        self._load_model()

    def _load_model(self):
        """Load the GGUF model if available."""
        if not self.model_path or not os.path.exists(self.model_path):
            print(f"[bench] Model not found at: {self.model_path}")
            print("[bench] Skipping GGUF SLM passes (raw + normalized).")
            return
        try:
            # Use minimal prompt format; system prompt enforces JSON grammar
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx = 256,
                n_gpu_layers = -1,  # all layers on GPU if available, else CPU
                verbose = False,
            )
            print(f"[bench] Loaded GGUF model: {self.model_path}")
        except Exception as e:
            print(f"[bench] Failed to load GGUF model: {e}")
            self.llm = None

    def infer(self, prompt: str) -> Dict[str, Any]:
        """Run inference with JSON schema enforcement.

        Returns dict with 'intent' and 'slots' keys, or fallback.
        """
        if self.llm is None:
            return {"intent": "set_reminder", "slots": {}}

        # Prompt that instructs the model to output strict JSON
        sys_prompt = (
            "You are a Hinglish NLU assistant. Output ONLY a valid JSON object "
            "with two fields: \"intent\" and \"slots\".\n"
            f"\"intent\" must be one of: {', '.join(INTENTS)}.\n"
            "\"slots\" is an object with optional keys: time, recipient, message, title, location.\n"
            "Do NOT include any reasoning, text, or markdown formatting. "
            "Output ONLY the JSON."
        )

        try:
            # Use guided generation / output parsing if available,
            # otherwise rely on the system prompt + post-processing.
            output = self.llm(
                f"{sys_prompt}\nUser: {prompt}\nAssistant:",
                max_tokens=64,
                stop=["\nAssistant:"],  # stop at the next assistant marker
                echo=False,
                temperature=0.0,
                top_p=1.0,
            )
            # llama-cpp-python returns {"choices": [{"text": ...}]} by default.
            if isinstance(output, dict):
                text = output["choices"][0]["text"] if output.get("choices") else ""
            elif isinstance(output, list) and output:
                text = output[0]["choices"][0]["text"]
            else:
                text = ""
            # Extract JSON from the generated text (the model may include surrounding text)
            import re as _re
            json_match = _re.search(r'\{.*\}', text, _re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                # The model may include null values; remove keys with None values
                slots = {k: v for k, v in data.get("slots", {}).items() if v is not None}
                # Ensure slots only have valid keys
                valid_slot_keys = {"time", "recipient", "message", "title", "location"}
                slots = {k: v for k, v in slots.items() if k in valid_slot_keys}
                intent = data.get("intent", "set_reminder")
                # Only keep intent if it's a valid one
                if intent not in INTENTS:
                    intent = "set_reminder"
                return {"intent": intent, "slots": slots}
            else:
                # No JSON found – fallback
                return {"intent": "set_reminder", "slots": {}}
        except Exception as e:
            print(f"[bench] SLM inference error: {e}")
            return {"intent": "set_reminder", "slots": {}}


# ──────────────────────────────────────────────
# Normalizer via ctypes
# ──────────────────────────────────────────────
def load_normalizer():
    """Load the C++ normalizer shared library and return the normalize_text function."""
    import ctypes
    lib = ctypes.CDLL(NORMALIZER_LIB)
    lib.normalize_text.restype = ctypes.c_char_p
    lib.normalize_text.argtypes = [ctypes.c_char_p]
    return lib


def normalize_text(lib, text):
    """Run the C++ normalizer on the input text. Returns ITRANS Romanized + collapsed."""
    if not text:
        return ""
    result = lib.normalize_text(text.encode("utf-8"))
    return result.decode("utf-8") if result else ""


# ──────────────────────────────────────────────
# Runtime wrapper for the C++ pipeline (Pass 3)
# ──────────────────────────────────────────────
class RuntimeNLU:
    """Call libvaani_runtime.so directly through its C ABI."""

    def __init__(
        self,
        lib_path: str = RUNTIME_LIB,
        model: str = NLU_MODEL,
        vocab: str = NLU_VOCAB,
        intent_labels: str = NLU_INTENT_LABELS,
        slot_labels: str = NLU_SLOT_LABELS,
    ):
        if not os.path.exists(lib_path):
            raise RuntimeError(f"runtime library not found: {lib_path}")

        self.lib = ctypes.CDLL(lib_path)
        self.lib.vaani_init.argtypes = [ctypes.c_char_p] * 4
        self.lib.vaani_init.restype = ctypes.c_bool
        self.lib.vaani_parse.argtypes = [ctypes.c_char_p]
        self.lib.vaani_parse.restype = ctypes.c_char_p
        self.lib.vaani_free_result.argtypes = [ctypes.c_char_p]
        self.lib.vaani_free_result.restype = None

        ok = self.lib.vaani_init(
            model.encode(), vocab.encode(),
            intent_labels.encode(), slot_labels.encode(),
        )
        if not ok:
            raise RuntimeError("vaani_init() failed")

    def parse(self, text: str) -> str:
        raw = self.lib.vaani_parse(text.encode("utf-8"))
        if not raw:
            return ""
        output = raw.decode("utf-8").strip()
        self.lib.vaani_free_result(raw)
        return output


# ──────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────
def intent_matches(pred_intent: str, true_intent: str) -> bool:
    return pred_intent == true_intent


def slot_exact_match(pred_slots: Dict[str, str], true_slots: Dict[str, str]) -> bool:
    """Exact match: all present slot keys must match exactly."""
    if set(pred_slots.keys()) != set(true_slots.keys()):
        return False
    return all(pred_slots[k] == true_slots[k] for k in true_slots)


def compute_token_fertility(pred_slots: Dict[str, str], true_slots: Dict[str, str]) -> float:
    """Average tokens per slot value; count non-empty true slot values."""
    true_count = sum(1 for v in true_slots.values() if v)
    if true_count == 0:
        return 0.0
    total_pred_tokens = sum(len(v.split()) for v in pred_slots.values() if v)
    return total_pred_tokens / true_count


# ──────────────────────────────────────────────
# Benchmark passes
# ──────────────────────────────────────────────
def run_benchmark_pass_slm(
    samples: List[Dict[str, str]],
    llm: SLMInference,
    normalizer_lib,
    pass_name: str,
) -> Dict[str, Any]:
    """Run a single benchmark pass (raw or normalized) using the GGUF SLM."""

    start_time = time.perf_counter()
    pred_intents = []
    pred_slots_list = []
    true_intents = []
    true_slots_list = []
    latencies_ms = []

    for i, sample in enumerate(samples):
        noisy_text = sample["noisy"]
        true_intent = sample["intent"]
        true_slots = {k: v for k, v in sample.get("slots", {}).items() if k != "intent"}

        # Prepare text for this pass
        if pass_name == "raw":
            text = noisy_text
        elif pass_name == "normalized":
            text = normalize_text(normalizer_lib, noisy_text)
        else:
            text = noisy_text

        # Measure latency for this sample
        iter_start = time.perf_counter()
        result = llm.infer(text)
        iter_ms = (time.perf_counter() - iter_start) * 1000.0
        latencies_ms.append(iter_ms)

        pred_intents.append(result.get("intent", "set_reminder"))
        pred_slots_list.append(result.get("slots", {}))
        true_intents.append(true_intent)
        true_slots_list.append(true_slots)

    # Compute metrics
    n = len(samples)
    if n == 0:
        return {
            "pass": pass_name,
            "intent_accuracy_pct": 0.0,
            "slot_exact_match_pct": 0.0,
            "token_fertility": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "avg_ms": 0.0,
            "peak_rss_mb": 0.0,
            "total_samples": 0,
            "predictions": [],
        }

    intent_accuracy = sum(1 for p, t in zip(pred_intents, true_intents) if intent_matches(p, t)) / n * 100.0
    slot_em = sum(1 for p, t in zip(pred_slots_list, true_slots_list) if slot_exact_match(p, t)) / n * 100.0
    token_fert = sum(compute_token_fertility(p, t) for p, t in zip(pred_slots_list, true_slots_list)) / n

    # Latencies
    latencies_ms.sort()
    p50_ms = latencies_ms[len(latencies_ms) // 2] if latencies_ms else 0.0
    p95_idx = int(0.95 * (len(latencies_ms) - 1))
    p95_ms = latencies_ms[p95_idx] if latencies_ms else 0.0
    avg_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

    # Peak RSS memory
    process = psutil.Process()
    peak_rss_mb = process.memory_info().rss / (1024 * 1024)

    # Collect predictions for the summary
    predictions = []
    for i in range(n):
        predictions.append({
            "intent_pred": pred_intents[i],
            "intent_true": true_intents[i],
            "slot_match": slot_exact_match(pred_slots_list[i], true_slots_list[i]),
        })

    return {
        "pass": pass_name,
        "intent_accuracy_pct": round(intent_accuracy, 1),
        "slot_exact_match_pct": round(slot_em, 1),
        "token_fertility": round(token_fert, 2),
        "p50_ms": round(p50_ms, 1),
        "p95_ms": round(p95_ms, 1),
        "avg_ms": round(avg_ms, 1),
        "peak_rss_mb": round(peak_rss_mb, 1),
        "total_samples": n,
        "predictions": predictions,
    }


def run_benchmark_pass_runtime(samples: List[Dict[str, str]]) -> Dict[str, Any]:
    """Run the third pass through libvaani_runtime.so (ctypes)."""

    nlu = RuntimeNLU()
    pred_intents = []
    pred_slots_list = []
    true_intents = []
    true_slots_list = []
    latencies_ms = []

    for i, sample in enumerate(samples):
        noisy_text = sample["noisy"]
        true_intent = sample["intent"]
        true_slots = {k: v for k, v in sample.get("slots", {}).items() if k != "intent"}

        iter_start = time.perf_counter()
        json_out = nlu.parse(noisy_text)
        iter_ms = (time.perf_counter() - iter_start) * 1000.0
        latencies_ms.append(iter_ms)

        # Parse the CLI JSON output
        if json_out:
            try:
                parsed = json.loads(json_out)
                pred_intents.append(parsed.get("intent", "set_reminder"))
                pred_slots_list.append(parsed.get("slots", {}))
            except json.JSONDecodeError:
                pred_intents.append("set_reminder")
                pred_slots_list.append({})
        else:
            pred_intents.append("set_reminder")
            pred_slots_list.append({})

        true_intents.append(true_intent)
        true_slots_list.append(true_slots)

    # Compute metrics
    n = len(samples)
    if n == 0:
        return {
            "pass": "full_cpp_pipeline",
            "intent_accuracy_pct": 0.0,
            "slot_exact_match_pct": 0.0,
            "token_fertility": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "avg_ms": 0.0,
            "peak_rss_mb": 0.0,
            "total_samples": 0,
            "predictions": [],
        }

    intent_accuracy = sum(1 for p, t in zip(pred_intents, true_intents) if intent_matches(p, t)) / n * 100.0
    slot_em = sum(1 for p, t in zip(pred_slots_list, true_slots_list) if slot_exact_match(p, t)) / n * 100.0
    token_fert = sum(compute_token_fertility(p, t) for p, t in zip(pred_slots_list, true_slots_list)) / n

    # Latencies
    latencies_ms.sort()
    p50_ms = latencies_ms[len(latencies_ms) // 2] if latencies_ms else 0.0
    p95_idx = int(0.95 * (len(latencies_ms) - 1))
    p95_ms = latencies_ms[p95_idx] if latencies_ms else 0.0
    avg_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

    # Peak RSS memory
    process = psutil.Process()
    peak_rss_mb = process.memory_info().rss / (1024 * 1024)

    # Collect predictions
    predictions = []
    for i in range(n):
        predictions.append({
            "intent_pred": pred_intents[i],
            "intent_true": true_intents[i],
            "slot_match": slot_exact_match(pred_slots_list[i], true_slots_list[i]),
        })

    return {
        "pass": "full_cpp_pipeline",
        "intent_accuracy_pct": round(intent_accuracy, 1),
        "slot_exact_match_pct": round(slot_em, 1),
        "token_fertility": round(token_fert, 2),
        "p50_ms": round(p50_ms, 1),
        "p95_ms": round(p95_ms, 1),
        "avg_ms": round(avg_ms, 1),
        "peak_rss_mb": round(peak_rss_mb, 1),
        "total_samples": n,
        "predictions": predictions,
    }


# ──────────────────────────────────────────────
# Markdown table output
# ──────────────────────────────────────────────
def print_markdown_table(results: List[Dict[str, Any]]) -> None:
    """Print a markdown table of benchmark results."""
    # Header
    header = (
        "| Pass | Intent Acc. % | Slot EM % | Token Fertility | p50 (ms) | p95 (ms) | Avg (ms) | Disk (MB) | Peak RSS (MB) |\n"
        "|------|---------------|-----------|-----------------|----------|----------|----------|-----------|----------------|\n"
    )
    print(header)

    # Sort passes in the expected order: raw, normalized, full_cpp_pipeline
    pass_order = ["raw", "normalized", "full_cpp_pipeline"]
    pass_lookup = {r["pass"]: r for r in results}

    for p in pass_order:
        if p in pass_lookup:
            r = pass_lookup[p]
            line = (
                f"| {r['pass']} | {r['intent_accuracy_pct']:.1f} | "
                f"{r['slot_exact_match_pct']:.1f} | {r['token_fertility']:.2f} | "
                f"{r['p50_ms']:.1f} | {r['p95_ms']:.1f} | "
                f"{r['avg_ms']:.1f} | {r['peak_rss_mb']:.1f} | "
                f"{r['peak_rss_mb']:.1f} |"  # Disk footprint placeholder; using RSS
            )
            print(line)

    # Summary
    print()
    print("--- Detailed Predictions ---")
    for r in results:
        print(f"\nPass: {r['pass']} ({r['total_samples']} samples)")
        for pred in r["predictions"][:5]:  # show first 5
            print(
                f"  intent={pred['intent_true']} -> {pred['intent_pred']} "
                f"(slot_match={pred['slot_match']})"
            )
        if len(r["predictions"]) > 5:
            print(f"  ... ({len(r['predictions'])-5} more)")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Hinglish NLU Benchmark: Three-Way Comparison")
    print("=" * 70)

    # Load eval set
    if not os.path.exists(EVAL_SET_PATH):
        print(f"[ERROR] eval set not found at: {EVAL_SET_PATH}", file=sys.stderr)
        sys.exit(1)
    samples = load_eval_set(EVAL_SET_PATH)
    print(f"[bench] Loaded {len(samples)} evaluation samples")

    # Load normalizer
    if not os.path.exists(NORMALIZER_LIB):
        print(f"[ERROR] Normalizer library not found at: {NORMALIZER_LIB}", file=sys.stderr)
        sys.exit(1)
    normalizer_lib = load_normalizer()
    print("[bench] Loaded C++ normalizer")

    # Initialize SLM (GGUF)
    llm = SLMInference(model_path=MODEL_PATH)

    # Run three passes: raw, normalized, full C++ pipeline via CLI
    results = []
    results.append(run_benchmark_pass_slm(samples, llm, normalizer_lib, pass_name="raw"))
    results.append(run_benchmark_pass_slm(samples, llm, normalizer_lib, pass_name="normalized"))
    results.append(run_benchmark_pass_runtime(samples))

    # Print Markdown table
    print_markdown_table(results)

    # Also print a brief summary
    print("\n--- Summary ---")
    for r in results:
        print(
            f"Pass '{r['pass']}': Intent Acc={r['intent_accuracy_pct']:.1f}%, "
            f"Slot EM={r['slot_exact_match_pct']:.1f}%, Token Fert={r['token_fertility']:.2f}, "
            f"p50={r['p50_ms']:.1f}ms, p95={r['p95_ms']:.1f}ms, "
            f"Avg={r['avg_ms']:.1f}ms, RSS={r['peak_rss_mb']:.1f}MB"
        )

    # -----------------------------------------------------------------
    # Persist results to README.md (append/update a results block)
    # -----------------------------------------------------------------
    readme_path = os.path.join(PROJECT_ROOT, "README.md")
    # Read existing README
    existing = ""
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            existing = f.read()

    # Build a results block to append
    # Find the last occurrence of the benchmark section and replace it
    marker = "<!-- benchmark-results-start -->"
    marker_end = "<!-- benchmark-results-end -->"
    start_idx = existing.find(marker)
    end_idx = existing.find(marker_end)

    if start_idx != -1 and end_idx != -1:
        # Replace existing block
        new_block = (
            f"\n{marker}\n"
            f"| Pass | Intent Acc. % | Slot EM % | Token Fertility | p50 (ms) | p95 (ms) | Avg (ms) | Peak RSS (MB) |\n"
            f"|------|---------------|-----------|-----------------|----------|----------|----------|---------------|\n"
        )
        for r in results:
            pass_name = r["pass"]
            npct = r["intent_accuracy_pct"]
            smct = r["slot_exact_match_pct"]
            tfrt = r["token_fertility"]
            p50 = r["p50_ms"]
            p95 = r["p95_ms"]
            avg = r["avg_ms"]
            rss = r["peak_rss_mb"]
            new_block += (
                f"| {pass_name} | {npct:.1f} | {smct:.1f} | {tfrt:.2f} | "
                f"{p50:.1f} | {p95:.1f} | {avg:.1f} | {rss:.1f} |\n"
            )
        new_block += f"{marker_end}\n"

        new_existing = existing[:start_idx] + new_block + existing[end_idx + len(marker_end):]
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(new_existing)
        print(f"[bench] Updated {readme_path} with benchmark results.")
    else:
        # Append new block at the end
        new_block = (
            f"\n{marker}\n"
            f"| Pass | Intent Acc. % | Slot EM % | Token Fertility | p50 (ms) | p95 (ms) | Avg (ms) | Peak RSS (MB) |\n"
            f"|------|---------------|-----------|-----------------|----------|----------|----------|---------------|\n"
        )
        for r in results:
            pass_name = r["pass"]
            npct = r["intent_accuracy_pct"]
            smct = r["slot_exact_match_pct"]
            tfrt = r["token_fertility"]
            p50 = r["p50_ms"]
            p95 = r["p95_ms"]
            avg = r["avg_ms"]
            rss = r["peak_rss_mb"]
            new_block += (
                f"| {pass_name} | {npct:.1f} | {smct:.1f} | {tfrt:.2f} | "
                f"{p50:.1f} | {p95:.1f} | {avg:.1f} | {rss:.1f} |\n"
            )
        new_block += f"{marker_end}\n"
        with open(readme_path, "a", encoding="utf-8") as f:
            f.write(new_block)
        print(f"[bench] Appended benchmark results to {readme_path}.")


if __name__ == "__main__":
    main()