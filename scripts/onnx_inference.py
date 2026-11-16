#!/usr/bin/env python3
"""ONNX inference backend used by libvaani_runtime.so.

Invoked as::

    python3 scripts/onnx_inference.py <input_ids> <model.onnx> [normalized_text]

``input_ids`` is a comma-separated list produced by the runtime tokenizer.
When ``normalized_text`` is supplied, BIO tags are merged back into slot values
using the original (normalized) words; otherwise slot keys are emitted with the
placeholder value ``"detected"``.

This subprocess path exists because the ONNX Runtime C/C++ headers are not
vendored in this repository. See README "Status and limitations".
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import onnxruntime as ort

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

from vaani_tokenizer import encode, load_vocab, tags_to_slots  # noqa: E402

DEFAULT_INTENTS = {
    0: "set_reminder",
    1: "send_message",
    2: "play_media",
    3: "check_weather",
    4: "set_alarm",
}
DEFAULT_SLOTS = {
    0: "O",
    1: "B-time",
    2: "I-time",
    3: "B-recipient",
    4: "I-recipient",
    5: "B-message",
    6: "I-message",
    7: "B-title",
    8: "I-title",
    9: "B-location",
    10: "I-location",
    11: "B-media_type",
    12: "I-media_type",
}


def _load_int_map(filename: str, fallback: dict) -> dict:
    try:
        with open(os.path.join(MODELS_DIR, filename), encoding="utf-8") as handle:
            return {int(key): value for key, value in json.load(handle).items()}
    except (OSError, ValueError, TypeError):
        return dict(fallback)


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"intent": "error", "slots": {}}))
        return 1

    input_ids = [int(value) for value in sys.argv[1].split(",") if value.strip()]
    model_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else os.environ.get("VAANI_MODEL_PATH", os.path.join(MODELS_DIR, "vaani_nlu.onnx"))
    )
    normalized_text = sys.argv[3] if len(sys.argv) > 3 else ""

    if not os.path.exists(model_path):
        print(json.dumps({"intent": "error", "slots": {}}))
        return 1

    vocab_path = os.path.join(MODELS_DIR, "vocab.txt")
    vocab = load_vocab(vocab_path) if os.path.exists(vocab_path) else {}

    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    except Exception:
        print(json.dumps({"intent": "error", "slots": {}}))
        return 1

    input_name = session.get_inputs()[0].name
    # The model is exported with a fixed 64-token sequence; right-pad with [PAD].
    pad_id = vocab.get("[PAD]", 0)
    padded = input_ids[:64] + [pad_id] * max(0, 64 - len(input_ids))
    outputs = session.run(None, {input_name: np.asarray([padded], dtype=np.int64)})
    input_ids = input_ids[:64]

    intent_map = _load_int_map("intent_labels.json", DEFAULT_INTENTS)
    slot_labels_map = _load_int_map("slot_labels.json", DEFAULT_SLOTS)

    intent_idx = int(np.argmax(outputs[0][0]))
    intent_name = intent_map.get(intent_idx, "set_reminder")

    token_tags = [
        slot_labels_map.get(int(index), "O")
        for index in np.argmax(outputs[1][0], axis=-1)
    ]

    slots = {}
    if normalized_text and vocab:
        ids, words, spans = encode(normalized_text, vocab, max_len=len(input_ids))
        if len(ids) == len(input_ids) and spans:
            word_tags = [
                token_tags[start] if start < len(token_tags) else "O"
                for start, _ in spans
            ]
            slots = tags_to_slots(words, word_tags)

    # If no text was supplied (or alignment failed), still report slot keys.
    if not slots:
        for tag in token_tags:
            if tag != "O" and "-" in tag:
                slots[tag.split("-", 1)[1]] = "detected"

    print(json.dumps({"intent": intent_name, "slots": slots}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
