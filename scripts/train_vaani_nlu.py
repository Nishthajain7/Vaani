#!/usr/bin/env python3
"""Train a compact intent + BIO slot model for Vaani NLU.

The exported ONNX model matches the runtime contract:

    input  : input_ids            int64  [batch, sequence]
    output : intent_logits        float  [batch, 5]
             slot_logits          float  [batch, sequence, 13]

It uses the word/character vocabulary produced by ``scripts/vaani_tokenizer.py``
and the C++ normalizer in ``build/libnormalizer.so`` so training and inference
see the same text.

Usage:
    python3 scripts/train_vaani_nlu.py [--samples 6000] [--epochs 15]

Outputs:
    models/vaani_nlu.onnx
    models/vocab.txt
    models/intent_labels.json
    models/slot_labels.json
    data/train_nlu.jsonl
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import re
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vaani_tokenizer import (
    CLS_TOKEN,
    SEP_TOKEN,
    build_vocab,
    encode,
    load_vocab,
    save_vocab,
    split_words,
    tags_to_slots,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
NORMALIZER_LIB = os.path.join(PROJECT_ROOT, "build", "libnormalizer.so")

MODEL_PATH = os.path.join(MODELS_DIR, "vaani_nlu.onnx")
VOCAB_PATH = os.path.join(MODELS_DIR, "vocab.txt")
INTENT_LABELS_PATH = os.path.join(MODELS_DIR, "intent_labels.json")
SLOT_LABELS_PATH = os.path.join(MODELS_DIR, "slot_labels.json")
TRAIN_PATH = os.path.join(DATA_DIR, "train_nlu.jsonl")

INTENT_LABELS = [
    "set_reminder",
    "send_message",
    "play_media",
    "check_weather",
    "set_alarm",
]

SLOT_LABELS = [
    "O",
    "B-time", "I-time",
    "B-recipient", "I-recipient",
    "B-message", "I-message",
    "B-title", "I-title",
    "B-location", "I-location",
    "B-media_type", "I-media_type",
]
SLOT_TO_ID = {label: index for index, label in enumerate(SLOT_LABELS)}

MAX_LEN = 64
SEED = 42

# ──────────────────────────────────────────────
# TOP intent mapping (from hinglish_top TSV parse)
# ──────────────────────────────────────────────
TOP_TO_VAANI_INTENT = {
    "CREATE_ALARM": "set_alarm",
    "CREATE_REMINDER": "set_reminder",
    "SEND_MESSAGE": "send_message",
    "PLAY_MUSIC": "play_media",
    "GET_WEATHER": "check_weather",
}

# ──────────────────────────────────────────────
# Slot-type mapping (TOP -> Vaani slot label)
# Only map types that directly correspond to Vaani's 13 slot labels.
# Unmapped types are ignored (emit 'O' tags).
# ──────────────────────────────────────────────
SLOT_MAP = {
    # Time-related
    "DATE_TIME": "time",
    "PERIOD": "time",
    "ALARM_NAME": "time",
    "DATE_TIME_ARRIVAL": "time",
    "DATE_TIME_DEPARTURE": "time",
    "DATE_TIME_NEW": "time",
    "DATE_TIME_RECURRING": "time",
    # Recipient/sender
    "RECIPIENT": "recipient",
    "CONTACT": "recipient",
    "SENDER": "recipient",
    # Message content
    "CONTENT_EXACT": "message",
    # Music slots
    "MUSIC_TRACK_TITLE": "title",
    "MUSIC_ALBUM_TITLE": "title",
    "MUSIC_ARTIST_NAME": "title",
    "MUSIC_GENRE": "title",
    "MUSIC_TYPE": "media_type",
    # Location
    "LOCATION": "location",
    "LOCATION_USER": "location",
    "LOCATION_MODIFIER": "location",
    "LOCATION_CURRENT": "location",
    "LOCATION_WORK": "location",
    # TODO types (not in Vaani slot labels -> emitted as 'O')
    "TODO": None,
    "REMINDER_TODO": None,
}

# ──────────────────────────────────────────────
# Synthetic data with span-level BIO labels
# ──────────────────────────────────────────────
TEMPLATES: Dict[str, List[str]] = {
    "set_reminder": [
        "set a reminder for {time} to {message}",
        "remind me to {message} at {time}",
        "set reminder at {time} for {message}",
        "wake me at {time} for {message}",
        "remind me at {time} to {message}",
    ],
    "send_message": [
        "send a message to {recipient} saying {message}",
        "text {recipient} saying {message}",
        "whatsapp {recipient} that {message}",
        "send message to {recipient} {message}",
        "message {recipient} {message}",
    ],
    "play_media": [
        "play some {media_type} {title}",
        "play {title} {media_type}",
        "listen to {title} on {media_type}",
        "play {media_type} {title}",
        "listen to {media_type} {title}",
    ],
    "check_weather": [
        "whats the weather in {location}",
        "forecast for {location} today",
        "hows the weather in {location}",
        "weather in {location} this weekend",
        "what will be the temperature in {location}",
    ],
    "set_alarm": [
        "set alarm at {time}",
        "wake me up at {time}",
        "set an alarm for {time}",
        "wake me at {time}",
        "set wake alarm at {time}",
    ],
}

SLOT_POOLS: Dict[str, Dict[str, List[str]]] = {
    "set_reminder": {
        "time": ["8 AM", "9 PM", "6:30 AM", "11 PM", "7 PM", "10 AM", "5:45 PM", "6 AM", "midnight", "noon"],
        "message": ["call mom", "send money", "finish project", "meet at cafe", "take out trash", "water plants", "call doctor", "pay bill", "send documents"],
    },
    "send_message": {
        "recipient": ["anita", "vikram", "sonu", "priya", "raj", "simran", "karan", "neha", "ajay", "pooja", "mom", "alex"],
        "message": ["call mom", "send money", "finish project", "water plants", "take out trash", "buy groceries", "call doctor", "pay bill", "send documents", "im running late", "i am late"],
    },
    "play_media": {
        "media_type": ["song", "music", "playlist", "album", "video"],
        "title": ["despacito", "hape shayad", "bollywood beats", "punjabi song", "old is gold", "romantic track", "dance number", "romantic song"],
    },
    "check_weather": {
        "location": ["delhi", "mumbai", "bangalore", "chennai", "kolkata", "hyderabad", "pune", "jaipur", "lucknow"],
    },
    "set_alarm": {
        "time": ["8 AM", "9 PM", "6:30 AM", "11 PM", "7 PM", "10 AM", "5:45 PM", "6 AM", "midnight", "noon"],
    },
}

FILLERS = ["ya", "sab", "kal", "bahut", "accha", "zara", "please", "thoda", "अभी", "जल्दी"]

DEVANAGARI_INTENT_WORDS = {
    "message": "संदेश",
    "reminder": "याद",
    "alarm": "अलार्म",
    "weather": "मौसम",
    "song": "गाना",
    "music": "संगीत",
    "send": "भेजो",
    "play": "चलाओ",
    "set": "लगाओ",
    "text": "लिखो",
    "whatsapp": "व्हाट्सएप",
    "forecast": "पूर्वानुमान",
    "wake": "जगाओ",
    "listen": "सुनो",
}

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _template_segments(template: str) -> List[Tuple[List[str], str | None]]:
    """Split a template into (words, slot_key) segments."""
    segments: List[Tuple[List[str], str | None]] = []
    cursor = 0
    for match in _PLACEHOLDER.finditer(template):
        literal = template[cursor:match.start()].strip()
        if literal:
            segments.append((split_words(literal), None))
        segments.append(([], match.group(1)))
        cursor = match.end()
    tail = template[cursor:].strip()
    if tail:
        segments.append((split_words(tail), None))
    return segments


def _noise_word(word: str, rng: random.Random) -> str:
    """Apply phonetic doubling, consonant deletion, vowel substitution, or Devanagari mixing."""
    # Devanagari mixing: 5% chance replace with Devanagari equivalent
    if rng.random() < 0.05 and word.isascii():
        # Pick a random Devanagari character approximating the first letter
        dev_map = {
            "a": "अ", "aa": "आ", "i": "इ", "ii": "ई", "u": "उ", "uu": "ऊ",
            "k": "क", "kh": "ख", "g": "ग", "gh": "घ", "ch": "च", "chh": "छ",
            "j": "ज", "jh": "झ", "ny": "ञ", "t": "ट", "th": "ठ", "d": "ड", "dh": "ढ",
            "n": "न", "p": "प", "ph": "फ", "b": "ब", "bh": "भ", "m": "म",
            "y": "य", "r": "र", "l": "ल", "v": "व", "sh": "श", "s": "स", "h": "ह",
        }
        first = word[0].lower()
        dev = dev_map.get(first, "")
        if dev:
            return dev + word[1:]
    # Phonetic doubling: 20% chance double a consonant inside the word
    roll = rng.random()
    if roll < 0.20 and len(word) > 3:
        index = rng.randrange(1, len(word) - 1)
        return word[:index] + word[index] + word[index:]
    # Consonant deletion: 15% chance delete a consonant inside the word
    if roll < 0.35 and len(word) > 4:
        index = rng.randrange(1, len(word) - 1)
        return word[:index] + word[index + 1:]
    # Vowel substitution: 10% chance replace first 'e' with 'a'
    if roll < 0.45:
        return word.replace("e", "a", 1)
    return word


def _noise_text(text: str, rng: random.Random) -> str:
    """Apply word-level noise to a full utterance, then normalise via C++ normaliser."""
    words = split_words(text)
    noisy_words = [_noise_word(w, rng) for w in words]
    # Join and normalise through the C++ libnormalizer so alignment is preserved
    joined = " ".join(noisy_words)
    # The normaliser expects UTF-8; we pass it through ctypes
    try:
        import ctypes as _ctypes
        lib = _ctypes.CDLL(os.path.join(PROJECT_ROOT, "build", "libnormalizer.so"))
        lib.normalize_text.argtypes = [ctypes.c_char_p]
        lib.normalize_text.restype = ctypes.c_char_p
        result = lib.normalize_text(joined.encode("utf-8"))
        return result.decode("utf-8") if result else joined
    except Exception:
        return joined


def make_sample(intent: str, rng: random.Random) -> Dict[str, object]:
    template = rng.choice(TEMPLATES[intent])
    values = {
        key: rng.choice(pool)
        for key, pool in SLOT_POOLS[intent].items()
    }

    words: List[str] = []
    tags: List[str] = []
    clean_words: List[str] = []
    clean_tags: List[str] = []

    for literal_words, slot_key in _template_segments(template):
        if slot_key is None:
            for word in literal_words:
                clean_words.append(word)
                clean_tags.append("O")
                words.append(_noise_word(word, rng))
                tags.append("O")
            continue

        value_words = split_words(values[slot_key])
        for index, word in enumerate(value_words):
            tag = ("B-" if index == 0 else "I-") + slot_key
            clean_words.append(word)
            clean_tags.append(tag)
            words.append(word)  # keep slot values exact
            tags.append(tag)

    # Insert fillers only at positions that do not split a slot span.
    for _ in range(rng.randint(0, 2)):
        candidates = [
            index for index in range(len(words) + 1)
            if (index == 0 or tags[index - 1] == "O")
            and (index == len(words) or tags[index] == "O")
        ]
        if not candidates:
            break
        position = rng.choice(candidates)
        words.insert(position, rng.choice(FILLERS))
        tags.insert(position, "O")

    return {
        "intent": intent,
        "words": words,
        "tags": tags,
        "clean_words": clean_words,
        "clean_tags": clean_tags,
    }


def build_dataset(num_samples: int, seed: int = SEED) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    samples = [make_sample(rng.choice(INTENT_LABELS), rng) for _ in range(num_samples)]
    return samples


# ──────────────────────────────────────────────
# Normalizer (C++ via ctypes)
# ──────────────────────────────────────────────
class Normalizer:
    def __init__(self, lib_path: str = NORMALIZER_LIB):
        if not os.path.exists(lib_path):
            raise SystemExit(
                f"normalizer library not found: {lib_path}\n"
                "Build it first: cmake --build build -j"
            )
        self.lib = ctypes.CDLL(lib_path)
        self.lib.normalize_text.argtypes = [ctypes.c_char_p]
        self.lib.normalize_text.restype = ctypes.c_char_p

    def __call__(self, text: str) -> str:
        result = self.lib.normalize_text(text.encode("utf-8"))
        return result.decode("utf-8") if result else ""


def normalize_words(normalizer: Normalizer, words: Sequence[str]) -> List[str]:
    """Normalize word by word so BIO alignment is preserved exactly."""
    out = []
    for word in words:
        normalized = normalizer(word).strip()
        out.append(normalized if normalized else word)
    return out


# ──────────────────────────────────────────────
# Hybrid data loading & ASR noise generation
# ──────────────────────────────────────────────
def load_hinglish_top_split(tsv_path: str) -> List[Dict[str, object]]:
    """Load a hinglish_top TSV split and convert to Vaani JSONL-like dicts."""
    records = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = __import__("csv").DictReader(f, delimiter="\t")
        for row in reader:
            en_query = row.get("en_query", "").strip()
            cs_query = row.get("cs_query", "").strip()
            text = cs_query if cs_query else en_query
            if not text:
                continue
            # Extract intent from TOP parse (prefer en_parse)
            intent = None
            for parse_col in ("en_parse", "cs_parse"):
                parse_text = row.get(parse_col, "")
                m = __import__("re").search(r"IN:(\w+)", parse_text)
                if m:
                    top_intent = m.group(1)
                    intent = TOP_TO_VAANI_INTENT.get(top_intent)
                    if intent is None:
                        continue  # unsupported intent -> skip
                    break
            if intent is None:
                continue

            # Extract slot spans from the parse column
            slot_text = row.get("en_parse", "") or row.get("cs_parse", "")
            slot_pattern = __import__("re").compile(r"\[SL:(\w+)\s+((?:\[[^\]]*\]|[^\[\]])*)\]")
            raw_slots = slot_pattern.findall(slot_text)

            # Build vaani slot label -> list of value strings
            slot_type_to_values: Dict[str, List[str]] = {}
            for top_type, value_str in raw_slots:
                vaani_label = SLOT_MAP.get(top_type)
                if vaani_label is None:
                    continue
                value_str = value_str.strip()
                if value_str:
                    slot_type_to_values.setdefault(vaani_label, []).append(value_str)

            # Build slots dict (first value per label)
            slots = {}
            for vaani_label, values in slot_type_to_values.items():
                if values:
                    slots[vaani_label] = values[0]

            records.append({
                "text": text,
                "intent": intent,
                "slots": slots,
            })
    return records


def augment_with_asr_noise(
    records: List[Dict[str, object]],
    normalizer: Normalizer,
    rng: random.Random,
) -> List[Dict[str, object]]:
    """Create ASR-noised copies of each record, applying phonetic doubling,
    split compound words, and random Devanagari script mixing."""
    noised = []
    for rec in records:
        # Normalise the original text first
        norm_text = normalizer(rec["text"])
        # Split into words after normalisation
        norm_words = norm_text.split()
        # Apply per-word noise
        noisy_words = [_noise_word(w, rng) for w in norm_words]
        # Occasionally split compound words: with 30% probability, split a
        # multi-word token at a natural boundary (e.g. "hwaiiime" -> "hwaii" + "me")
        if rng.random() < 0.30 and len(noisy_words) > 1:
            split_idx = rng.randrange(1, len(noisy_words))
            # Move the split point and add a space
            left = " ".join(noisy_words[:split_idx])
            right = " ".join(noisy_words[split_idx:])
            noisy_text = left + " " + right
        else:
            noisy_text = " ".join(noisy_words)
        # Re-normalise after noise so BIO alignment holds
        norm_noisy = normalizer(noisy_text)
        noised.append({
            "text": norm_noisy,
            "intent": rec["intent"],
            "slots": rec["slots"],
        })
    return noised


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
class TinyNLU(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_intents: int,
        num_slot_labels: int,
        hidden: int = 128,
        layers: int = 2,
        heads: int = 4,
        max_len: int = MAX_LEN,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.position = nn.Parameter(torch.zeros(1, max_len, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.intent_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_intents),
        )
        self.slot_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_slot_labels),
        )
        nn.init.normal_(self.position, std=0.02)

    def forward(self, input_ids: torch.Tensor):
        # input_ids: [B, T]; id 0 is [PAD]
        mask = input_ids.ne(0)
        length = input_ids.size(1)
        hidden = self.embed(input_ids) + self.position[:, :length, :]
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        intent_logits = self.intent_head(hidden[:, 0, :])
        slot_logits = self.slot_head(hidden)
        return intent_logits, slot_logits


class NLUDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, object]], vocab: Dict[str, int]):
        self.samples = []
        for row in rows:
            text = " ".join(row["words"])  # type: ignore[arg-type]
            ids, words, spans = encode(text, vocab, max_len=MAX_LEN)
            if not words:
                continue
            tags = row["tags"]
            word_tags = []
            for index in range(len(words)):
                if index < len(tags):
                    word_tags.append(tags[index])  # type: ignore[index]
                else:
                    word_tags.append("O")

            slot_ids = [-100] * len(ids)
            for word_index, (start, end) in enumerate(spans):
                for token_index in range(start, end):
                    slot_ids[token_index] = SLOT_TO_ID.get(word_tags[word_index], 0)

            self.samples.append(
                {
                    "input_ids": ids,
                    "slot_ids": slot_ids,
                    "intent_id": INTENT_LABELS.index(row["intent"]),  # type: ignore[arg-type]
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def collate(batch):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    slot_ids = torch.full((len(batch), max_len), -100, dtype=torch.long)
    intent_ids = torch.zeros(len(batch), dtype=torch.long)
    for row, item in enumerate(batch):
        length = len(item["input_ids"])
        input_ids[row, :length] = torch.tensor(item["input_ids"])
        slot_ids[row, :length] = torch.tensor(item["slot_ids"])
        intent_ids[row] = item["intent_id"]
    return input_ids, slot_ids, intent_ids


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
def evaluate_onnx(model_path: str, vocab: Dict[str, int], normalizer: Normalizer, data_path: str):
    import onnxruntime as ort

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def predict(text: str):
        ids, words, spans = encode(text, vocab, max_len=MAX_LEN)
        ids = ids[:MAX_LEN]
        padded = ids + [0] * (MAX_LEN - len(ids))
        array = np.asarray([padded], dtype=np.int64)
        intent_logits, slot_logits = session.run(None, {"input_ids": array})
        intent = INTENT_LABELS[int(np.argmax(intent_logits[0]))]
        token_tags = [
            SLOT_LABELS[int(index)]
            for index in np.argmax(slot_logits[0][: len(ids)], axis=-1)
        ]
        word_tags = [token_tags[start] if start < len(token_tags) else "O" for start, _ in spans]
        return intent, tags_to_slots(words, word_tags)

    rows = []
    with open(data_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    intent_ok = 0
    slot_ok = 0
    for row in rows:
        text = normalizer(row["noisy"])
        intent, slots = predict(text)
        truth = {key: value for key, value in row["slots"].items() if key != "intent"}
        intent_ok += int(intent == row["slots"]["intent"])
        slot_ok += int(slots == truth)

    total = len(rows)
    return intent_ok / total * 100.0, slot_ok / total * 100.0


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    normalizer = Normalizer()
    rng = random.Random(SEED)

    # -----------------------------------------------------------------
    # 1. Load existing hinglish_top splits (train + val)
    # -----------------------------------------------------------------
    hinglish_train = load_hinglish_top_split(
        os.path.join(DATA_DIR, "raw", "hinglish_top", "train.tsv")
    )
    hinglish_val = load_hinglish_top_split(
        os.path.join(DATA_DIR, "raw", "hinglish_top", "validation.tsv")
    )
    print(f"Loaded {len(hinglish_train)} hinglish_top train records, "
          f"{len(hinglish_val)} val records")

    # -----------------------------------------------------------------
    # 2. Create ASR-noised copies of the hinglish_top data (equal share)
    # -----------------------------------------------------------------
    noised_train = augment_with_asr_noise(hinglish_train, normalizer, rng)
    noised_val = augment_with_asr_noise(hinglish_val, normalizer, rng)
    print(f"Created {len(noised_train)} ASR-noised train records, "
          f"{len(noised_val)} val records")

    # -----------------------------------------------------------------
    # 3. Generate synthetic template samples (the original behaviour)
    # -----------------------------------------------------------------
    synthetic_samples = build_dataset(args.samples)
    print(f"Generated {len(synthetic_samples)} synthetic samples")

    # -----------------------------------------------------------------
    # 4. Combine all data: equal shares from each source
    # -----------------------------------------------------------------
    # We want equal shares: synthetic + hinglish_top_original + hinglish_top_noised
    # but adjusted so total == args.samples (approximately).
    # We'll take args.samples // 3 from each, then any remainder from synthetic.
    third = args.samples // 3
    rem = args.samples - 3 * third

    # Take 'rem' extra from synthetic (capped by available synthetic)
    synth_for_training = synthetic_samples[:third + rem]
    hinglish_original = hinglish_train[:third]
    hinglish_noised = noised_train[:third]

    # Also include the val portion of hinglish_top
    hinglish_val_original = hinglish_val[:third]
    hinglish_val_noised = noised_val[:third]

    all_train_rows = synth_for_training + hinglish_original + hinglish_noised
    all_val_rows = hinglish_val_original + hinglish_val_noised  # val set smaller

    # Balance: if we have too many, trim; too few, generate more synthetic
    if len(all_train_rows) < args.samples:
        extra_needed = args.samples - len(all_train_rows)
        extra_synth = build_dataset(extra_needed)
        all_train_rows += extra_synth
        print(f"Added {extra_needed} extra synthetic samples to reach target.")

    print(f"Training on {len(all_train_rows)} samples:")
    print(f"  synthetic: {len(synth_for_training)}")
    print(f"  hinglish_top original: {len(hinglish_original)}")
    print(f"  hinglish_top ASR-noised: {len(hinglish_noised)}")

    # -----------------------------------------------------------------
    # 4. Combine all data: equal shares from each source
    # -----------------------------------------------------------------
    # We want equal shares: synthetic + hinglish_top_original + hinglish_top_noised
    # but adjusted so total == args.samples (approximately).
    # We'll take args.samples // 3 from each, then any remainder from synthetic.
    third = args.samples // 3
    rem = args.samples - 3 * third

    # Take 'rem' extra from synthetic (capped by available synthetic)
    synth_for_training = synthetic_samples[:third + rem]

    # Load hinglish_top data (original, not yet noised)
    hinglish_original = load_hinglish_top_split(
        os.path.join(DATA_DIR, "raw", "hinglish_top", "train.tsv")
    )[:third]

    # Create ASR-noised copies of the hinglish train data
    noised_train = augment_with_asr_noise(hinglish_original, normalizer, rng)
    hinglish_noised = noised_train[:third]

    # Also include the val portion of hinglish_top
    hinglish_val_original = load_hinglish_top_split(
        os.path.join(DATA_DIR, "raw", "hinglish_top", "validation.tsv")
    )[:third]
    hinglish_val_noised = augment_with_asr_noise(
        hinglish_val_original, normalizer, rng
    )[:third]

    all_train_rows_raw = synth_for_training + hinglish_original + hinglish_noised
    all_val_rows_raw = hinglish_val_original + hinglish_val_noised  # val set smaller

    # Balance: if we have too many, trim; too few, generate more synthetic
    if len(all_train_rows_raw) < args.samples:
        extra_needed = args.samples - len(all_train_rows_raw)
        extra_synth = build_dataset(extra_needed)
        all_train_rows_raw += extra_synth
        print(f"Added {extra_needed} extra synthetic samples to reach target.")

    print(f"Training on {len(all_train_rows_raw)} samples:")
    print(f"  synthetic: {len(synth_for_training)}")
    print(f"  hinglish_top original: {len(hinglish_original)}")
    print(f"  hinglish_top ASR-noised: {len(hinglish_noised)}")

    # -----------------------------------------------------------------
    # 5. Normalise words and build alignment‑guarded rows
    # -----------------------------------------------------------------
    rows = []
    skipped = 0
    for sample in all_train_rows_raw:
        if "words" in sample and "tags" in sample:
            # Synthetic template data format (from make_sample)
            raw_words = sample["words"]
            tags = sample["tags"]
        elif "text" in sample and "intent" in sample:
            # Hinglish TOP data format (from load_hinglish_top_split)
            text = sample["text"]
            intent = sample["intent"]
            slots = sample.get("slots", {})
            # Tokenise the text using the same splitter
            raw_words = split_words(text)
            # Create BIO tags based on slot values
            tags = ["O"] * len(raw_words)
            # Build a mapping of lower-cased word -> list of positions
            lower_to_positions: Dict[str, List[int]] = {}
            for i, word in enumerate(raw_words):
                lw = word.lower()
                lower_to_positions.setdefault(lw, []).append(i)

            # For each slot label and its values, mark B-/I- tags
            for label, values in slots.items():
                if label == "task":
                    continue  # task not in Vaani slot labels
                for val in values:
                    val_lower = val.lower()
                    # Find all token positions whose lower-cased text matches val_lower
                    for pos in lower_to_positions.get(val_lower, []):
                        # Only set tag if currently 'O'
                        if tags[pos] == "O":
                            tags[pos] = f"B-{label}"
                        # Check subsequent tokens for I-<label>
                        i = pos + 1
                        while i < len(raw_words):
                            if raw_words[i].lower() == val_lower.split()[0] if " " in val else raw_words[i].lower() == val_lower:
                                if tags[i] == "O":
                                    tags[i] = f"I-{label}"
                                else:
                                    break  # stop at first non-matching or already-tagged token
                            i += 1
        else:
            continue  # unknown format, skip

        # Normalise words through the C++ normaliser so training sees normalised text
        norm_words = normalize_words(normalizer, raw_words)  # type: ignore[arg-type]
        text = " ".join(norm_words)
        # Alignment guard: the tokenizer must see exactly the words we tagged.
        expected = [word for word in norm_words if word]
        if split_words(text) != expected:
            skipped += 1
            continue
        rows.append(
            {
                "text": text,
                "intent": sample["intent"],
                "tags": tags,  # type: ignore[index]
                "words": norm_words,
            }
        )

    print(f"Kept {len(rows)} samples ({skipped} skipped for alignment)")

    # -----------------------------------------------------------------
    # 6. Write the combined train file
    # -----------------------------------------------------------------
    with open(TRAIN_PATH, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # -----------------------------------------------------------------
    # 7. Build vocabulary from the combined data
    # -----------------------------------------------------------------
    split = int(len(rows) * 0.9)
    train_rows, val_rows = rows[:split], rows[split:]

    print("Building vocabulary...")
    vocab_tokens = build_vocab([row["text"] for row in rows])
    vocab = {token: index for index, token in enumerate(vocab_tokens)}
    save_vocab(vocab_tokens, VOCAB_PATH)
    print(f"Vocabulary size: {len(vocab_tokens)}")

    with open(INTENT_LABELS_PATH, "w", encoding="utf-8") as handle:
        json.dump({str(i): label for i, label in enumerate(INTENT_LABELS)}, handle, indent=2)
    with open(SLOT_LABELS_PATH, "w", encoding="utf-8") as handle:
        json.dump({str(i): label for i, label in enumerate(SLOT_LABELS)}, handle, indent=2)

    # -----------------------------------------------------------------
    # 8. Create Datasets and DataLoaders
    # -----------------------------------------------------------------
    train_dataset = NLUDataset(train_rows, vocab)
    val_dataset = NLUDataset(val_rows, vocab)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # -----------------------------------------------------------------
    # 9. Train the model
    # -----------------------------------------------------------------
    model = TinyNLU(len(vocab_tokens), len(INTENT_LABELS), len(SLOT_LABELS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    intent_loss_fn = nn.CrossEntropyLoss()
    slot_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    print(f"Training {sum(p.numel() for p in model.parameters()):,} parameters...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for input_ids, slot_ids, intent_ids in train_loader:
            optimizer.zero_grad()
            intent_logits, slot_logits = model(input_ids)
            loss = intent_loss_fn(intent_logits, intent_ids) + 0.5 * slot_loss_fn(
                slot_logits.reshape(-1, len(SLOT_LABELS)), slot_ids.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss)
            correct += int((intent_logits.argmax(-1) == intent_ids).sum())
            seen += int(intent_ids.numel())

        model.eval()
        val_correct = 0
        val_seen = 0
        with torch.no_grad():
            for input_ids, _, intent_ids in val_loader:
                logits, _ = model(input_ids)
                val_correct += int((logits.argmax(-1) == intent_ids).sum())
                val_seen += int(intent_ids.numel())

        print(
            f"epoch {epoch:2d}/{args.epochs}  loss={total_loss / max(len(train_loader), 1):.4f}  "
            f"train_intent={correct / max(seen, 1):.3f}  val_intent={val_correct / max(val_seen, 1):.3f}"
        )

    # -----------------------------------------------------------------
    # 10. Export ONNX model
    # -----------------------------------------------------------------
    model.eval()
    dummy = torch.zeros(1, MAX_LEN, dtype=torch.long)
    dummy[0, 0] = vocab[CLS_TOKEN]
    dummy[0, 1] = vocab.get("play", 1)
    dummy[0, 2] = vocab.get("music", 1)
    dummy[0, 3] = vocab[SEP_TOKEN]
    torch.onnx.export(
        model,
        (dummy,),
        MODEL_PATH,
        input_names=["input_ids"],
        output_names=["intent_logits", "slot_logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "intent_logits": {0: "batch"},
            "slot_logits": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Exported ONNX model: {MODEL_PATH} ({os.path.getsize(MODEL_PATH) / 1e6:.2f} MB)")

    # -----------------------------------------------------------------
    # 11. Evaluate on (1) data/eval_set.jsonl (noisy ASR test set)
    # -----------------------------------------------------------------
    intent_acc_1, slot_em_1 = evaluate_onnx(MODEL_PATH, vocab, normalizer, os.path.join(DATA_DIR, "eval_set.jsonl"))
    print(f"\nEval on data/eval_set.jsonl (noisy ASR test set):")
    print(f"  intent accuracy : {intent_acc_1:.1f}%")
    print(f"  slot exact match: {slot_em_1:.1f}%")

    # -----------------------------------------------------------------
    # 12. Evaluate on (2) data/hinglish_top_test.jsonl (human-authored test split)
    # -----------------------------------------------------------------
    # For hinglish_top_test, we need to map its format (which already has intent
    # and slots in Vaani-style JSONL) to the evaluate_onnx function.
    # The function expects rows with "noisy" key; hinglish_top_test.jsonl has
    # "text" key. We'll normalise the text and treat it as the "noisy" field.
    intent_acc_2, slot_em_2 = evaluate_onnx_with_text(
        MODEL_PATH, vocab, normalizer, os.path.join(DATA_DIR, "hinglish_top_test.jsonl")
    )
    print(f"\nEval on data/hinglish_top_test.jsonl (human test split):")
    print(f"  intent accuracy : {intent_acc_2:.1f}%")
    print(f"  slot exact match: {slot_em_2:.1f}%")

    # -----------------------------------------------------------------
    # 13. Quick smoke test with the new tokenizer.
    # -----------------------------------------------------------------
    ids, words, spans = encode(normalizer("set a reminder at 8 AM"), vocab)
    print(f"\nTokenizer smoke test: {list(zip(words, [ids[start] for start, _ in spans]))}")


def evaluate_onnx_with_text(model_path: str, vocab: Dict[str, int], normalizer: Normalizer, data_path: str):
    """Evaluate ONNX model on a JSONL file where each line has 'text' and 'intent'/'slots' keys."""
    import onnxruntime as ort

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def predict(text: str):
        ids, words, spans = encode(text, vocab, max_len=MAX_LEN)
        ids = ids[:MAX_LEN]
        padded = ids + [0] * (MAX_LEN - len(ids))
        array = np.asarray([padded], dtype=np.int64)
        intent_logits, slot_logits = session.run(None, {"input_ids": array})
        intent = INTENT_LABELS[int(np.argmax(intent_logits[0]))]
        token_tags = [
            SLOT_LABELS[int(index)]
            for index in np.argmax(slot_logits[0][: len(ids)], axis=-1)
        ]
        word_tags = [token_tags[start] if start < len(token_tags) else "O" for start, _ in spans]
        return intent, tags_to_slots(words, word_tags)

    rows = []
    with open(data_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    intent_ok = 0
    slot_ok = 0
    for row in rows:
        text = normalizer(row["text"])
        intent, slots = predict(text)
        # row has "intent" and "slots" directly
        truth_intent = row.get("intent", "")
        truth_slots = {k: v for k, v in row.get("slots", {}).items() if k != "intent"}
        intent_ok += int(intent == truth_intent)
        slot_ok += int(slots == truth_slots)

    total = len(rows)
    return intent_ok / total * 100.0, slot_ok / total * 100.0


if __name__ == "__main__":
    main()