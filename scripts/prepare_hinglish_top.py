#!/usr/bin/env python3
"""
Prepare Hinglish TOP dataset for Vaani.

Reads data/raw/hinglish_top/{train,validation,test}.tsv and converts
matching records into Vaani's JSONL format:

    {"text": ..., "intent": ..., "slots": ..., "tokens": ..., "bio_tags": ...}

Mapping rules:
  * TOP intents -> Vaani's 5 intents (IN:CREATE_ALARM -> set_alarm,
    IN:CREATE_REMINDER -> set_reminder, IN:SEND_MESSAGE -> send_message,
    IN:PLAY_MUSIC -> play_media, IN:GET_WEATHER -> check_weather).
    Unsupported intents are skipped.
  * TOP slot-type -> Vaani slot label mapping (see SLOT_MAP).
    Leaf slot spans are turned into B-/I- tags aligned with word-level tokens.
  * Unmapped TOP slot tags are collected and printed at the end.

Output files written to data/:
    hinglish_top_train.jsonl
    hinglish_top_val.jsonl
    hinglish_top_test.jsonl
"""

import csv
import json
import os
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
RAW_TOP_DIR = os.path.join(DATA_ROOT, "raw", "hinglish_top")
OUT_DIR = DATA_ROOT

# ---------------------------------------------------------------------------
# Vaani intent mapping (TOP -> Vaani)
# ---------------------------------------------------------------------------
TOP_TO_VAANI_INTENT = {
    "CREATE_ALARM": "set_alarm",
    "CREATE_REMINDER": "set_reminder",
    "SEND_MESSAGE": "send_message",
    "PLAY_MUSIC": "play_media",
    "GET_WEATHER": "check_weather",
}

# ---------------------------------------------------------------------------
# Vaani slot-label mapping (Vaani label -> set of TOP slot types that map to it)
# ---------------------------------------------------------------------------
SLOT_LABELS = [
    "O",
    "B-time", "I-time",
    "B-recipient", "I-recipient",
    "B-message", "I-message",
    "B-title", "I-title",
    "B-location", "I-location",
]

# Map TOP slot type name -> Vaani slot label (the key without B-/I-)
SLOT_MAP = {
    "DATE_TIME": "time",
    "TODO": "task",
    "REMINDER_TODO": "task",
    "RECIPIENT": "recipient",
    "CONTACT": "recipient",
    "CONTENT_EXACT": "message",
    "MUSIC_TRACK_TITLE": "title",
    "MUSIC_ARTIST_NAME": "title",
    "MUSIC_GENRE": "title",
    "MUSIC_TYPE": "media_type",
    "LOCATION": "location",
}

# ---------------------------------------------------------------------------
# Tokeniser (simple Word‑Piece-like, matching the C++ runtime behaviour)
# ---------------------------------------------------------------------------
_WORD_RUN = re.compile(r"[A-Za-z0-9:.]+")

def split_words(text: str) -> list:
    """Split text into words, preserving internal ':' and '.'."""
    words = []
    for match in _WORD_RUN.finditer(text):
        word = match.group(0).strip(":.")
        if word:
            words.append(word)
    return words

def tokenize_utterance(text: str) -> list:
    """Return list of token strings (lower‑cased, runs of [A-Za-z0-9:.])."""
    return split_words(text)

# ---------------------------------------------------------------------------
# BIO tag assignment
# ---------------------------------------------------------------------------
def assign_bio_tags(tokens: list, slot_values: dict) -> list:
    """
    slot_values: dict mapping vaani_slot_label -> list of token indices
                 where the slot value occurs.
    Returns a list of BIO tag strings ('O' or 'B-<label>' or 'I-<label>')
    aligned with *tokens*.
    """
    bio = ["O"] * len(tokens)
    slot_first_done = {}

    for label, indices in slot_values.items():
        for idx in indices:
            if idx not in slot_first_done:
                bio[idx] = f"B-{label}"
                slot_first_done[idx] = True
            else:
                bio[idx] = f"I-{label}"
    return bio

# ---------------------------------------------------------------------------
# Parse a single TSV row
# ---------------------------------------------------------------------------
def parse_tsv_row(row, slot_map, top_to_vaani_intent):
    """
    row dict from csv.DictReader (columns: en_query, cs_query, en_parse,
    cs_parse, domain).

    Returns a dict as required for JSONL or None if the record should be skipped.
    """
    en_query = row.get("en_query", "").strip()
    cs_query = row.get("cs_query", "").strip()
    # Choose the utterance text: prefer cs_query (Hinglish/Hindi), fall back to en_query
    text = cs_query if cs_query else en_query
    if not text:
        return None

    # -----------------------------------------------------------------
    # Extract TOP intent from the parse columns (prefer en_parse)
    # -----------------------------------------------------------------
    intent = None
    for parse_col in ("en_parse", "cs_parse"):
        parse_text = row.get(parse_col, "")
        m = re.search(r"IN:(\w+)", parse_text)
        if m:
            top_intent = m.group(1)
            intent = top_to_vaani_intent.get(top_intent)
            if intent is None:
                # Unsupported TOP intent -> skip this record
                return None
            break

    if intent is None:
        return None

    # -----------------------------------------------------------------
    # Extract slot spans from the parse column (again prefer en_parse)
    # We look for [... [SL:TYPE value] ...] patterns.
    # -----------------------------------------------------------------
    slot_text = row.get("en_parse", "") or row.get("cs_parse", "")

    # Find all [... [SL:TYPE value] ...] with non-greedy outermost match.
    # This regex captures the slot type and the inner value text.
    slot_pattern = re.compile(
        r"\[SL:(\w+)\s+((?:\[[^\]]*\]|[^\[\]])*)\]"
    )

    raw_slots = slot_pattern.findall(slot_text)

    # Build a dict: vaani_slot_label -> list of value strings
    slot_type_to_values = defaultdict(list)

    for top_type, value_str in raw_slots:
        # Map TOP type to Vaani slot label
        vaani_label = slot_map.get(top_type)
        if vaani_label is None or vaani_label == "task":
            # unmapped or task -> skip
            continue
        # Normalise value: strip whitespace
        value_str = value_str.strip()
        if value_str:
            slot_type_to_values[vaani_label].append(value_str)

    # -----------------------------------------------------------------
    # Tokenise the utterance and align slot values to token indices
    # -----------------------------------------------------------------
    tokens = tokenize_utterance(text)

    # For each Vaani slot label, find token positions where the token text
    # exactly matches (case‑insensitive) one of the slot values.
    slot_label_to_indices = defaultdict(list)

    # Pre‑build a mapping from lower‑cased token -> list of positions.
    lower_to_positions = defaultdict(list)
    for i, tok in enumerate(tokens):
        lower_to_positions[tok.lower()].append(i)

    for vaani_label, values in slot_type_to_values.items():
        for val in values:
            val_lower = val.lower()
            for pos in lower_to_positions.get(val_lower, []):
                if pos not in slot_label_to_indices[vaani_label]:
                    slot_label_to_indices[vaani_label].append(pos)

    # -----------------------------------------------------------------
    # Assign BIO tags
    # -----------------------------------------------------------------
    bio_tags = assign_bio_tags(tokens, slot_label_to_indices)

    # -----------------------------------------------------------------
    # Build the slots dict (key -> first value per label)
    # -----------------------------------------------------------------
    slots = {}
    for vaani_label, values in slot_type_to_values.items():
        if values:
            slots[vaani_label] = values[0]

    # -----------------------------------------------------------------
    # Return the JSON‑serialisable record
    # -----------------------------------------------------------------
    return {
        "text": text,
        "intent": intent,
        "slots": slots,
        "tokens": tokens,
        "bio_tags": bio_tags,
    }

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def main():
    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # Per‑intent sample counts
    intent_counts = defaultdict(int)

    # Process each split
    splits = {
        "train": os.path.join(RAW_TOP_DIR, "train.tsv"),
        "val":   os.path.join(RAW_TOP_DIR, "validation.tsv"),
        "test":  os.path.join(RAW_TOP_DIR, "test.tsv"),
    }

    for split_name, tsv_path in splits.items():
        if not os.path.exists(tsv_path):
            print(f"[WARN] {tsv_path} not found – skipping {split_name}", file=sys.stderr)
            continue

        out_path = os.path.join(OUT_DIR, f"hinglish_top_{split_name}.jsonl")
        print(f"Processing {split_name} from {tsv_path} ...")

        with open(tsv_path, newline="", encoding="utf-8") as tsv_f:
            reader = csv.DictReader(tsv_f, delimiter="\t")
            with open(out_path, "w", encoding="utf-8") as out_f:
                for i, row in enumerate(reader):
                    record = parse_tsv_row(row, SLOT_MAP, TOP_TO_VAANI_INTENT)
                    if record is None:
                        # Skipped (unsupported intent or missing data)
                        continue

                    # Serialize to JSONL
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    # Count intents
                    intent_counts[record["intent"]] += 1

        print(f"  Wrote {split_name} records to {out_path}")

    # -----------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------
    print("\n=== Summary ===")
    for intent, cnt in sorted(intent_counts.items()):
        print(f"  Intent '{intent}': {cnt} samples")

    # Print the slot‑type mapping table for reference
    print("\nTOP slot-type -> Vaani slot label mapping used:")
    for top, vaani in SLOT_MAP.items():
        if vaani:
            print(f"  {top:25s} -> {vaani}")
    print("\n(Unmapped TOP slot types such as TODO/REMINDER_TODO are emitted as 'O' tags.)")

if __name__ == "__main__":
    main()