#!/usr/bin/env python3
"""Tokenizer and vocabulary helpers shared by training and inference.

The runtime (``src/runtime.cpp``) implements exactly this algorithm:

1. Split on characters outside ``[A-Za-z0-9:.]``.
2. Trim leading/trailing ``:`` and ``.`` from each word.
3. Lowercase the word and look it up in the vocabulary.
4. If absent, greedily match the longest prefix (``##`` prefix for non-initial
   pieces). If a character cannot be matched, emit ``[UNK]``.
5. Wrap the result with ``[CLS]`` and ``[SEP]``.

The vocabulary file is one token per line; the line number is the token id.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN]

_WORD_RUN = re.compile(r"[A-Za-z0-9:.]+")


def split_words(text: str) -> List[str]:
    """Split text into words, preserving case and internal ``:``/``.``."""
    words = []
    for match in _WORD_RUN.finditer(text):
        word = match.group(0).strip(":.")
        if word:
            words.append(word)
    return words


def load_vocab(path: str) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            token = line.rstrip("\n")
            if token:
                vocab[token] = index
    return vocab


def save_vocab(tokens: Sequence[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for token in tokens:
            handle.write(token + "\n")


def build_vocab(texts: Iterable[str], min_freq: int = 1) -> List[str]:
    """Build a deterministic word + character vocabulary."""
    counts: Dict[str, int] = {}
    for text in texts:
        for word in split_words(text):
            key = word.lower()
            counts[key] = counts.get(key, 0) + 1

    words = sorted(
        (word for word, count in counts.items() if count >= min_freq),
        key=lambda word: (-counts[word], word),
    )

    chars = set()
    for word in counts:
        chars.update(word)
    char_tokens = sorted(chars)

    tokens = list(SPECIAL_TOKENS)
    tokens.extend(words)
    tokens.extend(char_tokens)
    tokens.extend("##" + char for char in char_tokens)

    # Keep ids stable and unique.
    seen = set()
    unique = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique.append(token)
    return unique


def encode_word(word: str, vocab: Dict[str, int]) -> List[int]:
    """Greedy longest-match WordPiece encoding for one word."""
    lowered = word.lower()
    direct = vocab.get(lowered)
    if direct is not None:
        return [direct]

    unk = vocab.get(UNK_TOKEN, 1)
    ids: List[int] = []
    start = 0
    while start < len(lowered):
        match_id = None
        match_end = None
        for end in range(len(lowered), start, -1):
            piece = lowered[start:end]
            if start > 0:
                piece = "##" + piece
            token_id = vocab.get(piece)
            if token_id is not None:
                match_id = token_id
                match_end = end
                break
        if match_id is None:
            return [unk]
        ids.append(match_id)
        start = match_end
    return ids


def encode(
    text: str, vocab: Dict[str, int], max_len: int = 64
) -> Tuple[List[int], List[str], List[Tuple[int, int]]]:
    """Encode text.

    Returns ``(input_ids, words, word_spans)`` where each span is the
    half-open token range ``[start, end)`` for that word. ``input_ids``
    includes ``[CLS]`` at 0 and ``[SEP]`` at the end.
    """
    words = split_words(text)
    cls_id = vocab.get(CLS_TOKEN, 2)
    sep_id = vocab.get(SEP_TOKEN, 3)

    ids: List[int] = [cls_id]
    spans: List[Tuple[int, int]] = []
    for word in words:
        start = len(ids)
        ids.extend(encode_word(word, vocab))
        spans.append((start, len(ids)))
        if len(ids) >= max_len - 1:
            break
    ids.append(sep_id)
    spans = spans[: len(words)]
    return ids, words[: len(spans)], spans


def tags_to_slots(
    words: Sequence[str], word_tags: Sequence[str]
) -> Dict[str, str]:
    """Merge BIO tags into ``{slot_key: value}`` using the original words."""
    slots: Dict[str, str] = {}
    current_key = None
    current_words: List[str] = []

    def flush() -> None:
        nonlocal current_key, current_words
        if current_key and current_words:
            slots[current_key] = " ".join(current_words)
        current_key = None
        current_words = []

    for word, tag in zip(words, word_tags):
        if tag.startswith("B-"):
            flush()
            current_key = tag[2:]
            current_words = [word]
        elif tag.startswith("I-"):
            key = tag[2:]
            if current_key == key:
                current_words.append(word)
            else:
                flush()
                current_key = key
                current_words = [word]
        else:
            flush()
    flush()
    return slots
