#!/usr/bin/env python3
"""
Generate 100 realistic Hinglish voice commands across 5 intents,
with clean Romanized text, target JSON slots, and noisy ASR versions.
Output: data/eval_set.jsonl

Noise types applied:
  1. Phonetic spelling variations (e.g., 'baje' -> 'bajje'/'baaje')
  2. Split compound words (e.g., 'remind karna' -> 're mind karna')
  3. Random Devanagari script switching (e.g., 'Kal subah 8 बजे meeting hai')
"""

import json
import random
from collections import defaultdict

random.seed(42)

# ──────────────────────────────────────────────
# Intent templates (clean Romanized + slot extractors)
# ──────────────────────────────────────────────
TEMPLATES = {
    "set_reminder": {
        "templates": [
            "Set a reminder for {time} to {message}",
            "Remind me to {message} at {time}",
            "Set reminder at {time} for {message}",
            "Remember to {message} at {time}",
        ],
        "slot_keys": ["intent", "time", "message"],
        "time_vals": [
            "8 AM", "9 PM", "6:30 AM", "11 PM", "7 PM", "10 AM",
            "5:45 PM", "6 AM", "midnight", "noon",
        ],
        "message_vals": [
            "buy groceries", "call mom", "send money", "finish project",
            "meet at cafe", "take out trash", "water plants",
            "send documents", "call doctor", "pay bill",
        ],
    },
    "send_message": {
        "templates": [
            "Send message to {recipient} saying {message}",
            "Text {recipient} {message}",
            "WhatsApp {recipient} {message}",
            "Message {recipient} {message}",
        ],
        "slot_keys": ["intent", "recipient", "message"],
        "recipient_vals": [
            "anita", "vikram", "sonu", "priya", "raj", "simran",
            "karan", "neha", "ajay", "pooja",
        ],
        "message_vals": [
            "call mom", "send money", "finish project", "water plants",
            "take out trash", "buy groceries", "call doctor",
            "pay bill", "send documents", "meet at cafe",
        ],
    },
    "play_media": {
        "templates": [
            "Play {media_type} {title}",
            "Play some {media_type} {title}",
            "Listen to {title} on {media_type}",
            "Open {media_type} {title}",
        ],
        "slot_keys": ["intent", "media_type", "title"],
        "media_type_vals": ["song", "music", "playlist", "album", "video"],
        "title_vals": [
            "despacito", "hape shayad", "bollywood beats", "punjabi song",
            "old is gold", "romantic track", "dance number",
        ],
    },
    "check_weather": {
        "templates": [
            "Weather in {location}",
            "What's the weather in {location}",
            "How's the weather in {location}",
            "Forecast for {location}",
        ],
        "slot_keys": ["intent", "location"],
        "location_vals": [
            "delhi", "mumbai", "bangalore", "chennai", "kolkata",
            "hyderabad", "pune", "jaipur", "lucknow",
        ],
    },
    "set_alarm": {
        "templates": [
            "Set alarm at {time}",
            "Wake me up at {time}",
            "Set an alarm for {time}",
            "Wake me at {time}",
        ],
        "slot_keys": ["intent", "time"],
        "time_vals": [
            "8 AM", "9 PM", "6:30 AM", "11 PM", "7 PM", "10 AM",
            "5:45 PM", "6 AM", "midnight", "noon",
        ],
    },
}

# ──────────────────────────────────────────────
# Slot-value pools per intent
# ──────────────────────────────────────────────
INTENT_POOLS = {
    "set_reminder": {
        "time": ["8 AM", "9 PM", "6:30 AM", "11 PM", "7 PM", "10 AM", "5:45 PM", "6 AM", "midnight", "noon"],
        "message": ["buy groceries", "call mom", "send money", "finish project",
                     "meet at cafe", "take out trash", "water plants", "send documents", "call doctor", "pay bill"],
    },
    "send_message": {
        "recipient": ["anita", "vikram", "sonu", "priya", "raj", "simran", "karan", "neha", "ajay", "pooja"],
        "message": ["call mom", "send money", "finish project", "water plants", "take out trash",
                     "buy groceries", "call doctor", "pay bill", "send documents", "meet at cafe"],
    },
    "play_media": {
        "media_type": ["song", "music", "playlist", "album", "video"],
        "title": ["despacito", "hape shayad", "bollywood beats", "punjabi song", "old is gold", "romantic track", "dance number"],
    },
    "check_weather": {
        "location": ["delhi", "mumbai", "bangalore", "chennai", "kolkata", "hyderabad", "pune", "jaipur", "lucknow"],
    },
    "set_alarm": {
        "time": ["8 AM", "9 PM", "6:30 AM", "11 PM", "7 PM", "10 AM", "5:45 PM", "6 AM", "midnight", "noon"],
    },
}

# ──────────────────────────────────────────────
# Noiser functions
# ──────────────────────────────────────────────

def apply_phonetic_variation(text: str) -> str:
    """Simulate Whisper/Sherpa-onnx phonetic spelling variations."""
    result = []
    words = text.split()
    for word in words:
        if random.random() < 0.5 and len(word) >= 2 and word[-1] in ("a", "e", "o"):
            # Double the last vowel (e.g., 'baje' -> 'bajje', 'baaje')
            last_char = word[-1]
            new_word = word[:-1] + last_char + last_char
            result.append(new_word)
            continue
        # Sometimes swap vowel instead
        if random.random() < 0.3 and len(word) >= 2 and word[-1] in ("a", "e", "o"):
            vowels = {"a": "e", "e": "a", "o": "a"}
            swap = vowels.get(word[-1], "a")
            new_word = word[:-1] + swap
            result.append(new_word)
            continue
        result.append(word)
    return " ".join(result)


def apply_compound_split(text: str) -> str:
    """Split compound words at random positions, simulating ASR boundary errors."""
    words = text.split()
    if len(words) < 2:
        return text

    # Pick a split point between words
    split_idx = random.randint(1, len(words) - 1)

    # With ~60% chance, actually split; otherwise keep mostly original
    if random.random() < 0.6:
        left = words[:split_idx]
        right = words[split_idx:]

        # ~40% chance to further split the first/last word into sub-parts
        if left and random.random() < 0.4 and len(left[-1]) >= 3:
            prefix_len = random.randint(1, len(left[-1]) - 1)
            left[-1] = left[-1][:prefix_len]
            # Insert the fragment as a new word
            right.insert(0, left[-1][prefix_len:])

        # Prepend a spurious prefix to the first left word
        if random.random() < 0.5 and left:
            prefixes = ["re", "pre", "de", "mid", "trans"]
            left[0] = random.choice(prefixes) + left[0]

        return " ".join(left) + " " + " ".join(right)
    return text


def apply_devanagari_switch(text: str) -> str:
    """Randomly switch substrings to Devanagari script."""
    devanagari_pool = [
        "बजे", "आठ", "नौ", "दस", "पौने", "सात", "छ",
        "खाना", "पानी", "दूध", "नहीं", "है", "करना",
        "जाना", "रहना", "होगा", "हुआ", "है",
    ]

    result = []
    words = text.split()
    word_idx = 0

    # Randomly replace individual words with Devanagari variants
    while word_idx < len(words):
        word = words[word_idx]
        # 40% chance to replace this word with Devanagari if it's a known slot value
        if random.random() < 0.4 and word in INTENT_POOLS["set_reminder"]["time"] + INTENT_POOLS["set_reminder"]["message"] + \
           INTENT_POOLS["send_message"]["recipient"] + INTENT_POOLS["send_message"]["message"] + \
           INTENT_POOLS["play_media"]["title"] + INTENT_POOLS["play_media"]["media_type"] + \
           INTENT_POOLS["check_weather"]["location"]:
            result.append(random.choice(devanagari_pool))
        else:
            result.append(word)
        word_idx += 1

    text_after = " ".join(result)

    # ~30% chance to replace a contiguous phrase with Devanagari
    if random.random() < 0.3:
        phrase_parts = text_after.split()
        if len(phrase_parts) >= 2:
            start = random.randint(0, len(phrase_parts) - 1)
            end = random.randint(start + 1, min(len(phrase_parts), start + 3))
            devanagari_phrase = " ".join(random.choices(devanagari_pool, k=end - start))
            phrase_parts[start:end] = [devanagari_phrase]
            text_after = " ".join(phrase_parts)

    return text_after


def add_noise_to_command(clean_text: str) -> str:
    """Apply all three noise types with appropriate probabilities.
    At least one noise type is always applied."""
    noises_applied = []

    # Phonetic variation (~70% chance)
    if random.random() < 0.7:
        clean_text = apply_phonetic_variation(clean_text)
        noises_applied.append("phonetic")

    # Compound split (~50% chance)
    if random.random() < 0.5:
        clean_text = apply_compound_split(clean_text)
        noises_applied.append("compound_split")

    # Devanagari script switching (~40% chance)
    if random.random() < 0.4:
        clean_text = apply_devanagari_switch(clean_text)
        noises_applied.append("devanagari")

    # Guarantee at least one noise type by randomly triggering another if none fired
    if not noises_applied:
        choice = random.choice(["phonetic", "compound_split", "devanagari"])
        if choice == "phonetic":
            clean_text = apply_phonetic_variation(clean_text)
        elif choice == "compound_split":
            clean_text = apply_compound_split(clean_text)
        else:
            clean_text = apply_devanagari_switch(clean_text)

    return clean_text


# ──────────────────────────────────────────────
# Generate the eval set
# ──────────────────────────────────────────────

def generate_eval_set(n_samples: int = 100) -> list:
    samples = []
    intents = list(TEMPLATES.keys())

    # Distribute samples roughly evenly across intents
    per_intent = n_samples // len(intents)
    remainder = n_samples % len(intents)

    for idx, intent in enumerate(intents):
        template_info = TEMPLATES[intent]
        count = per_intent + (1 if idx < remainder else 0)

        for _ in range(count):
            # Pick a random template for this intent
            template = random.choice(template_info["templates"])

            # Fill in slot values based on intent
            pools = INTENT_POOLS[intent]
            slot_values = {}

            if intent == "set_reminder":
                time_val = random.choice(pools["time"])
                message_val = random.choice(pools["message"])
                slot_values = {"intent": intent, "time": time_val, "message": message_val}
                clean = template.format(time=time_val, message=message_val)

            elif intent == "send_message":
                recipient_val = random.choice(pools["recipient"])
                message_val = random.choice(pools["message"])
                slot_values = {"intent": intent, "recipient": recipient_val, "message": message_val}
                clean = template.format(recipient=recipient_val, message=message_val)

            elif intent == "play_media":
                media_val = random.choice(pools["media_type"])
                title_val = random.choice(pools["title"])
                slot_values = {"intent": intent, "media_type": media_val, "title": title_val}
                clean = template.format(media_type=media_val, title=title_val)

            elif intent == "check_weather":
                location_val = random.choice(pools["location"])
                slot_values = {"intent": intent, "location": location_val}
                clean = template.format(location=location_val)

            elif intent == "set_alarm":
                time_val = random.choice(pools["time"])
                slot_values = {"intent": intent, "time": time_val}
                clean = template.format(time=time_val)

            # Generate noisy ASR version (apply noise to a copy)
            noisy = add_noise_to_command(clean)

            samples.append({
                "clean": clean,
                "slots": slot_values,
                "noisy": noisy,
            })

    # Shuffle the samples
    random.shuffle(samples)
    return samples


# ──────────────────────────────────────────────
# Write to JSONL
# ──────────────────────────────────────────────

def main():
    samples = generate_eval_set(100)
    with open("data/eval_set.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Generated {len(samples)} samples to data/eval_set.jsonl")


if __name__ == "__main__":
    main()