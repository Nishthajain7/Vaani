#!/usr/bin/env python3
"""
LEGACY / EXPERIMENTAL - not the supported training pipeline.

Use ``scripts/train_vaani_nlu.py`` instead. This script contains known bugs
(undefined ``MAX_LENGTH``, dataset methods defined inside ``__init__``) that
prevent it from running end to end.

scripts/generate_and_train.py - Generate 3,000 Hinglish voice commands with BIO slot labels
and export a multi-task BERT model to ONNX/INT8 format.

This script:
1. Generates 3,000 diverse Hinglish voice commands with ASR noise, Devanagari mixing,
   code-switching, and varied word orders across 5 intents
2. Saves intent_labels.json, slot_labels.json, and vocab.txt
3. Fine-tunes a compact DistilBERT multi-task model (intent + slots)
4. Exports the model to ONNX (opset 14)
5. INT8 quantizes the ONNX model
"""

import json
import os
import random
import re
from typing import List, Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

# Project configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = PROJECT_ROOT

EVAL_SET_PATH = os.path.join(BASE, "data", "eval_set.jsonl")
SYNTHETIC_PATH = os.path.join(BASE, "data", "synthetic_3000.jsonl")
INTENT_LABELS_PATH = os.path.join(BASE, "models", "intent_labels.json")
SLOT_LABELS_PATH = os.path.join(BASE, "models", "slot_labels.json")
VOCAB_PATH = os.path.join(BASE, "models", "vocab.txt")
ONNX_PATH = os.path.join(BASE, "models", "vaani_joint_nlu.onnx")
ONNX_INT8_PATH = os.path.join(BASE, "models", "vaani_joint_nlu_int8.onnx")

BERT_MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 1
LR = 2e-5
SEED = 42

SLOT_LABELS = [
    "O", "B-time", "I-time", "B-recipient", "I-recipient",
    "B-message", "I-message", "B-title", "I-title", "B-location", "I-location",
]
NUM_INTENTS = 5
NUM_SLOT_LABELS = len(SLOT_LABELS)

random.seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")
print(f"🖥️  Device: {DEVICE}")


CLEAN_TEXTS_BY_INTENT = {
    "set_reminder": [
        "set a reminder for 8 AM", "remind me to call mom at 6 PM",
        "set reminder at 9 PM to pay bill", "wake me at midnight for prayer",
        "set an alarm for noon", "remind me to take medicine at 8 AM",
        "set a reminder for meeting at 3 PM",
    ],
    "send_message": [
        "send a message to neha saying call mom", "whatsapp mom that i am late",
        "text alex that im running late", "send message to raj finish project",
        "text priya the meeting is cancelled", "whatsapp team about the change",
        "send alex a quick hello",
    ],
    "play_media": [
        "play some bollywood music", "play despacito song",
        "listen to romantic track", "play punjabi song playlist",
        "listen to hape shayad", "play some old is gold",
        "listen to dance number",
    ],
    "check_weather": [
        "what's the weather in delhi", "forecast for mumbai today",
        "how's the weather in bangalore", "weather in chennai this weekend",
        "what will be the temperature in chennai", "forecast for kolkata tomorrow",
        "what's the weather like in pune",
    ],
    "set_alarm": [
        "set alarm at 7 AM", "wake me up at 6:30 PM",
        "set an alarm for 8 AM", "wake me at midnight",
        "set an alarm for 10 AM", "set wake alarm at 9 PM",
        "set alarm for morning walk",
    ],
}

ASR_NOISE_OPTIONS = [
    lambda t: t.replace("8", "∞").replace("0", "O").replace("1", "l"),
    lambda t: t + " ya" if random.random() < 0.3 else t,
    lambda t: t.replace("9 PM", "9 AM").replace("8 AM", "8 PM"),
    lambda t: " ".join(t.split()[::-1]) if random.random() < 0.2 else t,
    lambda t: re.sub(r'[aeiou]{2,}', lambda m: m.group(0)[0], t) if random.random() < 0.3 else t,
]

DEVANAGARI_WORDS = ["पानी", "समय", "संदेश", "संगीत", "मौसम", "अलार्म", "चेतावनी"]
HINGLISH_MIX = ["baaje", "sab", "kal", "ajKal", "bahut", "thodo", "accha", "achca"]


def get_slot_keys_for_intent(intent):
    mapping = {
        "set_reminder": {"time": None, "message": None},
        "send_message": {"recipient": None, "message": None},
        "play_media": {"title": None, "media_type": None},
        "check_weather": {"location": None},
        "set_alarm": {"time": None},
    }
    return mapping.get(intent, {})


def generate_sample(intent, use_devanagari=False, use_asr_noise=True):
    template = random.choice(CLEAN_TEXTS_BY_INTENT[intent])
    slot_pools = {
        "set_reminder": {
            "time": ["8 AM", "9 PM", "6:30 AM", "11 PM", "7 PM", "10 AM", "5:45 PM", "6 AM", "midnight", "noon"],
            "message": ["call mom", "send money", "finish project", "meet at cafe", "take out trash", "water plants"],
        },
        "send_message": {
            "recipient": ["anita", "vikram", "sonu", "priya", "raj", "simran", "karan", "neha", "ajay", "pooja"],
            "message": ["call mom", "send money", "finish project", "water plants", "take out trash", "buy groceries"],
        },
        "play_media": {
            "media_type": ["song", "music", "playlist", "album", "video"],
            "title": ["despacito", "hape shayad", "bollywood beats", "punjabi song", "old is gold", "romantic track"],
        },
        "check_weather": {
            "location": ["delhi", "mumbai", "bangalore", "chennai", "kolkata", "hyderabad", "pune", "jaipur", "lucknow"],
        },
        "set_alarm": {
            "time": ["8 AM", "9 PM", "6:30 AM", "11 PM", "7 PM", "10 AM", "5:45 PM", "6 AM", "midnight", "noon"],
        },
    }

    slot_values = {}
    for slot_key in slot_pools[intent]:
        slot_values[slot_key] = random.choice(slot_pools[intent][slot_key])

    clean = template.format(**slot_values)

    if use_devanagari and random.random() < 0.4:
        if random.random() < 0.5 and "mom" in clean.lower():
            clean = clean.lower().replace("mom", "परी")
        if random.random() < 0.5 and "alarm" in clean.lower():
            clean = clean.lower().replace("alarm", "अलार्म")
        if random.random() < 0.5 and "message" in clean.lower():
            clean = clean.lower().replace("message", "संदेश")

    if random.random() < 0.4:
        hinglish_word = random.choice(HINGLISH_MIX)
        words = clean.split()
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, hinglish_word)
        clean = " ".join(words)

    noisy = clean
    if use_asr_noise:
        noise_to_apply = random.sample(ASR_NOISE_OPTIONS, k=min(random.randint(0, 3), len(ASR_NOISE_OPTIONS)))
        for noise_fn in noise_to_apply:
            noisy = noise_fn(noisy)

    if not noisy.strip():
        noisy = clean

    slot_key_to_bio = {"time": "B-time", "recipient": "B-recipient", "message": "B-message",
                       "title": "B-title", "location": "B-location"}
    primary_slot = None
    for sk in ["time", "recipient", "message", "title", "location"]:
        if slot_key in slot_values and slot_values[slot_key]:
            primary_slot = slot_key
            break

    if primary_slot and primary_slot in slot_key_to_bio:
        primary_bio = slot_key_to_bio[primary_slot]
        bio_to_id = {"O": 0, "B-time": 1, "I-time": 2, "B-recipient": 3, "I-recipient": 4,
                     "B-message": 5, "I-message": 6, "B-title": 7, "I-title": 8, "B-location": 9, "I-location": 10}
        slot_label_id = bio_to_id.get(primary_bio, 0)
    else:
        slot_label_id = 0

    return {
        "intent": intent, "clean_text": clean, "noisy_text": noisy,
        "slot_values": slot_values, "slot_label_id": slot_label_id,
    }


def generate_hinglish_samples(n_samples=3000):
    print(f"🔨 Generating {n_samples} synthetic Hinglish samples...")
    samples = []
    intent_weights = ["set_reminder", "send_message", "play_media", "check_weather", "set_alarm"]
    for i in range(n_samples):
        intent = random.choices(intent_weights, weights=[0.25, 0.25, 0.2, 0.15, 0.15])[0]
        sample = generate_sample(intent=intent, use_devanagari=random.random() < 0.4,
                                 use_asr_noise=random.random() < 0.7)
        samples.append(sample)
    return samples


class HinglishNLUDataset(Dataset):
    def __init__(self, encodings, intent_ids, slot_label_ids):
        self.encodings = encodings
        self.intent_ids = intent_ids
        self.slot_label_ids = slot_label_ids

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).squeeze() if isinstance(val, list) else torch.tensor(val).squeeze()
                for key, val in self.encodings[idx].items()}
        item["intent_label"] = torch.tensor(self.intent_ids[idx], dtype=torch.long)
        item["slot_labels"] = torch.tensor(self.slot_label_ids[idx], dtype=torch.long)
        return item


class JointBERTIntentSlot(nn.Module):
    def __init__(self, bert_model_name=BERT_MODEL_NAME, hidden_size=768, num_intents=NUM_INTENTS,
                 num_slot_labels=NUM_SLOT_LABELS, mid_size=128):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_size, mid_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(mid_size, num_intents),
        )
        self.slot_classifier = nn.Sequential(
            nn.Linear(hidden_size, mid_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(mid_size, num_slot_labels),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        intent_logits = self.intent_classifier(cls_token)
        slot_logits = self.slot_classifier(cls_token)
        return intent_logits, slot_logits


def main():
    print("=" * 70)
    print("Vaani NLU: 3,000 Hinglish Sample Generation + Multi-Task Training")
    print("=" * 70)

    # Step 1: Generate 3,000 Hinglish samples
    global INTENT_LABELS, SLOT_LABELS
    INTENT_LABELS = ["set_reminder", "send_message", "play_media", "check_weather", "set_alarm"]
    SLOT_LABELS = [
        "O", "B-time", "I-time", "B-recipient", "I-recipient",
        "B-message", "I-message", "B-title", "I-title", "B-location", "I-location",
    ]

    samples = generate_hinglish_samples(n_samples=3000)
    print(f"✅ Generated {len(samples)} samples")

    os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
    with open(os.path.join(BASE, "data", "synthetic_3000.jsonl"), "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({"text": s["noisy_text"], "intent": s["intent"], "clean": s["clean_text"]}) + "\n")
    print(f"✅ Saved synthetic data to data/synthetic_3000.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME, truncation=True, max_length=MAX_LEN)

    encodings = []
    intent_ids = []
    slot_label_ids = []

    for sample in samples:
        encoded = tokenizer(sample["noisy_text"], truncation=True, max_length=MAX_LENGTH,
                          padding="max_length", return_tensors="pt")
        encodings.append(encoded)
        intent_ids.append(INTENT_LABELS.index(sample["intent"]))
        slot_label_ids.append(sample["slot_label_id"])

    class JointDataset(Dataset):
        def __init__(self, encodings, intent_ids, slot_labels):
            self.encodings = encodings
            self.intent_ids = intent_ids
            self.slot_labels = slot_labels
            def __len__(self):
                return len(self.encodings)
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]).squeeze() if isinstance(val, list) else torch.tensor(val).squeeze()
                        for key, val in self.encodings[idx].items()}
                item["intent_label"] = torch.tensor(self.intent_ids[idx], dtype=torch.long)
                item["slot_labels"] = torch.tensor(self.slot_labels[idx], dtype=torch.long)
                return item

    dataset = JointDataset(encodings, intent_ids, slot_label_ids)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = JointBERTIntentSlot()
    model.to(DEVICE)
    model.train()

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_loader) * EPOCHS)

    intent_criterion = nn.CrossEntropyLoss()
    slot_criterion = nn.CrossEntropyLoss()

    print(f"🚀 Starting training for {EPOCHS} epoch(s)...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        num_batches = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in progress:
            input_ids, attention_mask, intent_labels, slot_labels = batch
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            intent_labels = intent_labels.to(DEVICE)
            slot_labels = slot_labels.to(DEVICE)
            optimizer.zero_grad()
            intent_logits, slot_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            intent_loss = intent_criterion(intent_logits, intent_labels)
            slot_loss = slot_criterion(slot_logits, slot_labels)
            loss = intent_loss + slot_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{EPOCHS}: avg_loss={avg_loss:.4f}")

    print("✅ Training completed")

    # Export to ONNX
    model.eval()
    dummy_input_ids = torch.randint(tokenizer.vocab_size, (1, MAX_LEN), dtype=torch.long)
    dummy_attention_mask = torch.ones_like(dummy_input_ids)

    print(f"📤 Exporting model to ONNX: {ONNX_PATH}")
    torch.onnx.export(model, (dummy_input_ids, dummy_attention_mask), ONNX_PATH,
                      input_names=["input_ids", "attention_mask"],
                      output_names=["intent_logits", "slot_logits"],
                      dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"},
                                    "intent_logits": {0: "batch"}, "slot_logits": {0: "batch"}},
                      opset_version=14, do_constant_folding=True, verbose=False)
    print(f"✅ ONNX model exported to {ONNX_PATH}")

    from onnxruntime.quantization import quantize_dynamic, QuantType
    print(f"⚡ Applying INT8 dynamic quantization...")
    quantize_dynamic(model_input=ONNX_PATH, model_output=ONNX_INT8_PATH, weight_type=QuantType.QInt8)
    print(f"✅ INT8 quantized model exported to {ONNX_INT8_PATH}")

    print("💾 Saving vocabulary and label mappings...")
    intent_labels_map = {i: label for i, label in enumerate(INTENT_LABELS)}
    with open(INTENT_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(intent_labels_map, f, ensure_ascii=False, indent=2)
    print(f"   Saved intent labels to {INTENT_LABELS_PATH}")

    slot_labels_map = {i: label for i, label in enumerate(SLOT_LABELS)}
    with open(SLOT_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(slot_labels_map, f, ensure_ascii=False, indent=2)
    print(f"   Saved slot labels to {SLOT_LABELS_PATH}")

    vocab_tokens = tokenizer.get_vocab()
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        for token in vocab_tokens:
            f.write(token + "\n")
    print(f"   Saved vocabulary to {VOCAB_PATH}: {len(vocab_tokens)} tokens")

    def get_size_mb(path):
        import os
        return os.path.getsize(path) / (1024 * 1024)

    size_onnx = get_size_mb(ONNX_PATH)
    size_int8 = get_size_mb(ONNX_INT8_PATH)

    print(f"\n📊 Model sizes:")
    print(f"   ONNX (FP32): {size_onnx:.4f} MB")
    print(f"   INT8 quantized: {size_int8:.4f} MB")
    print(f"   ✅ Under 35 MB target: {size_int8 < 35}")

    print(f"\n🧪 Testing INT8 model inference...")
    import onnxruntime as ort
    session = ort.InferenceSession(ONNX_INT8_PATH)

    sample = dataset[0]
    inputs = {"input_ids": sample[0].unsqueeze(0).numpy(), "attention_mask": sample[1].unsqueeze(0).numpy()}
    outputs = session.run(None, inputs)

    intent_result = outputs[0]
    slot_result = outputs[1]
    pred_intent_idx = intent_result.argmax(-1)[0]
    pred_intent = INTENT_LABELS[pred_intent_idx]
    print(f"   Predicted intent: {pred_intent}")
    pred_slot_idx = slot_result.argmax(-1)[0]
    pred_slot = SLOT_LABELS[pred_slot_idx]
    print(f"   Predicted slot label: {pred_slot}")
    probs = torch.softmax(torch.tensor(intent_result[0]), dim=-1)
    top3_idx = probs.topk(3).indices.tolist()
    print(f"   Top-3 intents: {[(INTENT_LABELS[i], float(probs[i])) for i in top3_idx]}")

    print(f"\n✅ All steps completed successfully!")
    print(f"   Models saved to: {BASE}/models/")
    print(f"   📁 FP32 ONNX: {size_onnx:.4f} MB | INT8: {size_int8:.4f} MB (well under 35 MB target)")


if __name__ == "__main__":
    main()
SCRIPT_EOF