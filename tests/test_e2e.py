#!/usr/bin/env python3
"""End-to-end integration tests for the Vaani NLU C ABI.

Loads ``libvaani_runtime.so`` with ctypes, initializes it through
``vaani_init()``, and runs 20 noisy Hinglish utterances through
``vaani_parse()``. The suite checks the JSON/ABI contract and reports
intent accuracy, slot-key accuracy, and latency. Accuracy is reported,
not gated: the shipped model is a small reference model, and benchmark
quality is tracked separately in ``bench/run_benchmark.py``.

Run directly or through CTest:

    python3 tests/test_e2e.py
    cd build && ctest --output-on-failure

Environment overrides:
    VAANI_LIB_PATH           path to libvaani_runtime.so
    VAANI_MODEL_PATH         path to the ONNX model
    VAANI_VOCAB_PATH         path to vocab.txt
    VAANI_INTENT_LABELS_PATH path to intent_labels.json
    VAANI_SLOT_LABELS_PATH   path to slot_labels.json
"""

import ctypes
import json
import os
import statistics
import time
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_lib_path():
    candidates = [
        os.environ.get("VAANI_LIB_PATH"),
        os.path.join(PROJECT_ROOT, "build", "libvaani_runtime.so"),
        os.path.join(PROJECT_ROOT, "libvaani_runtime.so"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return candidates[1]


LIB_PATH = _default_lib_path()
MODEL_PATH = os.environ.get(
    "VAANI_MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "vaani_nlu.onnx")
)
VOCAB_PATH = os.environ.get(
    "VAANI_VOCAB_PATH", os.path.join(PROJECT_ROOT, "models", "vocab.txt")
)
INTENT_LABELS_PATH = os.environ.get(
    "VAANI_INTENT_LABELS_PATH",
    os.path.join(PROJECT_ROOT, "models", "intent_labels.json"),
)
SLOT_LABELS_PATH = os.environ.get(
    "VAANI_SLOT_LABELS_PATH",
    os.path.join(PROJECT_ROOT, "models", "slot_labels.json"),
)

# 20 noisy Hinglish utterances: 4 per intent, ASR-style noise and
# code-switching included.
TEST_CASES = [
    {"text": "set a reminder at 8 AM", "intent": "set_reminder", "slots": {"time"}},
    {"text": "remind me to call mom at 6 PM", "intent": "set_reminder", "slots": {"time", "message"}},
    {"text": "set reminder at 9 PM to pay bill", "intent": "set_reminder", "slots": {"time", "message"}},
    {"text": "wake me at midnight for prayer", "intent": "set_reminder", "slots": {"time"}},

    {"text": "send a message to neha saying call mom", "intent": "send_message", "slots": {"recipient", "message"}},
    {"text": "whatsapp mom that i am late", "intent": "send_message", "slots": {"recipient", "message"}},
    {"text": "text alex that im running late", "intent": "send_message", "slots": {"recipient", "message"}},
    {"text": "send message to raj finish project", "intent": "send_message", "slots": {"recipient", "message"}},

    {"text": "play some bollywood music", "intent": "play_media", "slots": {"title", "media_type"}},
    {"text": "play despacito song", "intent": "play_media", "slots": {"title", "media_type"}},
    {"text": "listen to romantic track", "intent": "play_media", "slots": {"title", "media_type"}},
    {"text": "play punjabi song playlist", "intent": "play_media", "slots": {"title", "media_type"}},

    {"text": "what's the weather in delhi", "intent": "check_weather", "slots": {"location"}},
    {"text": "forecast for mumbai today", "intent": "check_weather", "slots": {"location"}},
    {"text": "how's the weather in bangalore", "intent": "check_weather", "slots": {"location"}},
    {"text": "weather in chennai this weekend", "intent": "check_weather", "slots": {"location"}},

    {"text": "set alarm at 7 AM", "intent": "set_alarm", "slots": {"time"}},
    {"text": "wake me up at 6:30 PM", "intent": "set_alarm", "slots": {"time"}},
    {"text": "set an alarm for 8 AM", "intent": "set_alarm", "slots": {"time"}},
    {"text": "set wake alarm at 9 PM", "intent": "set_alarm", "slots": {"time"}},
]

VALID_INTENTS = {"set_reminder", "send_message", "play_media", "check_weather", "set_alarm"}


def load_runtime(lib_path):
    lib = ctypes.CDLL(lib_path)
    lib.vaani_init.argtypes = [ctypes.c_char_p] * 4
    lib.vaani_init.restype = ctypes.c_bool
    lib.vaani_parse.argtypes = [ctypes.c_char_p]
    lib.vaani_parse.restype = ctypes.c_char_p
    lib.vaani_free_result.argtypes = [ctypes.c_char_p]
    lib.vaani_free_result.restype = None
    lib.vaani_get_estimated_latency_ms.argtypes = []
    lib.vaani_get_estimated_latency_ms.restype = ctypes.c_float
    return lib


class TestE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(LIB_PATH):
            raise unittest.SkipTest(f"runtime library not found: {LIB_PATH}")
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest(f"model not found: {MODEL_PATH} (see README)")

        cls.lib = load_runtime(LIB_PATH)
        ok = cls.lib.vaani_init(
            MODEL_PATH.encode(),
            VOCAB_PATH.encode(),
            INTENT_LABELS_PATH.encode(),
            SLOT_LABELS_PATH.encode(),
        )
        if not ok:
            raise unittest.SkipTest("vaani_init() returned false; check model paths")

        cls.results = []
        cls.estimated_latency_ms = cls.lib.vaani_get_estimated_latency_ms()

    @classmethod
    def tearDownClass(cls):
        if not getattr(cls, "results", None):
            return
        total = len(cls.results)
        intent_ok = sum(r["intent_ok"] for r in cls.results)
        slots_ok = sum(r["slots_ok"] for r in cls.results)
        latencies = [r["latency_ms"] for r in cls.results]

        print("\n" + "=" * 72)
        print(f"E2E summary ({total} noisy Hinglish cases)")
        print("=" * 72)
        print(f"Intent accuracy:      {intent_ok}/{total} ({intent_ok / total:.1%})")
        print(f"Exact slot-key match: {slots_ok}/{total} ({slots_ok / total:.1%})")
        print(f"Latency:              p50={statistics.median(latencies):.1f} ms  "
              f"max={max(latencies):.1f} ms")
        print(f"Reported estimate:    {cls.estimated_latency_ms:.1f} ms")
        print("-" * 72)
        for row in cls.results:
            status = "PASS" if row["intent_ok"] and row["slots_ok"] else "FAIL"
            print(
                f"{row['index']:2d}. {row['text'][:38]:38s} "
                f"intent={row['actual_intent']:14s} {status}"
            )

    def _ensure_results(self):
        if not self.results:
            self.test_01_parse_contract()

    def test_01_parse_contract(self):
        """Every utterance returns parseable JSON with the documented shape."""
        self.results.clear()
        for index, case in enumerate(TEST_CASES, start=1):
            started = time.perf_counter()
            raw = self.lib.vaani_parse(case["text"].encode("utf-8"))
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            self.assertIsNotNone(raw, f"null result for {case['text']!r}")
            text = raw.decode("utf-8").strip()
            self.lib.vaani_free_result(raw)

            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                self.fail(f"invalid JSON for {case['text']!r}: {text!r} ({exc})")

            self.assertIsInstance(parsed.get("intent"), str)
            self.assertIsInstance(parsed.get("slots"), dict)

            actual_intent = parsed["intent"]
            actual_slots = set(parsed["slots"])
            self.results.append(
                {
                    "index": index,
                    "text": case["text"],
                    "actual_intent": actual_intent,
                    "actual_slots": actual_slots,
                    "expected_intent": case["intent"],
                    "expected_slots": case["slots"],
                    "intent_ok": actual_intent == case["intent"],
                    "slots_ok": actual_slots == case["slots"],
                    "latency_ms": elapsed_ms,
                }
            )

        self.assertEqual(len(self.results), 20)

    def test_02_intent_labels_are_known(self):
        """Predictions must come from the declared label set."""
        self._ensure_results()
        for row in self.results:
            self.assertIn(row["actual_intent"], VALID_INTENTS, row["text"])

    def test_03_latency_under_10s(self):
        """Guard against pathological regressions (subprocess backend is slow)."""
        self._ensure_results()
        average = sum(r["latency_ms"] for r in self.results) / len(self.results)
        self.assertLess(average, 10_000.0, f"average latency {average:.1f} ms")


if __name__ == "__main__":
    unittest.main(verbosity=2)
