#ifndef VAANI_NLU_H
#define VAANI_NLU_H

/**
 * vaani_nlu.h - C++20 Hinglish NLU runtime
 *
 * Provides vaani_parse() C ABI function that performs:
 *   1. Normalization via libnormalizer (Devanagari→ITRANS + phonetic collapse)
 *   2. Fast WordPiece tokenization
 *   3. ONNX Runtime INT8 inference
 *   4. BIO slot tags → JSON assembly
 *
 * Target latency: < 15 ms end-to-end on CPU
 */

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <string>

/**
 * @brief Initialize the Vaani NLU runtime.
 *
 * Loads vocabulary, intent labels, and slot labels from the specified
 * or default paths. Must be called before vaani_parse(), or vaani_parse()
 * will perform lazy initialization on first call.
 *
 * @param model_path Path to ONNX model file (default: models/vaani_nlu.onnx)
 * @param vocab_path Path to WordPiece vocabulary file (default: models/vocab.txt)
 * @param intent_labels_path Path to intent labels JSON (default: models/intent_labels.json)
 * @param slot_labels_path Path to slot labels JSON (default: models/slot_labels.json)
 * @return true if initialization succeeded, false otherwise
 */
extern "C" bool vaani_init(const char* model_path, const char* vocab_path,
                           const char* intent_labels_path, const char* slot_labels_path);

/**
 * @brief Parse a Hinglish voice command and return JSON with intent + slots.
 *
 * Execution pipeline:
 *   1. Normalize via C++17 normalizer (normalizer.cpp)
 *   2. Tokenize with pre-baked WordPiece vocabulary
 *   3. Run ONNX Runtime inference (via scripts/onnx_inference.py)
 *   4. Map logits → intent + BIO tags → slot values → JSON
 *
 * @param raw_text Input Hinglish text (UTF-8, may contain Devanagari or Romanized Hindi)
 * @return const char* JSON string in a thread-local buffer. Copy it if you need
 *         to retain it. Returns nullptr on fatal error.
 *
 * @note The returned pointer is valid until the next call to vaani_parse() on
 *       the same thread. vaani_free_result() is currently a no-op kept for ABI
 *       stability.
 *
 * @see vaani_init() for explicit initialization. If not called, vaani_parse()
 * will lazily initialize on first invocation.
 */
extern "C" const char* vaani_parse(const char* raw_text);

/**
 * @brief Free the JSON result returned by vaani_parse().
 *
 * Must be called once per call to vaani_parse() to avoid memory leaks.
 *
 * @param json_str Pointer returned by vaani_parse(). May be nullptr.
 */
extern "C" void vaani_free_result(const char* json_str);

/**
 * @brief Estimated latency of a single vaani_parse() call in milliseconds.
 *
 * @return float Placeholder constant (currently 12.0). This is not measured at
 *         runtime; see the benchmark for observed end-to-end latency.
 */
extern "C" float vaani_get_estimated_latency_ms();

#endif // VAANI_NLU_H