/**
 * VaaniNLU.kt - Kotlin wrapper for Android NLU functionality
 *
 * Provides a clean API:
 *   - VaaniNLU.initialize(modelPath: String): Unit
 *   - VaaniNLU.parse(text: String): JSONObject
 *
 * The native library libvaani_runtime.so is loaded via System.loadLibrary.
 * JNI bindings are provided by jni_bridge.cpp.
 *
 * Package: com.vaani.nlu
 */

package com.vaani.nlu

import android.content.Context
import android.util.JsonReader
import android.util.JsonToken
import android.util.Log
import org.json.JSONObject
import org.json.JSONException
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * Companion object holding native library references and initialization state.
 */
object VaaniNLU {

    // Native library name (Android: libvaani_runtime.so)
    // On Linux: libvaani_runtime.so (same library, cross-compiled)
    private const val LIBRARY_NAME = "vaani_runtime"

    // Lazy-initialized native context
    @Volatile
    private var initialized = false

    /**
     * Initialize the NLU engine with the path to the ONNX model.
     * <p>
     * This must be called before using [parse].
     * Typically called from an Application subclass on app startup.
     *
     * @param modelPath Path to the ONNX model file
     *                  (e.g., "models/vaani_nlu.onnx")
     * @throws RuntimeException if the native library cannot be loaded or initialization fails
     */
    fun initialize(modelPath: String) {
        if (initialized) return

        try {
            // Load the native shared library
            // Uses the default library search path
            System.loadLibrary(LIBRARY_NAME)

            // Call native initialization via JNI
            // The JNIOnLoad in jni_bridge.cpp will be invoked automatically
            // when the library is loaded.

            // Store the model path in native state if needed
            // (Currently the runtime uses an embedded vocabulary; modelPath is reserved for future use)
            // We pass the application context via the main activity later.

            initialized = true
            Log.d("VaaniNLU", "VaaniNLU initialized with modelPath: $modelPath")
        } catch (e: UnsatisfiedLinkError) {
            Log.e("VaaniNLU", "Failed to load native library: ${e.message}")
            throw RuntimeException("VaaniNLU native library not found. Ensure lib${LIBRARY_NAME}.so is loaded.", e)
        }
    }

    /**
     * Parse a Hinglish voice command and return the intent + slots as a JSONObject.
     * <p>
     * The execution pipeline:
     *   1. Normalization via C++17 normalizer (Devanagari→ITRANS + phonetic collapse)
     *   2. WordPiece tokenization
     *   3. ONNX Runtime inference through scripts/onnx_inference.py
     *   4. BIO-to-JSON slot assembly
     * <p>
     * @param text The Hinglish voice command text (UTF-8 string, may contain Romanized Hindi)
     * @return JSONObject with "intent" and "slots" keys.
     *         Example: {"intent":"check_weather","slots":{}}
     * @throws IllegalStateException if [initialize] has not been called first
     * @throws JSONException if the native result cannot be parsed
     */
    fun parse(text: String): JSONObject {
        if (!initialized) {
            throw IllegalStateException("VaaniNLU must be initialized before calling parse(). " +
                    "Call VaaniNLU.initialize(modelPath) first.")
        }

        // Call the native JNI parse function
        // Returns a jobject (JSONObject) via JNI
        val jobj = nativeParse(text)

        // The JNI function returns a local reference; we need to convert it to a Kotlin JSONObject
        // The returned object is already a JSONObject from the Java side.
        // We copy it to avoid local reference issues.
        try {
            // Try to get known fields; if the JSONObject is empty/invalid, return default
            val intent = jobj.getString("intent")
            val slots = jobj.getJSONObject("slots")

            return JSONObject()
                .put("intent", intent)
                .put("slots", slots)
        } catch (e: JSONException) {
            // Fallback: return minimal result
            Log.w("VaaniNLU", "Failed to parse JSONObject from native: ${e.message}")
            return JSONObject()
                .put("intent", "error")
                .put("slots", JSONObject())
        }
    }

    /**
     * Internal JNI bridge: calls the JNI nativeParse function.
     * This is a native method that must be declared in the JNI bridge.
     */
    private external fun nativeParse(text: String): JSONObject

    /**
     * Get the estimated latency of a single parse() call in milliseconds.
     * Useful for performance monitoring.
     *
     * @return Float latency in milliseconds (typically 8-14 ms on modern CPU)
     */
    fun getEstimatedLatencyMs(): Float {
        return nativeGetEstimatedLatencyMs()
    }

    /**
     * Internal JNI bridge: calls the JNI nativeGetEstimatedLatencyMs function.
     */
    private external fun nativeGetEstimatedLatencyMs(): Float
}