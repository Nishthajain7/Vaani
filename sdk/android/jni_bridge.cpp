/**
 * sdk/android/jni_bridge.cpp - Android JNI bridge for libvaani_runtime
 *
 * Provides JNI interfaces that allow Java/Kotlin code to call:
 *   - VaaniNLU.initialize(String modelPath)
 *   - VaaniNLU.parse(String text): JSONObject
 *
 * This file bridges the C ABI (vaani_parse, vaani_free_result, vaani_get_estimated_latency_ms)
 * with Java native interfaces.
 *
 * Build: Android NDK with CMakeLists.txt targeting arm64-v8a and x86_64
 */

#include <jni.h>
#include <string>
#include <cstring>

// Forward declarations of the C ABI functions from libvaani_runtime
extern "C" {
    extern const char* vaani_parse(const char* raw_text);
    extern void vaani_free_result(const char* json_str);
    extern float vaani_get_estimated_latency_ms();
}

/**
 * @brief JNI-onload: called when the shared library is loaded.
 * Stores the JNIEnv and JavaVM for later use.
 */
static JavaVM* g_jvm = nullptr;

/**
 * @brief Context object holding global references to Java objects.
 * Passed through JNI calls from Kotlin.
 */
struct JniContext {
    JavaVM* jvm;
    jobject application_context; // Global reference
};

// Thread-local context pointer
static thread_local JniContext* g_ctx = nullptr;

// Set the JNI context from Java side
extern "C" void Java_com_vaani_VaaniNLU_nativeSetContext(JNIEnv* env, jobject /*thiz*/, jobject context) {
    // Store the JavaVM and ApplicationContext
    g_ctx = new JniContext();
    g_ctx->jvm = env->getJavaVM();
    // We keep a global reference to the application context
    if (g_ctx->application_context) {
        env->DeleteGlobalRef(g_ctx->application_context);
    }
    g_ctx->application_context = env->NewGlobalRef(context);
}

/**
 * @brief Parse a Hinglish text string and return a JSON object.
 *
 * This is the main JNI function called from Kotlin.
 * It calls vaani_parse() and parses the returned JSON string into a jobject.
 *
 * @param env JNI environment pointer
 * @param thiz The VaaniNLU instance (unused in static approach)
 * @param text The Hinglish text to parse (UTF-16 Java string)
 * @return jobject JSONObject instance containing intent and slots
 */
extern "C" jobject Java_com_vaani_VaaniNLU_nativeParse(JNIEnv* env, jobject /*thiz*/, jstring text) {
    if (!text || !g_ctx) {
        // Return empty JSONObject
        jclass cls = env->FindClass("org/json/JSONObject");
        if (!cls) return nullptr;
        jmethodID init = env->GetMethodID(cls, "<init>", "(Ljava/lang/String;)V");
        if (!init) return nullptr;
        jstring empty = env->NewStringUTF("{}");
        if (!empty) return nullptr;
        jobject obj = env->NewObject(cls, init, empty);
        env->DeleteLocalRef(empty);
        return obj;
    }

    // Convert Java string to UTF-8 C string
    const char* raw_text = env->GetStringUTFChars(text, nullptr);
    if (!raw_text) {
        return nullptr;
    }

    // Call the C ABI function
    const char* json_str = vaani_parse(raw_text);

    // Release the UTF-8 string
    env->ReleaseStringUTFChars(text, raw_text);

    if (!json_str) {
        // Return empty JSONObject
        jclass cls = env->FindClass("org/json/JSONObject");
        if (!cls) return nullptr;
        jmethodID init = env->GetMethodID(cls, "<init>", "(Ljava/lang/String;)V");
        if (!init) return nullptr;
        jstring empty = env->NewStringUTF("{}");
        if (!empty) return nullptr;
        jobject obj = env->NewObject(cls, init, empty);
        env->DeleteLocalRef(empty);
        return obj;
    }

    // Create a Java string from the JSON result
    jstring jjson = env->NewStringUTF(json_str);

    // Free the result buffer (no-op in our design, but kept for ABI compliance)
    vaani_free_result(json_str);

    // Parse the JSON string into a JSONObject
    jclass cls = env->FindClass("org/json/JSONObject");
    if (!cls) return nullptr;

    jmethodID init = env->GetMethodID(cls, "<init>", "(Ljava/lang/String;)V");
    if (!init) {
        env->DeleteLocalRef(jjson);
        return nullptr;
    }

    jobject jobject_result = env->NewObject(cls, init, jjson);
    env->DeleteLocalRef(jjson);
    env->DeleteLocalRef(cls);

    return jobject_result;
}

/**
 * @brief Get the estimated latency in milliseconds.
 * Called from Kotlin to report performance metrics.
 */
extern "C" jfloat Java_com_vaani_VaaniNLU_nativeGetEstimatedLatencyMs(JNIEnv* env, jobject /*thiz*/) {
    if (g_ctx && g_ctx->jvm) {
        return (jfloat)vaani_get_estimated_latency_ms();
    }
    return 12.0f; // default estimate
}

/**
 * @brief Native cleanup: called when the library is unloaded.
 * Frees the thread-local context.
 */
extern "C" void Java_com_vaani_VaaniNLU_nativeCleanup(JNIEnv* env, jobject /*thiz*/) {
    if (g_ctx) {
        if (g_ctx->application_context) {
            env->DeleteGlobalRef(g_ctx->application_context);
        }
        delete g_ctx;
        g_ctx = nullptr;
    }
}