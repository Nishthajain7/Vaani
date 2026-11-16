/**
 * src/cli.cpp - Cross-platform CLI tool for local terminal testing of Vaani NLU
 *
 * Usage:
 *   ./vaani_cli "set a reminder at 8 AM"
 *   ./vaani_cli "play some music"
 *   ./vaani_cli "send a message to neha"
 *   ./vaani_cli "what's the weather in delhi"
 *   ./vaani_cli "set an alarm at 7 PM"
 *
 * Links against libvaani_runtime.so (Linux) or the Android-built equivalent.
 *
 * Build (Linux):
 *   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build -j
 *
 * Build (Android via CMake): see CMakeLists.txt.
 */

#include <iostream>
#include <string>
#include <cstring>

// Include the C ABI header
extern "C" {
#include "vaani_nlu.h"
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " \"Hinglish text\"\n";
        std::cerr << "Example: " << argv[0] << " \"set a reminder at 8 AM\"\n";
        return 1;
    }

    // Grab the input text (safely handle spaces)
    std::string raw_text;
    for (int i = 1; i < argc; ++i) {
        if (i > 1) raw_text += " ";
        raw_text += argv[i];
    }

    std::cout << "=== Vaani NLU CLI ===\n";
    std::cout << "Input: \"" << raw_text << "\"\n\n";

    // Step 1: Get estimated latency
    float latency = vaani_get_estimated_latency_ms();
    std::cout << "Estimated latency: " << latency << " ms\n\n";

    // Step 2: Parse the input
    const char* result = vaani_parse(raw_text.c_str());

    if (!result) {
        std::cerr << "Error: vaani_parse() returned nullptr\n";
        return 1;
    }

    std::cout << "Result: " << result << "\n\n";

    // Step 3: Free the result (no-op in our design, but kept for ABI compliance)
    vaani_free_result(result);

    return 0;
}