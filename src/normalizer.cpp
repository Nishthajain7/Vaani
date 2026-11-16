/**
 * src/normalizer.cpp
 * Hinglish text normalizer - C++17, no external dependencies.
 *
 * Provides:
 *   1. Devanagari -> ITRANS Romanization
 *   2. Whitespace canonicalization
 *   3. A small Hinglish spelling map (baaje -> baje)
 *
 * It deliberately does NOT collapse doubled letters in arbitrary words:
 * doing so corrupts English tokens ("message" -> "mesage", "mom" -> "mo").
 * Case is preserved so slot values can be reconstructed verbatim.
 */

#include <cstdint>
#include <cctype>
#include <iostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace hinglish::normalizer {

namespace {

// ──────────────────────────────────────────────
// UTF-8 helpers
// ──────────────────────────────────────────────
std::size_t decode_utf8(std::string_view s, std::size_t i, std::uint32_t& cp) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
        cp = c;
        return i + 1;
    }
    int extra = 0;
    if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; extra = 1; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; extra = 2; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; extra = 3; }
    else { cp = 0xFFFD; return i + 1; }

    if (i + extra >= s.size()) { cp = 0xFFFD; return i + 1; }
    for (int k = 1; k <= extra; ++k) {
        const unsigned char cc = static_cast<unsigned char>(s[i + k]);
        if ((cc & 0xC0) != 0x80) { cp = 0xFFFD; return i + k; }
        cp = (cp << 6) | (cc & 0x3F);
    }
    return i + extra + 1;
}

void append_utf8(std::string& out, std::uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// ──────────────────────────────────────────────
// Devanagari -> ITRANS tables
// ──────────────────────────────────────────────
const char* consonant(std::uint32_t cp) {
    switch (cp) {
        case 0x0915: return "k";   case 0x0916: return "kh";
        case 0x0917: return "g";   case 0x0918: return "gh";
        case 0x0919: return "ng";
        case 0x091A: return "ch";  case 0x091B: return "chh";
        case 0x091C: return "j";   case 0x091D: return "jh";
        case 0x091E: return "ny";
        case 0x091F: return "t";   case 0x0920: return "th";
        case 0x0921: return "d";   case 0x0922: return "dh";
        case 0x0923: return "n";
        case 0x0924: return "t";   case 0x0925: return "th";
        case 0x0926: return "d";   case 0x0927: return "dh";
        case 0x0928: return "n";
        case 0x0929: return "n";
        case 0x092A: return "p";   case 0x092B: return "ph";
        case 0x092C: return "b";   case 0x092D: return "bh";
        case 0x092E: return "m";
        case 0x092F: return "y";   case 0x0930: return "r";
        case 0x0931: return "r";
        case 0x0932: return "l";   case 0x0933: return "l";
        case 0x0934: return "l";
        case 0x0935: return "v";   case 0x0936: return "sh";
        case 0x0937: return "sh";  case 0x0938: return "s";
        case 0x0939: return "h";
        case 0x0958: return "q";   case 0x0959: return "kh";
        case 0x095A: return "gh";  case 0x095B: return "z";
        case 0x095C: return "r";   case 0x095D: return "rh";
        case 0x095E: return "f";   case 0x095F: return "y";
        default: return nullptr;
    }
}

const char* independent_vowel(std::uint32_t cp) {
    switch (cp) {
        case 0x0905: return "a";   case 0x0906: return "aa";
        case 0x0907: return "i";   case 0x0908: return "ii";
        case 0x0909: return "u";   case 0x090A: return "uu";
        case 0x090B: return "ri";  case 0x090D: return "e";
        case 0x090F: return "e";   case 0x0910: return "ai";
        case 0x0911: return "o";   case 0x0912: return "o";
        case 0x0913: return "o";   case 0x0914: return "au";
        default: return nullptr;
    }
}

const char* matra(std::uint32_t cp) {
    switch (cp) {
        case 0x093E: return "aa";  case 0x093F: return "i";
        case 0x0940: return "ii";  case 0x0941: return "u";
        case 0x0942: return "uu";  case 0x0943: return "ri";
        case 0x0945: return "e";   case 0x0947: return "e";
        case 0x0948: return "ai";  case 0x0949: return "o";
        case 0x094B: return "o";   case 0x094C: return "au";
        case 0x094E: return "e";   case 0x094F: return "au";
        default: return nullptr;
    }
}

bool is_devanagari(std::uint32_t cp) {
    return cp >= 0x0900 && cp <= 0x097F;
}

// Transliterate one Devanagari word. `word` is UTF-8.
std::string transliterate_word(std::string_view word) {
    std::string out;
    out.reserve(word.size());
    bool inherent_a = false;

    auto flush_inherent = [&]() {
        if (inherent_a) {
            out += 'a';
            inherent_a = false;
        }
    };

    std::size_t i = 0;
    while (i < word.size()) {
        std::uint32_t cp = 0;
        i = decode_utf8(word, i, cp);

        if (const char* c = consonant(cp)) {
            flush_inherent();
            out += c;
            inherent_a = true;
            continue;
        }
        if (const char* m = matra(cp)) {
            // A matra replaces the consonant's inherent "a".
            inherent_a = false;
            out += m;
            continue;
        }
        if (const char* v = independent_vowel(cp)) {
            flush_inherent();
            out += v;
            continue;
        }
        if (cp == 0x094D) {  // virama / halant
            inherent_a = false;
            continue;
        }
        if (cp == 0x0902 || cp == 0x0901) {  // anusvara / candrabindu
            flush_inherent();
            out += 'n';
            continue;
        }
        if (cp == 0x0903) {  // visarga
            flush_inherent();
            out += 'h';
            continue;
        }
        if (cp == 0x093C || cp == 0x200C || cp == 0x200D) {  // nukta / ZWJ
            continue;
        }

        flush_inherent();
        if (is_devanagari(cp)) {
            continue;  // unsupported Devanagari sign: drop
        }
        append_utf8(out, cp);
    }
    flush_inherent();

    // Hindi schwa deletion: drop a final inherent "a" on longer words.
    if (out.size() > 3 && out.back() == 'a') {
        out.pop_back();
    }
    return out;
}

// ──────────────────────────────────────────────
// Small Hinglish spelling map (applied to whole words, case-insensitive)
// ──────────────────────────────────────────────
const std::unordered_map<std::string, std::string>& hinglish_map() {
    static const std::unordered_map<std::string, std::string> map = {
        {"baaje", "baje"}, {"bajae", "baje"},
        {"thodo", "thoda"}, {"accha", "accha"}, {"achca", "accha"},
        {"ajkal", "aajkal"},
    };
    return map;
}

std::string lower_ascii(std::string_view s) {
    std::string out(s);
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

}  // namespace

std::string normalize(std::string_view input) {
    std::string out;
    out.reserve(input.size());

    std::size_t i = 0;
    bool pending_space = false;

    while (i < input.size()) {
        std::uint32_t cp = 0;
        const std::size_t next = decode_utf8(input, i, cp);

        if (cp < 128 && std::isspace(static_cast<unsigned char>(cp))) {
            if (!out.empty()) pending_space = true;
            i = next;
            continue;
        }

        if (pending_space) {
            out += ' ';
            pending_space = false;
        }

        if (is_devanagari(cp)) {
            // Collect the whole Devanagari word and transliterate it.
            const std::size_t start = i;
            std::size_t j = i;
            while (j < input.size()) {
                std::uint32_t wcp = 0;
                const std::size_t jnext = decode_utf8(input, j, wcp);
                if (!is_devanagari(wcp)) break;
                j = jnext;
            }
            out += transliterate_word(input.substr(start, j - start));
            i = j;
            continue;
        }

        // ASCII word: copy, then apply the small spelling map.
        if (cp < 128 && (std::isalnum(static_cast<unsigned char>(cp)) ||
                         cp == '\'' || cp == ':' || cp == '.')) {
            const std::size_t start = i;
            std::size_t j = i;
            while (j < input.size()) {
                std::uint32_t wcp = 0;
                const std::size_t jnext = decode_utf8(input, j, wcp);
                if (!(wcp < 128 && (std::isalnum(static_cast<unsigned char>(wcp)) ||
                                    wcp == '\'' || wcp == ':' || wcp == '.'))) {
                    break;
                }
                j = jnext;
            }
            std::string word(input.substr(start, j - start));
            const std::string key = lower_ascii(word);
            auto it = hinglish_map().find(key);
            out += (it != hinglish_map().end()) ? it->second : word;
            i = j;
            continue;
        }

        append_utf8(out, cp);
        i = next;
    }

    return out;
}

// ──────────────────────────────────────────────
// Extern C ABI for Python ctypes and cross-language FFI
// ──────────────────────────────────────────────
extern "C" {
    __attribute__((visibility("default")))
    const char* normalize_text(const char* input) {
        if (!input) return "";
        static thread_local std::string result;
        result = hinglish::normalizer::normalize(input);
        return result.c_str();
    }
}

}  // namespace hinglish::normalizer
