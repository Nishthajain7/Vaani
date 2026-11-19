# Contributing to Vaani

Thank you for wanting to contribute to Vaani! This project is an open-source Hinglish NLU runtime. Below you'll find guidelines on how to contribute, the development workflow, and how we handle issues and pull requests.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Development Setup](#development-setup)
3. [Building the Project](#building-the-project)
4. [Running Benchmarks](#running-benchmarks)
5. [Adding New Features](#adding-new-features)
6. [Reporting Issues](#reporting-issues)
7. [Pull Request Process](#pull-request-process)
8. [License](#license)

---

## Code of Conduct

This project is dedicated to providing a welcoming and open environment. Please read and follow the [Covenant of Code](https://contributor-covenant.org/version/2/1/code-of-conduct/), or other applicable code of conduct. We enforce this code of conduct across all project spaces.

---

## Development Setup

### Prerequisites

- **OS**: Linux (tested on Ubuntu/Debian), macOS, or Windows (WSL)
- **CMake**: >= 3.18
- **C++20 compiler** (GCC 10+, Clang 11+, MSVC 19.2+)
- **Python 3.8+** (for benchmark scripts and ONNX inference subprocess)
- **Git** (for cloning and contributing)

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/Nishthajain7/Vaani.git
cd Vaani

# Initialize submodules (if any) and install dependencies
git submodule update --init --recursive

# Create a build directory and configure
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
cmake --build . -j$(nproc)
```

### Directory Structure

```
Vaani/
├── bench/           # Benchmark scripts (run_benchmark.py)
├── data/            # Synthetic training/eval data
├── include/         # C++ headers (vaani_nlu.h)
├── models/          # ONNX model + vocab + label files
├── scripts/         # Python inference scripts
├── src/             # C++ source files (runtime.cpp, normalizer.cpp, cli.cpp)
├── tests/           # Unit / E2E tests
├── CMakeLists.txt   # Build configuration
└── CONTRIBUTING.md  # This file
```

---

## Building the Project

```bash
# From the project root
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

This produces:
- `build/libnormalizer.so` - C++17 Hinglish normalizer library
- `build/libvaani_runtime.so` - C ABI runtime library (the main artifact)
- `build/vaani_cli` - CLI tool for parsing commands

### Cross-Compilation for Android

```bash
# Requires Android NDK
cmake .. -DCMAKE_TOOLCHAIN_FILE=../android-ndk-r21b/build/cmake/android.toolchain.cmake \
  -DANDROID=ON -DANDROID_PLATFORM=android-21 -DANDROID_STL=c++_shared
cmake --build . -j$(nproc)
```

> **Note**: Android model assets must be placed in `assets/` and copied to the filesystem before calling `vaani_init()`. See issue #4 for the missing assets integration.

---

## Running Benchmarks

The benchmark suite compares three NLU pipelines:

```bash
# From the project root
python3 bench/run_benchmark.py
```

This produces:
- A Markdown table with intent accuracy, slot exact-match, token fertility, p50/p95 latency, and peak RSS memory
- Results appended to `README.md` under the `<!-- benchmark-results-start -->` marker

### Benchmark Passes

| Pass | Description |
|------|-------------|
| `raw` | Raw text → GGUF SLM (direct) |
| `normalized` | Normalized text → GGUF SLM |
| `full_cpp_pipeline` | CLI: `vaani_cli` → C++ pipeline output |

---

## Adding New Features

### 1. C++ Core Changes

- Follow the existing code style (C++20, no external dependencies beyond standard library and Python3)
- Add `extern "C"` wrappers for C ABI functions in `include/vaani_nlu.h`
- Thread-local state is used (`thread_local RuntimeState g_state`); document if your change affects threading
- ONNX Runtime C++ headers are **not** vendored; inference goes through `scripts/onnx_inference.py` — keep this Python subprocess path for now

### 2. Python Script Changes

- `scripts/onnx_inference.py` is the ONNX inference backend
- `scripts/vaani_tokenizer.py` contains the WordPiece tokenizer shared with the C++ runtime
- `scripts/generate_and_train.py` generates synthetic training data

### 3. New Models / Vocabulary

- Add tokens to `models/vocab.txt` (one per line, line number = token id)
- Regenerate or add intent/slot label JSON files in `models/`
- Update C++ runtime constants if slot/intent counts change

---

## Reporting Issues

### Before Opening an Issue

1. **Check existing issues** — search to avoid duplicates
2. **Search the codebase** — the problem may already be solved or documented
3. **Verify your environment** — ensure you're using the latest release

### How to File a Good Issue

- Use a clear, descriptive title
- Include steps to reproduce the problem
- Include `vaani_get_estimated_latency_ms()` output if relevant
- Include OS, compiler, and Python version
- Include relevant code snippets or JSON output
- Label appropriately (see labels used in this repo: `evaluation`, `synthetic-data`, `tokenizer`, `vocabulary`, `android`, `latency`, `benchmark`, etc.)

### Issue Labels (maintained by maintainers)

- `bug` — something isn't working
- `enhancement` — new feature or request
- `good first issue` — good for newcomers
- `help wanted` — extra attention needed
- `question` — further information requested
- `wontfix` — this will not be worked on

To add a new label: `gh label create --repo Nishthajain7/Vaani <label-name> --color <hex>`

---

## Pull Request Process

1. **Fork the repository** and create a branch from `main`
2. **Write a clear title and description** for the PR
3. **Include tests** if applicable (benchmark results, E2E test)
4. **Update documentation** if user-facing behavior changes
5. **Match the existing code style** (C++ style, Python conventions)
6. **Link any related issues** using ` closes #<number> ` in the description
7. **Wait for maintainer review** — address feedback and update commits as needed

### PR Checklist

- [ ] Code follows the project's C++/Python style
- [ ] Builds successfully (`cmake .. && cmake --build . -j`)
- [ ] Benchmarks pass (or new benchmarks are documented)
- [ ] Documentation updated if behavior changes
- [ ] No existing issues are blocked
- [ ] Related issues linked in description

---

## License

By contributing to Vaani, you agree that your contributions will be licensed under the [MIT License](LICENSE) that covers the project.

See `LICENSE` for the full copyright and license information.

---

## Questions?

- Open an issue on GitHub: https://github.com/Nishthajain7/Vaani/issues
- Discuss on the Discussions tab (if enabled)
- Email: maintainers@vaani.io (or your preferred contact)

Thank you for contributing!