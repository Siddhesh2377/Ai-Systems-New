# dLLM Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone C++ diffusion language model inference engine on raw ggml, test via ADB on Android.

**Architecture:** Pure ggml computation graph with bidirectional attention (no causal mask). MDLM sampling loop with Gumbel noise and confidence-based remasking. Own GGUF reader for Qwen3-0.6B-diffusion model weights.

**Tech Stack:** C++17, ggml (from `/home/home/dev/include/llama.cpp/ggml/`), CMake, Android NDK arm64-v8a, Python (conversion script only)

**Source directory:** `/home/home/dev/include/dllm/`

**ggml dependency:** `/home/home/dev/include/llama.cpp/ggml/`

---

## Task 1: Project Skeleton and CMake Build

**Files:**
- Create: `/home/home/dev/include/dllm/CMakeLists.txt`
- Create: `/home/home/dev/include/dllm/dllm.h`
- Create: `/home/home/dev/include/dllm/dllm_main.cpp`

**Step 1: Create the dllm directory**

Run:
```bash
mkdir -p /home/home/dev/include/dllm
```

**Step 2: Write CMakeLists.txt**

Create `/home/home/dev/include/dllm/CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.18)
project(dllm LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ggml — use the one from llama.cpp
set(GGML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/ggml)

# Build ggml as subdirectory
# Disable backends we don't need yet (enable later)
option(GGML_VULKAN "Enable Vulkan backend" OFF)
option(GGML_OPENCL "Enable OpenCL backend" OFF)
option(GGML_CUDA "Enable CUDA backend" OFF)
option(GGML_METAL "Enable Metal backend" OFF)

add_subdirectory(${GGML_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ggml_build)

# dllm static library
add_library(dllm STATIC
    dllm_model.cpp
    dllm_graph.cpp
    dllm_sampler.cpp
    dllm_tokenizer.cpp
)
target_include_directories(dllm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GGML_DIR}/include
)
target_link_libraries(dllm PUBLIC ggml)

# CLI test executable
add_executable(dllm_cli dllm_main.cpp)
target_link_libraries(dllm_cli PRIVATE dllm)
```

**Step 3: Write the public header (dllm.h) — just the struct declarations and forward decls**

Create `/home/home/dev/include/dllm/dllm.h`:
```cpp
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque context
typedef struct dllm_context dllm_context;

// Model hyperparameters (read from GGUF)
typedef struct {
    int32_t n_vocab;
    int32_t n_embd;
    int32_t n_head;
    int32_t n_head_kv;
    int32_t n_layer;
    int32_t n_ff;
    int32_t n_ctx_max;
    int32_t mask_token_id;
    float   rope_theta;
    float   rms_norm_eps;
} dllm_hparams;

// Engine parameters
typedef struct {
    int32_t n_threads;       // CPU threads (0 = auto)
    int32_t n_ctx;           // context size to allocate
    bool    use_mmap;        // memory-map model file
} dllm_params;

// Sampling parameters for MDLM
typedef struct {
    int32_t n_steps;         // diffusion steps (default: 64)
    int32_t gen_length;      // tokens to generate
    float   temperature;     // Gumbel temperature (0.0 = greedy)
    int32_t remasking;       // 0 = low_confidence, 1 = random
} dllm_sampling;

// Callback: called after each diffusion step
// Return false to cancel generation
typedef bool (*dllm_step_cb)(int step, int total_steps, const int32_t * tokens,
                             int n_tokens, void * user_data);

// Get default params
dllm_params    dllm_default_params(void);
dllm_sampling  dllm_default_sampling(void);

// Lifecycle
dllm_context * dllm_init(dllm_params params);
int            dllm_load_model(dllm_context * ctx, const char * gguf_path);
void           dllm_free(dllm_context * ctx);

// Info
const dllm_hparams * dllm_get_hparams(const dllm_context * ctx);
bool                 dllm_is_loaded(const dllm_context * ctx);

// Tokenize / detokenize
int  dllm_tokenize(const dllm_context * ctx, const char * text,
                   int32_t * tokens, int max_tokens);
int  dllm_detokenize(const dllm_context * ctx, const int32_t * tokens,
                     int n_tokens, char * buf, int buf_size);

// Generate (MDLM diffusion)
int  dllm_generate(dllm_context * ctx, const int32_t * prompt_tokens,
                   int n_prompt, dllm_sampling sampling,
                   dllm_step_cb callback, void * user_data,
                   int32_t * out_tokens, int max_out);

// Cancel (thread-safe)
void dllm_cancel(dllm_context * ctx);

#ifdef __cplusplus
}
#endif
```

**Step 4: Write minimal dllm_main.cpp stub**

Create `/home/home/dev/include/dllm/dllm_main.cpp`:
```cpp
#include "dllm.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s --model <path.gguf> [--prompt <text>] [--steps <N>] [--gen-length <N>] [--temp <float>]\n", prog);
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * prompt      = "Hello, how are you?";
    int          steps       = 64;
    int          gen_length  = 64;
    float        temperature = 0.0f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gen-length") == 0 && i + 1 < argc) {
            gen_length = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            temperature = (float)atof(argv[++i]);
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path) {
        print_usage(argv[0]);
        return 1;
    }

    printf("[dllm] Loading model: %s\n", model_path);

    dllm_params params = dllm_default_params();
    dllm_context * ctx = dllm_init(params);
    if (!ctx) {
        fprintf(stderr, "[dllm] Failed to init context\n");
        return 1;
    }

    if (dllm_load_model(ctx, model_path) != 0) {
        fprintf(stderr, "[dllm] Failed to load model\n");
        dllm_free(ctx);
        return 1;
    }

    const dllm_hparams * hp = dllm_get_hparams(ctx);
    printf("[dllm] Model loaded: n_vocab=%d n_embd=%d n_layer=%d n_head=%d mask_id=%d\n",
           hp->n_vocab, hp->n_embd, hp->n_layer, hp->n_head, hp->mask_token_id);

    // Tokenize prompt
    int32_t prompt_tokens[1024];
    int n_prompt = dllm_tokenize(ctx, prompt, prompt_tokens, 1024);
    if (n_prompt < 0) {
        fprintf(stderr, "[dllm] Tokenize failed\n");
        dllm_free(ctx);
        return 1;
    }
    printf("[dllm] Prompt tokens: %d\n", n_prompt);

    // Generate
    dllm_sampling samp = dllm_default_sampling();
    samp.n_steps    = steps;
    samp.gen_length = gen_length;
    samp.temperature = temperature;

    int32_t out_tokens[2048];
    auto step_cb = [](int step, int total, const int32_t * tokens,
                      int n_tokens, void * ud) -> bool {
        printf("\r[dllm] Step %d/%d", step + 1, total);
        fflush(stdout);
        return true;
    };

    int n_gen = dllm_generate(ctx, prompt_tokens, n_prompt, samp,
                              step_cb, nullptr, out_tokens, 2048);
    printf("\n");

    if (n_gen < 0) {
        fprintf(stderr, "[dllm] Generation failed\n");
        dllm_free(ctx);
        return 1;
    }

    // Detokenize and print
    char output_text[8192];
    int len = dllm_detokenize(ctx, out_tokens, n_gen, output_text, 8192);
    if (len > 0) {
        printf("[dllm] Generated (%d tokens):\n%s\n", n_gen, output_text);
    }

    dllm_free(ctx);
    return 0;
}
```

**Step 5: Create stub .cpp files so CMake compiles**

Create stubs for `dllm_model.cpp`, `dllm_graph.cpp`, `dllm_sampler.cpp`, `dllm_tokenizer.cpp`:

Each stub implements the API functions from dllm.h with placeholder returns.

```cpp
// dllm_model.cpp — stub
#include "dllm.h"
#include <cstdlib>
#include <cstring>

struct dllm_context {
    dllm_hparams hparams;
    dllm_params  params;
    bool         loaded;
    // TODO: ggml contexts, weight tensors, tokenizer data
};

dllm_params dllm_default_params(void) {
    return { .n_threads = 4, .n_ctx = 2048, .use_mmap = true };
}

dllm_sampling dllm_default_sampling(void) {
    return { .n_steps = 64, .gen_length = 64, .temperature = 0.0f, .remasking = 0 };
}

dllm_context * dllm_init(dllm_params params) {
    auto * ctx = (dllm_context *)calloc(1, sizeof(dllm_context));
    ctx->params = params;
    return ctx;
}

int dllm_load_model(dllm_context * ctx, const char * path) {
    // TODO: implement GGUF loading
    fprintf(stderr, "[dllm] dllm_load_model not yet implemented\n");
    return -1;
}

void dllm_free(dllm_context * ctx) {
    if (ctx) free(ctx);
}

const dllm_hparams * dllm_get_hparams(const dllm_context * ctx) {
    return &ctx->hparams;
}

bool dllm_is_loaded(const dllm_context * ctx) {
    return ctx && ctx->loaded;
}

void dllm_cancel(dllm_context * ctx) {
    // TODO: atomic cancel flag
}
```

```cpp
// dllm_graph.cpp — stub
#include "dllm.h"
// TODO: build_forward() — ggml graph for bidirectional transformer
```

```cpp
// dllm_sampler.cpp — stub
#include "dllm.h"

int dllm_generate(dllm_context * ctx, const int32_t * prompt_tokens,
                  int n_prompt, dllm_sampling sampling,
                  dllm_step_cb callback, void * user_data,
                  int32_t * out_tokens, int max_out) {
    // TODO: implement MDLM diffusion loop
    fprintf(stderr, "[dllm] dllm_generate not yet implemented\n");
    return -1;
}
```

```cpp
// dllm_tokenizer.cpp — stub
#include "dllm.h"

int dllm_tokenize(const dllm_context * ctx, const char * text,
                  int32_t * tokens, int max_tokens) {
    // TODO: BPE tokenizer
    fprintf(stderr, "[dllm] dllm_tokenize not yet implemented\n");
    return -1;
}

int dllm_detokenize(const dllm_context * ctx, const int32_t * tokens,
                    int n_tokens, char * buf, int buf_size) {
    // TODO: BPE detokenizer
    fprintf(stderr, "[dllm] dllm_detokenize not yet implemented\n");
    return -1;
}
```

**Step 6: Build and verify compilation**

Run:
```bash
cd /home/home/dev/include/dllm
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target dllm_cli 2>&1 | tail -5
```
Expected: Compiles successfully, links against ggml.

**Step 7: Commit**

```bash
cd /home/home/dev/include/dllm
git init
git add CMakeLists.txt dllm.h dllm_main.cpp dllm_model.cpp dllm_graph.cpp dllm_sampler.cpp dllm_tokenizer.cpp
git commit -m "feat: dllm project skeleton with API and build system"
```

---

## Task 2: GGUF Model Conversion Script

**Files:**
- Create: `/home/home/dev/include/dllm/scripts/convert_to_gguf.py`

**Step 1: Write the conversion script**

This converts `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` from HuggingFace safetensors to GGUF format.

Create `/home/home/dev/include/dllm/scripts/convert_to_gguf.py`:
```python
#!/usr/bin/env python3
"""Convert dllm-hub diffusion model (HuggingFace safetensors) to GGUF format.

Usage:
    python convert_to_gguf.py --model dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 --output model.gguf [--type q8_0]

Requires: pip install safetensors numpy transformers
"""

import argparse
import json
import struct
import numpy as np
from pathlib import Path

# GGUF constants
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF value types
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8

# ggml tensor types
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8

# ggml type info: (type_id, block_size, type_size_bytes)
GGML_TYPE_INFO = {
    "f32":  (GGML_TYPE_F32,  1, 4),
    "f16":  (GGML_TYPE_F16,  1, 2),
    "q4_0": (GGML_TYPE_Q4_0, 32, 18),  # 32 weights in 18 bytes (4-bit + 1 scale f16)
    "q8_0": (GGML_TYPE_Q8_0, 32, 34),  # 32 weights in 34 bytes (8-bit + 1 scale f16)
}

# HuggingFace → GGUF weight name mapping (Qwen3 architecture)
def hf_to_gguf_name(hf_name: str) -> str:
    """Map HuggingFace weight names to GGUF standard names."""
    # Embedding
    if hf_name == "model.embed_tokens.weight":
        return "token_embd.weight"
    # Output norm
    if hf_name == "model.norm.weight":
        return "output_norm.weight"
    # LM head
    if hf_name == "lm_head.weight":
        return "output.weight"

    # Layer weights: model.layers.{i}.{component}
    if hf_name.startswith("model.layers."):
        parts = hf_name.split(".")
        layer_idx = parts[2]
        rest = ".".join(parts[3:])

        mapping = {
            "input_layernorm.weight":           f"blk.{layer_idx}.attn_norm.weight",
            "self_attn.q_proj.weight":          f"blk.{layer_idx}.attn_q.weight",
            "self_attn.q_proj.bias":            f"blk.{layer_idx}.attn_q.bias",
            "self_attn.k_proj.weight":          f"blk.{layer_idx}.attn_k.weight",
            "self_attn.k_proj.bias":            f"blk.{layer_idx}.attn_k.bias",
            "self_attn.v_proj.weight":          f"blk.{layer_idx}.attn_v.weight",
            "self_attn.v_proj.bias":            f"blk.{layer_idx}.attn_v.bias",
            "self_attn.o_proj.weight":          f"blk.{layer_idx}.attn_output.weight",
            "post_attention_layernorm.weight":   f"blk.{layer_idx}.ffn_norm.weight",
            "mlp.gate_proj.weight":             f"blk.{layer_idx}.ffn_gate.weight",
            "mlp.up_proj.weight":               f"blk.{layer_idx}.ffn_up.weight",
            "mlp.down_proj.weight":             f"blk.{layer_idx}.ffn_down.weight",
        }
        if rest in mapping:
            return mapping[rest]

    return None  # skip unknown weights


def write_string(f, s: str):
    """Write GGUF string: uint64 length + bytes (no null terminator)."""
    encoded = s.encode("utf-8")
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)


def write_kv_string(f, key: str, value: str):
    """Write a string KV pair."""
    write_string(f, key)
    f.write(struct.pack("<i", GGUF_TYPE_STRING))
    write_string(f, value)


def write_kv_uint32(f, key: str, value: int):
    write_string(f, key)
    f.write(struct.pack("<i", GGUF_TYPE_UINT32))
    f.write(struct.pack("<I", value))


def write_kv_int32(f, key: str, value: int):
    write_string(f, key)
    f.write(struct.pack("<i", GGUF_TYPE_INT32))
    f.write(struct.pack("<i", value))


def write_kv_float32(f, key: str, value: float):
    write_string(f, key)
    f.write(struct.pack("<i", GGUF_TYPE_FLOAT32))
    f.write(struct.pack("<f", value))


def write_kv_bool(f, key: str, value: bool):
    write_string(f, key)
    f.write(struct.pack("<i", GGUF_TYPE_BOOL))
    f.write(struct.pack("<b", 1 if value else 0))


def quantize_q8_0(data: np.ndarray) -> bytes:
    """Quantize float32 array to Q8_0 format.
    Q8_0: blocks of 32 values, each block = 1 f16 scale + 32 int8 values = 34 bytes.
    """
    data = data.flatten().astype(np.float32)
    n = len(data)
    assert n % 32 == 0, f"Tensor size {n} not divisible by 32"
    n_blocks = n // 32

    result = bytearray()
    for i in range(n_blocks):
        block = data[i * 32:(i + 1) * 32]
        amax = np.max(np.abs(block))
        scale = amax / 127.0 if amax != 0 else 0.0
        # Write scale as float16
        result += np.float16(scale).tobytes()
        # Quantize and write int8 values
        if scale != 0:
            quantized = np.round(block / scale).astype(np.int8)
        else:
            quantized = np.zeros(32, dtype=np.int8)
        result += quantized.tobytes()

    return bytes(result)


def main():
    parser = argparse.ArgumentParser(description="Convert HF diffusion model to GGUF")
    parser.add_argument("--model", required=True, help="HuggingFace model directory or repo ID")
    parser.add_argument("--output", required=True, help="Output GGUF file path")
    parser.add_argument("--type", default="f16", choices=["f32", "f16", "q8_0"],
                        help="Quantization type for weight tensors")
    args = parser.parse_args()

    model_dir = Path(args.model)

    # If it's a repo ID, download it
    if not model_dir.exists():
        print(f"Downloading {args.model} from HuggingFace...")
        from huggingface_hub import snapshot_download
        model_dir = Path(snapshot_download(args.model))

    # Load config
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    print(f"Model config: {json.dumps(config, indent=2)}")

    n_vocab    = config["vocab_size"]
    n_embd     = config["hidden_size"]
    n_head     = config["num_attention_heads"]
    n_head_kv  = config.get("num_key_value_heads", n_head)
    n_layer    = config["num_hidden_layers"]
    n_ff       = config["intermediate_size"]
    n_ctx      = config.get("max_position_embeddings", 4096)
    rope_theta = config.get("rope_theta", 1000000.0)
    rms_eps    = config.get("rms_norm_eps", 1e-6)

    # Get mask token id from tokenizer config
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    mask_token_id = -1
    if tokenizer_config_path.exists():
        with open(tokenizer_config_path) as f:
            tok_config = json.load(f)
        # Try to find mask_token_id
        if "mask_token_id" in tok_config:
            mask_token_id = tok_config["mask_token_id"]
        elif "mask_token" in tok_config:
            # Need to look it up in tokenizer
            pass

    # Also check special_tokens_map.json
    special_tokens_path = model_dir / "special_tokens_map.json"
    if special_tokens_path.exists() and mask_token_id == -1:
        with open(special_tokens_path) as f:
            special = json.load(f)
        print(f"Special tokens: {special}")

    # Fallback: check config for mask_token_id
    if mask_token_id == -1:
        mask_token_id = config.get("mask_token_id", -1)

    if mask_token_id == -1:
        # Last resort: try loading tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        if hasattr(tokenizer, "mask_token_id") and tokenizer.mask_token_id is not None:
            mask_token_id = tokenizer.mask_token_id
        else:
            print("WARNING: Could not find mask_token_id! Using vocab_size - 1")
            mask_token_id = n_vocab - 1

    print(f"mask_token_id = {mask_token_id}")

    # Load weights from safetensors
    from safetensors import safe_open
    import glob

    st_files = sorted(glob.glob(str(model_dir / "*.safetensors")))
    if not st_files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")

    print(f"Loading weights from {len(st_files)} safetensors file(s)...")

    # Collect all tensors with GGUF names
    tensors = {}  # gguf_name -> (shape, numpy_array)
    for st_file in st_files:
        with safe_open(st_file, framework="numpy") as f:
            for hf_name in f.keys():
                gguf_name = hf_to_gguf_name(hf_name)
                if gguf_name is None:
                    print(f"  SKIP: {hf_name}")
                    continue
                data = f.get_tensor(hf_name)
                tensors[gguf_name] = (list(data.shape), data)
                print(f"  {hf_name} -> {gguf_name}  shape={data.shape}  dtype={data.dtype}")

    print(f"\nTotal tensors: {len(tensors)}")

    # Determine quantization type
    quant_type_name = args.type
    quant_type_id, block_size, type_size = GGML_TYPE_INFO[quant_type_name]

    # Prepare tensor data
    tensor_infos = []  # (name, n_dims, shape, ggml_type, data_bytes)
    for name, (shape, data) in tensors.items():
        data_f32 = data.astype(np.float32)

        # Don't quantize 1D tensors (norms, biases) — keep as f32
        if len(shape) == 1:
            tensor_data = data_f32.tobytes()
            tensor_type = GGML_TYPE_F32
        elif quant_type_name == "f32":
            tensor_data = data_f32.tobytes()
            tensor_type = GGML_TYPE_F32
        elif quant_type_name == "f16":
            tensor_data = data_f32.astype(np.float16).tobytes()
            tensor_type = GGML_TYPE_F16
        elif quant_type_name == "q8_0":
            # Pad to multiple of 32 if needed
            flat = data_f32.flatten()
            pad_to = ((len(flat) + 31) // 32) * 32
            if pad_to != len(flat):
                flat = np.pad(flat, (0, pad_to - len(flat)))
            tensor_data = quantize_q8_0(flat)
            tensor_type = GGML_TYPE_Q8_0
        else:
            raise ValueError(f"Unknown type: {quant_type_name}")

        # GGUF uses reversed dimensions (row-major → col-major convention)
        gguf_shape = list(reversed(shape))
        tensor_infos.append((name, len(gguf_shape), gguf_shape, tensor_type, tensor_data))

    # Write GGUF file
    print(f"\nWriting GGUF to {args.output}...")

    # Count KV pairs
    kv_pairs = [
        ("general.architecture", "string", "dllm"),
        ("general.name", "string", "dllm-qwen3-0.6b-diffusion"),
        ("dllm.context_length", "uint32", n_ctx),
        ("dllm.embedding_length", "uint32", n_embd),
        ("dllm.block_count", "uint32", n_layer),
        ("dllm.attention.head_count", "uint32", n_head),
        ("dllm.attention.head_count_kv", "uint32", n_head_kv),
        ("dllm.feed_forward_length", "uint32", n_ff),
        ("dllm.attention.layer_norm_rms_epsilon", "float32", rms_eps),
        ("dllm.rope.freq_base", "float32", rope_theta),
        ("dllm.mask_token_id", "int32", mask_token_id),
        ("dllm.vocab_size", "uint32", n_vocab),
        ("dllm.variant", "string", "mdlm"),
        ("dllm.is_diffusion", "bool", True),
    ]

    n_kv = len(kv_pairs)
    n_tensors = len(tensor_infos)

    with open(args.output, "wb") as f:
        # Header
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<q", n_tensors))
        f.write(struct.pack("<q", n_kv))

        # KV pairs
        for key, vtype, value in kv_pairs:
            if vtype == "string":
                write_kv_string(f, key, value)
            elif vtype == "uint32":
                write_kv_uint32(f, key, value)
            elif vtype == "int32":
                write_kv_int32(f, key, value)
            elif vtype == "float32":
                write_kv_float32(f, key, value)
            elif vtype == "bool":
                write_kv_bool(f, key, value)

        # Tensor infos (metadata only, data comes later)
        # First compute offsets
        data_offset = 0
        tensor_offsets = []
        for name, n_dims, shape, ttype, data_bytes in tensor_infos:
            # Align to GGUF_DEFAULT_ALIGNMENT
            aligned_offset = ((data_offset + GGUF_DEFAULT_ALIGNMENT - 1)
                              // GGUF_DEFAULT_ALIGNMENT) * GGUF_DEFAULT_ALIGNMENT
            tensor_offsets.append(aligned_offset)
            data_offset = aligned_offset + len(data_bytes)

        for i, (name, n_dims, shape, ttype, data_bytes) in enumerate(tensor_infos):
            write_string(f, name)
            f.write(struct.pack("<I", n_dims))
            for dim in shape:
                f.write(struct.pack("<q", dim))
            f.write(struct.pack("<i", ttype))
            f.write(struct.pack("<Q", tensor_offsets[i]))

        # Pad to alignment before tensor data blob
        current_pos = f.tell()
        aligned_pos = ((current_pos + GGUF_DEFAULT_ALIGNMENT - 1)
                       // GGUF_DEFAULT_ALIGNMENT) * GGUF_DEFAULT_ALIGNMENT
        f.write(b"\x00" * (aligned_pos - current_pos))

        blob_start = f.tell()

        # Write tensor data
        for i, (name, n_dims, shape, ttype, data_bytes) in enumerate(tensor_infos):
            # Pad to offset
            target = blob_start + tensor_offsets[i]
            current = f.tell()
            if target > current:
                f.write(b"\x00" * (target - current))
            f.write(data_bytes)

    file_size = Path(args.output).stat().st_size
    print(f"Done! Output: {args.output} ({file_size / 1024 / 1024:.1f} MB)")
    print(f"  Tensors: {n_tensors}")
    print(f"  Quantization: {quant_type_name}")
    print(f"  Mask token ID: {mask_token_id}")


if __name__ == "__main__":
    main()
```

**Step 2: Test the conversion script**

Run:
```bash
cd /home/home/dev/include/dllm/scripts
pip install safetensors numpy transformers huggingface_hub
python convert_to_gguf.py --model dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 --output ../test_model.gguf --type f16
```
Expected: Downloads model, converts to GGUF, prints tensor mapping and file size.

**Step 3: Commit**

```bash
cd /home/home/dev/include/dllm
git add scripts/convert_to_gguf.py
git commit -m "feat: add HF-to-GGUF conversion script for diffusion models"
```

---

## Task 3: GGUF Model Loader

**Files:**
- Modify: `/home/home/dev/include/dllm/dllm_model.cpp`
- Create: `/home/home/dev/include/dllm/dllm_internal.h`

**Step 1: Create internal header with weight structure**

Create `/home/home/dev/include/dllm/dllm_internal.h`:
```cpp
#pragma once

#include "dllm.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <atomic>
#include <vector>
#include <string>
#include <unordered_map>

// Per-layer weights
struct dllm_layer {
    struct ggml_tensor * attn_norm;      // [n_embd]
    struct ggml_tensor * attn_q;         // [n_embd, n_embd]
    struct ggml_tensor * attn_k;         // [n_embd, n_kv_dim]
    struct ggml_tensor * attn_v;         // [n_embd, n_kv_dim]
    struct ggml_tensor * attn_output;    // [n_embd, n_embd]
    struct ggml_tensor * attn_q_bias;    // [n_embd] (optional, Qwen3 has QKV bias)
    struct ggml_tensor * attn_k_bias;    // [n_kv_dim] (optional)
    struct ggml_tensor * attn_v_bias;    // [n_kv_dim] (optional)
    struct ggml_tensor * ffn_norm;       // [n_embd]
    struct ggml_tensor * ffn_gate;       // [n_embd, n_ff]
    struct ggml_tensor * ffn_up;         // [n_embd, n_ff]
    struct ggml_tensor * ffn_down;       // [n_ff, n_embd]
};

// Model weights container
struct dllm_model {
    struct ggml_tensor * tok_embeddings;  // [n_vocab, n_embd]
    struct ggml_tensor * output_norm;     // [n_embd]
    struct ggml_tensor * output;          // [n_vocab, n_embd] (lm_head)
    std::vector<dllm_layer> layers;
};

// Tokenizer data
struct dllm_tokenizer {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    int32_t mask_token_id;
    int32_t bos_token_id;
    int32_t eos_token_id;
    int32_t pad_token_id;
    // BPE merges
    std::vector<std::pair<std::string, std::string>> merges;
};

// Full context
struct dllm_context {
    dllm_hparams   hparams;
    dllm_params    params;
    bool           loaded;
    std::atomic<bool> cancelled;

    // Model weights
    dllm_model     model;

    // Tokenizer
    dllm_tokenizer tokenizer;

    // ggml memory
    struct gguf_context * gguf_ctx;
    struct ggml_context * weight_ctx;   // holds tensor metadata
    ggml_backend_buffer_t weight_buf;   // holds tensor data (mmap'd)
    ggml_backend_t        backend_cpu;

    // Computation graph memory (reused each forward pass)
    struct ggml_context * compute_ctx;
    ggml_gallocr_t        compute_alloc;
};

// Internal functions
int  dllm_load_weights(dllm_context * ctx, const char * path);
int  dllm_load_tokenizer_from_gguf(dllm_context * ctx);

// Graph building
struct ggml_cgraph * dllm_build_forward(
    dllm_context * ctx,
    const int32_t * tokens,
    int n_tokens,
    struct ggml_context * gctx    // graph context for this forward pass
);

// Returns pointer to logits tensor data after compute
float * dllm_compute_forward(dllm_context * ctx, const int32_t * tokens, int n_tokens);
```

**Step 2: Implement GGUF weight loading in dllm_model.cpp**

Replace `/home/home/dev/include/dllm/dllm_model.cpp` with full implementation:
```cpp
#include "dllm_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// --- Default params ---

dllm_params dllm_default_params(void) {
    dllm_params p = {};
    p.n_threads = 4;
    p.n_ctx     = 2048;
    p.use_mmap  = true;
    return p;
}

dllm_sampling dllm_default_sampling(void) {
    dllm_sampling s = {};
    s.n_steps     = 64;
    s.gen_length  = 64;
    s.temperature = 0.0f;
    s.remasking   = 0;
    return s;
}

// --- Lifecycle ---

dllm_context * dllm_init(dllm_params params) {
    auto * ctx = new dllm_context();
    memset(&ctx->hparams, 0, sizeof(dllm_hparams));
    ctx->params = params;
    ctx->loaded = false;
    ctx->cancelled.store(false);
    ctx->gguf_ctx     = nullptr;
    ctx->weight_ctx   = nullptr;
    ctx->weight_buf   = nullptr;
    ctx->backend_cpu  = nullptr;
    ctx->compute_ctx  = nullptr;
    ctx->compute_alloc = nullptr;
    return ctx;
}

void dllm_free(dllm_context * ctx) {
    if (!ctx) return;
    if (ctx->compute_alloc) ggml_gallocr_free(ctx->compute_alloc);
    if (ctx->compute_ctx)   ggml_free(ctx->compute_ctx);
    if (ctx->weight_buf)    ggml_backend_buffer_free(ctx->weight_buf);
    if (ctx->weight_ctx)    ggml_free(ctx->weight_ctx);
    if (ctx->gguf_ctx)      gguf_free(ctx->gguf_ctx);
    if (ctx->backend_cpu)   ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

const dllm_hparams * dllm_get_hparams(const dllm_context * ctx) {
    return &ctx->hparams;
}

bool dllm_is_loaded(const dllm_context * ctx) {
    return ctx && ctx->loaded;
}

void dllm_cancel(dllm_context * ctx) {
    if (ctx) ctx->cancelled.store(true);
}

// --- Helper: read GGUF metadata ---

static int64_t gguf_find_key_or_fail(struct gguf_context * gctx, const char * key) {
    int64_t id = gguf_find_key(gctx, key);
    if (id < 0) {
        fprintf(stderr, "[dllm] Missing GGUF key: %s\n", key);
    }
    return id;
}

static uint32_t gguf_get_u32(struct gguf_context * gctx, const char * key, uint32_t fallback) {
    int64_t id = gguf_find_key(gctx, key);
    if (id < 0) return fallback;
    return gguf_get_val_u32(gctx, id);
}

static int32_t gguf_get_i32(struct gguf_context * gctx, const char * key, int32_t fallback) {
    int64_t id = gguf_find_key(gctx, key);
    if (id < 0) return fallback;
    return gguf_get_val_i32(gctx, id);
}

static float gguf_get_f32(struct gguf_context * gctx, const char * key, float fallback) {
    int64_t id = gguf_find_key(gctx, key);
    if (id < 0) return fallback;
    return gguf_get_val_f32(gctx, id);
}

// --- Model loading ---

int dllm_load_model(dllm_context * ctx, const char * path) {
    printf("[dllm] Loading GGUF: %s\n", path);

    // 1. Init CPU backend
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "[dllm] Failed to init CPU backend\n");
        return -1;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->params.n_threads);

    // 2. Load GGUF file
    struct gguf_init_params gparams = {
        .no_alloc = true,
        .ctx      = &ctx->weight_ctx,
    };
    ctx->gguf_ctx = gguf_init_from_file(path, gparams);
    if (!ctx->gguf_ctx) {
        fprintf(stderr, "[dllm] Failed to load GGUF file: %s\n", path);
        return -1;
    }

    // 3. Read hyperparameters from metadata
    dllm_hparams * hp = &ctx->hparams;
    hp->n_vocab      = (int32_t)gguf_get_u32(ctx->gguf_ctx, "dllm.vocab_size", 0);
    hp->n_embd       = (int32_t)gguf_get_u32(ctx->gguf_ctx, "dllm.embedding_length", 0);
    hp->n_head       = (int32_t)gguf_get_u32(ctx->gguf_ctx, "dllm.attention.head_count", 0);
    hp->n_head_kv    = (int32_t)gguf_get_u32(ctx->gguf_ctx, "dllm.attention.head_count_kv", hp->n_head);
    hp->n_layer      = (int32_t)gguf_get_u32(ctx->gguf_ctx, "dllm.block_count", 0);
    hp->n_ff         = (int32_t)gguf_get_u32(ctx->gguf_ctx, "dllm.feed_forward_length", 0);
    hp->n_ctx_max    = (int32_t)gguf_get_u32(ctx->gguf_ctx, "dllm.context_length", 4096);
    hp->mask_token_id = gguf_get_i32(ctx->gguf_ctx, "dllm.mask_token_id", -1);
    hp->rope_theta    = gguf_get_f32(ctx->gguf_ctx, "dllm.rope.freq_base", 1000000.0f);
    hp->rms_norm_eps  = gguf_get_f32(ctx->gguf_ctx, "dllm.attention.layer_norm_rms_epsilon", 1e-6f);

    if (hp->n_vocab == 0 || hp->n_embd == 0 || hp->n_layer == 0) {
        fprintf(stderr, "[dllm] Invalid model params: n_vocab=%d n_embd=%d n_layer=%d\n",
                hp->n_vocab, hp->n_embd, hp->n_layer);
        return -1;
    }

    printf("[dllm] Hyperparams: vocab=%d embd=%d layers=%d heads=%d/%d ff=%d mask_id=%d\n",
           hp->n_vocab, hp->n_embd, hp->n_layer, hp->n_head, hp->n_head_kv,
           hp->n_ff, hp->mask_token_id);

    // 4. Allocate weight buffer from CPU backend
    ctx->weight_buf = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx->weight_ctx, ggml_backend_cpu_buffer_type());
    if (!ctx->weight_buf) {
        fprintf(stderr, "[dllm] Failed to allocate weight buffer\n");
        return -1;
    }

    // 5. Load tensor data from GGUF file into buffer
    // Memory-map the file and copy tensor data
    FILE * fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[dllm] Cannot open file: %s\n", path);
        return -1;
    }

    // Get data section offset
    size_t data_offset = gguf_get_data_offset(ctx->gguf_ctx);

    int64_t n_tensors = gguf_get_n_tensors(ctx->gguf_ctx);
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx->gguf_ctx, i);
        size_t offset = data_offset + gguf_get_tensor_offset(ctx->gguf_ctx, i);

        struct ggml_tensor * tensor = ggml_get_tensor(ctx->weight_ctx, name);
        if (!tensor) {
            fprintf(stderr, "[dllm] Warning: tensor '%s' not found in context\n", name);
            continue;
        }

        size_t tensor_size = ggml_nbytes(tensor);
        fseek(fp, offset, SEEK_SET);
        size_t read = fread(tensor->data, 1, tensor_size, fp);
        if (read != tensor_size) {
            fprintf(stderr, "[dllm] Failed to read tensor '%s': expected %zu, got %zu\n",
                    name, tensor_size, read);
            fclose(fp);
            return -1;
        }
    }
    fclose(fp);

    // 6. Map named tensors to model struct
    dllm_model * m = &ctx->model;
    m->tok_embeddings = ggml_get_tensor(ctx->weight_ctx, "token_embd.weight");
    m->output_norm    = ggml_get_tensor(ctx->weight_ctx, "output_norm.weight");
    m->output         = ggml_get_tensor(ctx->weight_ctx, "output.weight");

    if (!m->tok_embeddings || !m->output_norm) {
        fprintf(stderr, "[dllm] Missing essential tensors (token_embd / output_norm)\n");
        return -1;
    }

    // If no separate lm_head, tie to embeddings
    if (!m->output) {
        printf("[dllm] No output.weight found — tying to token_embd\n");
        m->output = m->tok_embeddings;
    }

    m->layers.resize(hp->n_layer);
    char name_buf[256];
    for (int i = 0; i < hp->n_layer; i++) {
        dllm_layer & layer = m->layers[i];

        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_norm.weight", i);
        layer.attn_norm = ggml_get_tensor(ctx->weight_ctx, name_buf);

        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_q.weight", i);
        layer.attn_q = ggml_get_tensor(ctx->weight_ctx, name_buf);

        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_k.weight", i);
        layer.attn_k = ggml_get_tensor(ctx->weight_ctx, name_buf);

        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_v.weight", i);
        layer.attn_v = ggml_get_tensor(ctx->weight_ctx, name_buf);

        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_output.weight", i);
        layer.attn_output = ggml_get_tensor(ctx->weight_ctx, name_buf);

        // Optional biases (Qwen3 has Q/K/V bias)
        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_q.bias", i);
        layer.attn_q_bias = ggml_get_tensor(ctx->weight_ctx, name_buf);  // may be null

        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_k.bias", i);
        layer.attn_k_bias = ggml_get_tensor(ctx->weight_ctx, name_buf);

        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_v.bias", i);
        layer.attn_v_bias = ggml_get_tensor(ctx->weight_ctx, name_buf);

        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_norm.weight", i);
        layer.ffn_norm = ggml_get_tensor(ctx->weight_ctx, name_buf);

        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_gate.weight", i);
        layer.ffn_gate = ggml_get_tensor(ctx->weight_ctx, name_buf);

        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_up.weight", i);
        layer.ffn_up = ggml_get_tensor(ctx->weight_ctx, name_buf);

        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_down.weight", i);
        layer.ffn_down = ggml_get_tensor(ctx->weight_ctx, name_buf);

        if (!layer.attn_norm || !layer.attn_q || !layer.attn_k || !layer.attn_v ||
            !layer.attn_output || !layer.ffn_norm || !layer.ffn_gate ||
            !layer.ffn_up || !layer.ffn_down) {
            fprintf(stderr, "[dllm] Missing weight tensors in layer %d\n", i);
            return -1;
        }
    }

    // 7. Set up computation allocator
    ctx->compute_alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());

    ctx->loaded = true;
    printf("[dllm] Model loaded successfully. %lld tensors, buffer=%.1f MB\n",
           n_tensors, (float)ggml_backend_buffer_get_size(ctx->weight_buf) / 1024.0f / 1024.0f);

    return 0;
}
```

**Step 3: Build and verify compilation**

Run:
```bash
cd /home/home/dev/include/dllm
cmake --build build --target dllm_cli 2>&1 | tail -10
```
Expected: Compiles (will fail at link if stubs don't match — fix any issues).

**Step 4: Commit**

```bash
git add dllm_internal.h dllm_model.cpp
git commit -m "feat: GGUF model loader with weight mapping for Qwen3 diffusion"
```

---

## Task 4: Bidirectional Transformer Forward Pass (Graph Builder)

**Files:**
- Modify: `/home/home/dev/include/dllm/dllm_graph.cpp`

This is the core — building the ggml computation graph for a full bidirectional transformer forward pass. The ONLY difference from a standard decoder is: **no causal mask**.

**Step 1: Implement dllm_build_forward**

Replace `/home/home/dev/include/dllm/dllm_graph.cpp`:
```cpp
#include "dllm_internal.h"

#include <cstdio>
#include <cmath>
#include <cstring>

// Build the full ggml computation graph for one forward pass
// Input: token IDs (including MASK tokens)
// Output: logits at ALL positions [n_tokens, n_vocab]
struct ggml_cgraph * dllm_build_forward(
    dllm_context * ctx,
    const int32_t * tokens,
    int n_tokens,
    struct ggml_context * gctx
) {
    const dllm_hparams * hp = &ctx->hparams;
    const dllm_model * m = &ctx->model;

    const int n_embd    = hp->n_embd;
    const int n_head    = hp->n_head;
    const int n_head_kv = hp->n_head_kv;
    const int n_layer   = hp->n_layer;
    const int n_vocab   = hp->n_vocab;
    const int n_ff      = hp->n_ff;
    const int head_dim  = n_embd / n_head;
    const int n_kv_dim  = head_dim * n_head_kv;

    const float rms_eps    = hp->rms_norm_eps;
    const float rope_theta = hp->rope_theta;
    const float kq_scale   = 1.0f / sqrtf((float)head_dim);

    // Create computation graph
    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, 4096 * n_layer, false);

    // Input token IDs tensor
    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);
    memcpy(inp_tokens->data, tokens, n_tokens * sizeof(int32_t));

    // Position IDs for RoPE [n_tokens]
    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);
    {
        int32_t * pos_data = (int32_t *)inp_pos->data;
        for (int i = 0; i < n_tokens; i++) {
            pos_data[i] = i;
        }
    }

    // Token embeddings: [n_tokens, n_embd]
    struct ggml_tensor * cur = ggml_get_rows(gctx, m->tok_embeddings, inp_tokens);
    // cur shape: [n_embd, n_tokens] (ggml convention: first dim is "columns")

    // Transformer layers
    for (int il = 0; il < n_layer; il++) {
        const dllm_layer & layer = m->layers[il];
        struct ggml_tensor * residual = cur;

        // 1. Pre-attention RMSNorm
        cur = ggml_rms_norm(gctx, cur, rms_eps);
        cur = ggml_mul(gctx, cur, layer.attn_norm);

        // 2. Q, K, V projections
        struct ggml_tensor * Q = ggml_mul_mat(gctx, layer.attn_q, cur);
        struct ggml_tensor * K = ggml_mul_mat(gctx, layer.attn_k, cur);
        struct ggml_tensor * V = ggml_mul_mat(gctx, layer.attn_v, cur);

        // Add biases if present (Qwen3 has QKV bias)
        if (layer.attn_q_bias) Q = ggml_add(gctx, Q, layer.attn_q_bias);
        if (layer.attn_k_bias) K = ggml_add(gctx, K, layer.attn_k_bias);
        if (layer.attn_v_bias) V = ggml_add(gctx, V, layer.attn_v_bias);

        // Reshape for multi-head: [head_dim, n_head, n_tokens]
        Q = ggml_reshape_3d(gctx, Q, head_dim, n_head,    n_tokens);
        K = ggml_reshape_3d(gctx, K, head_dim, n_head_kv, n_tokens);
        V = ggml_reshape_3d(gctx, V, head_dim, n_head_kv, n_tokens);

        // 3. RoPE on Q and K
        Q = ggml_rope(gctx, Q, inp_pos, head_dim, 0);
        K = ggml_rope(gctx, K, inp_pos, head_dim, 0);

        // 4. Attention: Q @ K^T / sqrt(d) → softmax → @ V
        //    NO CAUSAL MASK — this is the entire difference from autoregressive!

        // Permute for matmul: [head_dim, n_tokens, n_head]
        Q = ggml_permute(gctx, Q, 0, 2, 1, 3);  // [head_dim, n_tokens, n_head]
        K = ggml_permute(gctx, K, 0, 2, 1, 3);  // [head_dim, n_tokens, n_head_kv]

        // GQA: repeat K/V if n_head_kv < n_head
        if (n_head_kv < n_head) {
            int n_rep = n_head / n_head_kv;
            K = ggml_repeat(gctx, K, ggml_new_tensor_3d(gctx, K->type, head_dim, n_tokens, n_head));
            V = ggml_repeat(gctx, ggml_permute(gctx, V, 0, 2, 1, 3),
                           ggml_new_tensor_3d(gctx, V->type, head_dim, n_tokens, n_head));
        } else {
            V = ggml_permute(gctx, V, 0, 2, 1, 3);
        }

        // KQ = Q^T @ K  → [n_tokens, n_tokens, n_head]
        struct ggml_tensor * KQ = ggml_mul_mat(gctx, K, Q);

        // Scale
        KQ = ggml_scale(gctx, KQ, kq_scale);

        // Softmax — NO MASK (bidirectional attention)
        // Using ggml_soft_max with no mask tensor
        KQ = ggml_soft_max(gctx, KQ);

        // V is [head_dim, n_tokens, n_head]
        // Need V transposed: [n_tokens, head_dim, n_head]
        struct ggml_tensor * V_t = ggml_cont(gctx, ggml_transpose(gctx, V));

        // KQV = softmax(QK^T/√d) @ V → [head_dim, n_tokens, n_head]
        struct ggml_tensor * KQV = ggml_mul_mat(gctx, V_t, KQ);

        // Permute back: [head_dim, n_head, n_tokens] → reshape to [n_embd, n_tokens]
        KQV = ggml_permute(gctx, KQV, 0, 2, 1, 3);
        KQV = ggml_cont(gctx, KQV);
        KQV = ggml_reshape_2d(gctx, KQV, n_embd, n_tokens);

        // Output projection
        cur = ggml_mul_mat(gctx, layer.attn_output, KQV);

        // Residual connection
        cur = ggml_add(gctx, cur, residual);

        // 5. Post-attention: FFN
        residual = cur;

        // Pre-FFN RMSNorm
        cur = ggml_rms_norm(gctx, cur, rms_eps);
        cur = ggml_mul(gctx, cur, layer.ffn_norm);

        // SwiGLU: silu(x @ W_gate) * (x @ W_up)
        struct ggml_tensor * gate = ggml_mul_mat(gctx, layer.ffn_gate, cur);
        gate = ggml_silu(gctx, gate);
        struct ggml_tensor * up = ggml_mul_mat(gctx, layer.ffn_up, cur);
        cur = ggml_mul(gctx, gate, up);

        // Down projection
        cur = ggml_mul_mat(gctx, layer.ffn_down, cur);

        // Residual
        cur = ggml_add(gctx, cur, residual);
    }

    // Final RMSNorm
    cur = ggml_rms_norm(gctx, cur, rms_eps);
    cur = ggml_mul(gctx, cur, m->output_norm);

    // LM head: [n_embd, n_tokens] → [n_vocab, n_tokens]
    cur = ggml_mul_mat(gctx, m->output, cur);
    ggml_set_name(cur, "logits");

    ggml_build_forward_expand(gf, cur);

    return gf;
}

// Compute forward pass and return logits
float * dllm_compute_forward(dllm_context * ctx, const int32_t * tokens, int n_tokens) {
    // Allocate graph context (enough for all intermediate tensors)
    size_t ctx_size = ggml_tensor_overhead() * (4096 * ctx->hparams.n_layer + 64);
    ctx_size += n_tokens * sizeof(int32_t) * 2;  // input tokens + positions

    struct ggml_init_params gparams = {
        .mem_size   = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };

    struct ggml_context * gctx = ggml_init(gparams);
    if (!gctx) {
        fprintf(stderr, "[dllm] Failed to init graph context\n");
        return nullptr;
    }

    // Build the graph
    struct ggml_cgraph * gf = dllm_build_forward(ctx, tokens, n_tokens, gctx);
    if (!gf) {
        ggml_free(gctx);
        return nullptr;
    }

    // Reserve memory for the graph
    if (!ggml_gallocr_reserve(ctx->compute_alloc, gf)) {
        fprintf(stderr, "[dllm] Failed to reserve graph memory\n");
        ggml_free(gctx);
        return nullptr;
    }

    // Allocate tensors
    if (!ggml_gallocr_alloc_graph(ctx->compute_alloc, gf)) {
        fprintf(stderr, "[dllm] Failed to alloc graph\n");
        ggml_free(gctx);
        return nullptr;
    }

    // Set input data (tokens and positions were set during build)
    // Need to copy token data into allocated tensor
    struct ggml_tensor * inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    if (inp_tokens) {
        memcpy(inp_tokens->data, tokens, n_tokens * sizeof(int32_t));
    }

    struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        int32_t * pos = (int32_t *)inp_pos->data;
        for (int i = 0; i < n_tokens; i++) pos[i] = i;
    }

    // Compute
    ggml_backend_graph_compute(ctx->backend_cpu, gf);

    // Get logits
    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    float * result = nullptr;
    if (logits) {
        // Copy logits out (graph memory will be reused)
        size_t logits_size = ggml_nbytes(logits);
        result = (float *)malloc(logits_size);
        memcpy(result, logits->data, logits_size);
    }

    ggml_free(gctx);
    return result;  // caller must free()
}
```

**Step 2: Build and verify**

Run:
```bash
cd /home/home/dev/include/dllm
cmake --build build --target dllm_cli 2>&1 | tail -10
```
Expected: Compiles successfully.

**Step 3: Commit**

```bash
git add dllm_graph.cpp
git commit -m "feat: bidirectional transformer forward pass — no causal mask"
```

---

## Task 5: MDLM Diffusion Sampling Loop

**Files:**
- Modify: `/home/home/dev/include/dllm/dllm_sampler.cpp`

This is the novel part — the iterative denoising loop with Gumbel noise and confidence remasking.

**Step 1: Implement the MDLM sampler**

Replace `/home/home/dev/include/dllm/dllm_sampler.cpp`:
```cpp
#include "dllm_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <random>
#include <numeric>
#include <cfloat>

// --- Gumbel noise sampling ---

// Add Gumbel noise to logits for stochastic sampling
// CRITICAL: use float64 for Gumbel computation (low precision degrades quality)
static void add_gumbel_noise(float * logits, int n_vocab, float temperature, std::mt19937 & rng) {
    if (temperature == 0.0f) return;  // greedy — no noise

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (int i = 0; i < n_vocab; i++) {
        double u = std::max(uniform(rng), 1e-20);
        double gumbel_noise = std::pow(-std::log(u), (double)temperature);
        logits[i] = (float)(std::exp((double)logits[i]) / gumbel_noise);
    }
}

// Argmax over a row of logits
static int32_t argmax(const float * logits, int n_vocab) {
    int32_t best = 0;
    float best_val = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

// Softmax in-place over a row
static void softmax_inplace(float * logits, int n_vocab) {
    float max_val = *std::max_element(logits, logits + n_vocab);
    float sum = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n_vocab; i++) {
        logits[i] *= inv_sum;
    }
}

// --- Transfer schedule ---

// Linear even split: distribute n_masked tokens across n_steps
static std::vector<int> compute_transfer_schedule(int n_masked, int n_steps) {
    std::vector<int> schedule(n_steps);
    if (n_steps <= 0 || n_masked <= 0) return schedule;

    int base = n_masked / n_steps;
    int remainder = n_masked % n_steps;
    for (int i = 0; i < n_steps; i++) {
        schedule[i] = base + (i < remainder ? 1 : 0);
    }
    return schedule;
}

// --- Main MDLM generation ---

int dllm_generate(
    dllm_context * ctx,
    const int32_t * prompt_tokens,
    int n_prompt,
    dllm_sampling sampling,
    dllm_step_cb callback,
    void * user_data,
    int32_t * out_tokens,
    int max_out
) {
    if (!ctx || !ctx->loaded) return -1;

    const int n_vocab  = ctx->hparams.n_vocab;
    const int mask_id  = ctx->hparams.mask_token_id;
    const int n_steps  = sampling.n_steps;
    const int gen_len  = std::min(sampling.gen_length, max_out);
    const float temp   = sampling.temperature;
    const int remasking_strategy = sampling.remasking;

    if (mask_id < 0) {
        fprintf(stderr, "[dllm] No mask token ID set\n");
        return -1;
    }

    const int total_len = n_prompt + gen_len;
    printf("[dllm] Generating: prompt=%d gen=%d total=%d steps=%d temp=%.2f\n",
           n_prompt, gen_len, total_len, n_steps, temp);

    // 1. Initialize canvas: [prompt | MASK MASK ... MASK]
    std::vector<int32_t> canvas(total_len);
    memcpy(canvas.data(), prompt_tokens, n_prompt * sizeof(int32_t));
    for (int i = n_prompt; i < total_len; i++) {
        canvas[i] = mask_id;
    }

    // 2. Compute transfer schedule
    auto schedule = compute_transfer_schedule(gen_len, n_steps);

    // 3. RNG for Gumbel noise
    std::mt19937 rng(42);  // deterministic seed for reproducibility

    // 4. Diffusion loop
    ctx->cancelled.store(false);

    for (int step = 0; step < n_steps; step++) {
        if (ctx->cancelled.load()) {
            printf("[dllm] Generation cancelled at step %d\n", step);
            break;
        }

        // Count remaining masks
        int n_masked = 0;
        for (int i = n_prompt; i < total_len; i++) {
            if (canvas[i] == mask_id) n_masked++;
        }

        if (n_masked == 0) {
            printf("[dllm] All tokens unmasked at step %d — done early\n", step);
            break;
        }

        // 4a. Forward pass: get logits for ALL positions
        float * logits = dllm_compute_forward(ctx, canvas.data(), total_len);
        if (!logits) {
            fprintf(stderr, "[dllm] Forward pass failed at step %d\n", step);
            return -1;
        }

        // logits shape: [n_vocab, total_len] in ggml convention → accessed as logits[pos * n_vocab + v]
        // Actually ggml stores [total_len, n_vocab] but mul_mat output is [n_vocab, total_len]
        // We need to figure out the actual layout. Let's assume [total_len][n_vocab] = row-major.

        // 4b. For each position, sample with Gumbel noise and compute confidence

        // Allocate temporary arrays for this step
        std::vector<int32_t> x0(total_len);         // predicted tokens
        std::vector<float>   confidence(total_len);  // confidence scores

        // Make a copy of logits for confidence scoring (before Gumbel noise)
        std::vector<float> clean_logits(logits, logits + (size_t)total_len * n_vocab);

        // Add Gumbel noise to logits for sampling
        for (int pos = n_prompt; pos < total_len; pos++) {
            if (canvas[pos] != mask_id) continue;
            add_gumbel_noise(logits + (size_t)pos * n_vocab, n_vocab, temp, rng);
        }

        // Sample x0 = argmax(noisy_logits) for each masked position
        for (int pos = 0; pos < total_len; pos++) {
            if (pos < n_prompt || canvas[pos] != mask_id) {
                x0[pos] = canvas[pos];
                confidence[pos] = -1.0f;  // not a candidate
                continue;
            }

            x0[pos] = argmax(logits + (size_t)pos * n_vocab, n_vocab);

            // Confidence = softmax(clean_logits)[x0]
            if (remasking_strategy == 0) {
                // Low confidence remasking
                float * row = clean_logits.data() + (size_t)pos * n_vocab;
                softmax_inplace(row, n_vocab);
                confidence[pos] = row[x0[pos]];
            } else {
                // Random remasking
                confidence[pos] = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
            }
        }

        free(logits);

        // 4c. Select top-K most confident masked positions to unmask
        int k = schedule[step];
        k = std::min(k, n_masked);  // don't unmask more than available

        if (k > 0) {
            // Gather indices of masked positions
            std::vector<int> masked_indices;
            masked_indices.reserve(n_masked);
            for (int i = n_prompt; i < total_len; i++) {
                if (canvas[i] == mask_id) {
                    masked_indices.push_back(i);
                }
            }

            // Sort by confidence descending
            std::sort(masked_indices.begin(), masked_indices.end(),
                [&confidence](int a, int b) {
                    return confidence[a] > confidence[b];
                });

            // Unmask top-k
            for (int j = 0; j < k && j < (int)masked_indices.size(); j++) {
                int idx = masked_indices[j];
                canvas[idx] = x0[idx];
            }
        }

        // 4d. Callback
        if (callback) {
            bool cont = callback(step, n_steps, canvas.data() + n_prompt, gen_len, user_data);
            if (!cont) {
                printf("[dllm] Cancelled by callback at step %d\n", step);
                break;
            }
        }
    }

    // 5. Copy generated tokens to output
    int n_out = std::min(gen_len, max_out);
    memcpy(out_tokens, canvas.data() + n_prompt, n_out * sizeof(int32_t));

    // Report any remaining masks
    int remaining_masks = 0;
    for (int i = 0; i < n_out; i++) {
        if (out_tokens[i] == mask_id) remaining_masks++;
    }
    if (remaining_masks > 0) {
        printf("[dllm] Warning: %d tokens still masked after generation\n", remaining_masks);
    }

    return n_out;
}
```

**Step 2: Build and verify**

Run:
```bash
cd /home/home/dev/include/dllm
cmake --build build --target dllm_cli 2>&1 | tail -10
```
Expected: Compiles.

**Step 3: Commit**

```bash
git add dllm_sampler.cpp
git commit -m "feat: MDLM diffusion sampler — Gumbel noise + confidence remasking"
```

---

## Task 6: BPE Tokenizer

**Files:**
- Modify: `/home/home/dev/include/dllm/dllm_tokenizer.cpp`

The tokenizer needs to load vocabulary and BPE merges from the GGUF file (or a sidecar tokenizer.json). For initial testing, we'll load from HuggingFace's `tokenizer.json` format.

**Step 1: Implement basic BPE tokenizer**

Replace `/home/home/dev/include/dllm/dllm_tokenizer.cpp`:
```cpp
#include "dllm_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>

// Simple JSON string value extractor (no dependency on JSON lib)
// Finds "key": "value" and returns value
static std::string json_get_string(const std::string & json, const std::string & key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find("\"", pos + search.length() + 1);  // skip colon and whitespace
    if (pos == std::string::npos) return "";
    pos++;
    size_t end = json.find("\"", pos);
    if (end == std::string::npos) return "";
    return json.substr(pos, end - pos);
}

// Load tokenizer from tokenizer.json (HuggingFace format)
// Expected to be in same directory as the GGUF model, or passed separately
static int load_tokenizer_json(dllm_tokenizer * tok, const char * path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "[dllm] Cannot open tokenizer: %s\n", path);
        return -1;
    }

    // Read entire file
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    f.close();

    // Parse vocabulary from "vocab" section
    // Format: "vocab": { "token": id, "token2": id2, ... }
    size_t vocab_start = content.find("\"vocab\"");
    if (vocab_start == std::string::npos) {
        fprintf(stderr, "[dllm] No 'vocab' section found in tokenizer.json\n");
        return -1;
    }

    // Find the opening brace of the vocab object
    size_t brace_start = content.find("{", vocab_start + 7);
    if (brace_start == std::string::npos) return -1;

    // Parse token:id pairs
    int max_id = -1;
    size_t pos = brace_start + 1;
    int count = 0;

    while (pos < content.length()) {
        // Skip whitespace
        while (pos < content.length() && (content[pos] == ' ' || content[pos] == '\n' ||
               content[pos] == '\r' || content[pos] == '\t' || content[pos] == ','))
            pos++;

        if (content[pos] == '}') break;

        // Parse "token"
        if (content[pos] != '"') break;
        pos++;
        size_t tok_end = pos;
        // Handle escaped characters in token string
        while (tok_end < content.length()) {
            if (content[tok_end] == '\\') { tok_end += 2; continue; }
            if (content[tok_end] == '"') break;
            tok_end++;
        }
        std::string token = content.substr(pos, tok_end - pos);
        pos = tok_end + 1;

        // Skip ":"
        while (pos < content.length() && (content[pos] == ' ' || content[pos] == ':')) pos++;

        // Parse id (integer)
        size_t id_start = pos;
        while (pos < content.length() && (content[pos] >= '0' && content[pos] <= '9')) pos++;
        int32_t id = atoi(content.substr(id_start, pos - id_start).c_str());

        // Unescape basic sequences
        std::string unescaped;
        for (size_t i = 0; i < token.length(); i++) {
            if (token[i] == '\\' && i + 1 < token.length()) {
                char c = token[i + 1];
                if (c == 'n') { unescaped += '\n'; i++; }
                else if (c == 't') { unescaped += '\t'; i++; }
                else if (c == '\\') { unescaped += '\\'; i++; }
                else if (c == '"') { unescaped += '"'; i++; }
                else { unescaped += token[i]; }
            } else {
                unescaped += token[i];
            }
        }

        tok->token_to_id[unescaped] = id;
        if (id > max_id) max_id = id;
        count++;
    }

    // Build id_to_token
    tok->id_to_token.resize(max_id + 1);
    for (auto & [token, id] : tok->token_to_id) {
        if (id >= 0 && id < (int32_t)tok->id_to_token.size()) {
            tok->id_to_token[id] = token;
        }
    }

    printf("[dllm] Tokenizer loaded: %d tokens, max_id=%d\n", count, max_id);

    // Parse BPE merges from "merges" section
    size_t merges_start = content.find("\"merges\"");
    if (merges_start != std::string::npos) {
        size_t arr_start = content.find("[", merges_start);
        if (arr_start != std::string::npos) {
            size_t mpos = arr_start + 1;
            while (mpos < content.length()) {
                while (mpos < content.length() && content[mpos] != '"' && content[mpos] != ']') mpos++;
                if (content[mpos] == ']') break;
                mpos++;  // skip opening "
                size_t mend = content.find("\"", mpos);
                if (mend == std::string::npos) break;
                std::string merge = content.substr(mpos, mend - mpos);
                mpos = mend + 1;

                // Split on space
                size_t sp = merge.find(' ');
                if (sp != std::string::npos) {
                    tok->merges.emplace_back(merge.substr(0, sp), merge.substr(sp + 1));
                }
            }
        }
        printf("[dllm] BPE merges loaded: %zu\n", tok->merges.size());
    }

    return 0;
}

// Simple BPE encode: character-level split then apply merges
static std::vector<int32_t> bpe_encode(const dllm_tokenizer * tok, const std::string & text) {
    if (text.empty()) return {};

    // Start with individual bytes/characters as tokens
    // For Qwen3: uses byte-level BPE (similar to GPT-2)
    std::vector<std::string> tokens;
    for (size_t i = 0; i < text.length(); ) {
        // Try to match longest token first (greedy)
        int best_len = 0;
        int32_t best_id = -1;
        for (int len = std::min((int)(text.length() - i), 32); len >= 1; len--) {
            std::string sub = text.substr(i, len);
            auto it = tok->token_to_id.find(sub);
            if (it != tok->token_to_id.end()) {
                best_len = len;
                best_id = it->second;
                break;
            }
        }
        if (best_len > 0) {
            tokens.push_back(text.substr(i, best_len));
            i += best_len;
        } else {
            // Fallback: single byte
            tokens.push_back(text.substr(i, 1));
            i++;
        }
    }

    // Apply BPE merges
    for (const auto & [left, right] : tok->merges) {
        std::string merged = left + right;
        for (size_t i = 0; i + 1 < tokens.size(); ) {
            if (tokens[i] == left && tokens[i + 1] == right) {
                tokens[i] = merged;
                tokens.erase(tokens.begin() + i + 1);
            } else {
                i++;
            }
        }
    }

    // Convert to token IDs
    std::vector<int32_t> ids;
    ids.reserve(tokens.size());
    for (const auto & t : tokens) {
        auto it = tok->token_to_id.find(t);
        if (it != tok->token_to_id.end()) {
            ids.push_back(it->second);
        } else {
            fprintf(stderr, "[dllm] Unknown token: '%s'\n", t.c_str());
            // Use first byte as fallback
            ids.push_back((int32_t)(unsigned char)t[0]);
        }
    }

    return ids;
}

// --- Public API ---

int dllm_tokenize(const dllm_context * ctx, const char * text,
                  int32_t * tokens, int max_tokens) {
    if (!ctx || !text) return -1;

    auto ids = bpe_encode(&ctx->tokenizer, std::string(text));
    int n = std::min((int)ids.size(), max_tokens);
    memcpy(tokens, ids.data(), n * sizeof(int32_t));
    return n;
}

int dllm_detokenize(const dllm_context * ctx, const int32_t * tokens,
                    int n_tokens, char * buf, int buf_size) {
    if (!ctx || !tokens || !buf) return -1;

    std::string result;
    for (int i = 0; i < n_tokens; i++) {
        int32_t id = tokens[i];
        if (id == ctx->hparams.mask_token_id) {
            result += "[MASK]";
        } else if (id >= 0 && id < (int32_t)ctx->tokenizer.id_to_token.size()) {
            result += ctx->tokenizer.id_to_token[id];
        } else {
            result += "<?>";
        }
    }

    int len = std::min((int)result.length(), buf_size - 1);
    memcpy(buf, result.c_str(), len);
    buf[len] = '\0';
    return len;
}

// Load tokenizer — called from dllm_load_model
// Tries to find tokenizer.json next to the GGUF file
int dllm_load_tokenizer_from_gguf(dllm_context * ctx) {
    // For now, look for tokenizer.json in same directory as model
    // TODO: embed tokenizer in GGUF metadata
    fprintf(stderr, "[dllm] Note: tokenizer loading from tokenizer.json not yet auto-detected.\n");
    fprintf(stderr, "[dllm] Call dllm_load_tokenizer() manually with the tokenizer.json path.\n");
    return 0;
}
```

Also add a public function to load tokenizer separately. Add to `dllm.h`:
```c
// Load tokenizer from a tokenizer.json file (HuggingFace format)
int dllm_load_tokenizer(dllm_context * ctx, const char * tokenizer_json_path);
```

And implement it in `dllm_tokenizer.cpp`:
```cpp
int dllm_load_tokenizer(dllm_context * ctx, const char * path) {
    return load_tokenizer_json(&ctx->tokenizer, path);
}
```

**Step 2: Update dllm_main.cpp to accept --tokenizer flag**

Add to argument parsing:
```cpp
const char * tokenizer_path = nullptr;
// ...
} else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
    tokenizer_path = argv[++i];
}
// After model load:
if (tokenizer_path) {
    if (dllm_load_tokenizer(ctx, tokenizer_path) != 0) {
        fprintf(stderr, "[dllm] Failed to load tokenizer\n");
    }
}
```

**Step 3: Build and verify**

Run:
```bash
cd /home/home/dev/include/dllm
cmake --build build --target dllm_cli 2>&1 | tail -10
```

**Step 4: Commit**

```bash
git add dllm_tokenizer.cpp dllm.h dllm_main.cpp
git commit -m "feat: BPE tokenizer with HuggingFace tokenizer.json support"
```

---

## Task 7: Integration Test — Full Pipeline on Host

**Files:**
- No new files — test existing code

**Step 1: Convert model to GGUF**

```bash
cd /home/home/dev/include/dllm/scripts
python convert_to_gguf.py \
    --model dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 \
    --output /home/home/dev/include/dllm/test_model_f16.gguf \
    --type f16
```

**Step 2: Copy tokenizer.json from the downloaded model**

```bash
# Find the HF cache directory for the model
find ~/.cache/huggingface -name "tokenizer.json" -path "*Qwen3*diffusion*" 2>/dev/null
# Copy it
cp <found_path> /home/home/dev/include/dllm/tokenizer.json
```

**Step 3: Run the CLI on host (x86_64)**

```bash
cd /home/home/dev/include/dllm
./build/dllm_cli \
    --model test_model_f16.gguf \
    --tokenizer tokenizer.json \
    --prompt "The capital of France is" \
    --steps 32 \
    --gen-length 32 \
    --temp 0.0
```

Expected: Model loads, tokenizes prompt, runs 32 diffusion steps, prints generated tokens (may be garbage at first — correctness comes from debugging).

**Step 4: Debug and fix any issues**

Common issues to watch for:
- Tensor shape mismatches in ggml graph (especially attention permutations)
- Logits layout (row vs column major)
- Memory allocation too small for graph
- GQA repeat logic for K/V when n_head_kv < n_head

**Step 5: Commit any fixes**

```bash
git add -u
git commit -m "fix: integration test fixes for full forward pass pipeline"
```

---

## Task 8: Android NDK Cross-Compilation and ADB Testing

**Files:**
- Modify: `/home/home/dev/include/dllm/CMakeLists.txt` (add Android toolchain support)
- Create: `/home/home/dev/include/dllm/scripts/build_android.sh`

**Step 1: Create Android build script**

Create `/home/home/dev/include/dllm/scripts/build_android.sh`:
```bash
#!/bin/bash
set -e

# Detect NDK
if [ -z "$ANDROID_NDK" ]; then
    # Try common locations
    for ndk in ~/Android/Sdk/ndk/*/; do
        if [ -d "$ndk" ]; then
            export ANDROID_NDK="$ndk"
            break
        fi
    done
fi

if [ -z "$ANDROID_NDK" ]; then
    echo "ERROR: ANDROID_NDK not set. Set it to your NDK path."
    exit 1
fi

echo "Using NDK: $ANDROID_NDK"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DLLM_DIR="$(dirname "$SCRIPT_DIR")"

cd "$DLLM_DIR"

cmake -B build-android \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_NATIVE_API_LEVEL=28 \
    -DANDROID_STL=c++_shared \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=OFF \
    -DGGML_OPENCL=OFF

cmake --build build-android --target dllm_cli -j$(nproc)

echo ""
echo "Build complete: build-android/dllm_cli"
echo ""
echo "To test on device:"
echo "  adb push build-android/dllm_cli /data/local/tmp/"
echo "  adb push test_model_f16.gguf /data/local/tmp/"
echo "  adb push tokenizer.json /data/local/tmp/"
echo "  adb shell /data/local/tmp/dllm_cli --model /data/local/tmp/test_model_f16.gguf --tokenizer /data/local/tmp/tokenizer.json --prompt 'Hello' --steps 16 --gen-length 32"
```

**Step 2: Build for Android**

```bash
chmod +x /home/home/dev/include/dllm/scripts/build_android.sh
/home/home/dev/include/dllm/scripts/build_android.sh
```

**Step 3: Push to device and test**

```bash
adb push /home/home/dev/include/dllm/build-android/dllm_cli /data/local/tmp/
adb push /home/home/dev/include/dllm/test_model_f16.gguf /data/local/tmp/
adb push /home/home/dev/include/dllm/tokenizer.json /data/local/tmp/

# Also push libc++_shared.so if using c++_shared STL
adb push $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so /data/local/tmp/

adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. ./dllm_cli --model test_model_f16.gguf --tokenizer tokenizer.json --prompt 'Hello world' --steps 16 --gen-length 32 --temp 0.0"
```

Expected: Loads model on device, runs diffusion inference, outputs text.

**Step 4: Commit**

```bash
git add scripts/build_android.sh
git commit -m "feat: Android NDK build script for ADB testing"
```

---

## Task 9: Performance Benchmarking and Correctness Validation

**Files:**
- Modify: `/home/home/dev/include/dllm/dllm_main.cpp` (add --benchmark flag)

**Step 1: Add timing to the sampling loop**

Add to `dllm_sampler.cpp` inside the diffusion loop:
```cpp
#include <chrono>

// Inside the loop, around the forward pass:
auto t0 = std::chrono::high_resolution_clock::now();
float * logits = dllm_compute_forward(ctx, canvas.data(), total_len);
auto t1 = std::chrono::high_resolution_clock::now();
float forward_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
printf("[dllm] Step %d/%d: forward=%.1fms, masked=%d, unmasking=%d\n",
       step + 1, n_steps, forward_ms, n_masked, k);
```

**Step 2: Add --benchmark mode to CLI**

```cpp
// In dllm_main.cpp:
if (benchmark_mode) {
    auto total_start = std::chrono::high_resolution_clock::now();
    // ... generate ...
    auto total_end = std::chrono::high_resolution_clock::now();
    float total_ms = std::chrono::duration<float, std::milli>(total_end - total_start).count();
    printf("\n[benchmark] Total: %.1fms, %.1f ms/step, %.1f tok/s (diffusion steps)\n",
           total_ms, total_ms / steps, gen_length * 1000.0f / total_ms);
}
```

**Step 3: Run correctness test**

Compare output against Python reference:
```bash
# Python reference:
cd /path/to/dllm
python -c "
from dllm import Sampler
from transformers import AutoModelForMaskedLM, AutoTokenizer
model = AutoModelForMaskedLM.from_pretrained('dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1')
tok = AutoTokenizer.from_pretrained('dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1')
sampler = Sampler(model)
out = sampler.sample(tok.encode('The capital of France is'), steps=32, gen_length=32, temperature=0.0)
print(tok.decode(out))
"
```

Then compare with C++ output at temperature=0 (greedy, deterministic).

**Step 4: Commit**

```bash
git add dllm_sampler.cpp dllm_main.cpp
git commit -m "feat: add benchmarking and timing instrumentation"
```

---

## Roadmap (Future Tasks — Not in This Plan)

### Phase 2: BD3LM Support
- Block-causal attention mask builder
- KV cache for prefix blocks
- Block-level semi-autoregressive generation loop

### Phase 3: Optimizations
- Fast-dLLM prefix cache (approximate KV reuse between steps)
- Confidence-aware parallel decoding (threshold-based instead of top-K)
- ARM NEON-specific Gumbel sampling optimization
- Vulkan GPU backend integration

### Phase 4: AiSystems SDK Integration
- JNI bridge (`DLLMNativeLib.kt`)
- Kotlin API wrapper (`DLLMEngine.kt`)
- Integration with ToolNeuron UI (new "Diffusion" tab in chat)
- GGUF download from HuggingFace model store

### Phase 5: Advanced Features
- Classifier-Free Guidance (CFG)
- Native infilling API
- EditFlow (insertion/deletion/substitution operations)
- Multiple remasking strategies (entropy, margin)
