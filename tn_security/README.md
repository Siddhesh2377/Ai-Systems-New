# tn_security

Unified diagnostic + crash + log capture SDK used by every other module in this repo. Sits at the bottom of the dependency graph; `gguf_lib`, `ai_sd`, and `ai_sherpa` all pull it in via `api(project(":tn_security"))`.

- Package: `com.dark.tn_security`
- Min SDK: 29
- ABI: `arm64-v8a`
- Prefab publishing: enabled (C++ headers exported to consumers)

## What it provides

| Surface | Purpose |
|---|---|
| Signal handlers | Catch `SIGSEGV` / `SIGABRT` / `SIGBUS` / `SIGILL` / `SIGFPE`, write JSON crash file before re-raising |
| Ring buffer | Fixed 256-entry circular buffer of recent log events; serialized into crash file at signal time |
| Structured errors | Tagged by module / op-id / stage / `tn_code`, with user-actionable suggestion text |
| Event streaming | Every log / error / cancellation / crash flows through one Kotlin `SharedFlow<TnEvent>` |
| Crash drain | `TnSecurity.drainCrashFiles(dir)` reads + deletes any crash JSON from prior process death and replays as `TnEvent.Crash` |

The crash handler is **async-signal-safe**: pre-allocated 16 KB buffer, only `snprintf` / `write` / `fsync`, no `malloc`, no mutex. The ring buffer write inside the handler is wait-free.

## Quick start

```kotlin
class MyApp : Application() {
    override fun onCreate() {
        super.onCreate()

        TnSecurity.init(
            context = this,
            module = TnModule.TN_APP,                       // your module tag
            crashDir = File(filesDir, "crashes"),
            installSignalHandlers = true,
        )
        TnSecurity.addSink(LogcatSink())                    // bundled reference sink
        TnSecurity.drainCrashFiles(File(filesDir, "crashes"))   // replay last run's crash, if any
    }
}
```

Then in a worker / repository:

```kotlin
TnSecurity.withOp("generate-image-42") {
    // tn_security stamps op-id on every log + error emitted in this scope,
    // including ones from native code, until the block exits.
    sdManager.generateImage(params)
}
```

Add a custom sink (must be thread-safe + non-blocking):

```kotlin
TnSecurity.addSink { event ->
    when (event) {
        is TnEvent.Crash -> uploadCrashReport(event)
        is TnEvent.Error -> showSnackbarIf(event.code, event.suggestion)
        is TnEvent.Log,
        is TnEvent.Cancellation -> Unit
    }
}
```

## Event model

```kotlin
sealed class TnEvent {
    data class Log(...)
    data class Error(code: TnCode, stage: TnStage, message: String, suggestion: String?, cause: String?)
    data class Cancellation(reason: String?)
    data class Crash(signal: Int, signalName: String, pid: Int, faultAddr: Long, crashFilePath: String, ring: List<Log>)
}
```

`TnCode` enumerates 100+ stable numeric codes covering resources (`OOM`, `DISK_FULL`), I/O (`FILE_NOT_FOUND`, `PERMISSION_DENIED`), models (`MODEL_LOAD_FAIL`, `QUANT_UNSUPPORTED`), inference (`DECODE_FAIL`, `TOKENIZE_FAIL`), backends (`QNN_HTP_UNAVAILABLE`, `GPU_UNAVAILABLE`), IPC (`AIDL_DEAD_OBJECT`, `AIDL_TIMEOUT`), plugins, and native crashes (`NATIVE_CRASH`, `NATIVE_ABORT`). **Values are stable** — never renumbered.

`TnStage` is granular enough to attribute errors to a specific pipeline phase: `LOAD`, `WARMUP`, `TOKENIZE`, `PROMPT_EVAL`, `DECODE`, `SAMPLE`, `DETOKENIZE`, `VLM_PROJECT`, `STT_DECODE`, `TTS_GENERATE`, `SD_UNET` / `SD_CLIP` / `SD_VAE` / `SD_UPSCALE` / etc.

`TnModule` covers every component that emits events: `LLAMA_CPP`, `GGML`, `SHERPA_ONNX`, `ONNX_RT`, `MNN`, `QNN`, `GGUF_LIB`, `AI_SHERPA`, `AI_SD`, `TN_SERVICE`, `TN_APP`, `TN_PLUGIN`, `TN_HXS`.

## Native C API

```c
// Lifecycle
int  tn_sec_init(void);
void tn_sec_shutdown(void);
int  tn_sec_set_crash_file_pattern(const char* pattern);    // supports %m, %p, %t
int  tn_sec_install_signal_handlers(void);

// Thread-local op tracking
void tn_sec_set_op(const char* op_id);
void tn_sec_clear_op(void);
const char* tn_sec_current_op(void);

// Emit
void tn_sec_log(tn_level level, tn_module module, const char* tag,
                const char* op_id, const char* file, int line, const char* func,
                const char* fmt, ...);
void tn_sec_emit_error(const tn_error_init* init, const char* fmt, ...);
void tn_sec_emit_cancellation(tn_module module, const char* op_id, const char* reason);
```

C++ headers (in `include/tn_security/`) ship via Prefab; consumers `find_package(tn_security REQUIRED CONFIG)` in their CMake.

## C / C++ macros (per-file)

```cpp
#define TN_MODULE TN_MODULE_GGUF_LIB     // pick once at the top of each TU
#define TN_TAG    "graph"                // optional

#include <tn_security/tn_security_macros.h>

TN_I("loaded %d layers", n_layers);
TN_W("falling back to CPU");
TN_E("vk device lost during prefill");

TN_ERR(TN_CODE_DECODE_FAIL, TN_STAGE_DECODE,
       "decode returned %d on batch %d", rc, batch);

TN_ERR_FIX(TN_CODE_QNN_HTP_UNAVAILABLE, TN_STAGE_INIT,
           "Use the 'min' QNN variant or fall back to MNN",
           "HTP V%d not supported by this model", htp_version);

TN_CANCEL("user requested stop");

{
    TN_OP_SCOPE("generate-token-1234");
    // op-id is stamped on every emit inside this scope; cleared on scope exit
    run_decode();
}
```

The `TN_T/D/I/W/E/F` macros auto-fill `__FILE__` / `__LINE__` / `__func__`.

## JNI integration

`tn_security` provides the JNI sink that **every** native log line in the project goes through. The sink does one critical thing: it builds Java strings via `String(byte[], CharsetDecoder)` with replacement, instead of `NewStringUTF`. Upstream code (llama.cpp tokenizer merges, MNN error messages, sherpa-onnx logs) routinely emits invalid UTF-8 byte sequences that strict Modified-UTF-8 decode would reject and crash on; the lenient path replaces them with U+FFFD and keeps the event flowing.

If you're writing a new C/C++ SDK that needs to log to the same stream:

```cmake
find_package(tn_security REQUIRED CONFIG)
target_link_libraries(your_sdk PRIVATE tn_security::tn_security)
```

```cpp
#define TN_MODULE TN_MODULE_YOUR_SDK   // request a new enum value if needed
#include <tn_security/tn_security_macros.h>
```

That's it — no need to add your own JNI bridge, your own crash handler, or your own log routing. There's one of each, in this module.

## Crash file format

```json
{
  "signal": 11,
  "signal_name": "SIGSEGV",
  "timestamp_ms": 1714060000000,
  "pid": 4242,
  "tid": 4250,
  "module": 8,
  "module_slug": "gguf_lib",
  "si_code": 1,
  "fault_addr": "0x0",
  "ring": [
    {"ts": ..., "lvl": "I", "mod": 8, "tag": "graph", "op": "...", "msg": "..."},
    ...
  ]
}
```

File path is resolved from the pattern set via `tn_sec_set_crash_file_pattern()`. Supports:

| Token | Expansion |
|---|---|
| `%m` | Module slug (e.g. `gguf_lib`) — best-effort from last ring entry |
| `%p` | Process ID |
| `%t` | Epoch milliseconds |

Default pattern is `crash-%m-%p-%t.json`.

## Surprising / non-obvious

- **No AIDL.** The unified event stream is a Kotlin `SharedFlow`. AIDL boundaries (between processes, e.g. ToolNeuron's LLM service) are wired by the host app, not here.
- **`AndroidManifest.xml` is empty** — pure library; no service, activity, or permission declared.
- **`tn_sec_current_op()` returns a pointer**, not a copy, into a 128-byte thread-local. Don't outlive the thread.
- **Release minify is enabled** in this module's `build.gradle.kts` — the only SDK in the repo where R8 is on. The security audit (see `../SECURITY.md`) flagged the others.
- **Module enum values are stable** — never renumber. They show up in serialized crash JSON and may be read by external tooling.

## Layout

```
src/main/
├── cpp/
│   ├── tn_security.cpp           ring buffer, signal handlers, crash JSON writer
│   ├── tn_security_jni.cpp       JNI sink with lenient UTF-8 decode
│   └── include/tn_security/
│       ├── tn_security.h         C API (lifecycle, log, error, cancel, op tracking)
│       └── tn_security_macros.h  TN_T/D/I/W/E/F, TN_ERR, TN_ERR_FIX, TN_CANCEL, TN_OP_SCOPE
└── java/com/dark/tn_security/
    ├── TnSecurity.kt             singleton: init/shutdown, sinks, op tracking, crash drain
    ├── TnEvent.kt                sealed: Log, Error, Cancellation, Crash
    ├── TnSink.kt                 fun interface { onEvent(TnEvent) }
    ├── LogcatSink.kt             reference sink
    ├── TnModule.kt               module enum (stable values)
    ├── TnLevel.kt                TRACE…FATAL
    ├── TnCode.kt                 100+ stable error codes
    └── TnStage.kt                pipeline-stage enum
```

## Build prerequisites

- NDK 27.3.13750724
- CMake 3.31.4
- C++17

Headers are exported to consuming Gradle modules via Prefab, so most consumers don't need to know any of the above.
