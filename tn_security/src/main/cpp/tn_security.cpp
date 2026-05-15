#include <tn_security/tn_security.h>

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <android/log.h>

namespace {

constexpr int kMaxMsg     = 1024;
constexpr int kMaxRingMsg = 256;
constexpr int kRingSize   = 256;
constexpr int kMaxPath    = 1024;

struct RingEntry {
    int64_t timestamp_ms;
    int     kind;     // 0=log, 1=error, 2=cancellation
    int     level;
    int     module;
    int     code;
    int     stage;
    int     tid;
    int     line;
    char    tag[32];
    char    op_id[64];
    char    file[64];
    char    func[48];
    char    message[kMaxRingMsg];
};

struct State {
    std::mutex            mu;
    tn_sec_sink_fn        sink      = nullptr;
    void*                 sink_user = nullptr;
    char                  crash_pattern[kMaxPath] = "";
    std::atomic<bool>     handlers_installed{false};
    std::atomic<uint64_t> ring_next{0};
    RingEntry             ring[kRingSize] = {};
    std::atomic<bool>     initialized{false};
};

State* g_state() {
    static State s;
    return &s;
}

thread_local char tl_op_id[128] = "";

int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

int gettid_compat() {
    return (int)syscall(SYS_gettid);
}

void safe_strncpy(char* dst, size_t cap, const char* src) {
    if (!dst || cap == 0) return;
    if (!src) { dst[0] = '\0'; return; }
    size_t n = 0;
    while (src[n] && n < cap - 1) { dst[n] = src[n]; ++n; }
    dst[n] = '\0';
}

void push_ring(int kind, int level, int module, int code, int stage,
               const char* tag, const char* op_id, const char* file, int line,
               const char* func, const char* msg) {
    State* s = g_state();
    uint64_t idx = s->ring_next.fetch_add(1, std::memory_order_relaxed) % kRingSize;
    RingEntry& e = s->ring[idx];
    e.timestamp_ms = now_ms();
    e.kind = kind;
    e.level = level;
    e.module = module;
    e.code = code;
    e.stage = stage;
    e.tid = gettid_compat();
    e.line = line;
    safe_strncpy(e.tag,     sizeof(e.tag),     tag);
    safe_strncpy(e.op_id,   sizeof(e.op_id),   op_id);
    safe_strncpy(e.file,    sizeof(e.file),    file);
    safe_strncpy(e.func,    sizeof(e.func),    func);
    safe_strncpy(e.message, sizeof(e.message), msg);
}

void dispatch(int kind, int level, int module, int code, int stage,
              const char* tag, const char* op_id, const char* file,
              int line, const char* func, const char* message,
              const char* suggestion) {
    State* s = g_state();
    tn_sec_sink_fn sink;
    void* user;
    {
        std::lock_guard<std::mutex> lk(s->mu);
        sink = s->sink;
        user = s->sink_user;
    }
    if (sink) {
        sink(kind, level, module, code, stage, tag, op_id, file, line, func,
             message, suggestion, now_ms(), gettid_compat(), user);
        return;
    }
    // Fallback: logcat. Useful before the JNI sink is wired.
    int prio;
    switch (level) {
        case TN_LEVEL_TRACE: prio = ANDROID_LOG_VERBOSE; break;
        case TN_LEVEL_DEBUG: prio = ANDROID_LOG_DEBUG;   break;
        case TN_LEVEL_WARN:  prio = ANDROID_LOG_WARN;    break;
        case TN_LEVEL_ERROR: prio = ANDROID_LOG_ERROR;   break;
        case TN_LEVEL_FATAL: prio = ANDROID_LOG_FATAL;   break;
        case TN_LEVEL_INFO:
        default:             prio = ANDROID_LOG_INFO;    break;
    }
    const char* slug = tn_sec_module_slug((tn_module)module);
    __android_log_print(prio, tag ? tag : "tn_sec", "[%s] %s",
                        slug, message ? message : "");
}

// ─── Signal handler — async-signal-safe ──────────────────────────────────
//
// AS-safe rules: only open/write/close/fsync/raise/snprintf with non-locale
// format specifiers. No malloc, no mutex. Reads ring without locking;
// tolerates partial corruption (last entry might be half-written — accepted).

constexpr int kCrashBufSize = 16 * 1024;
char g_crash_buf[kCrashBufSize];  // pre-allocated; reused by signal handler

void resolve_crash_path(const char* pattern, tn_module module,
                        char* out, size_t cap) {
    if (!pattern || pattern[0] == '\0') { if (cap) out[0] = '\0'; return; }
    const char* slug = tn_sec_module_slug(module);
    int     pid = getpid();
    int64_t ts  = now_ms();
    size_t  pos = 0;
    for (const char* p = pattern; *p && pos + 32 < cap; ++p) {
        if (*p == '%' && p[1]) {
            char k = p[1];
            ++p;
            int w = 0;
            if      (k == 'm') w = snprintf(out + pos, cap - pos, "%s",   slug);
            else if (k == 'p') w = snprintf(out + pos, cap - pos, "%d",   pid);
            else if (k == 't') w = snprintf(out + pos, cap - pos, "%lld", (long long)ts);
            else {
                if (pos + 2 < cap) { out[pos++] = '%'; out[pos++] = k; }
                continue;
            }
            if (w > 0) pos += (size_t)w;
        } else {
            out[pos++] = *p;
        }
    }
    out[pos] = '\0';
}

size_t buf_append(char* buf, size_t pos, size_t cap, const char* s) {
    if (!s) return pos;
    while (*s && pos + 1 < cap) buf[pos++] = *s++;
    buf[pos] = '\0';
    return pos;
}

size_t buf_appendf(char* buf, size_t pos, size_t cap, const char* fmt, ...) {
    if (pos + 1 >= cap) return pos;
    va_list ap;
    va_start(ap, fmt);
    int w = vsnprintf(buf + pos, cap - pos, fmt, ap);
    va_end(ap);
    if (w > 0) pos += (size_t)w;
    if (pos >= cap) pos = cap - 1;
    return pos;
}

size_t buf_json_str(char* buf, size_t pos, size_t cap, const char* s) {
    if (pos + 2 >= cap) return pos;
    buf[pos++] = '"';
    if (s) {
        for (; *s && pos + 8 < cap; ++s) {
            unsigned char c = (unsigned char)*s;
            if      (c == '"')  { buf[pos++] = '\\'; buf[pos++] = '"';  }
            else if (c == '\\') { buf[pos++] = '\\'; buf[pos++] = '\\'; }
            else if (c == '\n') { buf[pos++] = '\\'; buf[pos++] = 'n';  }
            else if (c == '\r') { buf[pos++] = '\\'; buf[pos++] = 'r';  }
            else if (c == '\t') { buf[pos++] = '\\'; buf[pos++] = 't';  }
            else if (c < 0x20) {
                int w = snprintf(buf + pos, cap - pos, "\\u%04x", c);
                if (w > 0) pos += (size_t)w;
            } else {
                buf[pos++] = (char)c;
            }
        }
    }
    if (pos + 1 < cap) buf[pos++] = '"';
    buf[pos] = '\0';
    return pos;
}

void write_crash_file(int sig, siginfo_t* info) {
    State* s = g_state();

    // Best-effort module pick: most-recent ring entry (last writer wins).
    tn_module mod = TN_MODULE_UNKNOWN;
    uint64_t next = s->ring_next.load(std::memory_order_relaxed);
    if (next > 0) {
        uint64_t idx = (next - 1) % kRingSize;
        int m = s->ring[idx].module;
        if (m > 0) mod = (tn_module)m;
    }

    char path[kMaxPath];
    resolve_crash_path(s->crash_pattern, mod, path, sizeof(path));
    if (path[0] == '\0') return;

    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return;

    char*  b   = g_crash_buf;
    size_t cap = sizeof(g_crash_buf);
    size_t pos = 0;
    pos = buf_appendf(b, pos, cap, "{\"signal\":%d,", sig);
    pos = buf_append (b, pos, cap, "\"signal_name\":");
    pos = buf_json_str(b, pos, cap, tn_sec_signal_name(sig));
    pos = buf_appendf(b, pos, cap, ",\"timestamp_ms\":%lld",  (long long)now_ms());
    pos = buf_appendf(b, pos, cap, ",\"pid\":%d,\"tid\":%d",  (int)getpid(), gettid_compat());
    pos = buf_appendf(b, pos, cap, ",\"module\":%d", (int)mod);
    pos = buf_append (b, pos, cap, ",\"module_slug\":");
    pos = buf_json_str(b, pos, cap, tn_sec_module_slug(mod));

    if (info) {
        pos = buf_appendf(b, pos, cap,
            ",\"si_code\":%d,\"fault_addr\":\"%p\"",
            info->si_code, info->si_addr);
    }

    pos = buf_append(b, pos, cap, ",\"ring\":[");
    int count = (next < (uint64_t)kRingSize) ? (int)next : kRingSize;
    int start = (int)((next - (uint64_t)count) % kRingSize);
    for (int i = 0; i < count && pos + 256 < cap; ++i) {
        const RingEntry& e = s->ring[(start + i) % kRingSize];
        if (i > 0) pos = buf_append(b, pos, cap, ",");
        pos = buf_appendf(b, pos, cap,
            "{\"ts\":%lld,\"kind\":%d,\"lvl\":%d,\"mod\":%d,\"code\":%d,\"stage\":%d,\"tid\":%d,\"line\":%d,",
            (long long)e.timestamp_ms, e.kind, e.level, e.module,
            e.code, e.stage, e.tid, e.line);
        pos = buf_append (b, pos, cap, "\"tag\":"); pos = buf_json_str(b, pos, cap, e.tag);
        pos = buf_append (b, pos, cap, ",\"op\":"); pos = buf_json_str(b, pos, cap, e.op_id);
        pos = buf_append (b, pos, cap, ",\"file\":");pos = buf_json_str(b, pos, cap, e.file);
        pos = buf_append (b, pos, cap, ",\"func\":");pos = buf_json_str(b, pos, cap, e.func);
        pos = buf_append (b, pos, cap, ",\"msg\":"); pos = buf_json_str(b, pos, cap, e.message);
        pos = buf_append (b, pos, cap, "}");
    }
    pos = buf_append(b, pos, cap, "]}");

    ssize_t _ = write(fd, b, pos);
    (void)_;
    fsync(fd);
    close(fd);
}

void crash_handler(int sig, siginfo_t* info, void* /*ctx*/) {
    write_crash_file(sig, info);
    signal(sig, SIG_DFL);
    raise(sig);
}

void install_one(int sig) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = crash_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigaction(sig, &sa, nullptr);
}

} // namespace


// ─── Public API ─────────────────────────────────────────────────────────────

extern "C" {

void tn_sec_init(void) { g_state()->initialized.store(true); }

void tn_sec_shutdown(void) {
    State* s = g_state();
    std::lock_guard<std::mutex> lk(s->mu);
    s->sink      = nullptr;
    s->sink_user = nullptr;
    s->initialized.store(false);
}

void tn_sec_set_crash_file_pattern(const char* pattern) {
    State* s = g_state();
    std::lock_guard<std::mutex> lk(s->mu);
    if (!pattern) { s->crash_pattern[0] = '\0'; return; }
    safe_strncpy(s->crash_pattern, sizeof(s->crash_pattern), pattern);
}

void tn_sec_install_signal_handlers(void) {
    State* s = g_state();
    bool expected = false;
    if (!s->handlers_installed.compare_exchange_strong(expected, true)) return;
    install_one(SIGSEGV);
    install_one(SIGABRT);
    install_one(SIGBUS);
    install_one(SIGILL);
    install_one(SIGFPE);
}

void tn_sec_set_op(const char* op_id) {
    if (!op_id) { tl_op_id[0] = '\0'; return; }
    safe_strncpy(tl_op_id, sizeof(tl_op_id), op_id);
}

void tn_sec_clear_op(void) { tl_op_id[0] = '\0'; }

const char* tn_sec_current_op(void) { return tl_op_id[0] ? tl_op_id : nullptr; }

void tn_sec_set_sink(tn_sec_sink_fn fn, void* user_data) {
    State* s = g_state();
    std::lock_guard<std::mutex> lk(s->mu);
    s->sink      = fn;
    s->sink_user = user_data;
}

void tn_sec_log_v(tn_level level, tn_module module, const char* tag,
                  const char* op_id, const char* file, int line,
                  const char* func, const char* fmt, va_list ap) {
    char msg[kMaxMsg];
    if (fmt) vsnprintf(msg, sizeof(msg), fmt, ap);
    else     msg[0] = '\0';
    const char* op = op_id ? op_id : tn_sec_current_op();
    push_ring(0, (int)level, (int)module, 0, 0, tag, op, file, line, func, msg);
    dispatch (0, (int)level, (int)module, 0, 0, tag, op, file, line, func, msg, nullptr);
}

void tn_sec_log(tn_level level, tn_module module, const char* tag,
                const char* op_id, const char* file, int line,
                const char* func, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    tn_sec_log_v(level, module, tag, op_id, file, line, func, fmt, ap);
    va_end(ap);
}

void tn_sec_emit_error(const tn_error_init* init, const char* fmt, ...) {
    if (!init) return;
    char msg[kMaxMsg];
    if (fmt) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(msg, sizeof(msg), fmt, ap);
        va_end(ap);
    } else {
        msg[0] = '\0';
    }
    const char* op = init->op_id ? init->op_id : tn_sec_current_op();
    push_ring(1, (int)TN_LEVEL_ERROR, (int)init->module,
              (int)init->code, (int)init->stage,
              nullptr, op, init->file, init->line, init->func, msg);
    dispatch (1, (int)TN_LEVEL_ERROR, (int)init->module,
              (int)init->code, (int)init->stage,
              nullptr, op, init->file, init->line, init->func,
              msg, init->suggestion);
}

void tn_sec_emit_cancellation(tn_module module, const char* op_id, const char* reason) {
    const char* op = op_id ? op_id : tn_sec_current_op();
    push_ring(2, (int)TN_LEVEL_INFO, (int)module, (int)TN_CODE_CANCELLED, 0,
              nullptr, op, nullptr, 0, nullptr, reason ? reason : "");
    dispatch (2, (int)TN_LEVEL_INFO, (int)module, (int)TN_CODE_CANCELLED, 0,
              nullptr, op, nullptr, 0, nullptr, reason ? reason : "", nullptr);
}

const char* tn_sec_module_slug(tn_module module) {
    switch (module) {
        case TN_MODULE_TN_SECURITY: return "tn_security";
        case TN_MODULE_LLAMA_CPP:   return "llama.cpp";
        case TN_MODULE_GGML:        return "ggml";
        case TN_MODULE_SHERPA_ONNX: return "sherpa-onnx";
        case TN_MODULE_ONNX_RT:     return "onnxruntime";
        case TN_MODULE_MNN:         return "MNN";
        case TN_MODULE_QNN:         return "QNN";
        case TN_MODULE_GGUF_LIB:    return "gguf_lib";
        case TN_MODULE_AI_SHERPA:   return "ai_sherpa";
        case TN_MODULE_AI_SD:       return "ai_sd";
        case TN_MODULE_TN_SERVICE:  return "tn_service";
        case TN_MODULE_TN_APP:      return "tn_app";
        case TN_MODULE_TN_PLUGIN:   return "tn_plugin";
        case TN_MODULE_TN_HXS:      return "tn_hxs";
        case TN_MODULE_UNKNOWN:
        default:                    return "unknown";
    }
}

const char* tn_sec_signal_name(int sig) {
    switch (sig) {
        case SIGSEGV: return "SIGSEGV";
        case SIGABRT: return "SIGABRT";
        case SIGBUS:  return "SIGBUS";
        case SIGILL:  return "SIGILL";
        case SIGFPE:  return "SIGFPE";
        case SIGKILL: return "SIGKILL";
        case SIGSTOP: return "SIGSTOP";
        case SIGTERM: return "SIGTERM";
        case SIGTRAP: return "SIGTRAP";
        default:      return "UNKNOWN";
    }
}

} // extern "C"
