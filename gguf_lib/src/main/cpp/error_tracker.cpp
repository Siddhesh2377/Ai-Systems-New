#include "error_tracker.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include <fcntl.h>
#include <signal.h>
#include <unistd.h>

#include <android/log.h>
#define ERR_TAG "TnError"
#define ELOGI(...) __android_log_print(ANDROID_LOG_INFO,  ERR_TAG, __VA_ARGS__)
#define ELOGE(...) __android_log_print(ANDROID_LOG_ERROR, ERR_TAG, __VA_ARGS__)

namespace {

constexpr size_t kStateBufSize = 1024;
constexpr size_t kErrorBufSize = 2048;
constexpr size_t kPathBufSize  = 512;

std::mutex     g_state_mutex;
char           g_state_buf[kStateBufSize] = "{}";

std::mutex     g_error_mutex;
char           g_error_buf[kErrorBufSize] = "{}";

std::mutex     g_path_mutex;
char           g_crash_log_path[kPathBufSize] = "";

std::atomic<bool> g_initialized{false};

void escape_into(char* dst, size_t dst_size, const char* src) {
    size_t out = 0;
    if (!src) { dst[0] = '\0'; return; }
    for (size_t i = 0; src[i] != '\0' && out + 7 < dst_size; ++i) {
        unsigned char c = (unsigned char) src[i];
        if (c == '"' || c == '\\') {
            dst[out++] = '\\';
            dst[out++] = (char) c;
        } else if (c == '\n') {
            dst[out++] = '\\'; dst[out++] = 'n';
        } else if (c == '\r') {
            dst[out++] = '\\'; dst[out++] = 'r';
        } else if (c == '\t') {
            dst[out++] = '\\'; dst[out++] = 't';
        } else if (c < 0x20) {
            int n = snprintf(dst + out, dst_size - out, "\\u%04x", c);
            if (n > 0) out += (size_t) n;
        } else {
            dst[out++] = (char) c;
        }
    }
    dst[out] = '\0';
}

long long now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

void crash_handler(int sig, siginfo_t* info, void* ctx) {
    (void) info; (void) ctx;

    char path[kPathBufSize];
    {
        path[0] = '\0';
        std::strncpy(path, g_crash_log_path, sizeof(path) - 1);
        path[sizeof(path) - 1] = '\0';
    }
    if (path[0] == '\0') goto reraise;

    {
        int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) goto reraise;

        const char* sig_name = tn_error_signal_name(sig);
        char prefix[256];
        int n = snprintf(prefix, sizeof(prefix),
            "{\"signal\":%d,\"signal_name\":\"%s\",\"timestamp\":%lld,\"current_op\":",
            sig, sig_name, now_ms());
        if (n > 0) write(fd, prefix, (size_t) n);

        size_t state_len = strlen(g_state_buf);
        if (state_len > 0) write(fd, g_state_buf, state_len);
        else                write(fd, "null", 4);

        const char* tail = ",\"last_error\":";
        write(fd, tail, strlen(tail));
        size_t err_len = strlen(g_error_buf);
        if (err_len > 0) write(fd, g_error_buf, err_len);
        else              write(fd, "null", 4);

        write(fd, "}", 1);
        fsync(fd);
        close(fd);
    }

reraise:
    struct sigaction sa{};
    sa.sa_handler = SIG_DFL;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(sig, &sa, nullptr);
    raise(sig);
}

void install_handler(int sig) {
    struct sigaction sa{};
    sa.sa_sigaction = crash_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigaction(sig, &sa, nullptr);
}

}

extern "C" {

void tn_error_init(void) {
    bool was = g_initialized.exchange(true);
    if (was) return;
    install_handler(SIGSEGV);
    install_handler(SIGABRT);
    install_handler(SIGFPE);
    install_handler(SIGBUS);
    install_handler(SIGILL);
    ELOGI("Crash handlers installed");
}

void tn_error_set_crash_log_path(const char* path) {
    if (!path) return;
    std::lock_guard<std::mutex> lock(g_path_mutex);
    std::strncpy(g_crash_log_path, path, sizeof(g_crash_log_path) - 1);
    g_crash_log_path[sizeof(g_crash_log_path) - 1] = '\0';
}

void tn_error_set_op(const char* op, const char* detail) {
    char op_esc[256];
    char detail_esc[512];
    escape_into(op_esc, sizeof(op_esc), op);
    escape_into(detail_esc, sizeof(detail_esc), detail);

    std::lock_guard<std::mutex> lock(g_state_mutex);
    snprintf(g_state_buf, sizeof(g_state_buf),
        "{\"op\":\"%s\",\"detail\":\"%s\",\"started_ms\":%lld}",
        op_esc, detail_esc, now_ms());
}

void tn_error_clear_op(void) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    std::strcpy(g_state_buf, "{}");
}

void tn_error_set_last(int code, const char* category, const char* message) {
    char cat_esc[128];
    char msg_esc[1024];
    escape_into(cat_esc, sizeof(cat_esc), category);
    escape_into(msg_esc, sizeof(msg_esc), message);

    char op_snapshot[kStateBufSize];
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        std::strncpy(op_snapshot, g_state_buf, sizeof(op_snapshot) - 1);
        op_snapshot[sizeof(op_snapshot) - 1] = '\0';
    }

    std::lock_guard<std::mutex> lock(g_error_mutex);
    snprintf(g_error_buf, sizeof(g_error_buf),
        "{\"code\":%d,\"category\":\"%s\",\"message\":\"%s\",\"op_at_time\":%s,\"timestamp\":%lld}",
        code, cat_esc, msg_esc, op_snapshot, now_ms());

    ELOGE("%s [%s]: %s", category ? category : "?", op_snapshot, message ? message : "?");
}

void tn_error_clear_last(void) {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    std::strcpy(g_error_buf, "{}");
}

const char* tn_error_get_last_json(void) {
    return g_error_buf;
}

const char* tn_error_signal_name(int sig) {
    switch (sig) {
        case SIGSEGV: return "SIGSEGV";
        case SIGABRT: return "SIGABRT";
        case SIGFPE:  return "SIGFPE";
        case SIGBUS:  return "SIGBUS";
        case SIGILL:  return "SIGILL";
        default:      return "UNKNOWN";
    }
}

}
