package com.dark.tn_security

import android.content.Context
import android.util.Log
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import org.json.JSONException
import org.json.JSONObject
import java.io.File
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Unified diagnostic + error capture singleton. Entry point for the SDK.
 *
 * ## Wiring
 *
 * In each process that loads tn_security:
 *
 * ```
 * TnSecurity.init(
 *     context  = appContext,
 *     module   = TnModule.TN_SERVICE,        // or TN_APP / TN_PLUGIN
 *     crashDir = File(context.filesDir, "tn_security/crashes"),
 * )
 * TnSecurity.addSink(LogcatSink())            // dev
 * // app process additionally:
 * TnSecurity.addSink(HxsSink(hexStorage))     // persistent + queryable
 * ```
 *
 * ## Producers
 *
 * Native (C/C++): include `<tn_security/tn_security_macros.h>` and use
 * `TN_I(...) / TN_E(...) / TN_ERR(code, stage, fmt, ...)`. The C core ships
 * events to this singleton via JNI.
 *
 * Kotlin: call [log], [error], [cancel], or wrap an operation in [withOp].
 */
object TnSecurity {

    @Volatile private var initialized   = false
    @Volatile private var processModule = TnModule.UNKNOWN

    private val sinks = CopyOnWriteArrayList<TnSink>()

    private val _events = MutableSharedFlow<TnEvent>(
        replay              = 0,
        extraBufferCapacity = 256,
        onBufferOverflow    = BufferOverflow.DROP_OLDEST,
    )

    /** Stream of every event emitted in this process (logs, errors, cancellations). */
    val events: SharedFlow<TnEvent> = _events.asSharedFlow()

    /**
     * Load native lib, init core, install signal handlers, set crash-file
     * pattern. Safe to call multiple times — second-and-later calls update
     * config only.
     */
    @JvmStatic
    fun init(
        context:  Context,
        module:   TnModule,
        crashDir: File? = null,
        installSignalHandlers: Boolean = true,
    ) {
        if (!initialized) {
            try {
                System.loadLibrary("tn_security")
            } catch (t: Throwable) {
                Log.e(LOG_TAG, "loadLibrary failed", t)
                return
            }
            nativeInit()
            initialized = true
        }
        processModule = module

        if (crashDir != null) {
            crashDir.mkdirs()
            // %m=module slug, %p=pid, %t=epoch ms
            val pattern = File(crashDir, "crash_%m_%p_%t.json").absolutePath
            nativeSetCrashFilePattern(pattern)
        }

        if (installSignalHandlers) nativeInstallSignalHandlers()
    }

    @JvmStatic
    fun shutdown() {
        if (!initialized) return
        nativeShutdown()
        initialized = false
        sinks.clear()
    }

    // ── Sinks ──────────────────────────────────────────────────────────────

    fun addSink(sink: TnSink): TnSink {
        sinks.add(sink)
        return sink
    }

    fun removeSink(sink: TnSink) { sinks.remove(sink) }

    // ── Op tracking ────────────────────────────────────────────────────────

    fun setOp(opId: String?)           { if (initialized) nativeSetOp(opId) }
    fun clearOp()                       { if (initialized) nativeClearOp() }
    fun currentOp(): String?            = if (initialized) nativeCurrentOp() else null

    /** Run [block] with an op-id stamped on every error/log emitted during it. */
    inline fun <T> withOp(opId: String, block: () -> T): T {
        val previous = currentOp()
        setOp(opId)
        try {
            return block()
        } finally {
            if (previous != null) setOp(previous) else clearOp()
        }
    }

    // ── Emission ───────────────────────────────────────────────────────────

    @JvmOverloads
    fun log(
        level:   TnLevel,
        module:  TnModule = processModule,
        message: String,
        tag:     String?  = null,
        opId:    String?  = null,
        file:    String?  = null,
        line:    Int       = 0,
        func:    String?  = null,
    ) {
        if (!initialized) return
        nativeLog(level.value, module.value, tag, opId, file, line, func, message)
    }

    @JvmOverloads
    fun error(
        code:       TnCode,
        stage:      TnStage,
        message:    String,
        module:     TnModule = processModule,
        suggestion: String?  = null,
        opId:       String?  = null,
        file:       String?  = null,
        line:       Int       = 0,
        func:       String?  = null,
    ) {
        if (!initialized) return
        nativeEmitError(module.value, code.value, stage.value,
                        opId, file, line, func, suggestion, message)
    }

    @JvmOverloads
    fun cancel(
        module: TnModule = processModule,
        opId:   String?  = null,
        reason: String?  = null,
    ) {
        if (!initialized) return
        nativeEmitCancellation(module.value, opId, reason)
    }

    // ── Crash-file drain ──────────────────────────────────────────────────
    //
    // After a service-process death, the signal handler has written a crash
    // file to the configured directory. On next service rebind the app reads
    // and replays each as a [TnEvent.Crash] through the normal sink pipeline.
    //
    // To make crash JSON available for post-mortem (e.g. `adb pull`) without
    // re-spamming the log on every process start, each replayed file is
    // RENAMED from `crash_*.json` to `crash_*.json.seen`. Subsequent drains
    // only pick up the bare `.json` files (i.e. brand-new crashes), so the
    // same crash is broadcast exactly once across all processes that start.
    // FIFO eviction (cap [MAX_RETAINED_CRASH_FILES]) considers BOTH
    // extensions so the on-disk retention is bounded regardless of whether
    // a file has been replayed yet.

    private const val MAX_RETAINED_CRASH_FILES = 5

    fun drainCrashFiles(crashDir: File): List<TnEvent.Crash> {
        if (!crashDir.isDirectory) return emptyList()
        // Strict: only new (unread) crash files. Already-replayed files have
        // a `.seen` suffix and are skipped here on purpose.
        val freshFiles = crashDir.listFiles { f ->
            f.name.startsWith("crash_") && f.name.endsWith(".json")
        } ?: emptyArray()

        val out = mutableListOf<TnEvent.Crash>()
        for (f in freshFiles) {
            val crash = runCatching { parseCrashFile(f) }.getOrNull() ?: continue
            out += crash
            dispatch(crash)
            // Mark replayed so future drainCrashFiles calls (from sibling
            // processes or app relaunches) don't re-broadcast.
            runCatching { f.renameTo(File(f.parentFile, f.name + ".seen")) }
        }
        evictOldCrashFiles(listAllCrashFiles(crashDir))
        return out
    }

    private fun listAllCrashFiles(crashDir: File): List<File> {
        // Pick up both fresh and already-replayed crash files for the FIFO
        // retention pass — we want a hard upper bound on disk usage no
        // matter how many crashes have been seen.
        return crashDir.listFiles { f ->
            f.name.startsWith("crash_") &&
                (f.name.endsWith(".json") || f.name.endsWith(".json.seen"))
        }?.toList().orEmpty()
    }

    private fun crashFileSortKey(f: File): Long {
        // Pattern: crash_<module>_<pid>_<epochMs>.json[.seen] — last
        // underscore-separated chunk before ".json[.seen]" is the
        // timestamp. Falls back to file mtime if the trailing component
        // isn't numeric.
        val stem = f.name
            .removePrefix("crash_")
            .removeSuffix(".seen")
            .removeSuffix(".json")
        val tail = stem.substringAfterLast('_', missingDelimiterValue = "")
        return tail.toLongOrNull() ?: f.lastModified()
    }

    private fun evictOldCrashFiles(files: List<File>) {
        if (files.size <= MAX_RETAINED_CRASH_FILES) return
        val toDelete = files
            .sortedByDescending { crashFileSortKey(it) }
            .drop(MAX_RETAINED_CRASH_FILES)
        for (f in toDelete) runCatching { f.delete() }
    }

    private fun parseCrashFile(file: File): TnEvent.Crash {
        val json = JSONObject(file.readText())
        val ring = mutableListOf<TnEvent.Log>()
        val ja = json.optJSONArray("ring")
        if (ja != null) {
            for (i in 0 until ja.length()) {
                val o = ja.getJSONObject(i)
                ring += TnEvent.Log(
                    timestampMs = o.optLong("ts"),
                    module      = TnModule.fromInt(o.optInt("mod")),
                    opId        = o.optString("op").ifBlank { null },
                    tid         = o.optInt("tid"),
                    level       = TnLevel.fromInt(o.optInt("lvl")),
                    tag         = o.optString("tag").ifBlank { null },
                    file        = o.optString("file").ifBlank { null },
                    line        = o.optInt("line"),
                    func        = o.optString("func").ifBlank { null },
                    message     = o.optString("msg"),
                )
            }
        }
        return TnEvent.Crash(
            timestampMs   = json.optLong("timestamp_ms"),
            module        = TnModule.fromInt(json.optInt("module")),
            opId          = ring.lastOrNull()?.opId,
            tid           = json.optInt("tid"),
            signal        = json.optInt("signal"),
            signalName    = json.optString("signal_name", "UNKNOWN"),
            pid           = json.optInt("pid"),
            faultAddr     = json.optString("fault_addr").ifBlank { null },
            crashFilePath = file.absolutePath,
            ring          = ring,
        )
    }

    // ── JNI sink — called from native code ─────────────────────────────────

    @JvmStatic
    @Suppress("unused", "LongParameterList")
    private fun onNativeEvent(
        kind:        Int,
        level:       Int,
        module:      Int,
        code:        Int,
        stage:       Int,
        tag:         String?,
        opId:        String?,
        file:        String?,
        line:        Int,
        func:        String?,
        message:     String?,
        suggestion:  String?,
        timestampMs: Long,
        tid:         Int,
    ) {
        val ev: TnEvent = when (kind) {
            1 -> TnEvent.Error(
                timestampMs = timestampMs,
                module      = TnModule.fromInt(module),
                opId        = opId,
                tid         = tid,
                code        = TnCode.fromInt(code),
                stage       = TnStage.fromInt(stage),
                file        = file,
                line        = line,
                func        = func,
                message     = message ?: "",
                suggestion  = suggestion,
            )
            2 -> TnEvent.Cancellation(
                timestampMs = timestampMs,
                module      = TnModule.fromInt(module),
                opId        = opId,
                tid         = tid,
                reason      = message,
            )
            else -> TnEvent.Log(
                timestampMs = timestampMs,
                module      = TnModule.fromInt(module),
                opId        = opId,
                tid         = tid,
                level       = TnLevel.fromInt(level),
                tag         = tag,
                file        = file,
                line        = line,
                func        = func,
                message     = message ?: "",
            )
        }
        dispatch(ev)
    }

    private fun dispatch(ev: TnEvent) {
        _events.tryEmit(ev)
        for (sink in sinks) {
            try {
                sink.onEvent(ev)
            } catch (t: Throwable) {
                Log.e(LOG_TAG, "sink threw", t)
            }
        }
    }

    // ── Test endpoint — only callable in instrumentation tests ─────────────

    @Suppress("unused")
    internal fun raiseSignalForTest(sig: Int) {
        if (initialized) nativeRaiseSignalForTest(sig)
    }

    // ── Native bridges ─────────────────────────────────────────────────────

    private external fun nativeInit()
    private external fun nativeShutdown()
    private external fun nativeSetCrashFilePattern(pattern: String?)
    private external fun nativeInstallSignalHandlers()
    private external fun nativeSetOp(opId: String?)
    private external fun nativeClearOp()
    private external fun nativeCurrentOp(): String?
    private external fun nativeLog(
        level: Int, module: Int,
        tag: String?, opId: String?, file: String?, line: Int, func: String?,
        message: String,
    )
    private external fun nativeEmitError(
        module: Int, code: Int, stage: Int,
        opId: String?, file: String?, line: Int, func: String?,
        suggestion: String?, message: String,
    )
    private external fun nativeEmitCancellation(module: Int, opId: String?, reason: String?)
    private external fun nativeModuleSlug(module: Int): String
    private external fun nativeSignalName(sig: Int): String
    private external fun nativeApiVersion(): Int
    private external fun nativeRaiseSignalForTest(sig: Int)

    private const val LOG_TAG = "TnSecurity"
}
