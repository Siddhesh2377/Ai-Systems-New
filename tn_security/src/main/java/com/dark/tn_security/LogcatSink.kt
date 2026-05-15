package com.dark.tn_security

import android.util.Log

/**
 * Reference sink — writes every event to logcat. Useful in dev. Production
 * should pair this with [HxsSink][com.dark.tn_security.HxsSink] (in
 * ToolNeuron) for persistence.
 */
class LogcatSink(
    private val tagPrefix: String = "tn",
) : TnSink {
    override fun onEvent(event: TnEvent) {
        val tag = "$tagPrefix:${event.module.slug}"
        when (event) {
            is TnEvent.Log -> {
                val msg = buildString {
                    event.opId?.let { append("[op=$it] ") }
                    event.tag?.let { append("[$it] ") }
                    append(event.message)
                    if (event.file != null && event.line > 0) {
                        append("  @ ${event.file}:${event.line}")
                    }
                }
                when (event.level) {
                    TnLevel.TRACE -> Log.v(tag, msg)
                    TnLevel.DEBUG -> Log.d(tag, msg)
                    TnLevel.INFO  -> Log.i(tag, msg)
                    TnLevel.WARN  -> Log.w(tag, msg)
                    TnLevel.ERROR -> Log.e(tag, msg)
                    TnLevel.FATAL -> Log.wtf(tag, msg)
                }
            }
            is TnEvent.Error -> {
                Log.e(tag, buildString {
                    event.opId?.let { append("[op=$it] ") }
                    append("code=${event.code} stage=${event.stage}: ")
                    append(event.message)
                    event.suggestion?.let { append("  (hint: $it)") }
                    if (event.file != null && event.line > 0) {
                        append("  @ ${event.file}:${event.line}")
                    }
                })
            }
            is TnEvent.Cancellation -> {
                Log.i(tag, "[cancelled]${event.opId?.let { " op=$it" } ?: ""}: ${event.reason ?: ""}")
            }
            is TnEvent.Crash -> {
                Log.e(tag, "[crash] ${event.signalName} pid=${event.pid} tid=${event.tid} faultAddr=${event.faultAddr ?: "-"} file=${event.crashFilePath} ring_size=${event.ring.size}")
            }
        }
    }
}
