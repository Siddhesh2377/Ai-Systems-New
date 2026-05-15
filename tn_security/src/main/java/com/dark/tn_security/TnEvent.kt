package com.dark.tn_security

/**
 * One diagnostic event. Logs, errors, cancellations, and crashes are all
 * represented here. Persisted to hxs, streamed over AIDL, and rendered by the
 * crash-detail Activity.
 */
sealed class TnEvent {
    abstract val timestampMs: Long
    abstract val module:      TnModule
    abstract val opId:        String?
    abstract val tid:         Int

    data class Log(
        override val timestampMs: Long,
        override val module:      TnModule,
        override val opId:        String?,
        override val tid:         Int,
        val level:   TnLevel,
        val tag:     String?,
        val file:    String?,
        val line:    Int,
        val func:    String?,
        val message: String,
    ) : TnEvent()

    data class Error(
        override val timestampMs: Long,
        override val module:      TnModule,
        override val opId:        String?,
        override val tid:         Int,
        val code:       TnCode,
        val stage:      TnStage,
        val file:       String?,
        val line:       Int,
        val func:       String?,
        val message:    String,
        val suggestion: String?,
        val cause:      Error? = null,
    ) : TnEvent()

    data class Cancellation(
        override val timestampMs: Long,
        override val module:      TnModule,
        override val opId:        String?,
        override val tid:         Int,
        val reason: String?,
    ) : TnEvent()

    /**
     * A native signal handler caught a fatal signal. Constructed by draining
     * `<crashFile>` JSON written by the C signal handler — the service
     * process is already dead when this is materialised on the app side.
     */
    data class Crash(
        override val timestampMs: Long,
        override val module:      TnModule,
        override val opId:        String?,
        override val tid:         Int,
        val signal:        Int,
        val signalName:    String,
        val pid:           Int,
        val faultAddr:     String?,
        val crashFilePath: String,
        /** Last N events captured in the ring buffer leading up to the crash. */
        val ring:          List<Log>,
    ) : TnEvent()
}
