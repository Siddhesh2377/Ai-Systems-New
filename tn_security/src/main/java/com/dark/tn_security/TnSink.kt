package com.dark.tn_security

/**
 * Consumer of TnEvents. Register with [TnSecurity.addSink].
 *
 * Sinks are called on whichever thread emitted the event — JNI threads,
 * background workers, UI thread. Implementations must be thread-safe and
 * must not block (the event-emit path is in the hot inference loop).
 */
fun interface TnSink {
    fun onEvent(event: TnEvent)
}
