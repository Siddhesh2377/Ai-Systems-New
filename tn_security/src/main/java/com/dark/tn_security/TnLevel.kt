package com.dark.tn_security

/** Log/event severity level. Numeric values mirror `tn_level` in tn_security.h. */
enum class TnLevel(val value: Int) {
    TRACE(0),
    DEBUG(1),
    INFO(2),
    WARN(3),
    ERROR(4),
    FATAL(5);

    companion object {
        fun fromInt(v: Int): TnLevel = entries.firstOrNull { it.value == v } ?: INFO
    }
}
