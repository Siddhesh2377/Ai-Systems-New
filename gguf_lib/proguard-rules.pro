# ============================================================================
# Tool-Neuron gguf_lib — Library ProGuard Rules
# Applied when building the library with minification enabled (release builds).
# Consumer rules in consumer-rules.pro are automatically included — no duplication needed.
# ============================================================================

# --- Source map preservation for crash reports ---
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile

# --- Kotlin metadata and signatures ---
-keepattributes *Annotation*,Signature,InnerClasses,EnclosingMethod,RuntimeVisibleAnnotations

# --- Kotlin intrinsics ---
# R8 can inline these but some versions produce broken bytecode without this guard.
-dontwarn kotlin.Unit
-dontwarn kotlin.**

# --- Coroutines ---
# Keep only what's needed: internal state machine classes and flow infrastructure.
# Do NOT blanket-keep all of kotlinx.coroutines — that defeats minification.
-keep class kotlinx.coroutines.flow.** { *; }
-keepclassmembers class * {
    @kotlinx.coroutines.** *;
}
-dontwarn kotlinx.coroutines.**

# --- Kotlin callbackFlow / awaitClose internals ---
# Used by GGMLEngine streaming flows — the lambda closures must survive.
-keepclassmembers class * extends kotlinx.coroutines.channels.ProducerScope { *; }

# --- R8 / ProGuard compatibility ---
-ignorewarnings
