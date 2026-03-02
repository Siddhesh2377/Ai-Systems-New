# ============================================================================
# Tool-Neuron gguf_lib — Library ProGuard Rules
# Applied when building the library itself with minification enabled.
# ============================================================================

# Keep everything in consumer-rules.pro (auto-included),
# plus library-internal classes needed for correct operation.

# JNI bridge — all native methods must be kept verbatim
-keep class com.dark.gguf_lib.GGUFNativeLib {
    native <methods>;
    # Static init block loads the .so
    static { *; }
}

# Callback interfaces — method signatures must match JNI lookups
-keep interface com.dark.gguf_lib.models.StreamCallback { *; }
-keep interface com.dark.gguf_lib.models.EmbeddingCallback { *; }

# Data classes constructed or inspected from native/JSON
-keep class com.dark.gguf_lib.models.** { *; }
-keep class com.dark.gguf_lib.toolcalling.** { *; }

# Public SDK API
-keep class com.dark.gguf_lib.GGMLEngine { public protected *; }
-keep class com.dark.gguf_lib.ToolManager { public protected *; }
-keep class com.dark.gguf_lib.CharacterEngine { public protected *; }
-keep class com.dark.gguf_lib.EmbeddingEngine { public protected *; }
-keep class com.dark.gguf_lib.RAGEngine { public protected *; }

# Keep source file + line numbers for crash reports
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile

# Keep annotations (used by Kotlin, coroutines, etc.)
-keepattributes *Annotation*,Signature,InnerClasses,EnclosingMethod

# Kotlin coroutines
-keep class kotlinx.coroutines.** { *; }
-dontwarn kotlinx.coroutines.**
