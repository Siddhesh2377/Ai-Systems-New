# ============================================================================
# Tool-Neuron gguf_lib — Consumer ProGuard Rules
# Bundled with the AAR; auto-applied when the consuming app enables R8.
# ============================================================================

# --- JNI native methods (called by name from C++ via RegisterNatives/dlsym) ---
-keep class com.dark.gguf_lib.GGUFNativeLib {
    native <methods>;
}

# --- Callback interfaces invoked from JNI via env->CallVoidMethod ---
# Method names and signatures must match exactly what native code looks up.
-keep interface com.dark.gguf_lib.models.StreamCallback { *; }
-keep interface com.dark.gguf_lib.models.EmbeddingCallback { *; }

# --- Data classes constructed from JNI via env->NewObject ---
-keep class com.dark.gguf_lib.models.EmbeddingResult { *; }

# --- Data classes parsed from JSON (field names must survive for org.json) ---
-keep class com.dark.gguf_lib.models.RAGResult { *; }
-keep class com.dark.gguf_lib.models.DecodingMetrics { *; }
-keep class com.dark.gguf_lib.models.GenerationEvent { *; }
-keep class com.dark.gguf_lib.models.GenerationEvent$* { *; }

# --- Public API classes (keep for SDK consumers) ---
-keep class com.dark.gguf_lib.GGMLEngine { public *; }
-keep class com.dark.gguf_lib.ToolManager { public *; }
-keep class com.dark.gguf_lib.CharacterEngine { public *; }
-keep class com.dark.gguf_lib.EmbeddingEngine { public *; }
-keep class com.dark.gguf_lib.RAGEngine { public *; }

# --- Tool calling models (serialized to/from JSON) ---
-keep class com.dark.gguf_lib.toolcalling.ToolCall { *; }
-keep class com.dark.gguf_lib.toolcalling.ToolCallingConfig { *; }
-keep class com.dark.gguf_lib.toolcalling.GrammarMode { *; }
-keep class com.dark.gguf_lib.toolcalling.ToolDefinitionBuilder { public *; }

# --- Keep Kotlin coroutine continuations (used by suspend functions) ---
-keep class kotlin.coroutines.Continuation { *; }

# --- Suppress warnings for internal llama.cpp JNI symbols ---
-dontwarn com.dark.gguf_lib.**
