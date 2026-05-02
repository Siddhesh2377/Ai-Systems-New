# ============================================================================
# Tool-Neuron gguf_lib — Consumer ProGuard Rules
# Bundled with the AAR; auto-applied when the consuming app enables R8/ProGuard.
# ============================================================================

# --- JNI bridge ---
# Method names are looked up by the native layer at runtime via GetMethodID.
# Renaming or removing ANY method here will cause a crash at the JNI lookup site.
# Kotlin object — must keep INSTANCE field so callers can dispatch native methods.
-keep class com.dark.gguf_lib.GGUFNativeLib {
    public static ** INSTANCE;
    native <methods>;
}
-keepclassmembers class com.dark.gguf_lib.GGUFNativeLib {
    public static ** INSTANCE;
}

# --- JNI callback interfaces ---
# Called from C++ via env->GetMethodID + env->CallVoidMethod.
# Method names and signatures must match exactly what native code expects.
-keep interface com.dark.gguf_lib.models.StreamCallback { *; }
-keep interface com.dark.gguf_lib.models.EmbeddingCallback { *; }
-keep interface com.dark.gguf_lib.models.AgentCallback { *; }

# --- Data classes constructed from JNI via env->NewObject ---
-keep class com.dark.gguf_lib.models.EmbeddingResult { *; }

# --- Sealed classes / data classes parsed from JSON or used across AIDL ---
-keep class com.dark.gguf_lib.models.RAGResult { *; }
-keep class com.dark.gguf_lib.models.DecodingMetrics { *; }
-keep class com.dark.gguf_lib.models.GenerationEvent { *; }
-keep class com.dark.gguf_lib.models.GenerationEvent$* { *; }

# --- Public SDK classes ---
# Keep public surface so consuming apps can call them after minification.
-keep class com.dark.gguf_lib.GGMLEngine { public *; }
-keep class com.dark.gguf_lib.GGMLEngine$* { public *; }          # companion object + nested types
-keep class com.dark.gguf_lib.ToolManager { public *; }
-keep class com.dark.gguf_lib.CharacterEngine { public *; }
-keep class com.dark.gguf_lib.EmbeddingEngine { public *; }
-keep class com.dark.gguf_lib.RAGEngine { public *; }
-keep class com.dark.gguf_lib.TextDigest { *; }
-keep class com.dark.gguf_lib.TextDigest$* { *; }

# --- Enums used by the SDK ---
# Enum names are accessed via name()/ordinal() in Kotlin/Java and must survive.
-keep enum com.dark.gguf_lib.DeviceTier { *; }
-keep enum com.dark.gguf_lib.Mood { *; }
-keep enum com.dark.gguf_lib.toolcalling.GrammarMode { *; }
-keep enum com.dark.gguf_lib.TaskProfileMode { *; }
-keep enum com.dark.gguf_lib.DocKind { *; }

# --- Data classes part of public API ---
-keep class com.dark.gguf_lib.LoadingParams { *; }
-keep class com.dark.gguf_lib.GenerationResult { *; }
-keep class com.dark.gguf_lib.Personality { *; }
-keep class com.dark.gguf_lib.ControlVectorConfig { *; }

# --- Tool calling API ---
-keep class com.dark.gguf_lib.toolcalling.ToolCall { *; }
-keep class com.dark.gguf_lib.toolcalling.ToolCallingConfig { *; }
-keep class com.dark.gguf_lib.toolcalling.ToolDefinitionBuilder { public *; }
-keep class com.dark.gguf_lib.toolcalling.ToolDefinitionBuilder$* { public *; }

# --- Kotlin coroutine continuations ---
# Suspend functions generate Continuation subclasses; these must not be removed.
-keep class kotlin.coroutines.Continuation
-keepclassmembers class * implements kotlin.coroutines.Continuation { *; }

# --- Kotlin metadata ---
# Required for reflection-based access (e.g. Gson, kotlinx.serialization, AIDL stubs).
-keepattributes *Annotation*,Signature,InnerClasses,EnclosingMethod,RuntimeVisibleAnnotations

# --- Suppress warnings for internal implementation details ---
-dontwarn com.dark.gguf_lib.GGUFNativeLib
