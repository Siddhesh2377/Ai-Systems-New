# Consumer ProGuard rules for com.dark.gguf_lib (gguf_lib AAR).
# Bundled with the AAR; auto-applied when the consuming app enables R8/ProGuard.

# JNI bridge object — native method names are resolved by C++ at runtime via
# auto-discovery (Java_com_dark_gguf_1lib_GGUFNativeLib_*). Renaming or
# stripping any method here crashes at the lookup site.
-keep class com.dark.gguf_lib.GGUFNativeLib {
    public static ** INSTANCE;
    native <methods>;
}
-keepclassmembers class com.dark.gguf_lib.GGUFNativeLib {
    public static ** INSTANCE;
}

# JNI callback interfaces — native code resolves methods via env->GetMethodID.
-keep interface com.dark.gguf_lib.models.StreamCallback { *; }
-keep interface com.dark.gguf_lib.models.EmbeddingCallback { *; }
-keep interface com.dark.gguf_lib.models.AgentCallback { *; }

# Constructed from JNI via env->NewObject.
-keep class com.dark.gguf_lib.models.EmbeddingResult { *; }

# Surface across AIDL / JSON / sealed-class hierarchies — keep names + members.
-keep class com.dark.gguf_lib.models.RAGResult { *; }
-keep class com.dark.gguf_lib.models.DecodingMetrics { *; }
-keep class com.dark.gguf_lib.models.GenerationEvent { *; }
-keep class com.dark.gguf_lib.models.GenerationEvent$* { *; }

# Public SDK classes.
-keep class com.dark.gguf_lib.GGMLEngine { public *; }
-keep class com.dark.gguf_lib.GGMLEngine$* { public *; }
-keep class com.dark.gguf_lib.ToolManager { public *; }
-keep class com.dark.gguf_lib.CharacterEngine { public *; }
-keep class com.dark.gguf_lib.EmbeddingEngine { public *; }
-keep class com.dark.gguf_lib.RAGEngine { public *; }
-keep class com.dark.gguf_lib.TextDigest { *; }
-keep class com.dark.gguf_lib.TextDigest$* { *; }

# Enums — name()/ordinal() are used at runtime.
-keep enum com.dark.gguf_lib.DeviceTier { *; }
-keep enum com.dark.gguf_lib.Mood { *; }
-keep enum com.dark.gguf_lib.toolcalling.GrammarMode { *; }
-keep enum com.dark.gguf_lib.DocKind { *; }

# Public data classes.
-keep class com.dark.gguf_lib.LoadingParams { *; }
-keep class com.dark.gguf_lib.GenerationResult { *; }
-keep class com.dark.gguf_lib.Personality { *; }
-keep class com.dark.gguf_lib.ControlVectorConfig { *; }

# Tool calling API.
-keep class com.dark.gguf_lib.toolcalling.ToolCall { *; }
-keep class com.dark.gguf_lib.toolcalling.ToolCallingConfig { *; }
-keep class com.dark.gguf_lib.toolcalling.ToolDefinitionBuilder { public *; }
-keep class com.dark.gguf_lib.toolcalling.ToolDefinitionBuilder$* { public *; }

# Suspend functions generate Continuation subclasses; preserve them.
-keep class kotlin.coroutines.Continuation
-keepclassmembers class * implements kotlin.coroutines.Continuation { *; }

-keepattributes *Annotation*,Signature,InnerClasses,EnclosingMethod,RuntimeVisibleAnnotations

-dontwarn com.dark.gguf_lib.GGUFNativeLib
