# JNI — consuming apps must not rename these; method names are resolved by native code at runtime
-keepclasseswithmembernames,includedescriptorclasses class com.dark.ai_sd.SDNativeLib {
    native <methods>;
}

# SDCallback — method names called from C++ via GetMethodID; renaming breaks the callback chain
-keep interface com.dark.ai_sd.SDCallback { *; }
-keep class * implements com.dark.ai_sd.SDCallback { *; }

# Public API surface
-keep class com.dark.ai_sd.StableDiffusionManager { public *; }
-keep class com.dark.ai_sd.StableDiffusionManager$Companion { *; }
-keep class com.dark.ai_sd.DiffusionManager { public *; }
-keep class com.dark.ai_sd.DiffusionManager$Companion { *; }

# Data classes used by consumers
-keep class com.dark.ai_sd.DiffusionModelConfig { *; }
-keep class com.dark.ai_sd.DiffusionGenerationParams { *; }
-keep class com.dark.ai_sd.DiffusionRuntimeConfig { *; }
-keep class com.dark.ai_sd.LoRAConfig { *; }

# Sealed class hierarchies — subclasses needed for when-expressions and instanceof checks
-keep class com.dark.ai_sd.DiffusionBackendState { *; }
-keep class com.dark.ai_sd.DiffusionBackendState$* { *; }
-keep class com.dark.ai_sd.DiffusionGenerationState { *; }
-keep class com.dark.ai_sd.DiffusionGenerationState$* { *; }
-keep class com.dark.ai_sd.DiffusionGenerationResult { *; }
-keep class com.dark.ai_sd.DiffusionGenerationResult$* { *; }
-keep class com.dark.ai_sd.UpscaleState { *; }
-keep class com.dark.ai_sd.UpscaleState$* { *; }
-keep class com.dark.ai_sd.SegmenterState { *; }
-keep class com.dark.ai_sd.SegmenterState$* { *; }
-keep class com.dark.ai_sd.LamaState { *; }
-keep class com.dark.ai_sd.LamaState$* { *; }
-keep class com.dark.ai_sd.DepthState { *; }
-keep class com.dark.ai_sd.DepthState$* { *; }
-keep class com.dark.ai_sd.StyleState { *; }
-keep class com.dark.ai_sd.StyleState$* { *; }
-keep class com.dark.ai_sd.LoRAState { *; }
-keep class com.dark.ai_sd.LoRAState$* { *; }
-keep class com.dark.ai_sd.RuntimeSetupState { *; }
-keep class com.dark.ai_sd.RuntimeSetupState$* { *; }

# Apache Commons Compress + XZ — declared as api dependency; keep for transitive consumers
-dontwarn org.apache.commons.compress.**
-keep class org.apache.commons.compress.archivers.tar.** { *; }
-keep class org.apache.commons.compress.compressors.xz.** { *; }
-keep class org.tukaani.xz.** { *; }
