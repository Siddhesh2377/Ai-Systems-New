# Source info for crash reports
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile

# Kotlin metadata, generics, checked exceptions
-keepattributes Exceptions,Signature,InnerClasses,EnclosingMethod

# JNI — method names must match C++ extern "C" declarations exactly.
# includedescriptorclasses keeps parameter/return types from being renamed.
-keepclasseswithmembernames,includedescriptorclasses class com.dark.ai_sd.SDNativeLib {
    native <methods>;
}

# SDCallback methods are invoked from C++ via GetMethodID.
# Every method name and descriptor must be exact.
-keep interface com.dark.ai_sd.SDCallback { *; }
-keep class * implements com.dark.ai_sd.SDCallback { *; }

# Public API entry points
-keep class com.dark.ai_sd.StableDiffusionManager { public *; }
-keep class com.dark.ai_sd.DiffusionManager { public *; }

# Data classes passed to / returned from the API
-keep class com.dark.ai_sd.DiffusionModelConfig { *; }
-keep class com.dark.ai_sd.DiffusionGenerationParams { *; }
-keep class com.dark.ai_sd.DiffusionRuntimeConfig { *; }
-keep class com.dark.ai_sd.LoRAConfig { *; }

# Sealed class hierarchies — all subclasses needed for when-expressions and instanceof checks
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

# Apache Commons Compress + XZ — used for tar.xz QNN lib extraction at runtime
-dontwarn org.apache.commons.compress.**
-keep class org.apache.commons.compress.archivers.tar.** { *; }
-keep class org.apache.commons.compress.compressors.xz.** { *; }
-keep class org.tukaani.xz.** { *; }
