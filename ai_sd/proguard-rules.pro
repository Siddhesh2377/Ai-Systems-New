# ============================================================================
# ai_sd — Module ProGuard/R8 Rules (applied when building the library AAR)
# ============================================================================

# Preserve line numbers for crash reports
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile

# Keep all public SDK API surface
-keep public class com.dark.ai_sd.** { public *; }

# JNI: native methods must retain exact signatures
-keepclasseswithmembernames class com.dark.ai_sd.SDNativeLib {
    native <methods>;
}

# JNI callback: C++ calls these via GetMethodID reflection
-keep interface com.dark.ai_sd.SDCallback { *; }
-keep class * implements com.dark.ai_sd.SDCallback { *; }

# Sealed class subclasses — needed for when-expressions and instanceof checks
-keep class com.dark.ai_sd.DiffusionBackendState$* { *; }
-keep class com.dark.ai_sd.DiffusionGenerationState$* { *; }
-keep class com.dark.ai_sd.DiffusionGenerationResult$* { *; }
-keep class com.dark.ai_sd.UpscaleState$* { *; }
-keep class com.dark.ai_sd.RuntimeSetupState$* { *; }

# Apache Commons Compress + XZ (tar.xz extraction for QNN libs)
-dontwarn org.apache.commons.compress.**
-keep class org.apache.commons.compress.archivers.tar.** { *; }
-keep class org.apache.commons.compress.compressors.xz.** { *; }
-keep class org.tukaani.xz.** { *; }
