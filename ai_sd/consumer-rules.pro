# ============================================================================
# ai_sd — Consumer ProGuard/R8 Rules
# These rules are automatically applied to any app that depends on ai_sd.
# ============================================================================

# --- JNI Native Methods ---
# SDNativeLib uses JNI — method names must match C++ extern declarations exactly.
# R8 must not rename or remove native methods or the class itself.
-keep class com.dark.ai_sd.SDNativeLib {
    native <methods>;
    <init>();
}

# --- JNI Callback Interface ---
# SDCallback methods are invoked from C++ via JNI reflection (GetMethodID).
# Renaming any method breaks the native -> Java callback chain.
-keep interface com.dark.ai_sd.SDCallback { *; }
-keep class * implements com.dark.ai_sd.SDCallback { *; }

# --- Public API Classes ---
# Keep all public-facing SDK classes and their public/protected members.
-keep class com.dark.ai_sd.StableDiffusionManager {
    public *;
}
-keep class com.dark.ai_sd.DiffusionManager {
    public *;
}

# --- Data Classes & Sealed Hierarchies ---
# These are used by consumers and may be checked with `is` / `when`.
# Sealed class hierarchies need all subclasses kept for exhaustive when-expressions.
-keep class com.dark.ai_sd.DiffusionModelConfig { *; }
-keep class com.dark.ai_sd.DiffusionGenerationParams { *; }
-keep class com.dark.ai_sd.DiffusionRuntimeConfig { *; }

-keep class com.dark.ai_sd.DiffusionBackendState { *; }
-keep class com.dark.ai_sd.DiffusionBackendState$* { *; }

-keep class com.dark.ai_sd.DiffusionGenerationState { *; }
-keep class com.dark.ai_sd.DiffusionGenerationState$* { *; }

-keep class com.dark.ai_sd.DiffusionGenerationResult { *; }
-keep class com.dark.ai_sd.DiffusionGenerationResult$* { *; }

-keep class com.dark.ai_sd.UpscaleState { *; }
-keep class com.dark.ai_sd.UpscaleState$* { *; }

-keep class com.dark.ai_sd.RuntimeSetupState { *; }
-keep class com.dark.ai_sd.RuntimeSetupState$* { *; }

# --- Apache Commons Compress (tar.xz extraction for QNN libs) ---
-dontwarn org.apache.commons.compress.**
-keep class org.apache.commons.compress.archivers.tar.** { *; }
-keep class org.apache.commons.compress.compressors.xz.** { *; }
-keep class org.tukaani.xz.** { *; }
