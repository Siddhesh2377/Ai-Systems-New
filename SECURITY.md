# AiSystems SDK Security Audit Report

**Date:** 2026-02-28 (updated 2026-03-02)
**Total Findings:** 8 (1 Critical, 1 High, 3 Medium, 3 Low)
**Modules Audited:** `gguf_lib`, `ai_sd`, `ai_supertonic_tts`

## Executive Summary

The AiSystems SDK has SDK-level licensing via native Ed25519 signature verification in `gguf_lib` (see `license_guard.cpp`). However, obfuscation is disabled across all modules and several other security gaps exist.

---

## CATEGORY 1: PREMIUM/MONETIZATION BYPASS

### FINDING 1 — CRITICAL: Obfuscation Completely Disabled Across ALL Modules

Every single module has `isMinifyEnabled = false` in its release build type.

**Affected files:**

- `gguf_lib/build.gradle.kts`: `isMinifyEnabled = false`
- `ai_sd/build.gradle.kts`: `isMinifyEnabled = false`
- `ai_supertonic_tts/build.gradle.kts`: `isMinifyEnabled = false`

**Fix:** Set `isMinifyEnabled = true` in all release builds. Write proper keep rules only for JNI and serialization classes.

---

## CATEGORY 2: MOD-ABILITY / TAMPERING

### FINDING 2 — HIGH: No Tamper Detection or APK Signature Verification

The SDK has zero tamper detection:

- No APK signature verification at runtime
- No checksum validation on SDK `.so` libraries
- No detection of Frida, Xposed, or other hooking frameworks
- No detection of debugger attachment

**Note:** `gguf_lib` has anti-tamper in `license_integrity.cpp` (TracerPid, Frida, APK cert check). This should be extended to other modules.

**Fix:** Extend runtime tamper detection to all modules.

---

## CATEGORY 3: INFORMATION DISCLOSURE / REVERSE ENGINEERING

### FINDING 3 — MEDIUM: Verbose Debug Logging Exposes Internal State

Extensive logging in `ai_sd/DiffusionManager.kt` and other modules persists in release builds (ProGuard/R8 is disabled).

**Fix:** Strip debug logs from release builds via ProGuard rules:

```proguard
-assumenosideeffects class android.util.Log {
    public static int d(...);
    public static int v(...);
    public static int i(...);
    public static int w(...);
}
```

---

### FINDING 4 — MEDIUM: Path Traversal Risk in Tar.xz Extraction

**File:** `ai_sd/src/main/java/com/dark/ai_sd/util.kt`

The extraction does not validate that paths stay within the target directory (Zip Slip / Tar Slip vulnerability).

**Fix:** Validate resolved paths stay within target directory:

```kotlin
val canonical = outputFile.canonicalPath
require(canonical.startsWith(targetDir.canonicalPath)) { "Path traversal detected" }
```

---

### FINDING 5 — MEDIUM: All Native JNI Methods Are Public

All JNI bridge classes expose their native methods as public:

- `gguf_lib/.../GGUFNativeLib.kt` — 50+ public `external fun` methods
- `ai_sd/.../SDNativeLib.kt` — all methods public
- `ai_supertonic_tts/.../SupertonicNativeLib.kt` — all methods public

**Fix:** Make JNI wrapper classes `internal` so they cannot be accessed from consuming apps directly.

---

### FINDING 6 — LOW: No String Encryption

All strings are in plaintext including system prompts and error messages.

**Fix:** Use string encryption for sensitive constants.

---

### FINDING 7 — LOW: local.properties Contains SDK Path

If committed to a public repository, reveals the developer's home directory path.

**Fix:** Ensure `local.properties` is in `.gitignore` (verified: it is).

---

### FINDING 8 — LOW: ToolDefinitionBuilder Fields Are Public

**File:** `gguf_lib/.../toolcalling/ToolDefinitionBuilder.kt`

External code can directly modify the builder's internal state.

**Fix:** Make builder properties private. Expose only the builder methods.

---

## Summary Table

| # | Finding | Severity | Category |
|---|---------|----------|----------|
| 1 | Obfuscation disabled in ALL modules | CRITICAL | Reverse Engineering |
| 2 | No tamper detection in non-gguf_lib modules | HIGH | Mod-ability |
| 3 | Verbose debug logging in release builds | MEDIUM | Information Disclosure |
| 4 | Path traversal risk in tar.xz extraction | MEDIUM | Code Injection |
| 5 | All JNI methods are public | MEDIUM | SDK Bypass |
| 6 | No string encryption | LOW | Reverse Engineering |
| 7 | local.properties has developer path | LOW | Info Disclosure |
| 8 | Builder internal fields are public | LOW | State Protection |

---

## Priority Fix Order

### Tier 1 — Must fix before release

1. Enable R8/ProGuard (`isMinifyEnabled = true`) in all release builds
2. Add path traversal protection in `extractTarXzWithCommonsCompress()`

### Tier 2 — Should fix before public release

3. Extend tamper detection from gguf_lib to other modules
4. Strip debug logs via ProGuard rules
5. Make JNI wrapper classes `internal`

### Tier 3 — Hardening

6. Add string encryption for sensitive constants
7. Make `ToolDefinitionBuilder` fields private
