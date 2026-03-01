# AiSystems SDK Security Audit Report

**Date:** 2026-02-28
**Total Findings:** 13 (3 Critical, 4 High, 3 Medium, 3 Low)
**Modules Audited:** `ai_gguf`, `ai_sd`, `ai_supertonic_tts`, `backend_manager`, `backend_plugin_api`, `app`

## Executive Summary

The AiSystems SDK is currently an open SDK with no monetization gating, no licensing, and no tamper protection. All features are fully accessible to any consumer without restriction. Any premium gating added on the consuming app (ToolNeuron) side will be trivially bypassable because the SDK itself has zero access controls. The SDK is also fully unobfuscated, making reverse engineering trivial.

---

## CATEGORY 1: PREMIUM/MONETIZATION BYPASS

### FINDING 1 — CRITICAL: Obfuscation Completely Disabled Across ALL Modules

Every single module has `isMinifyEnabled = false` in its release build type. The published AAR/APK contains fully readable class names, method names, and string literals with zero obfuscation.

**Affected files:**

- `app/build.gradle.kts` line 25: `isMinifyEnabled = false`
- `ai_gguf/build.gradle.kts` line 60: `isMinifyEnabled = false`
- `ai_sd/build.gradle.kts` line 41: `isMinifyEnabled = false`
- `ai_supertonic_tts/build.gradle.kts` line 31: `isMinifyEnabled = false`
- `backend_manager/build.gradle.kts` line 26: `isMinifyEnabled = false`
- `backend_plugin_api/build.gradle.kts` line 20: `isMinifyEnabled = false`

Additionally, `ai_gguf/proguard-rules.pro` explicitly keeps ALL classes:

```proguard
-keep class com.mp.ai_gguf.** { *; }
```

**Exploitation:** `jadx` or `apktool` reveals the entire SDK source with original names. Every internal API, callback, and state management pattern is fully visible.

**Fix:** Set `isMinifyEnabled = true` in all release builds. Write proper keep rules only for JNI and serialization classes.

---

### FINDING 2 — CRITICAL: No Licensing or Premium Gating in the SDK

The SDK has zero licensing, feature gating, or premium checks anywhere:

- `GGUFNativeLib` exposes all native methods publicly with no access control
- `SupertonicTTS` is fully functional without any license check
- `DiffusionManager` / `StableDiffusionManager` have no premium verification
- `BackendPluginManager` loads and runs any backend plugin without license validation
- No SharedPreferences, license key fields, or server-side validation calls exist

**Impact:** If ToolNeuron implements premium checks, an attacker can:

1. Decompile ToolNeuron (trivial since no obfuscation)
2. Patch the premium check boolean
3. Repackage the APK

Or simpler: build a new app that imports the SDK libraries and calls them directly.

**Fix:** Design SDK-level licensing. Consider moving critical logic to native code (harder to patch than Kotlin bytecode). Add license key validation that gates `loadModel()` calls.

---

### FINDING 3 — CRITICAL: Plugin Loader Has No Signature or Integrity Verification

**File:** `backend_manager/src/main/java/com/dark/backend_manager/loader/PluginLoader.kt`, lines 32-72

```kotlin
fun load(manifest: BackendManifest, installPath: String): Result<BackendPlugin> {
    return runCatching {
        require(manifest.apiVersion == PluginManager.API_VERSION) { ... }

        val pluginDir = File(installPath)
        require(pluginDir.exists()) { ... }

        // Loads native libraries from disk WITHOUT signature verification
        loadNativeLibs(manifest.nativeLibs, pluginDir)

        // Loads DEX file WITHOUT any hash/signature check
        val dexPath = File(pluginDir, "classes.dex")
        val classLoader = DexClassLoader(
            dexPath.absolutePath, ...)

        // Instantiates arbitrary class via reflection
        val pluginClass = classLoader.loadClass(manifest.entryClass)
        val plugin = pluginClass.getDeclaredConstructor().newInstance()
        ...
    }
}
```

The `nativeLibs` loading at lines 88-109 calls `System.load()` on any file found on disk:

```kotlin
private fun loadNativeLibs(libs: List<String>, pluginDir: File) {
    // ...
    System.load(libFile.absolutePath) // Loads any .so file from disk
}
```

**Exploitation:** An attacker with file system access can:

1. Replace `classes.dex` in the plugins directory with a malicious DEX
2. Replace any `.so` native library with a trojaned version
3. Modify `manifest.json` to point `entryClass` to their malicious class
4. The `PluginLoader` loads and executes malicious code within the app's process

**Fix:** Sign plugin packages with a private key. Verify the signature in `PluginLoader` before loading any DEX or SO file. Use a checksum allowlist for known-good plugins.

---

## CATEGORY 2: MOD-ABILITY / TAMPERING

### FINDING 4 — HIGH: Manifest Parser Trusts Arbitrary JSON Without Validation

**File:** `backend_manager/src/main/java/com/dark/backend_manager/registry/ManifestParser.kt`, lines 13-43

```kotlin
fun parse(pluginDir: File): Result<BackendManifest> = runCatching {
    val manifestFile = File(pluginDir, "manifest.json")
    val json = JSONObject(manifestFile.readText())

    BackendManifest(
        id = json.getString("id"),
        entryClass = json.getString("entryClass"), // Attacker controls this
        nativeLibs = ...,  // Attacker controls which .so files to load
        ...
    )
}
```

**Exploitation:** An attacker can craft a `manifest.json` that points to any class or library, controlling what code gets loaded.

**Fix:** Validate manifest fields against an allowlist. Combine with plugin signature verification from Finding 3.

---

### FINDING 5 — HIGH: Application Backup Enabled

**File:** `app/src/main/AndroidManifest.xml`, line 7

```xml
android:allowBackup="true"
```

**Exploitation:** On Android 12 or lower, `adb backup` extracts the entire app data directory including plugin directories with DEX and native libraries, cached model data, and any future SharedPreferences with premium/license state. An attacker can modify the backup and restore a tampered state.

**Fix:** Set `android:allowBackup="false"`.

---

### FINDING 6 — HIGH: No Tamper Detection or APK Signature Verification

The entire SDK has zero tamper detection:

- No APK signature verification at runtime
- No checksum validation on SDK `.so` libraries
- No detection of Frida, Xposed, or other hooking frameworks
- No detection of rooted/modified devices
- No detection of debugger attachment
- No runtime integrity checks on loaded code

**Impact:** The app can be freely decompiled with `apktool`, modified, and repackaged with a different signing key. The SDK functions identically.

**Fix:** Add runtime APK signature verification. Detect hooking frameworks. Implement integrity checks on native libraries.

---

### FINDING 7 — HIGH: Plugin System Accessible via Public API Without Access Control

**File:** `backend_manager/src/main/java/com/dark/backend_manager/BackendPluginManager.kt`

```kotlin
// Anyone can register arbitrary backends
override fun registerInstalled(manifest: BackendManifest, installPath: String) {
    registry.register(manifest, installPath)
}

// Anyone can request and use any backend
override suspend fun requestBackend(model: ModelMetadata): Result<BackendPlugin>
```

**Exploitation:** Any consuming app can register arbitrary backend plugins from any directory, load and use all capabilities (text gen, image gen, TTS, embeddings) without restriction. There is no concept of "authorized consumer."

**Fix:** Add an authorization layer. Require API key or signature verification before allowing backend registration and usage.

---

## CATEGORY 3: INFORMATION DISCLOSURE / REVERSE ENGINEERING

### FINDING 8 — MEDIUM: Verbose Debug Logging Exposes Internal State

The `backend_manager` module logs detailed internal state that persists in release builds (ProGuard/R8 is disabled):

```kotlin
// BackendPluginManager.kt
Log.i(TAG, "Initialized with ${registry.entries.value.size} installed backends")
Log.i(TAG, "Backend '$backendId' ready with model '${model.name}'")

// PluginLoader.kt
Log.d(TAG, "Loaded native lib: ${libFile.absolutePath}")
Log.i(TAG, "Loaded backend: ${manifest.id} v${manifest.version} [${manifest.capabilities.joinToString()}]")
```

Also extensive logging in `ai_sd/DiffusionManager.kt` (lines 68, 90, 105, 112, 155, 178, etc.)

**Exploitation:** `adb logcat` reveals exact plugin paths, native library locations, backend IDs and versions, model names and capabilities, and file paths used for model loading.

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

### FINDING 9 — MEDIUM: Path Traversal Risk in Tar.xz Extraction

**File:** `ai_sd/src/main/java/com/dark/ai_sd/util.kt`, lines 19-73

```kotlin
fun extractTarXzWithCommonsCompress(tarXzFile: File, targetDir: File) {
    while (tarIn.nextEntry.also { entry = it } != null) {
        val entryName = entry!!.name
        // Only strips root directory name, no path traversal check
        val outputFile = File(targetDir, relativePath)
        outputFile.outputStream().use { output ->
            tarIn.copyTo(output)
        }
    }
}
```

The extraction does not validate that `relativePath` stays within `targetDir`. A maliciously crafted tar.xz with entries like `../../data/data/com.app/files/evil.so` could write files outside the intended directory (Zip Slip / Tar Slip vulnerability).

**Fix:** Validate resolved paths stay within target directory:

```kotlin
val canonical = outputFile.canonicalPath
require(canonical.startsWith(targetDir.canonicalPath)) { "Path traversal detected" }
```

---

### FINDING 10 — MEDIUM: All Native JNI Methods Are Public

All JNI bridge classes expose their native methods as public:

- `ai_gguf/src/main/java/com/mp/ai_gguf/GGUFNativeLib.kt` — 50+ public `external fun` methods
- `ai_sd/src/main/java/com/dark/ai_sd/SDNativeLib.kt` — all methods public
- `ai_supertonic_tts/src/main/java/com/mp/ai_supertonic_tts/SupertonicNativeLib.kt` — all methods public

**Exploitation:** Any app that loads these native libraries can call any JNI method directly, bypassing any Kotlin-level wrapper logic. Even if you add a license check in `SupertonicTTS.synthesize()`, an attacker can call `SupertonicNativeLib` directly.

**Fix:** Make JNI wrapper classes `internal` (`internal class GGUFNativeLib`) so they cannot be accessed from consuming apps directly — only through your public API layer.

---

### FINDING 11 — LOW: No String Encryption

All strings are in plaintext, including:

- TAG constants for logging
- System prompts (`LUNA_SYSTEM` and `LUNA_PERSONALITY` in `ChatViewModel.kt`)
- Error messages that reveal internal architecture
- File paths and directory names

**File:** `app/src/main/java/com/dark/gguf_android/ui/chat/ChatViewModel.kt`, lines 62-81

```kotlin
private const val LUNA_SYSTEM = "You are Luna, a 26-year-old woman who works as a creative ..."
```

These strings are trivially extractable from the APK using `strings` or any decompiler.

**Fix:** Use string encryption for sensitive constants. Consider DexGuard or manual encryption with runtime decryption.

---

### FINDING 12 — LOW: local.properties Contains SDK Path

**File:** `local.properties`, line 10

```
sdk.dir=/home/home/Android/Sdk
```

If committed to a public repository, reveals the developer's home directory path.

**Fix:** Ensure `local.properties` is in `.gitignore` (it should already be).

---

### FINDING 13 — LOW: ToolDefinitionBuilder Fields Are Public

**File:** `ai_gguf/src/main/java/com/mp/ai_gguf/toolcalling/ToolDefinition.kt`, lines 133-138

```kotlin
class ToolDefinitionBuilder(
     val name: String,
     val description: String
) {
     val parameters = mutableMapOf<String, ToolParameter>()
     val required = mutableListOf<String>()
```

External code can directly modify the builder's internal state.

**Fix:** Make builder properties private. Expose only the builder methods.

---

## Summary Table

| # | Finding | Severity | Category |
|---|---------|----------|----------|
| 1 | Obfuscation disabled in ALL modules | CRITICAL | Reverse Engineering |
| 2 | No licensing/premium gating in SDK | CRITICAL | Monetization Bypass |
| 3 | Plugin loader has no signature/integrity verification | CRITICAL | Code Injection |
| 4 | Manifest parser trusts arbitrary JSON | HIGH | Tampering |
| 5 | Application backup enabled | HIGH | Data Extraction |
| 6 | No tamper detection / APK signature verification | HIGH | Mod-ability |
| 7 | Plugin system fully public with no access control | HIGH | SDK Bypass |
| 8 | Verbose debug logging in release builds | MEDIUM | Information Disclosure |
| 9 | Path traversal risk in tar.xz extraction | MEDIUM | Code Injection |
| 10 | All JNI methods are public | MEDIUM | SDK Bypass |
| 11 | No string encryption | LOW | Reverse Engineering |
| 12 | local.properties has developer path | LOW | Info Disclosure |
| 13 | Builder internal fields are public | LOW | State Protection |

---

## Priority Fix Order

### Tier 1 — Must fix before any release (blocks all monetization)

1. Enable R8/ProGuard (`isMinifyEnabled = true`) in all release builds. Write proper keep rules only for JNI and serialization classes.
2. Add plugin signature verification in `PluginLoader` — sign plugin packages with a private key and verify before loading DEX/SO files.
3. Add path traversal protection in `extractTarXzWithCommonsCompress()`.

### Tier 2 — Should fix before public release

4. Add runtime tamper detection — verify APK signing certificate at startup, detect Frida/Xposed.
5. Set `android:allowBackup="false"` in the app manifest.
6. Strip debug logs from release builds via ProGuard rules.
7. Make JNI wrapper classes `internal` so they can't be accessed directly from consuming apps.
8. Add authorization layer to the plugin system.

### Tier 3 — Hardening

9. Design SDK-level licensing if selling the SDK itself.
10. Move critical logic to native code (harder to patch than Kotlin bytecode).
11. Add string encryption for sensitive constants.
12. Make `ToolDefinitionBuilder` fields private.
