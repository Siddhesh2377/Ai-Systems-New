package com.dark.ai_sd

import android.app.ActivityManager
import android.content.Context
import org.json.JSONObject
import java.io.File

/**
 * Estimates memory + runtime cost of an SD generation before launching it, so
 * a host app can warn the user, downscale, or pick a different model variant
 * instead of running into an OOM mid-generation.
 *
 * Why this exists: [DiffusionManager.loadModel] makes an unconditional call
 * into JNI and only fails after MNN/QNN have already opened files and parsed
 * graphs (which can be hundreds of MB). A pre-flight check is much cheaper.
 *
 * The numbers below come from measured headroom on Snapdragon 7s Gen 3 with
 * the xororz QNN bundles — they are intentionally conservative. Real peak
 * usage during generation is dominated by:
 *   - UNet weights (the bulk; 700-900 MB for SD1.5 QNN, 4-6 GB for SDXL)
 *   - VAE decoder weights (60-100 MB)
 *   - CLIP weights (50-200 MB)
 *   - latent + intermediate buffers (a few MB total)
 *   - QNN HTP shared memory + libs (~30-50 MB)
 */
class SDContextCalculator(private val context: Context) {

    /** Aggregated estimate result. */
    data class Estimate(
        val components: List<Component>,
        val totalMb: Long,
        val availableMb: Long,
        val totalRamMb: Long,
        val safetyMarginMb: Long,
        val canRun: Boolean,
        val warnings: List<String>,
        val estimatedSecondsPerStep: Float,
        val estimatedTotalSeconds: Float
    ) {
        fun toReadableSummary(): String = buildString {
            appendLine("=== SD Context Estimate ===")
            appendLine("Available RAM: $availableMb / $totalRamMb MB  (safety margin $safetyMarginMb MB)")
            appendLine("Estimated peak: $totalMb MB   ${if (canRun) "OK" else "WILL OOM"}")
            appendLine("Estimated time: ${"%.1f".format(estimatedTotalSeconds)} s "
                + "(${"%.2f".format(estimatedSecondsPerStep)} s/step)")
            if (warnings.isNotEmpty()) {
                appendLine("Warnings:")
                warnings.forEach { appendLine("  - $it") }
            }
            appendLine("Breakdown:")
            for (c in components) {
                appendLine("  ${c.name.padEnd(20)} ${c.estimateMb.toString().padStart(6)} MB  (${c.source})")
            }
        }
    }

    data class Component(val name: String, val estimateMb: Long, val source: String)

    /** Configuration to estimate for. */
    data class Request(
        val modelDir: String,
        val width: Int = 512,
        val height: Int = 512,
        val steps: Int = 20,
        val cfgScale: Float = 7f,
        val runOnCpu: Boolean = false,
        val useCpuClip: Boolean = false,
        val useSafetyChecker: Boolean = false,
        val useUpscaler: Boolean = false,
        /// QNN HTP version reported by [SDNativeLib.nativeGetSocInfo]. 0 = unknown.
        val htpVersion: Int = 0
    )

    /**
     * Run the estimate. Pure data — no JNI calls, no model load.
     */
    fun estimate(req: Request): Estimate {
        val components = mutableListOf<Component>()
        val warnings = mutableListOf<String>()

        // --- Weights on disk ---
        val dir = File(req.modelDir)
        val clipFile = pickFirst(
            File(dir, if (req.runOnCpu || req.useCpuClip) "clip.mnn" else "clip.bin"),
            File(dir, "clip_v2.mnn"),
            File(dir, "clip.mnn")
        )
        val unetFile = File(dir, if (req.runOnCpu) "unet.mnn" else "unet.bin")
        val vaeDecFile = File(dir, if (req.runOnCpu) "vae_decoder.mnn" else "vae_decoder.bin")
        val vaeEncFile = File(dir, if (req.runOnCpu) "vae_encoder.mnn" else "vae_encoder.bin")
        val tokenizerFile = File(dir, "tokenizer.json")

        if (clipFile != null && clipFile.exists()) {
            components += Component("CLIP weights", mb(clipFile.length()), clipFile.name)
        } else {
            warnings += "CLIP weights not found in $dir"
        }
        if (unetFile.exists()) {
            components += Component("UNet weights", mb(unetFile.length()), unetFile.name)
        } else {
            warnings += "UNet weights missing: ${unetFile.name}"
        }
        if (vaeDecFile.exists()) {
            components += Component("VAE decoder", mb(vaeDecFile.length()), vaeDecFile.name)
        } else {
            warnings += "VAE decoder missing"
        }
        if (vaeEncFile.exists()) {
            components += Component("VAE encoder", mb(vaeEncFile.length()), vaeEncFile.name)
        }
        if (tokenizerFile.exists()) {
            components += Component("tokenizer", mb(tokenizerFile.length()), tokenizerFile.name)
        }

        // QNN runtime overhead (libs + HTP context + shared mem). Fixed-ish.
        if (!req.runOnCpu) {
            components += Component("QNN runtime", 50, "estimate")
        }

        // Latent buffers (batch=2 by default; 4 channels; latent = 1/8 pixel).
        val latentMb = run {
            val sampleW = req.width / 8
            val sampleH = req.height / 8
            val perTensor = 4L * sampleH * sampleW * 4L  // 4 channels × float
            // Pipeline allocates: latents_in, latents_out, noise_pred, scratch, etc.
            // Roughly 6× per-tensor across batched buffers.
            mb(2 * 6 * perTensor)
        }
        components += Component("Latent + scratch", latentMb, "computed")

        // Text embedding buffer: 2 × 77 × 768 × 4B = ~470 KB. Tiny.
        components += Component("Text embeddings", 1, "computed")

        // Output RGB image buffers (intermediate previews + final).
        val imageBufMb = mb(req.width.toLong() * req.height * 3L * 6L) // 6 buffers max
        components += Component("Image buffers", imageBufMb, "computed")

        if (req.useSafetyChecker) {
            components += Component("Safety checker", 60, "estimate")
        }

        if (req.useUpscaler) {
            // Real-ESRGAN x4plus ~65 MB + tile scratch.
            val outDim = (req.width * 4).coerceAtMost(2048)
            components += Component("Upscaler", 65 + mb(outDim.toLong() * outDim * 3L), "estimate")
        }

        val totalMb = components.sumOf { it.estimateMb }

        // --- Available RAM ---
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val availMb = mi.availMem / 1_048_576L
        val totalRamMb = mi.totalMem / 1_048_576L

        // 256 MB matches the backend_manager safety margin used elsewhere in
        // this monorepo. Below that the OOM killer becomes very aggressive.
        val safetyMb = 256L
        val canRun = availMb - totalMb >= safetyMb
        if (!canRun) {
            warnings += "Estimated peak ($totalMb MB) exceeds safe budget " +
                "(avail=$availMb, margin=$safetyMb MB). Consider a smaller resolution, " +
                "the QNN \"min\" variant, or fewer concurrent buffers."
        }
        if (mi.lowMemory) {
            warnings += "Device is in low-memory state right now."
        }

        // --- Time estimate (rough) ---
        val secPerStep = stepTime(req)
        // CLIP + VAE add ~0.5–2 s extra; CFG=1 halves UNet time on QNN.
        val pipelineOverheadSec = if (req.runOnCpu) 5f else 1.5f
        val totalSec = secPerStep * req.steps + pipelineOverheadSec

        return Estimate(
            components = components,
            totalMb = totalMb,
            availableMb = availMb,
            totalRamMb = totalRamMb,
            safetyMarginMb = safetyMb,
            canRun = canRun,
            warnings = warnings,
            estimatedSecondsPerStep = secPerStep,
            estimatedTotalSeconds = totalSec
        )
    }

    /**
     * Suggest a config that should fit on this device. Picks resolution +
     * step count + cpu/qnn mode based on available RAM and HTP version. The
     * returned Request is a starting point — not a final config.
     */
    fun recommend(modelDir: String, htpVersion: Int = 0): Request {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val availMb = mi.availMem / 1_048_576L

        // QNN model needs ~1.2 GB for SD1.5; CPU mode needs ~2 GB+.
        val useQnn = availMb > 1500 && htpVersion >= 68
        val res = when {
            availMb > 3500 -> 768 to 768
            availMb > 2200 -> 640 to 640
            else -> 512 to 512
        }
        val steps = if (useQnn) 20 else 12
        return Request(
            modelDir = modelDir,
            width = res.first,
            height = res.second,
            steps = steps,
            cfgScale = if (useQnn) 7f else 1f,  // CPU benefits from cfg=1 LCM-style
            runOnCpu = !useQnn,
            useCpuClip = useQnn,  // xororz QNN bundles ship CLIP as MNN
            htpVersion = htpVersion
        )
    }

    /**
     * Per-step UNet time in seconds. Calibrated against measured numbers on
     * Snapdragon 7s Gen 3 (HTP V73): ~1.1 s/step at 512x512 in QNN, ~6 s/step
     * on CPU MNN. Other SoCs scale proportionally to HTP version.
     */
    private fun stepTime(req: Request): Float {
        val pixelScale = (req.width * req.height) / (512f * 512f)

        // CPU MNN: dominated by core count + cache. Benchmark anchor: 6 s/step
        // at 512×512 on 4× A78 + 4× A55.
        val cpuStep = 6f * pixelScale

        if (req.runOnCpu) return cpuStep

        // QNN HTP: scales with version. V73 ~ 1.1 s, V69 ~ 1.5 s, V68 ~ 1.8 s.
        // Newer versions (V75/V79/V81) are roughly 0.7-0.9 s.
        val qnnStepBase = when {
            req.htpVersion >= 79 -> 0.75f
            req.htpVersion >= 75 -> 0.9f
            req.htpVersion >= 73 -> 1.1f
            req.htpVersion >= 69 -> 1.5f
            else -> 1.8f
        } * pixelScale

        // CFG=1 fast path (after the cfg=1 skip-uncond change in unet_runner)
        // halves QNN per-step time because only one of two passes runs.
        val cfgFactor = if (kotlin.math.abs(req.cfgScale - 1f) < 1e-3f) 0.55f else 1.0f
        return qnnStepBase * cfgFactor
    }

    private fun mb(bytes: Long): Long = (bytes + 1_048_575L) / 1_048_576L

    private fun pickFirst(vararg candidates: File): File? =
        candidates.firstOrNull { it.exists() }

    /**
     * Read the SoC info JSON from native and extract the HTP version. Returns
     * 0 if anything fails.
     */
    fun probeHtpVersion(): Int = runCatching {
        val sd = StableDiffusionManager.getInstance(context)
        val json = JSONObject(sd.getSocInfo())
        json.optInt("htp_version", 0)
    }.getOrDefault(0)
}
