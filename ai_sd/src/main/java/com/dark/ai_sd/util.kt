package com.dark.ai_sd

import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.apache.commons.compress.archivers.tar.TarArchiveEntry
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.tukaani.xz.XZInputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.util.Base64

/**
 * Extension functions and utility helpers for Stable Diffusion operations
 */

/**
 * Calculate optimal buffer size based on available device RAM.
 * Low memory: 8KB, normal: 64KB, high: 256KB.
 */
fun getAdaptiveBufferSize(context: Context): Int {
    val am = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
    val memInfo = ActivityManager.MemoryInfo()
    am?.getMemoryInfo(memInfo)
    val availMb = memInfo.availMem / (1024 * 1024)
    return when {
        memInfo.lowMemory || availMb < 256 -> 8 * 1024       // 8KB — critically low
        availMb < 512                      -> 32 * 1024      // 32KB — low
        availMb < 1024                     -> 64 * 1024      // 64KB — normal
        else                               -> 256 * 1024     // 256KB — plenty
    }
}

/**
 * Copy stream with progress reporting and adaptive buffer size.
 * @param onProgress called with (bytesWritten, totalBytes). totalBytes is -1 if unknown.
 */
fun InputStream.copyToWithProgress(
    out: OutputStream,
    bufferSize: Int,
    totalBytes: Long = -1L,
    onProgress: ((bytesWritten: Long, totalBytes: Long) -> Unit)? = null
): Long {
    val buffer = ByteArray(bufferSize)
    var bytesCopied = 0L
    var bytesRead: Int
    while (read(buffer).also { bytesRead = it } >= 0) {
        out.write(buffer, 0, bytesRead)
        bytesCopied += bytesRead
        onProgress?.invoke(bytesCopied, totalBytes)
    }
    return bytesCopied
}

/**
 * Extract a tar.xz archive with progress reporting and adaptive buffer sizing.
 *
 * @param tarXzFile The tar.xz archive to extract
 * @param targetDir Target directory to extract into
 * @param bufferSize I/O buffer size (use [getAdaptiveBufferSize] for RAM-aware sizing)
 * @param onProgress Called per extracted file with (filesExtracted, currentFileName)
 */
fun extractTarXzWithCommonsCompress(
    tarXzFile: File,
    targetDir: File,
    bufferSize: Int = 64 * 1024,
    onProgress: ((filesExtracted: Int, currentFile: String) -> Unit)? = null
) {
    try {
        var filesExtracted = 0

        tarXzFile.inputStream().buffered(bufferSize).use { inputStream ->
        XZInputStream(inputStream).use { xzIn ->
        TarArchiveInputStream(xzIn).use { tarIn ->

            var entry: TarArchiveEntry?
            var rootDirName: String? = null
            val buf = ByteArray(bufferSize)

            while (tarIn.nextEntry.also { entry = it } != null) {
                val currentEntry = entry ?: continue
                val entryName = currentEntry.name

                // Detect and strip the root directory
                val pathComponents = entryName.split("/")
                if (rootDirName == null && pathComponents.size > 1) {
                    rootDirName = pathComponents[0]
                }

                // Strip root directory if present
                val relativePath = if (rootDirName != null && entryName.startsWith("$rootDirName/")) {
                    entryName.substring(rootDirName.length + 1)
                } else {
                    entryName
                }

                // Skip if it's just the root directory itself or empty path
                if (relativePath.isEmpty()) continue

                val outputFile = File(targetDir, relativePath)

                // Path traversal protection: ensure extracted file stays within target directory
                if (!outputFile.canonicalPath.startsWith(targetDir.canonicalPath + File.separator)) {
                    throw SecurityException("Path traversal detected: ${currentEntry.name}")
                }

                if (currentEntry.isDirectory) {
                    outputFile.mkdirs()
                } else {
                    outputFile.parentFile?.mkdirs()
                    FileOutputStream(outputFile).use { fos ->
                        var bytesRead: Int
                        while (tarIn.read(buf).also { bytesRead = it } >= 0) {
                            fos.write(buf, 0, bytesRead)
                        }
                    }

                    // Set executable permissions if needed
                    if (currentEntry.mode and 0x40 != 0) { // Check execute bit
                        outputFile.setExecutable(true)
                    }

                    filesExtracted++
                    onProgress?.invoke(filesExtracted, relativePath)
                    Log.d("Model-Runtime", "Extracted [$filesExtracted]: $relativePath (${currentEntry.size} bytes)")
                }
            }

        } // tarIn auto-closed
        } // xzIn auto-closed
        } // inputStream auto-closed

        Log.d("Model-Runtime", "Successfully extracted $filesExtracted files using Apache Commons Compress")
    } catch (e: Exception) {
        Log.e("Model-Runtime", "Failed to extract tar.xz with Commons Compress", e)
        throw IOException("Failed to extract tar.xz archive", e)
    }
}


// Bitmap extensions

/**
 * Convert bitmap to raw RGB byte array for img2img operations via JNI
 */
fun Bitmap.toRgbByteArray(): ByteArray {
    val pixels = IntArray(width * height)
    getPixels(pixels, 0, width, 0, 0, width, height)

    val rgbBytes = ByteArray(width * height * 3)
    for (i in pixels.indices) {
        val pixel = pixels[i]
        val index = i * 3
        rgbBytes[index] = ((pixel shr 16) and 0xFF).toByte()     // R
        rgbBytes[index + 1] = ((pixel shr 8) and 0xFF).toByte()  // G
        rgbBytes[index + 2] = (pixel and 0xFF).toByte()          // B
    }

    return rgbBytes
}

/**
 * Convert bitmap to base64 RGB string for img2img operations
 */
fun Bitmap.toBase64Rgb(): String {
    return Base64.getEncoder().encodeToString(toRgbByteArray())
}

/**
 * Save bitmap to file
 */
fun Bitmap.saveToFile(
    file: File, format: Bitmap.CompressFormat = Bitmap.CompressFormat.PNG, quality: Int = 100
): Boolean {
    return try {
        FileOutputStream(file).use { out ->
            compress(format, quality, out)
        }
        true
    } catch (e: IOException) {
        e.printStackTrace()
        false
    }
}

/**
 * Get bitmap size in bytes
 */
fun Bitmap.sizeInBytes(): Int {
    return allocationByteCount
}

// Context extensions

/**
 * Get models directory
 */
fun Context.getModelsDirectory(): File {
    return File(filesDir, "models").apply {
        if (!exists()) mkdirs()
    }
}

/**
 * Get temp directory for temporary files
 */
fun Context.getTempDirectory(): File {
    return File(cacheDir, "temp").apply {
        if (!exists()) mkdirs()
    }
}

/**
 * Clean temp directory
 */
fun Context.cleanTempDirectory() {
    getTempDirectory().deleteRecursively()
    getTempDirectory().mkdirs()
}

// Generation state extensions

/**
 * Check if state represents an active generation
 */
fun DiffusionGenerationState.isActive(): Boolean {
    return this is DiffusionGenerationState.Progress
}

/**
 * Check if state represents completion
 */
fun DiffusionGenerationState.isComplete(): Boolean {
    return this is DiffusionGenerationState.Complete
}

/**
 * Check if state represents an error
 */
fun DiffusionGenerationState.isError(): Boolean {
    return this is DiffusionGenerationState.Error
}

/**
 * Get progress value or 0 if not in progress state
 */
fun DiffusionGenerationState.getProgress(): Float {
    return when (this) {
        is DiffusionGenerationState.Progress -> progress
        else -> 0f
    }
}

// Backend state extensions

/**
 * Check if backend is ready for generation
 */
fun DiffusionBackendState.isReady(): Boolean {
    return this is DiffusionBackendState.Running
}

/**
 * Check if backend has an error
 */
fun DiffusionBackendState.hasError(): Boolean {
    return this is DiffusionBackendState.Error
}

// Model config builder

/**
 * Builder for ModelConfig
 */
class ModelConfigBuilder {
    private var name: String = ""
    private var modelDir: String = ""
    private var textEmbeddingSize: Int = 768
    private var runOnCpu: Boolean = false
    private var useCpuClip: Boolean = false
    private var isPony: Boolean = false
    private var safetyMode: Boolean = false

    fun name(name: String) = apply { this.name = name }
    fun modelDir(dir: String) = apply { this.modelDir = dir }
    fun textEmbeddingSize(size: Int) = apply { this.textEmbeddingSize = size }
    fun runOnCpu(cpu: Boolean) = apply { this.runOnCpu = cpu }
    fun useCpuClip(cpuClip: Boolean) = apply { this.useCpuClip = cpuClip }
    fun isPony(pony: Boolean) = apply { this.isPony = pony }
    fun setSafetyMode(safetyMode: Boolean) = apply { this.safetyMode = safetyMode }

    fun build(): DiffusionModelConfig {
        require(name.isNotEmpty()) { "Model name is required" }
        require(modelDir.isNotEmpty()) { "Model directory is required" }

        return DiffusionModelConfig(
            name = name,
            modelDir = modelDir,
            textEmbeddingSize = textEmbeddingSize,
            runOnCpu = runOnCpu,
            useCpuClip = useCpuClip,
            isPony = isPony,
            safetyMode = safetyMode
        )
    }
}

/**
 * DSL function for creating ModelConfig
 */
fun modelConfig(block: ModelConfigBuilder.() -> Unit): DiffusionModelConfig {
    return ModelConfigBuilder().apply(block).build()
}

// Generation params builder

/**
 * Builder for GenerationParams
 */
class GenerationParamsBuilder {
    private var prompt: String = ""
    private var negativePrompt: String = ""
    private var steps: Int = 28
    private var cfgScale: Float = 7f
    private var seed: Long? = null
    private var width: Int = 512
    private var height: Int = 512
    private var scheduler: String = "dpm"
    private var useOpenCL: Boolean = false
    private var inputImage: String? = null
    private var mask: String? = null
    private var denoiseStrength: Float = 0.6f
    private var showDiffusionProcess: Boolean = false
    private var showDiffusionStride: Int = 1

    fun prompt(prompt: String) = apply { this.prompt = prompt }
    fun negativePrompt(negative: String) = apply { this.negativePrompt = negative }
    fun steps(steps: Int) = apply { this.steps = steps }
    fun cfgScale(cfg: Float) = apply { this.cfgScale = cfg }
    fun seed(seed: Long?) = apply { this.seed = seed }
    fun resolution(width: Int, height: Int) = apply {
        this.width = width
        this.height = height
    }

    fun scheduler(scheduler: String) = apply { this.scheduler = scheduler }
    fun useOpenCL(use: Boolean) = apply { this.useOpenCL = use }
    fun inputImage(image: String?) = apply { this.inputImage = image }
    fun mask(mask: String?) = apply { this.mask = mask }
    fun denoiseStrength(strength: Float) = apply { this.denoiseStrength = strength }
    fun showProcess(show: Boolean, stride: Int = 1) = apply {
        this.showDiffusionProcess = show
        this.showDiffusionStride = stride
    }

    fun build(): DiffusionGenerationParams {
        require(prompt.isNotEmpty()) { "Prompt is required" }

        return DiffusionGenerationParams(
            prompt = prompt,
            negativePrompt = negativePrompt,
            steps = steps,
            cfgScale = cfgScale,
            seed = seed,
            width = width,
            height = height,
            scheduler = scheduler,
            useOpenCL = useOpenCL,
            inputImage = inputImage,
            mask = mask,
            denoiseStrength = denoiseStrength,
            showDiffusionProcess = showDiffusionProcess,
            showDiffusionStride = showDiffusionStride
        )
    }
}

/**
 * DSL function for creating GenerationParams
 */
fun generationParams(block: GenerationParamsBuilder.() -> Unit): DiffusionGenerationParams {
    return GenerationParamsBuilder().apply(block).build()
}

// Common schedulers
object Schedulers {
    const val DPM = "dpm"
    const val EULER = "euler"
    const val EULER_A = "euler_a"
    const val LCM = "lcm"
}

// Common resolutions
object Resolutions {
    val SD15_512 = 512 to 512
    val SD15_768 = 768 to 768
    val LANDSCAPE_768x512 = 768 to 512
    val PORTRAIT_512x768 = 512 to 768
}