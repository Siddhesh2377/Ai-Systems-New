package com.dark.demon_system.data

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File

data class ModelSpec(
    val id: String,
    val displayName: String,
    val url: String,
    val approxBytes: Long
) {
    companion object {
        val ABSOLUTE_REALITY_QNN_MIN = ModelSpec(
            id = "AbsoluteReality_qnn2.28_min",
            displayName = "AbsoluteReality (QNN, min variant)",
            url = "https://huggingface.co/xororz/sd-qnn/resolve/main/AbsoluteReality_qnn2.28_min.zip?download=true",
            approxBytes = 993_451_663L
        )
    }
}

sealed class ModelInstallState {
    object Idle : ModelInstallState()
    data class Downloading(
        val bytesDownloaded: Long,
        val totalBytes: Long,
        val bytesPerSecond: Long
    ) : ModelInstallState()
    data class Extracting(
        val filesExtracted: Int,
        val currentFile: String,
        val uncompressedBytesSoFar: Long
    ) : ModelInstallState()
    object Finalizing : ModelInstallState()
    data class Installed(val modelDir: File, val files: List<String>) : ModelInstallState()
    data class Error(val message: String) : ModelInstallState()
    object Cancelled : ModelInstallState()
}

/**
 * Coordinates download, extraction, and finalization of a single model into
 * `context.filesDir/models/<id>/`. A `.installed` marker file inside the
 * extracted directory marks a successful install — partial extractions are
 * deleted on the next attempt.
 */
class ModelInstaller(private val context: Context) {

    companion object {
        private const val TAG = "ModelInstaller"
        private const val MARKER = ".installed"
    }

    private val downloader = ModelDownloader()
    private val extractor = ZipExtractor()

    private val _state = MutableStateFlow<ModelInstallState>(ModelInstallState.Idle)
    val state: StateFlow<ModelInstallState> = _state.asStateFlow()

    fun modelDir(spec: ModelSpec): File =
        File(File(context.filesDir, "models"), spec.id)

    fun isInstalled(spec: ModelSpec): Boolean {
        val dir = modelDir(spec)
        return dir.isDirectory && File(dir, MARKER).exists()
    }

    /**
     * Returns the current partial-download progress (0..1) for the given spec,
     * or null if no partial exists. Useful for resuming the UI after process
     * death.
     */
    fun partialDownloadProgress(spec: ModelSpec): Float? {
        val part = File(context.cacheDir, "${spec.id}.zip.part")
        if (!part.exists() || part.length() == 0L) return null
        return (part.length().toFloat() / spec.approxBytes).coerceIn(0f, 1f)
    }

    suspend fun install(spec: ModelSpec) {
        try {
            if (isInstalled(spec)) {
                // Idempotent: repairs layout for installs created before normalization existed.
                normalizeXororzLayout(modelDir(spec))
                _state.value = ModelInstallState.Installed(
                    modelDir(spec),
                    modelDir(spec).listFiles()?.map { it.name }.orEmpty()
                )
                return
            }

            // Clear any half-extracted dir.
            val targetDir = modelDir(spec)
            if (targetDir.exists()) targetDir.deleteRecursively()

            val zipFile = File(context.cacheDir, "${spec.id}.zip")

            _state.value = ModelInstallState.Downloading(
                bytesDownloaded = File(context.cacheDir, "${spec.id}.zip.part").let {
                    if (it.exists()) it.length() else 0L
                },
                totalBytes = spec.approxBytes,
                bytesPerSecond = 0L
            )

            downloader.download(spec.url, zipFile) { p ->
                _state.value = ModelInstallState.Downloading(
                    bytesDownloaded = p.bytesDownloaded,
                    totalBytes = if (p.totalBytes > 0) p.totalBytes else spec.approxBytes,
                    bytesPerSecond = p.bytesPerSecond
                )
            }

            _state.value = ModelInstallState.Extracting(0, "", 0L)
            extractor.extract(zipFile, targetDir) { p ->
                _state.value = ModelInstallState.Extracting(
                    filesExtracted = p.filesExtracted,
                    currentFile = p.currentFile,
                    uncompressedBytesSoFar = p.uncompressedBytesSoFar
                )
            }

            _state.value = ModelInstallState.Finalizing
            normalizeXororzLayout(targetDir)
            File(targetDir, MARKER).writeText(spec.id)
            // Free ~1 GB of cache once we've successfully extracted.
            zipFile.delete()

            _state.value = ModelInstallState.Installed(
                targetDir,
                targetDir.listFiles()?.map { it.name }.orEmpty()
            )
            Log.i(TAG, "Installed ${spec.id} -> ${targetDir.absolutePath}")
        } catch (e: CancellationException) {
            _state.value = ModelInstallState.Cancelled
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Install failed for ${spec.id}", e)
            _state.value = ModelInstallState.Error(e.message ?: "Install failed")
        }
    }

    /**
     * Deletes the installed model directory and any partial download.
     */
    fun uninstall(spec: ModelSpec) {
        modelDir(spec).deleteRecursively()
        File(context.cacheDir, "${spec.id}.zip").delete()
        File(context.cacheDir, "${spec.id}.zip.part").delete()
        _state.value = ModelInstallState.Idle
    }

    fun resetState() {
        if (_state.value is ModelInstallState.Error || _state.value is ModelInstallState.Cancelled) {
            _state.value = ModelInstallState.Idle
        }
    }

    /**
     * xororz QNN bundles wrap files inside a `qnn_models_<variant>/` directory and
     * use `clip_v2.mnn` instead of the canonical `clip.mnn` that ai_sd expects.
     * Flatten the layout and rename the CLIP file in-place so DiffusionManager
     * can find everything by its expected name.
     */
    private fun normalizeXororzLayout(modelDir: File) {
        // Flatten: if modelDir has exactly one entry and it's a directory, hoist
        // its contents up. Tolerates extra hidden files (.DS_Store etc.).
        val entries = modelDir.listFiles().orEmpty()
        val files = entries.filter { it.isFile && !it.name.startsWith(".") }
        val dirs = entries.filter { it.isDirectory }
        if (files.isEmpty() && dirs.size == 1) {
            val inner = dirs[0]
            inner.listFiles()?.forEach { f ->
                val dest = File(modelDir, f.name)
                if (dest.exists()) dest.deleteRecursively()
                if (!f.renameTo(dest)) {
                    f.copyRecursively(dest, overwrite = true)
                    f.deleteRecursively()
                }
            }
            inner.delete()
            Log.i(TAG, "Flattened single-subdir model layout (was ${inner.name}/)")
        }

        // xororz QNN bundles ship CLIP as the v2 variant (1x77x768 input_embedding
        // input rather than 1x77 input_ids). ai_sd's model_loader auto-detects it
        // by looking for clip_v2.mnn next to clip.mnn and switching input wiring.
        // Earlier revisions of this method incorrectly renamed clip_v2.mnn ->
        // clip.mnn, which broke that detection and caused a SIGSEGV in
        // MNN::Interpreter::resizeTensor (input name not found). Repair that
        // here by renaming back: any standalone clip.mnn in this dir is the v2
        // model that should be named clip_v2.mnn.
        val clipMnn = File(modelDir, "clip.mnn")
        val clipV2 = File(modelDir, "clip_v2.mnn")
        if (clipMnn.exists() && !clipV2.exists()) {
            if (clipMnn.renameTo(clipV2)) {
                Log.i(TAG, "Renamed clip.mnn -> clip_v2.mnn so ai_sd auto-detects v2 CLIP")
            }
        }
    }
}
