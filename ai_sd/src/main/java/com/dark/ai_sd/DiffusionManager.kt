package com.dark.ai_sd

import android.annotation.SuppressLint
import android.content.Context
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File
import java.io.FileNotFoundException
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Main manager class for Stable Diffusion backend operations
 * Handles model loading, process management, and resource cleanup
 */
class DiffusionManager(private val context: Context) {

    companion object {
        private const val TAG = "DiffusionManager"
        private const val RUNTIME_DIR = "runtime_libs"
        private const val EXECUTABLE_NAME = "libstable_diffusion_core.so"
        private const val SAFETY_CHECKER_FILE = "safety_checker/safety_checker.mnn"

        @SuppressLint("StaticFieldLeak")
        @Volatile
        private var instance: DiffusionManager? = null

        fun getInstance(context: Context): DiffusionManager {
            return instance ?: synchronized(this) {
                instance ?: DiffusionManager(context.applicationContext).also { instance = it }
            }
        }
    }

    private val safetyCheckerFile = File(context.filesDir, SAFETY_CHECKER_FILE)

    // State management
    private val _Diffusion_backendState = MutableStateFlow<DiffusionBackendState>(DiffusionBackendState.Idle)
    val diffusionBackendState: StateFlow<DiffusionBackendState> = _Diffusion_backendState.asStateFlow()

    // Runtime configuration
    private lateinit var runtimeDir: File
    private var process: Process? = null
    private var currentModel: DiffusionModelConfig? = null
    private var monitorThread: Thread? = null

    // Runtime setup
    private var isRuntimePrepared = false

    /**
     * Initialize the runtime environment
     * Must be called before loading models
     */
    fun setupRuntime(config: DiffusionRuntimeConfig = DiffusionRuntimeConfig(RUNTIME_DIR)) {
        if (isRuntimePrepared) {
            Log.i(TAG, "Runtime already prepared")
            return
        }

        try {
            prepareRuntimeDirectory(config)

            if (config.safetyCheckerEnabled) {
                prepareSafetyChecker()
            }

            isRuntimePrepared = true
            Log.i(TAG, "Runtime setup completed successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Runtime setup failed", e)
            updateState(DiffusionBackendState.Error("Runtime setup failed: ${e.message}"))
            throw RuntimeException("Failed to setup runtime environment", e)
        }
    }

    /**
     * Load a model and start the backend server
     */
    fun loadModel(diffusionModelConfig: DiffusionModelConfig, width: Int = 512, height: Int = 512): Boolean {
        if (!isRuntimePrepared) {
            Log.e(TAG, "Runtime not prepared. Call setupRuntime() first")
            updateState(DiffusionBackendState.Error("Runtime not prepared"))
            return false
        }

        // Stop existing backend if running
        if (_Diffusion_backendState.value is DiffusionBackendState.Running) {
            Log.i(TAG, "Stopping existing backend before loading new model")
            stopBackend()
        }

        return startBackend(diffusionModelConfig, width, height)
    }

    /**
     * Restart the backend with the same model
     */
    fun restartBackend(): Boolean {
        if (currentModel == null) {
            Log.e(TAG, "Cannot restart: no model loaded")
            return false
        }
        Log.i(TAG, "Restarting backend with model: ${currentModel!!.name}")
        stopBackend()
        return startBackend(currentModel!!, 512, 512)
    }

    /**
     * Stop the backend server
     */
    fun stopBackend() {
        Log.i(TAG, "Stopping backend")

        process?.let { proc ->
            try {
                // Try graceful shutdown first
                proc.destroy()

                // Force kill if not stopped within timeout
                if (!proc.waitFor(5, TimeUnit.SECONDS)) {
                    Log.w(TAG, "Graceful shutdown timeout, forcing termination")
                    proc.destroyForcibly()
                }

                val exitCode = proc.exitValue()
                Log.i(TAG, "Backend process stopped with exit code: $exitCode")

                updateState(DiffusionBackendState.Idle)
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping backend", e)
                updateState(DiffusionBackendState.Error("Error stopping backend: ${e.message}"))
            } finally {
                process = null
            }
        }

        // Stop monitor thread
        monitorThread?.interrupt()
        monitorThread = null

        currentModel = null
    }

    /**
     * Get the current model configuration
     */
    fun getCurrentModel(): DiffusionModelConfig? = currentModel

    /**
     * Check if backend is running
     */
    fun isBackendRunning(): Boolean = _Diffusion_backendState.value is DiffusionBackendState.Running

    /**
     * Cleanup resources
     */
    fun cleanup() {
        stopBackend()
        isRuntimePrepared = false
    }

    // Private helper methods

    private fun prepareRuntimeDirectory(config: DiffusionRuntimeConfig) {
        runtimeDir = File(context.filesDir, config.runtimeDir).apply {
            if (!exists()) {
                mkdirs()
            }
        }

        try {
            val markerFile = File(runtimeDir, ".extracted")

            if (markerFile.exists() && runtimeDir.listFiles()?.isNotEmpty() == true) {
                Log.i(TAG, "QNN libraries already exist, skipping extraction")

                runtimeDir.listFiles()?.forEach { file ->
                    file.setReadable(true, true)
                    file.setExecutable(true, true)
                }

                runtimeDir.setReadable(true, true)
                runtimeDir.setExecutable(true, true)

                Log.i(TAG, "QNN libraries prepared in: ${runtimeDir.absolutePath}")
                Log.i(TAG, "Runtime files: ${runtimeDir.list()?.joinToString()}")
                return
            }

            val tarXzAssetPath = "${config.qnnLibsAssetPath}/qnnlibs.tar.xz"
            val tarXzFile = File(context.cacheDir, "qnnlibs.tar.xz")

            Log.i(TAG, "Extracting QNN libraries from tar.xz")

            context.assets.open(tarXzAssetPath).use { input ->
                tarXzFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            Log.d(TAG, "Copied tar.xz to cache: ${tarXzFile.absolutePath}")

            extractTarXzWithCommonsCompress(tarXzFile, runtimeDir)

            markerFile.createNewFile()

            tarXzFile.delete()

            Log.i(TAG, "QNN libraries extracted successfully")

            runtimeDir.listFiles()?.forEach { file ->
                file.setReadable(true, true)
                file.setExecutable(true, true)
            }

            runtimeDir.setReadable(true, true)
            runtimeDir.setExecutable(true, true)

            Log.i(TAG, "QNN libraries prepared in: ${runtimeDir.absolutePath}")
            Log.i(TAG, "Runtime files: ${runtimeDir.list()?.joinToString()}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to prepare QNN libraries from assets", e)
            throw RuntimeException("Failed to prepare QNN libraries from assets", e)
        }
    }

    private fun prepareSafetyChecker(assetPath: String = "safety_checker.mnn") {
        try {
            // Ensure parent directory exists
            if (!safetyCheckerFile.parentFile?.exists()!!) {
                safetyCheckerFile.parentFile?.mkdirs()
                Log.d(TAG, "Created safety checker directory: ${safetyCheckerFile.parentFile?.absolutePath}")
            }

            val safetyCheckerSource = context.assets.open(assetPath)
            val safetyCheckerTarget = safetyCheckerFile

            safetyCheckerSource.use { input ->
                safetyCheckerTarget.outputStream().use { output ->
                    input.copyTo(output)
                }
            }

            safetyCheckerTarget.setReadable(true, true)
            Log.i(TAG, "Safety checker model copied to: ${safetyCheckerTarget.absolutePath}")
        } catch (e: IOException) {
            Log.e(TAG, "Failed to copy safety checker model", e)
            throw RuntimeException("Failed to copy safety checker model", e)
        }
    }

    private fun startBackend(model: DiffusionModelConfig, width: Int, height: Int): Boolean {
        Log.i(TAG, "Starting backend - Model: ${model.name}, Resolution: ${width}×${height}")
        updateState(DiffusionBackendState.Starting)

        try {
            val nativeDir = context.applicationInfo.nativeLibraryDir
            val modelsDir = File(model.modelDir)

            val executableFile = File(nativeDir, EXECUTABLE_NAME)
            if (!executableFile.exists()) {
                Log.e(TAG, "Executable not found: ${executableFile.absolutePath}")
                updateState(DiffusionBackendState.Error("Executable not found"))
                return false
            }

            // Build command
            var command = buildCommand(model, modelsDir, executableFile, width, height)

            if (model.safetyMode){
                if (!safetyCheckerFile.exists()){
                    throw FileNotFoundException("Safety checker model not found: ${safetyCheckerFile.absolutePath}")
                }
                val tempCommand = command
                val safetyCommand= listOf(
                    "--safety_checker",
                    safetyCheckerFile.absolutePath
                )
                command = tempCommand + safetyCommand
            }

            // Build environment
            val env = buildEnvironment()

            // Log execution details
            logExecutionDetails(command, env)

            // Start process
            val processBuilder = ProcessBuilder(command).apply {
                directory(File(nativeDir))
                redirectErrorStream(true)
                environment().putAll(env)
            }

            process = processBuilder.start()
            currentModel = model

            // Start monitoring thread
            startMonitorThread()

            updateState(DiffusionBackendState.Running)
            Log.i(TAG, "Backend started successfully")
            return true

        } catch (e: Exception) {
            Log.e(TAG, "Failed to start backend", e)
            updateState(DiffusionBackendState.Error("Backend start failed: ${e.message}"))
            return false
        }
    }

    private fun buildCommand(
        model: DiffusionModelConfig, modelsDir: File, executableFile: File, width: Int, height: Int
    ): List<String> {
        val preferences = context.getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
        val useImg2img = preferences.getBoolean("use_img2img", true)

        if (model.runOnCpu) {
            return buildCpuCommand(model, modelsDir, executableFile, useImg2img)
        }

        // GPU command
        val clipFilename = if (model.useCpuClip) "clip.mnn" else "clip.bin"

        var command = listOf(
            executableFile.absolutePath,
            "--clip",
            File(modelsDir, clipFilename).absolutePath,
            "--unet",
            File(modelsDir, "unet.bin").absolutePath,
            "--vae_decoder",
            File(modelsDir, "vae_decoder.bin").absolutePath,
            "--tokenizer",
            File(modelsDir, "tokenizer.json").absolutePath,
            "--backend",
            File(runtimeDir, "libQnnHtp.so").absolutePath,
            "--system_library",
            File(runtimeDir, "libQnnSystem.so").absolutePath,
            "--port",
            model.httpPort.toString(),
            "--text_embedding_size",
            model.textEmbeddingSize.toString()
        )

        // Add resolution patch if needed
        if (width != 512 || height != 512) {
            command = addResolutionPatch(command, modelsDir, width, height)
        }

        // Add img2img support
        if (useImg2img) {
            command = command + listOf(
                "--vae_encoder", File(modelsDir, "vae_encoder.bin").absolutePath
            )
        }

        // Add model-specific flags
        if (model.isPony) {
            command += "--ponyv55"
        }

        if (model.useCpuClip) {
            command += "--use_cpu_clip"
        }

        return command
    }

    private fun buildCpuCommand(
        model: DiffusionModelConfig, modelsDir: File, executableFile: File, useImg2img: Boolean
    ): List<String> {
        var command = listOf(
            executableFile.absolutePath,
            "--clip",
            File(modelsDir, "clip.mnn").absolutePath,
            "--unet",
            File(modelsDir, "unet.mnn").absolutePath,
            "--vae_decoder",
            File(modelsDir, "vae_decoder.mnn").absolutePath,
            "--tokenizer",
            File(modelsDir, "tokenizer.json").absolutePath,
            "--port",
            model.httpPort.toString(),
            "--text_embedding_size",
            model.textEmbeddingSize.toString(),
            "--cpu"
        )



        if (useImg2img) {
            command = command + listOf(
                "--vae_encoder", File(modelsDir, "vae_encoder.mnn").absolutePath
            )
        }

        return command
    }

    private fun addResolutionPatch(
        command: List<String>, modelsDir: File, width: Int, height: Int
    ): List<String> {
        val patchFile = if (width == height) {
            val squarePatch = File(modelsDir, "${width}.patch")
            if (squarePatch.exists()) squarePatch else File(modelsDir, "${width}x${height}.patch")
        } else {
            File(modelsDir, "${width}x${height}.patch")
        }

        return if (patchFile.exists()) {
            Log.i(TAG, "Using patch file: ${patchFile.name}")
            command + listOf("--patch", patchFile.absolutePath)
        } else {
            Log.w(TAG, "Patch file not found: ${patchFile.absolutePath}, using 512×512")
            command
        }
    }

    private fun buildEnvironment(): Map<String, String> {
        val env = mutableMapOf<String, String>()

        val systemLibPaths = mutableListOf(
            runtimeDir.absolutePath, "/system/lib64", "/vendor/lib64", "/vendor/lib64/egl"
        )

        // Add Mali GPU paths if available
        try {
            val maliSymlink = File("/system/vendor/lib64/egl/libGLES_mali.so")
            if (maliSymlink.exists()) {
                val realPath = maliSymlink.canonicalPath
                val soc = realPath.split("/").getOrNull(realPath.split("/").size - 2)

                soc?.let {
                    val socPaths = listOf(
                        "/vendor/lib64/$it", "/vendor/lib64/egl/$it"
                    )
                    socPaths.forEach { path ->
                        if (!systemLibPaths.contains(path)) {
                            systemLibPaths.add(path)
                            Log.d(TAG, "Added SoC path: $path")
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to resolve Mali paths: ${e.message}")
        }

        env["LD_LIBRARY_PATH"] = systemLibPaths.joinToString(":")
        env["DSP_LIBRARY_PATH"] = runtimeDir.absolutePath

        return env
    }

    private fun startMonitorThread() {
        monitorThread = Thread {
            try {
                process?.let { proc ->
                    proc.inputStream.bufferedReader().use { reader ->
                        var line: String?
                        while (reader.readLine().also { line = it } != null) {
                            Log.i(TAG, "Backend: $line")
                        }
                    }

                    val exitCode = proc.waitFor()
                    Log.i(TAG, "Backend process exited with code: $exitCode")

                    if (_Diffusion_backendState.value is DiffusionBackendState.Running) {
                        updateState(DiffusionBackendState.Error("Backend process exited unexpectedly (code: $exitCode)"))
                    }
                }
            } catch (e: InterruptedException) {
                Log.d(TAG, "Monitor thread interrupted")
            } catch (e: Exception) {
                Log.e(TAG, "Monitor thread error", e)
                updateState(DiffusionBackendState.Error("Monitor error: ${e.message}"))
            }
        }.apply {
            isDaemon = true
            name = "BackendMonitor"
            start()
        }
    }

    private fun logExecutionDetails(command: List<String>, env: Map<String, String>) {
        Log.d(TAG, "=== Backend Execution Details ===")
        Log.d(TAG, "Command: ${command.joinToString(" ")}")
        Log.d(TAG, "Working directory: ${context.applicationInfo.nativeLibraryDir}")
        Log.d(TAG, "LD_LIBRARY_PATH: ${env["LD_LIBRARY_PATH"]}")
        Log.d(TAG, "DSP_LIBRARY_PATH: ${env["DSP_LIBRARY_PATH"]}")
        Log.d(TAG, "================================")
    }

    private fun updateState(state: DiffusionBackendState) {
        _Diffusion_backendState.value = state
    }
}