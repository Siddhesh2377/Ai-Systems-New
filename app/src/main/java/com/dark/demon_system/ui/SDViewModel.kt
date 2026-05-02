package com.dark.demon_system.ui

import android.app.Application
import android.graphics.Bitmap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.dark.ai_sd.DiffusionBackendState
import com.dark.ai_sd.DiffusionGenerationParams
import com.dark.ai_sd.DiffusionGenerationState
import com.dark.ai_sd.DiffusionModelConfig
import com.dark.ai_sd.RuntimeSetupState
import com.dark.ai_sd.StableDiffusionManager
import com.dark.demon_system.data.ModelInstallState
import com.dark.demon_system.data.ModelInstaller
import com.dark.demon_system.data.ModelSpec
import kotlinx.coroutines.Job
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class GenerationForm(
    val prompt: String = "a cinematic portrait of a fox in a misty forest, golden hour, 35mm",
    val negativePrompt: String = "lowres, blurry, jpeg artifacts, watermark, text",
    val steps: Int = 20,
    val cfgScale: Float = 7f,
    val seedText: String = "",
    val width: Int = 512,
    val height: Int = 512,
    val scheduler: String = "dpm",
    val runOnCpu: Boolean = false
)

class SDViewModel(app: Application) : AndroidViewModel(app) {

    private val sd = StableDiffusionManager.getInstance(app)
    private val installer = ModelInstaller(app)
    private val spec = ModelSpec.ABSOLUTE_REALITY_QNN_MIN

    val installState: StateFlow<ModelInstallState> = installer.state
    val runtimeState: StateFlow<RuntimeSetupState> = sd.runtimeSetupState
    val backendState: StateFlow<DiffusionBackendState> = sd.diffusionBackendState
    val generationState: StateFlow<DiffusionGenerationState> = sd.diffusionGenerationState
    val isGenerating: StateFlow<Boolean> = sd.isGenerating

    private val _form = MutableStateFlow(GenerationForm())
    val form: StateFlow<GenerationForm> = _form.asStateFlow()

    private val _socInfo = MutableStateFlow<String?>(null)
    val socInfo: StateFlow<String?> = _socInfo.asStateFlow()

    val modelSpec: ModelSpec = spec
    val isInstalled: Boolean get() = installer.isInstalled(spec)

    private var installJob: Job? = null

    init {
        viewModelScope.launch {
            _socInfo.value = withContext(Dispatchers.IO) { runCatching { sd.getSocInfo() }.getOrNull() }
        }
        // Reflect existing install in state so the UI shows "Installed" on cold start.
        if (installer.isInstalled(spec)) {
            // Touch state via a fresh install call short-circuit.
            viewModelScope.launch { installer.install(spec) }
        }
    }

    fun updateForm(transform: (GenerationForm) -> GenerationForm) {
        _form.value = transform(_form.value)
    }

    fun installModel() {
        if (installJob?.isActive == true) return
        installJob = viewModelScope.launch {
            installer.install(spec)
        }
    }

    fun cancelInstall() {
        installJob?.cancel()
    }

    fun uninstallModel() {
        installer.uninstall(spec)
    }

    fun resetInstallError() {
        installer.resetState()
    }

    /**
     * Runs runtime setup (extract QNN libs from assets) then loads the model.
     */
    fun loadModel() {
        viewModelScope.launch {
            val dir = installer.modelDir(spec)

            // Pre-validate so we surface a precise error before hitting JNI.
            val runOnCpu = _form.value.runOnCpu
            val unet = if (runOnCpu) "unet.mnn" else "unet.bin"
            val vae = if (runOnCpu) "vae_decoder.mnn" else "vae_decoder.bin"
            // ai_sd accepts either clip.mnn (v1) or clip_v2.mnn (v2, used by xororz);
            // the native loader auto-detects v2 when present beside clip.mnn.
            val hasClip = java.io.File(dir, "clip.mnn").exists() ||
                java.io.File(dir, "clip_v2.mnn").exists()
            val missing = buildList {
                if (!hasClip) add("clip.mnn or clip_v2.mnn")
                if (!java.io.File(dir, unet).exists()) add(unet)
                if (!java.io.File(dir, vae).exists()) add(vae)
                if (!java.io.File(dir, "tokenizer.json").exists()) add("tokenizer.json")
            }
            if (missing.isNotEmpty()) {
                _modelLoadError.value = "Missing model files: ${missing.joinToString()}"
                return@launch
            }
            _modelLoadError.value = null

            try {
                sd.initialize()
            } catch (e: Exception) {
                return@launch
            }
            val cfg = DiffusionModelConfig(
                name = spec.displayName,
                modelDir = dir.absolutePath,
                textEmbeddingSize = 768,
                runOnCpu = runOnCpu,
                useCpuClip = !runOnCpu, // QNN mode: CLIP runs on CPU as MNN for xororz models
                isPony = false,
                safetyMode = false
            )
            withContext(Dispatchers.IO) {
                sd.loadModel(cfg, _form.value.width, _form.value.height)
            }
        }
    }

    private val _modelLoadError = MutableStateFlow<String?>(null)
    val modelLoadError: StateFlow<String?> = _modelLoadError.asStateFlow()

    fun unloadModel() {
        viewModelScope.launch(Dispatchers.IO) { sd.stopBackend() }
    }

    fun generate() {
        val f = _form.value
        val params = DiffusionGenerationParams(
            prompt = f.prompt,
            negativePrompt = f.negativePrompt,
            steps = f.steps,
            cfgScale = f.cfgScale,
            seed = f.seedText.toLongOrNull(),
            width = f.width,
            height = f.height,
            scheduler = f.scheduler,
            useOpenCL = false,
            showDiffusionProcess = true,
            showDiffusionStride = 2
        )
        sd.generateImage(params)
    }

    fun cancelGeneration() {
        sd.cancelGeneration()
    }

    fun resetGenerationState() {
        sd.resetGenerationState()
    }

    /** Latest result bitmap if the generation state holds one, else null. */
    fun latestBitmap(): Bitmap? = when (val s = generationState.value) {
        is DiffusionGenerationState.Complete -> s.bitmap
        is DiffusionGenerationState.Progress -> s.intermediateImage
        else -> null
    }

    companion object {
        val Factory = object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: androidx.lifecycle.viewmodel.CreationExtras): T {
                val app = extras[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY] as Application
                return SDViewModel(app) as T
            }
        }
    }
}
