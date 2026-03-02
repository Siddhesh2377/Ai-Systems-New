package com.dark.ai_systems.tts

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Environment
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.dark.ai_chatterbox.ChatterboxConfig
import com.dark.ai_chatterbox.ChatterboxManager
import com.dark.ai_chatterbox.ChatterboxState
import com.dark.ai_chatterbox.ChatterboxVariant
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

data class TTSUiState(
    val modelDir: String = "",
    val tokenizerPath: String = "",
    val voicePresetDir: String = "",
    val inputText: String = "Hello! This is a test of the Chatterbox text to speech system.",
    val statusMessage: String = "Enter model paths and load",
    val isModelLoaded: Boolean = false,
    val isGenerating: Boolean = false,
    val isScanningPaths: Boolean = false,
    val variant: ChatterboxVariant = ChatterboxVariant.TURBO,
    val exaggeration: Float = 1.0f,
    val lastAudioSamples: Int = 0,
    val lastTokenCount: Int = 0,
    val generationTimeMs: Long = 0
)

class TTSViewModel : ViewModel() {

    companion object {
        private const val TAG = "TTSViewModel"
        private const val SAMPLE_RATE = 24000
    }

    private val manager = ChatterboxManager.getInstance()

    private val _uiState = MutableStateFlow(TTSUiState())
    val uiState: StateFlow<TTSUiState> = _uiState.asStateFlow()

    private var audioTrack: AudioTrack? = null
    private var lastPcmData: ShortArray? = null

    init {
        // Observe engine state
        viewModelScope.launch {
            manager.state.collect { state ->
                when (state) {
                    is ChatterboxState.Idle -> updateStatus("Idle")
                    is ChatterboxState.Loading -> updateStatus("Loading models...")
                    is ChatterboxState.Ready -> {
                        _uiState.value = _uiState.value.copy(
                            isModelLoaded = true,
                            isGenerating = false,
                            statusMessage = "Ready — enter text and tap Synthesize"
                        )
                    }
                    is ChatterboxState.Generating -> {
                        _uiState.value = _uiState.value.copy(
                            isGenerating = true,
                            lastTokenCount = state.tokensGenerated,
                            statusMessage = "Generating: ${state.tokensGenerated} speech tokens..."
                        )
                    }
                    is ChatterboxState.Complete -> {
                        _uiState.value = _uiState.value.copy(
                            isGenerating = false,
                            lastAudioSamples = state.pcmData.size,
                            statusMessage = "Done! ${state.pcmData.size} samples " +
                                    "(${String.format("%.1f", state.pcmData.size / SAMPLE_RATE.toFloat())}s)"
                        )
                    }
                    is ChatterboxState.Error -> {
                        _uiState.value = _uiState.value.copy(
                            isGenerating = false,
                            statusMessage = "Error: ${state.message}"
                        )
                    }
                }
            }
        }

        // Auto-scan for models on common paths
        viewModelScope.launch { scanForModels() }
    }

    fun updateModelDir(path: String) {
        _uiState.value = _uiState.value.copy(modelDir = path)
    }

    fun updateTokenizerPath(path: String) {
        _uiState.value = _uiState.value.copy(tokenizerPath = path)
    }

    fun updateVoicePresetDir(path: String) {
        _uiState.value = _uiState.value.copy(voicePresetDir = path)
    }

    fun updateInputText(text: String) {
        _uiState.value = _uiState.value.copy(inputText = text)
    }

    fun updateVariant(variant: ChatterboxVariant) {
        _uiState.value = _uiState.value.copy(variant = variant)
    }

    fun updateExaggeration(value: Float) {
        _uiState.value = _uiState.value.copy(exaggeration = value)
    }

    fun loadModel() {
        val state = _uiState.value
        if (state.modelDir.isBlank() || state.tokenizerPath.isBlank()) {
            updateStatus("Error: model dir and tokenizer path required")
            return
        }

        viewModelScope.launch {
            val config = ChatterboxConfig(
                modelDir = state.modelDir.trim(),
                tokenizerPath = state.tokenizerPath.trim(),
                voicePresetDir = state.voicePresetDir.trim().ifBlank { null },
                variant = state.variant,
                exaggeration = state.exaggeration
            )

            Log.i(TAG, "Loading model: $config")
            val success = manager.loadModel(config)
            if (!success) {
                updateStatus("Failed to load model — check paths and logcat")
            }
        }
    }

    fun loadVoicePreset() {
        val dir = _uiState.value.voicePresetDir.trim()
        if (dir.isBlank()) {
            updateStatus("Error: voice preset dir is empty")
            return
        }

        val success = manager.loadVoicePreset(dir)
        if (success) {
            updateStatus("Voice preset loaded from $dir")
        } else {
            updateStatus("Failed to load voice preset")
        }
    }

    fun synthesize() {
        val text = _uiState.value.inputText.trim()
        if (text.isBlank()) {
            updateStatus("Error: text is empty")
            return
        }

        if (!_uiState.value.isModelLoaded) {
            updateStatus("Error: model not loaded")
            return
        }

        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isGenerating = true)

            val startTime = System.currentTimeMillis()
            val pcm = manager.synthesize(text)
            val elapsed = System.currentTimeMillis() - startTime

            if (pcm != null && pcm.isNotEmpty()) {
                lastPcmData = pcm
                _uiState.value = _uiState.value.copy(
                    generationTimeMs = elapsed,
                    statusMessage = "Done! ${pcm.size} samples " +
                            "(${String.format("%.1f", pcm.size / SAMPLE_RATE.toFloat())}s) " +
                            "in ${String.format("%.1f", elapsed / 1000f)}s"
                )
                Log.i(TAG, "Synthesis complete: ${pcm.size} samples in ${elapsed}ms")
            } else {
                _uiState.value = _uiState.value.copy(
                    isGenerating = false,
                    generationTimeMs = elapsed,
                    statusMessage = "Synthesis returned no audio (${elapsed}ms)"
                )
            }
        }
    }

    fun playAudio() {
        val pcm = lastPcmData ?: run {
            updateStatus("No audio to play — synthesize first")
            return
        }

        viewModelScope.launch(Dispatchers.IO) {
            try {
                stopAudio()

                val bufferSize = AudioTrack.getMinBufferSize(
                    SAMPLE_RATE,
                    AudioFormat.CHANNEL_OUT_MONO,
                    AudioFormat.ENCODING_PCM_16BIT
                )

                val track = AudioTrack.Builder()
                    .setAudioAttributes(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build()
                    )
                    .setAudioFormat(
                        AudioFormat.Builder()
                            .setSampleRate(SAMPLE_RATE)
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .build()
                    )
                    .setBufferSizeInBytes(maxOf(bufferSize, pcm.size * 2))
                    .setTransferMode(AudioTrack.MODE_STATIC)
                    .build()

                track.write(pcm, 0, pcm.size)
                track.play()
                audioTrack = track

                withContext(Dispatchers.Main) {
                    updateStatus("Playing audio (${String.format("%.1f", pcm.size / SAMPLE_RATE.toFloat())}s)...")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Playback failed", e)
                withContext(Dispatchers.Main) {
                    updateStatus("Playback error: ${e.message}")
                }
            }
        }
    }

    fun stopAudio() {
        try {
            audioTrack?.let { track ->
                if (track.playState == AudioTrack.PLAYSTATE_PLAYING) {
                    track.stop()
                }
                track.release()
            }
            audioTrack = null
        } catch (e: Exception) {
            Log.e(TAG, "Stop audio failed", e)
        }
    }

    fun stopGeneration() {
        manager.stop()
        updateStatus("Stop requested")
    }

    private fun updateStatus(msg: String) {
        _uiState.value = _uiState.value.copy(statusMessage = msg)
    }

    private suspend fun scanForModels() {
        withContext(Dispatchers.IO) {
            _uiState.value = _uiState.value.copy(isScanningPaths = true)

            // Common locations where models might be placed
            val searchDirs = listOf(
                File(Environment.getExternalStorageDirectory(), "chatterbox"),
                File(Environment.getExternalStorageDirectory(), "Download/chatterbox"),
                File(Environment.getExternalStorageDirectory(), "models/chatterbox"),
                File("/sdcard/chatterbox"),
                File("/sdcard/Download/chatterbox")
            )

            for (dir in searchDirs) {
                if (!dir.exists()) continue

                // Check for model files
                val embedTokens = File(dir, "embed_tokens.onnx")
                val languageModel = File(dir, "language_model.onnx")
                val tokenizer = File(dir, "tokenizer.json")

                if (embedTokens.exists() && languageModel.exists()) {
                    val modelDir = dir.absolutePath
                    val tokPath = if (tokenizer.exists()) tokenizer.absolutePath else ""

                    // Check for voice preset subdirectory
                    var voiceDir = ""
                    val voiceSubdirs = listOf("voice", "voice_preset", "preset", "style")
                    for (sub in voiceSubdirs) {
                        val candidate = File(dir, sub)
                        if (candidate.exists() && File(candidate, "cond_emb.bin").exists()) {
                            voiceDir = candidate.absolutePath
                            break
                        }
                    }

                    // Also check if voice files are directly in the model dir
                    if (voiceDir.isEmpty() && File(dir, "cond_emb.bin").exists()) {
                        voiceDir = dir.absolutePath
                    }

                    withContext(Dispatchers.Main) {
                        _uiState.value = _uiState.value.copy(
                            modelDir = modelDir,
                            tokenizerPath = tokPath,
                            voicePresetDir = voiceDir,
                            statusMessage = "Found models at $modelDir",
                            isScanningPaths = false
                        )
                    }
                    Log.i(TAG, "Auto-detected models at $modelDir")
                    return@withContext
                }
            }

            withContext(Dispatchers.Main) {
                _uiState.value = _uiState.value.copy(
                    isScanningPaths = false,
                    statusMessage = "No models found — enter paths manually"
                )
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        stopAudio()
        manager.release()
    }
}
