package com.dark.litert_lib

import android.content.Context
import android.os.Build
import android.util.Log
import com.dark.unified_inference.capability.AccelerationBackend
import com.dark.unified_inference.capability.NPUAccelerable
import com.dark.unified_inference.model.ModelDescriptor
import com.dark.unified_inference.model.ModelFormat
import com.dark.unified_inference.model.ModelSource
import com.dark.unified_inference.text.TextEngine
import com.dark.unified_inference.text.TextEvent
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext

class LiteRTEngine(private val context: Context) : TextEngine, NPUAccelerable {

    companion object {
        private const val TAG = "LiteRTEngine"
        private val NPU_SOC_PREFIXES = listOf("SM", "MT")
    }

    // ── State ──

    private var engineInstance: Any? = null  // Will be typed to actual LiteRT Engine class
    private var currentBackend: AccelerationBackend = AccelerationBackend.CPU

    // ── InferenceEngine ──

    override val engineId: String = "litert"
    override val displayName: String = "LiteRT"
    override val providerTag: String = "Google"
    override val supportedFormats: List<ModelFormat> = listOf(ModelFormat.TFLite, ModelFormat.LiteRTLM)

    override fun isModelLoaded(): Boolean = engineInstance != null

    override suspend fun loadModel(descriptor: ModelDescriptor, params: String?): Boolean =
        withContext(Dispatchers.IO) {
            try {
                unload()

                val modelPath = when (val source = descriptor.source) {
                    is ModelSource.FilePath -> source.path
                    is ModelSource.Directory -> source.path
                    else -> return@withContext false
                }

                // TODO: Verify LiteRT-LM SDK API — class names and constructors may differ
                // Expected API: Engine(EngineConfig) -> engine.initialize()
                // The actual Maven artifact needs to be resolved to confirm imports
                Log.d(TAG, "Loading LiteRT model: $modelPath with backend: $currentBackend")

                // Placeholder — actual LiteRT SDK initialization goes here
                // once the Maven dependency resolves and we can verify the API
                engineInstance = modelPath  // placeholder marker
                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load LiteRT model", e)
                false
            }
        }

    override suspend fun unload(): Unit = withContext(Dispatchers.IO) {
        try {
            // TODO: Call engine.close() on actual LiteRT Engine instance
            engineInstance = null
        } catch (e: Exception) {
            Log.e(TAG, "Error unloading LiteRT model", e)
            engineInstance = null
        }
    }

    override fun stopGeneration() {
        // LiteRT-LM doesn't expose a cancel API — conversation close handles cleanup
    }

    // ── TextEngine ──

    override fun generateFlow(prompt: String, maxTokens: Int): Flow<TextEvent> = callbackFlow {
        if (engineInstance == null) {
            trySend(TextEvent.Error("Engine not loaded"))
            close()
            return@callbackFlow
        }

        try {
            // TODO: Replace with actual LiteRT-LM conversation API
            // Expected: engine.createConversation().use { conv ->
            //     conv.sendMessageAsync(prompt).collect { chunk -> trySend(TextEvent.Token(chunk)) }
            // }
            trySend(TextEvent.Error("LiteRT generation not yet wired — SDK API verification pending"))
            trySend(TextEvent.Done)
        } catch (e: Exception) {
            trySend(TextEvent.Error(e.message ?: "LiteRT generation error"))
        }

        close()
        awaitClose { }
    }.flowOn(Dispatchers.IO)

    override fun generateMultiTurnFlow(messagesJson: String, maxTokens: Int): Flow<TextEvent> =
        callbackFlow {
            if (engineInstance == null) {
                trySend(TextEvent.Error("Engine not loaded"))
                close()
                return@callbackFlow
            }

            try {
                // TODO: Replace with actual LiteRT-LM multi-turn conversation API
                // Expected: parse messagesJson, replay prior messages, stream last
                trySend(TextEvent.Error("LiteRT multi-turn not yet wired — SDK API verification pending"))
                trySend(TextEvent.Done)
            } catch (e: Exception) {
                trySend(TextEvent.Error(e.message ?: "LiteRT multi-turn error"))
            }

            close()
            awaitClose { }
        }.flowOn(Dispatchers.IO)

    // ── NPUAccelerable ──

    override fun isNPUAvailable(): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S) return false
        @Suppress("USELESS_ELVIS")
        val soc = Build.SOC_MODEL ?: return false
        return NPU_SOC_PREFIXES.any { soc.startsWith(it) }
    }

    override fun setAccelerationBackend(backend: AccelerationBackend): Boolean {
        if (backend == AccelerationBackend.NPU && !isNPUAvailable()) return false
        currentBackend = backend
        return true
    }
}
