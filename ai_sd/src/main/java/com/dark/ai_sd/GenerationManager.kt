package com.dark.ai_sd

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.core.graphics.createBitmap
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.util.Base64
import java.util.concurrent.TimeUnit

/**
 * Manager class for handling image generation operations
 * Communicates with the backend server started by DiffusionManager
 */
class GenerationManager(private val context: Context) {

    companion object {
        private const val TAG = "GenerationManager"
        private const val DEFAULT_TIMEOUT_SECONDS = 3600L
        
        @Volatile
        private var instance: GenerationManager? = null
        
        fun getInstance(context: Context): GenerationManager {
            return instance ?: synchronized(this) {
                instance ?: GenerationManager(context.applicationContext).also { instance = it }
            }
        }
    }

    // State management
    private val _generationState = MutableStateFlow<GenerationState>(GenerationState.Idle)
    val generationState: StateFlow<GenerationState> = _generationState.asStateFlow()

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    // Coroutine management
    private var generationJob: Job? = null
    private val generationScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    // HTTP client
    private val httpClient: OkHttpClient by lazy {
        OkHttpClient.Builder()
            .connectTimeout(DEFAULT_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .readTimeout(DEFAULT_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .writeTimeout(DEFAULT_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .callTimeout(DEFAULT_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .retryOnConnectionFailure(true)
            .build()
    }

    /**
     * Start image generation with the given parameters
     * Returns immediately, use generationState to monitor progress
     */
    fun generateImage(params: GenerationParams) {
        if (_isGenerating.value) {
            Log.w(TAG, "Generation already in progress, cancelling previous job")
            cancelGeneration()
        }

        generationJob = generationScope.launch {
            try {
                _isGenerating.value = true
                updateState(GenerationState.Idle)
                runGeneration(params)
            } catch (e: CancellationException) {
                Log.i(TAG, "Generation cancelled")
                updateState(GenerationState.Error("Generation cancelled"))
            } catch (e: Exception) {
                Log.e(TAG, "Generation failed", e)
                updateState(GenerationState.Error(e.message ?: "Unknown error"))
            } finally {
                _isGenerating.value = false
            }
        }
    }

    /**
     * Generate image synchronously and return the result
     * Suspending function for use in coroutines
     */
    suspend fun generateImageSync(params: GenerationParams): GenerationResult = withContext(Dispatchers.IO) {
        try {
            _isGenerating.value = true
            updateState(GenerationState.Idle)
            runGeneration(params)
            
            // Wait for completion
            when (val state = _generationState.value) {
                is GenerationState.Complete -> {
                    GenerationResult.Success(
                        bitmap = state.bitmap,
                        seed = state.seed,
                        width = state.width,
                        height = state.height
                    )
                }
                is GenerationState.Error -> GenerationResult.Failure(state.message)
                else -> GenerationResult.Failure("Unexpected state: $state")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Synchronous generation failed", e)
            GenerationResult.Failure(e.message ?: "Unknown error")
        } finally {
            _isGenerating.value = false
        }
    }

    /**
     * Cancel ongoing generation
     */
    fun cancelGeneration() {
        generationJob?.cancel()
        generationJob = null
        updateState(GenerationState.Idle)
        _isGenerating.value = false
    }

    /**
     * Reset state to idle
     */
    fun resetState() {
        if (!_isGenerating.value) {
            updateState(GenerationState.Idle)
        }
    }

    /**
     * Cleanup resources
     */
    fun cleanup() {
        cancelGeneration()
        generationScope.cancel()
    }

    // Private helper methods

    private suspend fun runGeneration(params: GenerationParams) = withContext(Dispatchers.IO) {
        try {
            updateState(GenerationState.Progress(0f))

            val jsonObject = buildRequestJson(params)
            val request = buildHttpRequest(jsonObject, params)

            httpClient.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    throw IOException("Request failed with code: ${response.code}")
                }

                response.body?.let { responseBody ->
                    processStreamingResponse(responseBody, params.width, params.height)
                } ?: throw IOException("Empty response body")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Generation error", e)
            throw e
        }
    }

    private fun buildRequestJson(params: GenerationParams): JSONObject {
        return JSONObject().apply {
            put("prompt", params.prompt)
            put("negative_prompt", params.negativePrompt)
            put("steps", params.steps)
            put("cfg", params.cfgScale)
            put("use_cfg", true)
            put("width", params.width)
            put("height", params.height)
            put("denoise_strength", params.denoiseStrength)
            put("use_opencl", params.useOpenCL)
            put("scheduler", params.scheduler)
            put("show_diffusion_process", params.showDiffusionProcess)
            put("show_diffusion_stride", params.showDiffusionStride)
            
            params.seed?.let { put("seed", it) }
            params.inputImage?.let { put("image", it) }
            params.mask?.let { put("mask", it) }
        }
    }

    private fun buildHttpRequest(jsonObject: JSONObject, params: GenerationParams): Request {
        // Determine port from model config or use default
        val port = 8081 // Could be made configurable
        
        return Request.Builder()
            .url("http://localhost:$port/generate")
            .post(jsonObject.toString().toRequestBody("application/json".toMediaTypeOrNull()))
            .build()
    }

    private suspend fun processStreamingResponse(
        responseBody: ResponseBody,
        width: Int,
        height: Int
    ) = withContext(Dispatchers.IO) {
        val reader = BufferedReader(InputStreamReader(responseBody.byteStream()))
        
        try {
            while (isActive) {
                val line = reader.readLine() ?: break

                if (line.startsWith("data: ")) {
                    val data = line.substring(6).trim()
                    if (data == "[DONE]") break

                    val message = JSONObject(data)
                    processMessage(message, width, height)
                }
            }
        } finally {
            reader.close()
        }
    }

    private fun processMessage(message: JSONObject, width: Int, height: Int) {
        when (message.optString("type")) {
            "progress" -> processProgressMessage(message, width, height)
            "complete" -> processCompleteMessage(message)
            "error" -> {
                val errorMsg = message.optString("message", "Unknown error")
                Log.e(TAG, "Received error message: $errorMsg")
                throw IOException(errorMsg)
            }
        }
    }

    private fun processProgressMessage(message: JSONObject, width: Int, height: Int) {
        val step = message.optInt("step")
        val totalSteps = message.optInt("total_steps")
        val progress = step.toFloat() / totalSteps

        val intermediateImage = message.optString("image").takeIf { it.isNotEmpty() }?.let { b64 ->
            decodeBase64Image(b64, width, height)
        }

        updateState(
            GenerationState.Progress(
                progress = progress,
                currentStep = step,
                totalSteps = totalSteps,
                intermediateImage = intermediateImage
            )
        )
    }

    private fun processCompleteMessage(message: JSONObject) {
        val startTime = System.currentTimeMillis()
        
        val base64Image = message.optString("image")
        val seed = message.optLong("seed", -1).takeIf { it != -1L }
        val width = message.optInt("width", 512)
        val height = message.optInt("height", 512)

        if (base64Image.isNullOrEmpty()) {
            throw IOException("No image data in response")
        }

        // Decode image
        val decodeStart = System.currentTimeMillis()
        val imageBytes = Base64.getDecoder().decode(base64Image)
        val decodeTime = System.currentTimeMillis() - decodeStart

        // Create bitmap
        val bitmapStart = System.currentTimeMillis()
        val bitmap = createBitmapFromRgb(imageBytes, width, height)
        val bitmapTime = System.currentTimeMillis() - bitmapStart

        val totalTime = System.currentTimeMillis() - startTime
        Log.d(TAG, "Image processing: decode=${decodeTime}ms, bitmap=${bitmapTime}ms, total=${totalTime}ms")

        updateState(
            GenerationState.Complete(
                bitmap = bitmap,
                seed = seed,
                width = width,
                height = height
            )
        )
    }

    private fun decodeBase64Image(base64: String, width: Int, height: Int): Bitmap? {
        return try {
            val imageBytes = Base64.getDecoder().decode(base64)
            createBitmapFromRgb(imageBytes, width, height)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to decode intermediate image", e)
            null
        }
    }

    private fun createBitmapFromRgb(imageBytes: ByteArray, width: Int, height: Int): Bitmap {
        val bitmap = createBitmap(width, height)
        val pixels = IntArray(width * height)

        for (i in 0 until width * height) {
            val index = i * 3
            if (index + 2 < imageBytes.size) {
                val r = imageBytes[index].toInt() and 0xFF
                val g = imageBytes[index + 1].toInt() and 0xFF
                val b = imageBytes[index + 2].toInt() and 0xFF
                pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }

    private fun updateState(state: GenerationState) {
        _generationState.value = state
    }
}