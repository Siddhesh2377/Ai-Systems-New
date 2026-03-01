package com.dark.gguf_android.ui.chat

import android.annotation.SuppressLint
import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.dark.gguf_android.engine.EngineRepository
import com.mp.ai_gguf_android.DecodingMetrics
import com.mp.ai_gguf_android.StreamCallback
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.BufferedReader
import java.io.InputStreamReader

data class ChatMessage(
    val role: String,
    val text: String,
    val isStreaming: Boolean = false
)

/** Active loading/processing stage for progress display. */
enum class LoadingStage {
    NONE,
    LOADING_MODEL,
    LOADING_VISION,
    EMBEDDING_RAG,
    SEARCHING_WEB,
    SEARCHING_DOCS,
    ANALYZING_IMAGE,
    GENERATING
}

data class ChatUiState(
    val messages: List<ChatMessage> = emptyList(),
    val inputText: String = "",
    val isGenerating: Boolean = false,
    val modelLoaded: Boolean = false,
    val visionLoaded: Boolean = false,
    val modelName: String = "",
    val tokPerSec: Float = 0f,
    val ttft: Long = 0L,
    val genTokens: Int = 0,
    val pendingImage: Bitmap? = null,
    val pendingImageUri: Uri? = null,
    val visionImageSize: Int = 512,
    val webSearchOn: Boolean = false,
    val ragDocs: Int = 0,
    val ragReady: Boolean = false,
    val showLoadDialog: Boolean = false,
    val error: String? = null,
    val loadingStage: LoadingStage = LoadingStage.NONE
)

private const val LUNA_SYSTEM = "You are Luna, a 26-year-old woman who works as a creative " +
    "AI researcher. You're warm, playful, and genuinely curious about everything. You speak " +
    "naturally like a close friend — casual, honest, sometimes witty, sometimes nerdy. You " +
    "love science, art, music, and late-night conversations about the universe. You have a " +
    "dry sense of humor and aren't afraid to tease. Never say 'As an AI' or break character. " +
    "Keep responses concise but engaging."

private const val LUNA_PERSONALITY = """{
  "name": "Luna",
  "system_prompt": "$LUNA_SYSTEM",
  "temp_early": 1.25,
  "temp_mid": 1.0,
  "temp_late": 0.80,
  "attn_gate_mid": 0.92,
  "ffn_gate_mid": 0.94,
  "sampling_temp": 0.85,
  "sampling_top_k": 50,
  "sampling_top_p": 0.93,
  "sampling_rep_penalty": 1.15
}"""

// Max chars of RAG/web context to inject (keeps well within 2048 token ctx)
private const val MAX_CONTEXT_CHARS = 600

// Stall prompt — Luna's filler text while waiting for web/tool results
private const val STALL_PROMPT = "<|im_start|>assistant\nhmm, let me look that up real quick..."
private const val STALL_MAX_TOKENS = 15

@SuppressLint("DefaultLocale")
class ChatViewModel : ViewModel() {

    private val _state = MutableStateFlow(ChatUiState())
    val state: StateFlow<ChatUiState> = _state.asStateFlow()

    private val repo = EngineRepository

    fun updateInput(text: String) {
        _state.value = _state.value.copy(inputText = text)
    }

    fun showLoadDialog() = run { _state.value = _state.value.copy(showLoadDialog = true) }
    fun hideLoadDialog() = run { _state.value = _state.value.copy(showLoadDialog = false) }
    fun dismissError() = run { _state.value = _state.value.copy(error = null) }
    fun toggleWebSearch() = run { _state.value = _state.value.copy(webSearchOn = !_state.value.webSearchOn) }

    // ── Model Loading ────────────────────────────────────────────────────────

    fun loadModel(fd: Int) {
        viewModelScope.launch(Dispatchers.IO) {
            _state.value = _state.value.copy(loadingStage = LoadingStage.LOADING_MODEL)
            try {
                if (repo.getLibOrNull() == null) repo.create()
                val lib = repo.getLib()
                val ok = lib.loadModel(fd, 2048)
                if (ok) {
                    lib.setPersonalityJson(LUNA_PERSONALITY)
                    lib.setSampling(0.85f, 50, 0.93f, 1.15f)
                    lib.setAttentionTemperatureProfile(1.25f, 1.0f, 0.8f)
                    lib.setResidualGates(0.92f, 0.94f)

                    val info = try { lib.getModelInfo() } catch (_: Exception) { "{}" }
                    val name = Regex("\"name\"\\s*:\\s*\"([^\"]+)\"")
                        .find(info)?.groupValues?.get(1) ?: "Model"

                    _state.value = _state.value.copy(
                        modelLoaded = true, modelName = name,
                        showLoadDialog = false, error = null,
                        loadingStage = LoadingStage.NONE,
                        // Reset vision/RAG state — old model's resources were freed
                        visionLoaded = false, ragDocs = 0, ragReady = false,
                        messages = emptyList(), isGenerating = false,
                        tokPerSec = 0f, ttft = 0L, genTokens = 0
                    )
                } else {
                    _state.value = _state.value.copy(
                        error = "Failed to load model",
                        loadingStage = LoadingStage.NONE
                    )
                }
            } catch (e: Exception) {
                _state.value = _state.value.copy(
                    error = e.message, loadingStage = LoadingStage.NONE
                )
            }
        }
    }

    fun loadVisionModel(fd: Int) {
        viewModelScope.launch(Dispatchers.IO) {
            _state.value = _state.value.copy(loadingStage = LoadingStage.LOADING_VISION)
            try {
                val lib = repo.getLib()
                val ok = lib.loadVisionModel(fd)
                if (ok) {
                    _state.value = _state.value.copy(
                        visionLoaded = true,
                        visionImageSize = lib.getVisionImageSize(),
                        showLoadDialog = false,
                        loadingStage = LoadingStage.NONE
                    )
                } else {
                    _state.value = _state.value.copy(
                        error = "Failed to load vision model — make sure it's a mmproj GGUF",
                        loadingStage = LoadingStage.NONE
                    )
                }
            } catch (e: Exception) {
                _state.value = _state.value.copy(
                    error = e.message, loadingStage = LoadingStage.NONE
                )
            }
        }
    }

    // ── Image ────────────────────────────────────────────────────────────────

    fun attachImage(resolver: ContentResolver, uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val stream = resolver.openInputStream(uri) ?: return@launch
                val bmp = BitmapFactory.decodeStream(stream)
                stream.close()
                val sz = _state.value.visionImageSize
                val scaled = Bitmap.createScaledBitmap(bmp, sz, sz, true)
                if (scaled !== bmp) bmp.recycle()
                _state.value = _state.value.copy(pendingImage = scaled, pendingImageUri = uri)
            } catch (e: Exception) {
                _state.value = _state.value.copy(error = "Image load failed: ${e.message}")
            }
        }
    }

    fun clearImage() {
        _state.value.pendingImage?.recycle()
        _state.value = _state.value.copy(pendingImage = null, pendingImageUri = null)
    }

    // ── RAG Document Import ──────────────────────────────────────────────────

    fun ragImportFile(resolver: ContentResolver, uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            _state.value = _state.value.copy(loadingStage = LoadingStage.EMBEDDING_RAG)
            try {
                val lib = repo.getLib()
                val source = uri.lastPathSegment ?: "document"

                // Check MIME type — only accept text-based files
                val mimeType = resolver.getType(uri) ?: ""
                val isText = mimeType.startsWith("text/") ||
                    mimeType == "application/json" ||
                    mimeType == "application/xml" ||
                    source.endsWith(".txt") || source.endsWith(".md") ||
                    source.endsWith(".json") || source.endsWith(".csv") ||
                    source.endsWith(".xml") || source.endsWith(".html") ||
                    source.endsWith(".log") || source.endsWith(".kt") ||
                    source.endsWith(".java") || source.endsWith(".py") ||
                    source.endsWith(".cpp") || source.endsWith(".h")

                if (!isText) {
                    _state.value = _state.value.copy(
                        error = "Only text files supported for RAG (got ${mimeType.ifEmpty { source.substringAfterLast('.') }}). PDF/binary files are not supported.",
                        loadingStage = LoadingStage.NONE
                    )
                    return@launch
                }

                val stream = resolver.openInputStream(uri) ?: run {
                    _state.value = _state.value.copy(
                        error = "Cannot open file", loadingStage = LoadingStage.NONE)
                    return@launch
                }
                val reader = BufferedReader(InputStreamReader(stream))
                val text = reader.readText()
                reader.close()

                // Sanity check: reject if >30% non-printable chars (binary data)
                val nonPrintable = text.count { it < ' ' && it != '\n' && it != '\r' && it != '\t' }
                if (text.isNotEmpty() && nonPrintable.toFloat() / text.length > 0.3f) {
                    _state.value = _state.value.copy(
                        error = "File appears to contain binary data, not text",
                        loadingStage = LoadingStage.NONE
                    )
                    return@launch
                }

                lib.ragIngest(text, source)
                _state.value = _state.value.copy(
                    ragDocs = _state.value.ragDocs + 1,
                    ragReady = false
                )
                // Embed in background — this is slow (runs forward pass per chunk)
                lib.ragEmbedChunks()
                _state.value = _state.value.copy(
                    ragReady = true, loadingStage = LoadingStage.NONE
                )
            } catch (e: Exception) {
                _state.value = _state.value.copy(
                    error = "RAG import failed: ${e.message}",
                    loadingStage = LoadingStage.NONE
                )
            }
        }
    }

    // ── Send Message ─────────────────────────────────────────────────────────

    /** Add a system info message to chat. */
    private fun addSystemMsg(text: String) {
        val cur = _state.value
        val msgs = cur.messages.toMutableList()
        // Insert before the last message (the streaming assistant placeholder)
        val insertIdx = (msgs.size - 1).coerceAtLeast(0)
        msgs.add(insertIdx, ChatMessage("system", text))
        _state.value = cur.copy(messages = msgs)
    }

    fun send() {
        val s = _state.value
        val text = s.inputText.trim()
        if (text.isEmpty() || s.isGenerating || !s.modelLoaded) return

        val lib = repo.getLib()
        val userMsg = ChatMessage("user", text)
        val assistantMsg = ChatMessage("assistant", "", isStreaming = true)

        _state.value = s.copy(
            messages = s.messages + userMsg + assistantMsg,
            isGenerating = true, inputText = "",
            tokPerSec = 0f, ttft = 0L, genTokens = 0
        )

        viewModelScope.launch(Dispatchers.IO) {
            val sb = StringBuilder()

            val callback = object : StreamCallback {
                override fun onToken(token: String) {
                    sb.append(token)
                    val cur = _state.value
                    val msgs = cur.messages.toMutableList()
                    msgs[msgs.lastIndex] = ChatMessage("assistant", sb.toString(), true)
                    _state.value = cur.copy(messages = msgs)
                }

                override fun onToolCall(name: String, argsJson: String) {}

                override fun onDone() {
                    val cur = _state.value
                    val msgs = cur.messages.toMutableList()
                    msgs[msgs.lastIndex] = ChatMessage("assistant", sb.toString(), false)
                    _state.value = cur.copy(
                        messages = msgs, isGenerating = false,
                        loadingStage = LoadingStage.NONE
                    )
                }

                override fun onError(message: String) {
                    val cur = _state.value
                    val msgs = cur.messages.toMutableList()
                    val txt = sb.toString().ifEmpty { "Error: $message" }
                    msgs[msgs.lastIndex] = ChatMessage("assistant", txt, false)
                    _state.value = cur.copy(
                        messages = msgs, isGenerating = false,
                        error = message, loadingStage = LoadingStage.NONE
                    )
                }

                override fun onMetrics(metrics: DecodingMetrics) {
                    _state.value = _state.value.copy(
                        tokPerSec = metrics.tokensPerSecond,
                        ttft = metrics.timeToFirstToken,
                        genTokens = metrics.generatedTokens
                    )
                }
            }

            try {
                var context = ""
                val needsWebSearch = s.webSearchOn

                // ── Stall + Web search in parallel ──
                // Stall streams filler text ("hmm, let me look that up...") on the
                // model thread while web search runs on a network thread.
                // When web search completes it signals stopStall().
                if (needsWebSearch) {
                    _state.value = _state.value.copy(loadingStage = LoadingStage.SEARCHING_WEB)
                    coroutineScope {
                        val webDeferred = async(Dispatchers.IO) {
                            var webCtx = ""
                            try {
                                addSystemMsg("searching the web...")
                                val webResult = lib.webSearch(text)
                                if (webResult.isNotEmpty() && webResult != "[]" && webResult != "{}") {
                                    webCtx = extractTextFromJson(webResult, MAX_CONTEXT_CHARS / 2)
                                    addSystemMsg("found: ${webCtx.take(120)}...")
                                } else {
                                    addSystemMsg("no web results found")
                                }
                            } catch (e: Exception) {
                                addSystemMsg("web search failed: ${e.message}")
                            } finally {
                                lib.stopStall()
                            }
                            webCtx
                        }

                        // Stream filler tokens while web search runs (blocks until
                        // stopStall() is called or max tokens reached)
                        lib.generateStall(STALL_PROMPT, STALL_MAX_TOKENS, callback)

                        // Stall exited — get web context
                        context = webDeferred.await()
                    }
                    // Clear stall filler text — real response replaces it
                    sb.clear()
                }

                // ── RAG retrieval (sequential — uses model compute backend) ──
                if (s.ragReady && s.ragDocs > 0) {
                    _state.value = _state.value.copy(loadingStage = LoadingStage.SEARCHING_DOCS)
                    addSystemMsg("searching documents...")
                    try {
                        val ragResult = lib.ragSearch(text, 3)
                        val ragText = extractTextFromJson(ragResult, MAX_CONTEXT_CHARS / 2)
                        if (ragText.isNotEmpty()) {
                            context = if (context.isEmpty()) ragText
                            else "$context\n$ragText"
                            addSystemMsg("found: ${ragText.take(120)}...")
                        } else {
                            addSystemMsg("no matching documents")
                        }
                    } catch (e: Exception) {
                        addSystemMsg("document search failed: ${e.message}")
                    }
                }

                // Truncate total context
                if (context.length > MAX_CONTEXT_CHARS) {
                    context = context.take(MAX_CONTEXT_CHARS) + "..."
                }

                // ── VLM path ──
                val bitmap = s.pendingImage
                if (bitmap != null && s.visionLoaded) {
                    _state.value = _state.value.copy(loadingStage = LoadingStage.ANALYZING_IMAGE)
                    addSystemMsg("analyzing image...")
                    val w = bitmap.width; val h = bitmap.height
                    val pixels = IntArray(w * h)
                    bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
                    val sysPrompt = buildPrompt(context)
                    _state.value = _state.value.copy(loadingStage = LoadingStage.GENERATING)
                    lib.generateWithImage(pixels, w, h, sysPrompt, text, 512, callback)
                    _state.value = _state.value.copy(pendingImage = null, pendingImageUri = null)
                } else if (context.isNotEmpty()) {
                    // Has context — use generate() with explicit system prompt
                    _state.value = _state.value.copy(loadingStage = LoadingStage.GENERATING)
                    val sysPrompt = buildPrompt(context)
                    lib.generate(sysPrompt, text, 512, callback)
                } else {
                    // Pure text — characterChat for personality + conversation history
                    _state.value = _state.value.copy(loadingStage = LoadingStage.GENERATING)
                    lib.characterChat(text, 512, callback)
                }
            } catch (e: Exception) {
                callback.onError(e.message ?: "Generation failed")
            }
        }
    }

    /** Extract "text" fields from RAG/web JSON and truncate to maxChars. */
    private fun extractTextFromJson(json: String, maxChars: Int): String {
        val sb = StringBuilder()
        // Quick regex to pull "text":"..." values from the JSON array
        val textPattern = Regex("\"text\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"")
        for (match in textPattern.findAll(json)) {
            if (sb.length >= maxChars) break
            val chunk = match.groupValues[1]
                .replace("\\n", " ")
                .replace("\\t", " ")
                .replace("\\\"", "\"")
                .replace("\\\\", "\\")
                .trim()
            if (chunk.isNotEmpty()) {
                if (sb.isNotEmpty()) sb.append(" ")
                val remaining = maxChars - sb.length
                if (chunk.length > remaining) {
                    sb.append(chunk.take(remaining))
                } else {
                    sb.append(chunk)
                }
            }
        }
        return sb.toString()
    }

    private fun buildPrompt(context: String): String {
        return if (context.isNotEmpty()) {
            "$LUNA_SYSTEM\n\nRelevant info:\n$context"
        } else LUNA_SYSTEM
    }

    fun stop() {
        try { repo.getLib().stopGeneration() } catch (_: Exception) { }
    }

    fun clearChat() {
        try { repo.getLib().reset() } catch (_: Exception) { }
        _state.value = _state.value.copy(
            messages = emptyList(), isGenerating = false,
            tokPerSec = 0f, ttft = 0L, genTokens = 0, error = null
        )
    }

    override fun onCleared() {
        _state.value.pendingImage?.recycle()
        super.onCleared()
    }
}
