package com.mp.ai_systems

import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.mp.ai_gguf.GGUFNativeLib
import com.mp.ai_gguf.models.DecodingMetrics
import com.mp.ai_gguf.models.StreamCallback
import com.mp.ai_gguf.toolcalling.ToolCallManager
import com.mp.ai_gguf.toolcalling.tool
import com.mp.ai_systems.ui.theme.AiSystemsTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.*
import android.util.Log

sealed class Message {
    data class User(val text: String) : Message()
    data class Assistant(val text: String) : Message()
    data class Reasoning(val text: String) : Message() // Model's thinking before tool call
    data class ToolCall(val name: String, val args: String, val result: String?) : Message()
    data class Error(val text: String) : Message()
    data class System(val text: String) : Message() // System notifications
}

// ViewModel with new SDK
class ToolCallingViewModel : ViewModel() {
    companion object {
        private const val TAG = "ToolCallingVM"
    }

    private val gguf = GGUFNativeLib()
    private val toolCallManager = ToolCallManager(gguf)

    private val _messages = MutableStateFlow<List<Message>>(emptyList())
    val messages: StateFlow<List<Message>> = _messages.asStateFlow()

    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    private val _modelLoaded = MutableStateFlow(false)
    val modelLoaded: StateFlow<Boolean> = _modelLoaded.asStateFlow()

    private val _currentResponse = MutableStateFlow("")
    val currentResponse: StateFlow<String> = _currentResponse.asStateFlow()

    private val _progressMessage = MutableStateFlow("")
    val progressMessage: StateFlow<String> = _progressMessage.asStateFlow()

    init {
        // Register tools using the new SDK
        registerTools()
    }

    private fun registerTools() {
        toolCallManager.registerTools(
            tool("get_current_time", "Get the current date and/or time") {
                stringParam(
                    "format",
                    "Format type: 'full' for both date and time, 'time' for time only, 'date' for date only",
                    required = false,
                    enum = listOf("full", "time", "date")
                )
            },
            tool("show_toast", "Display a toast message to the user") {
                stringParam("message", "The message to display", required = true)
                stringParam(
                    "duration",
                    "Duration to show the toast",
                    required = false,
                    enum = listOf("short", "long")
                )
            },
            tool("get_device_info", "Get information about the Android device") {
                stringParam(
                    "info_type",
                    "Type of information: 'basic' for device name, 'system' for OS info, 'all' for everything",
                    required = false,
                    enum = listOf("basic", "system", "all")
                )
            }
        )
    }

    fun loadModel(context: android.content.Context, uri: Uri) {
        Log.i(TAG, "loadModel: Starting model load from URI: $uri")
        viewModelScope.launch(Dispatchers.IO) {
            try {
                Log.d(TAG, "Opening file descriptor...")
                val fd = context.contentResolver.openFileDescriptor(uri, "r")?.detachFd()
                    ?: throw Exception("Failed to open file descriptor")

                Log.i(TAG, "File descriptor obtained: $fd")
                Log.d(TAG, "Loading model from FD...")

                val success = gguf.nativeLoadModelFromFd(
                    fd = fd,
                    threads = 4,
                    ctxSize = 2048,
                    temp = 0.7f,
                    topK = 40,
                    topP = 0.9f,
                    minP = 0.05f,
                    mirostat = 0,
                    mirostatTau = 5.0f,
                    mirostatEta = 0.1f,
                    seed = -1
                )

                Log.i(TAG, "nativeLoadModelFromFd returned: $success")

                if (success) {
                    Log.d(TAG, "Checking model compatibility...")
                    // Check if model supports tool calling
                    if (!toolCallManager.isModelCompatible()) {
                        val arch = toolCallManager.modelArchitecture
                        Log.w(TAG, "Model not compatible for tool calling. Architecture: $arch")
                        withContext(Dispatchers.Main) {
                            _messages.value += Message.Error(
                                "Model loaded but tool calling is not supported. " +
                                "Model architecture: $arch. Only Qwen models support tool calling."
                            )
                        }
                        _modelLoaded.value = true
                        return@launch
                    }

                    Log.i(TAG, "Model is compatible (Qwen). Enabling tool calling...")
                    // Enable tool calling with multi-step reasoning
                    if (toolCallManager.enable(enableMultiStep = true)) {
                        Log.i(TAG, "Tool calling enabled successfully")
                        _modelLoaded.value = true
                        withContext(Dispatchers.Main) {
                            _messages.value += Message.System(
                                "✅ Qwen model loaded with multi-step tool calling!\n\n" +
                                "Features:\n" +
                                "• Multi-step reasoning - Model explains its thinking\n" +
                                "• Automatic chaining - Tools execute in sequence\n" +
                                "• Context preservation - Results fed back automatically\n\n" +
                                "Available tools:\n" +
                                "• get_current_time - Get the current date/time\n" +
                                "• show_toast - Display a message\n" +
                                "• get_device_info - Get device information\n\n" +
                                "Try: 'Show me the time and then device info'"
                            )
                        }
                    } else {
                        Log.e(TAG, "Failed to enable tool calling: ${toolCallManager.lastError}")
                        throw Exception("Failed to enable tool calling: ${toolCallManager.lastError}")
                    }
                } else {
                    Log.e(TAG, "Failed to load model from FD")
                    throw Exception("Failed to load model")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Exception in loadModel", e)
                withContext(Dispatchers.Main) {
                    _messages.value += Message.Error("Error loading model: ${e.message}")
                }
            }
        }
    }

    fun sendMessage(text: String, context: android.content.Context) {
        if (text.isBlank() || !_modelLoaded.value) {
            Log.w(TAG, "sendMessage: blank text or model not loaded")
            return
        }

        Log.i(TAG, "=== Starting new message: $text ===")
        viewModelScope.launch {
            // Add user message and reset conversation for new request
            _messages.value += Message.User(text)
            toolCallManager.resetConversation()
            toolCallManager.conversationHistory.addUserMessage(text)

            Log.d(TAG, "Conversation reset, starting generation loop")

            // Start generation loop (supports multi-step)
            executeGenerationLoop(text, context)
        }
    }

    private suspend fun executeGenerationLoop(
        initialPrompt: String,
        context: android.content.Context,
        isContinuation: Boolean = false
    ) {
        Log.d(TAG, "executeGenerationLoop: isContinuation=$isContinuation")
        _isLoading.value = true
        _currentResponse.value = ""
        _progressMessage.value = if (isContinuation) "Continuing..." else "Generating..."

        try {
            var fullResponse = ""
            var isToolCall = false
            var isReasoning = false
            var tokenCount = 0

            // Build prompt (initial or continuation)
            val prompt = if (isContinuation) {
                val contPrompt = toolCallManager.buildContinuationPrompt()
                Log.d(TAG, "Continuation prompt: $contPrompt")
                contPrompt
            } else {
                Log.d(TAG, "Initial prompt: $initialPrompt")
                initialPrompt
            }

            val callback = object : StreamCallback {
                override fun onToken(token: String) {
                    tokenCount++
                    fullResponse += token
                    _currentResponse.value = fullResponse
                    _progressMessage.value = "Tokens: $tokenCount"

                    if (tokenCount % 10 == 0) {
                        Log.v(TAG, "Token $tokenCount: Current response length=${fullResponse.length}")
                    }

                    // Detect if this looks like a tool call (starts with {)
                    if (fullResponse.trim().startsWith("{")) {
                        isToolCall = true
                        Log.d(TAG, "Detected tool call pattern")
                    } else if (fullResponse.isNotBlank() && !isToolCall) {
                        isReasoning = true
                    }
                }

                override fun onToolCall(name: String, argsJson: String) {
                    Log.i(TAG, "onToolCall: name=$name, args=$argsJson")
                    isToolCall = true
                    viewModelScope.launch {
                        _currentResponse.value = ""
                        _progressMessage.value = "Executing tool: $name"

                        // Show reasoning if any
                        val reasoningText = fullResponse.replace(argsJson, "").trim()
                        Log.d(TAG, "Reasoning text: $reasoningText")

                        if (reasoningText.isNotBlank() && !reasoningText.startsWith("{")) {
                            Log.i(TAG, "Adding reasoning message")
                            _messages.value += Message.Reasoning(reasoningText)
                            toolCallManager.conversationHistory.addAssistantReasoning(reasoningText)
                        }

                        // Parse and execute tool call
                        val toolCall = toolCallManager.parseToolCall(argsJson)
                        if (toolCall != null) {
                            Log.i(TAG, "Executing tool: ${toolCall.name}")
                            val result = executeToolCall(toolCall, context)
                            Log.i(TAG, "Tool result: $result")

                            // Record in history
                            toolCallManager.recordToolResult(toolCall.name, result, true)

                            // Show in UI
                            _messages.value += Message.ToolCall(toolCall.name, argsJson, result)

                            // Auto-continue if needed
                            val shouldCont = toolCallManager.shouldContinue()
                            val toolCount = toolCallManager.conversationHistory.getToolCallCount()
                            Log.d(TAG, "shouldContinue=$shouldCont, toolCount=$toolCount, maxSteps=${toolCallManager.maxAutoSteps}")

                            if (shouldCont) {
                                Log.i(TAG, "Auto-continuing to next step")
                                _messages.value += Message.System("🔄 Continuing to next step...")
                                executeGenerationLoop(initialPrompt, context, isContinuation = true)
                            } else {
                                Log.i(TAG, "No more continuation needed")
                                _isLoading.value = false
                                _progressMessage.value = ""
                            }
                        } else {
                            Log.e(TAG, "Failed to parse tool call: ${toolCallManager.lastError}")
                            _messages.value += Message.Error("Failed to parse tool call: ${toolCallManager.lastError}")
                            _isLoading.value = false
                            _progressMessage.value = ""
                        }
                    }
                }

                override fun onError(error: String) {
                    Log.e(TAG, "onError: $error")
                    viewModelScope.launch {
                        _messages.value += Message.Error(error)
                        _currentResponse.value = ""
                        _isLoading.value = false
                        _progressMessage.value = ""
                    }
                }

                override fun onDone() {
                    Log.d(TAG, "onDone: isToolCall=$isToolCall, fullResponse.length=${fullResponse.length}, isContinuation=$isContinuation")
                    viewModelScope.launch {
                        if (!isToolCall && fullResponse.isNotBlank()) {
                            if (isReasoning && isContinuation) {
                                // This is likely the final answer after tools
                                Log.i(TAG, "Adding final assistant response (continuation)")
                                _messages.value += Message.Assistant(fullResponse)
                                toolCallManager.conversationHistory.addAssistantResponse(fullResponse)
                            } else if (!isContinuation) {
                                // Regular response without tools
                                Log.i(TAG, "Adding regular assistant response")
                                _messages.value += Message.Assistant(fullResponse)
                            } else {
                                Log.w(TAG, "onDone: Response not added - isToolCall=$isToolCall, fullResponse='$fullResponse'")
                            }
                        }
                        _currentResponse.value = ""
                        _isLoading.value = false
                        _progressMessage.value = ""
                        Log.i(TAG, "=== Generation complete ===")
                    }
                }

                override fun onMetrics(metrics: DecodingMetrics) {
                    Log.d(TAG, "Metrics: tokens=${metrics.totalTokens}, tps=${metrics.tokensPerSecond}")
                }
            }

            Log.i(TAG, "Starting nativeGenerateStream...")
            withContext(Dispatchers.IO) {
                val success = gguf.nativeGenerateStream(prompt, 256, callback)
                Log.d(TAG, "nativeGenerateStream returned: $success")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Exception in executeGenerationLoop", e)
            _messages.value += Message.Error("Error: ${e.message}")
            _isLoading.value = false
            _currentResponse.value = ""
            _progressMessage.value = ""
        }
    }

    private suspend fun executeToolCall(
        toolCall: com.mp.ai_gguf.toolcalling.ToolCall,
        context: android.content.Context
    ): String {
        return withContext(Dispatchers.IO) {
            try {
                when (toolCall.name) {
                    "get_current_time" -> {
                        val format = toolCall.getString("format", "full")
                        getCurrentTime(format)
                    }
                    "show_toast" -> {
                        val message = toolCall.getString("message", "Hello!")
                        val duration = toolCall.getString("duration", "short")
                        showToast(context, message, duration)
                    }
                    "get_device_info" -> {
                        val infoType = toolCall.getString("info_type", "basic")
                        getDeviceInfo(infoType)
                    }
                    else -> "Unknown tool: ${toolCall.name}"
                }
            } catch (e: Exception) {
                "Error executing tool: ${e.message}"
            }
        }
    }

    private fun getCurrentTime(format: String): String {
        val calendar = Calendar.getInstance()
        return when (format) {
            "time" -> SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(calendar.time)
            "date" -> SimpleDateFormat("EEEE, MMMM dd, yyyy", Locale.getDefault()).format(calendar.time)
            else -> SimpleDateFormat("EEEE, MMMM dd, yyyy 'at' HH:mm:ss", Locale.getDefault()).format(calendar.time)
        }
    }

    private suspend fun showToast(context: android.content.Context, message: String, duration: String): String {
        withContext(Dispatchers.Main) {
            val toastDuration = if (duration == "long") Toast.LENGTH_LONG else Toast.LENGTH_SHORT
            Toast.makeText(context, message, toastDuration).show()
        }
        return "Toast displayed: $message"
    }

    private fun getDeviceInfo(infoType: String): String {
        val info = StringBuilder()

        when (infoType) {
            "basic" -> {
                info.append("Device: ${android.os.Build.BRAND} ${android.os.Build.MODEL}\n")
                info.append("Manufacturer: ${android.os.Build.MANUFACTURER}")
            }
            "system" -> {
                info.append("Android Version: ${android.os.Build.VERSION.RELEASE}\n")
                info.append("API Level: ${android.os.Build.VERSION.SDK_INT}\n")
                info.append("ABIs: ${android.os.Build.SUPPORTED_ABIS.joinToString(", ")}")
            }
            "all" -> {
                info.append("Device: ${android.os.Build.BRAND} ${android.os.Build.MODEL}\n")
                info.append("Manufacturer: ${android.os.Build.MANUFACTURER}\n")
                info.append("Android: ${android.os.Build.VERSION.RELEASE} (API ${android.os.Build.VERSION.SDK_INT})")
            }
            else -> info.append("Device: ${android.os.Build.BRAND} ${android.os.Build.MODEL}")
        }

        return info.toString()
    }

    override fun onCleared() {
        super.onCleared()
        gguf.nativeRelease()
    }
}

// UI
class ToolCallingActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            AiSystemsTheme {
                ToolCallingScreen()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ToolCallingScreen(viewModel: ToolCallingViewModel = viewModel()) {
    val context = LocalContext.current
    val messages by viewModel.messages.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()
    val modelLoaded by viewModel.modelLoaded.collectAsState()
    val currentResponse by viewModel.currentResponse.collectAsState()
    val progressMessage by viewModel.progressMessage.collectAsState()

    var inputText by remember { mutableStateOf("Show me the time") }
    val listState = rememberLazyListState()

    // File picker launcher
    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        uri?.let {
            viewModel.loadModel(context, it)
        }
    }

    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.size - 1)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Tool Calling Demo") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
            // Model picker button
            if (!modelLoaded) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Button(
                        onClick = { filePickerLauncher.launch(arrayOf("*/*")) }
                    ) {
                        Text("Pick Model from SAF")
                    }
                }
            } else {
                // Progress indicator
                if (isLoading && progressMessage.isNotBlank()) {
                    LinearProgressIndicator(
                        modifier = Modifier.fillMaxWidth()
                    )
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 8.dp, vertical = 4.dp),
                        horizontalArrangement = Arrangement.Center
                    ) {
                        Text(
                            text = progressMessage,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                        )
                    }
                }

                // Chat messages
                LazyColumn(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                        .padding(8.dp),
                    state = listState
                ) {
                    items(messages) { message ->
                        ChatMessageItem(message)
                    }

                    // Show current streaming response
                    if (currentResponse.isNotBlank()) {
                        item {
                            ChatMessageItem(Message.Assistant(currentResponse))
                        }
                    }
                }

                // Input field
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    OutlinedTextField(
                        value = inputText,
                        onValueChange = { inputText = it },
                        modifier = Modifier.weight(1f),
                        placeholder = { Text("Ask me something...") },
                        enabled = !isLoading
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    IconButton(
                        onClick = {
                            if (inputText.isNotBlank()) {
                                viewModel.sendMessage(inputText, context)
                                inputText = ""
                            }
                        },
                        enabled = !isLoading && inputText.isNotBlank()
                    ) {
                        Icon(Icons.AutoMirrored.Filled.Send, "Send")
                    }
                }
            }
        }
    }
}

@Composable
fun ChatMessageItem(message: Message) {
    val alignment = when (message) {
        is Message.User -> Alignment.CenterEnd
        else -> Alignment.CenterStart
    }

    val backgroundColor = when (message) {
        is Message.User -> MaterialTheme.colorScheme.primaryContainer
        is Message.Assistant -> MaterialTheme.colorScheme.secondaryContainer
        is Message.Reasoning -> MaterialTheme.colorScheme.surfaceVariant
        is Message.ToolCall -> MaterialTheme.colorScheme.tertiaryContainer
        is Message.Error -> MaterialTheme.colorScheme.errorContainer
        is Message.System -> MaterialTheme.colorScheme.surfaceVariant
    }

    val textColor = when (message) {
        is Message.User -> MaterialTheme.colorScheme.onPrimaryContainer
        is Message.Assistant -> MaterialTheme.colorScheme.onSecondaryContainer
        is Message.Reasoning -> MaterialTheme.colorScheme.onSurfaceVariant
        is Message.ToolCall -> MaterialTheme.colorScheme.onTertiaryContainer
        is Message.Error -> MaterialTheme.colorScheme.onErrorContainer
        is Message.System -> MaterialTheme.colorScheme.onSurfaceVariant
    }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        contentAlignment = alignment
    ) {
        Surface(
            shape = RoundedCornerShape(12.dp),
            color = backgroundColor,
            modifier = Modifier.widthIn(max = 300.dp)
        ) {
            Column(
                modifier = Modifier.padding(12.dp)
            ) {
                when (message) {
                    is Message.User -> {
                        Text(
                            text = message.text,
                            color = textColor
                        )
                    }
                    is Message.Assistant -> {
                        Text(
                            text = message.text,
                            color = textColor
                        )
                    }
                    is Message.Reasoning -> {
                        Text(
                            text = "💭 ${message.text}",
                            color = textColor,
                            fontStyle = androidx.compose.ui.text.font.FontStyle.Italic,
                            fontSize = 13.sp
                        )
                    }
                    is Message.ToolCall -> {
                        Text(
                            text = "🔧 Tool: ${message.name}",
                            fontWeight = FontWeight.Bold,
                            color = textColor,
                            fontSize = 12.sp
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        message.result?.let {
                            Text(
                                text = "✅ $it",
                                color = textColor,
                                fontSize = 14.sp
                            )
                        }
                    }
                    is Message.System -> {
                        Text(
                            text = message.text,
                            color = textColor,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Medium
                        )
                    }
                    is Message.Error -> {
                        Text(
                            text = "❌ ${message.text}",
                            color = textColor
                        )
                    }
                }
            }
        }
    }
}