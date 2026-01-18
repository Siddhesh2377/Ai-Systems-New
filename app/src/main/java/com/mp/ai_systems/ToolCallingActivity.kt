package com.mp.ai_systems

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
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
import com.mp.ai_systems.ui.theme.AiSystemsTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.*

// Data classes
data class Tools(
    val toolName: String,
    val description: String,
    val args: Map<String, Any?>
)

sealed class Message {
    data class User(val text: String) : Message()
    data class Assistant(val text: String) : Message()
    data class ToolCall(val name: String, val args: String, val result: String?) : Message()
    data class Error(val text: String) : Message()
}

// ViewModel
class ToolCallingViewModel : ViewModel() {
    private val gguf = GGUFNativeLib()

    private val _messages = MutableStateFlow<List<Message>>(emptyList())
    val messages: StateFlow<List<Message>> = _messages.asStateFlow()

    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    private val _modelLoaded = MutableStateFlow(false)
    val modelLoaded: StateFlow<Boolean> = _modelLoaded.asStateFlow()

    private val _currentResponse = MutableStateFlow("")
    val currentResponse: StateFlow<String> = _currentResponse.asStateFlow()

    private val toolCallingChatTemplate = """
        {%- if professional is defined or emotional is defined -%}
        <|im_start|>system
        The assistant should modulate style accordingly while staying accurate.
        <|im_end|>
        {%- endif -%}
        {%- if gbnf is defined and gbnf|length > 0 -%}
        <|im_start|>system
        The assistant's NEXT message MUST conform to the following GBNF grammar.
        If a token would violate the grammar, do not emit it.
        <GBNF>
        {{ gbnf }}
        </GBNF>
        <|im_end|>
        {%- endif -%}
        {%- for m in messages -%}
        <|im_start|>{{ m['role'] }}
        {{ m['content'] }}
        <|im_end|>
        {%- endfor -%}
        {%- if add_generation_prompt -%}
        <|im_start|>assistant
        {%- endif -%}
    """.trimIndent()

    private val toolCallingSystemPrompt = """
        You are a function-calling assistant. When tools are available, respond ONLY with a JSON object in this EXACT format:

        {
          "tool_calls": [{
            "name": "toolName",
            "arguments": {
              "param1": "value1",
              "param2": "value2"
            }
          }]
        }

        CRITICAL RULES:
        1. Use "arguments" as an object containing all parameters
        2. NEVER put parameters directly in the tool_calls object
        3. NEVER include any text before or after the JSON
        4. The "arguments" field must be a JSON object, not a string
        5. Match parameter names exactly as defined in the tool schema

        If no tool is needed, respond with plain text.
    """.trimIndent()

    // Available tools
    private val availableTools = listOf(
        Tools(
            toolName = "get_current_time",
            description = "Get the current date and/or time. Use 'full' for both, 'time' for time only, 'date' for date only.",
            args = mapOf("format" to "full")
        ),
        Tools(
            toolName = "show_toast",
            description = "Display a toast message to the user",
            args = mapOf(
                "message" to "",
                "duration" to "short"
            )
        ),
        Tools(
            toolName = "get_device_info",
            description = "Get information about the Android device",
            args = mapOf("info_type" to "basic")
        )
    )

    fun loadModel(context: android.content.Context, uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val fd = context.contentResolver.openFileDescriptor(uri, "r")?.detachFd()
                    ?: throw Exception("Failed to open file descriptor")

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

                if (success) {
                    // Set system prompt and chat template
                    gguf.nativeSetSystemPrompt(toolCallingSystemPrompt)
                    gguf.nativeSetChatTemplate(toolCallingChatTemplate)

                    // Build and set tools JSON
                    val toolsJson = buildToolsJson()
                    gguf.nativeSetToolsJson(toolsJson)

                    _modelLoaded.value = true
                    withContext(Dispatchers.Main) {
                        _messages.value += Message.Assistant("Model loaded! I can now call tools. Try asking: 'What time is it?'")
                    }
                } else {
                    throw Exception("Failed to load model")
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    _messages.value += Message.Error("Error loading model: ${e.message}")
                }
            }
        }
    }

    fun sendMessage(text: String, context: android.content.Context) {
        if (text.isBlank() || !_modelLoaded.value) return

        viewModelScope.launch {
            _messages.value += Message.User(text)
            _isLoading.value = true
            _currentResponse.value = ""

            try {
                var fullResponse = ""
                var toolCallDetected = false

                val callback = object : StreamCallback {
                    override fun onToken(token: String) {
                        fullResponse += token
                        _currentResponse.value = fullResponse
                    }

                    override fun onToolCall(name: String, argsJson: String) {
                        toolCallDetected = true
                        viewModelScope.launch {
                            // Execute tool
                            val result = executeToolCall(name, argsJson, context)
                            _messages.value += Message.ToolCall(name, argsJson, result)
                            _currentResponse.value = ""
                        }
                    }

                    override fun onError(error: String) {
                        viewModelScope.launch {
                            _messages.value += Message.Error(error)
                            _currentResponse.value = ""
                        }
                    }

                    override fun onDone() {
                        viewModelScope.launch {
                            if (!toolCallDetected && fullResponse.isNotBlank()) {
                                _messages.value += Message.Assistant(fullResponse)
                            }
                            _currentResponse.value = ""
                            _isLoading.value = false
                        }
                    }

                    override fun onMetrics(metrics: DecodingMetrics) {
                        // Optional: handle metrics
                    }
                }

                withContext(Dispatchers.IO) {
                    gguf.nativeGenerateStream(text, 256, callback)
                }
            } catch (e: Exception) {
                _messages.value += Message.Error("Error: ${e.message}")
                _isLoading.value = false
                _currentResponse.value = ""
            }
        }
    }

    private suspend fun executeToolCall(
        name: String,
        argsJson: String,
        context: android.content.Context
    ): String {
        return withContext(Dispatchers.IO) {
            try {
                val args = JSONObject(argsJson)

                when (name) {
                    "get_current_time" -> {
                        val format = args.optString("format", "full")
                        getCurrentTime(format)
                    }
                    "show_toast" -> {
                        val message = args.optString("message", "Hello!")
                        val duration = args.optString("duration", "short")
                        showToast(context, message, duration)
                    }
                    "get_device_info" -> {
                        val infoType = args.optString("info_type", "basic")
                        getDeviceInfo(infoType)
                    }
                    else -> "Unknown tool: $name"
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

    private fun buildToolsJson(): String {
        val toolsArray = JSONArray()

        availableTools.forEach { tool ->
            val toolDef = toolDefinitionBuilder(tool)
            toolsArray.put(toolDef.getJSONObject(0))
        }

        return toolsArray.toString()
    }

    private fun toolDefinitionBuilder(tool: Tools): JSONArray {
        val properties = JSONObject()
        val required = mutableListOf<String>()

        tool.args.forEach { (key, value) ->
            val type = when (value) {
                is Int, is Double, is Float -> "number"
                is Boolean -> "boolean"
                else -> "string"
            }
            properties.put(key, JSONObject().put("type", type))
            if (value != null) required.add(key)
        }

        val parameters = JSONObject()
            .put("type", "object")
            .put("properties", properties)
            .put("required", JSONArray(required))

        val function = JSONObject()
            .put("name", tool.toolName)
            .put("description", tool.description)
            .put("parameters", parameters)

        return JSONArray().put(
            JSONObject()
                .put("type", "function")
                .put("function", function)
        )
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

    var inputText by remember { mutableStateOf("What time is it?") }
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
                        Icon(Icons.Default.Send, "Send")
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
        is Message.ToolCall -> MaterialTheme.colorScheme.tertiaryContainer
        is Message.Error -> MaterialTheme.colorScheme.errorContainer
    }

    val textColor = when (message) {
        is Message.User -> MaterialTheme.colorScheme.onPrimaryContainer
        is Message.Assistant -> MaterialTheme.colorScheme.onSecondaryContainer
        is Message.ToolCall -> MaterialTheme.colorScheme.onTertiaryContainer
        is Message.Error -> MaterialTheme.colorScheme.onErrorContainer
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