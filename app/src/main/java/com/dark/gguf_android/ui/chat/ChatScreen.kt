package com.dark.gguf_android.ui.chat

import android.annotation.SuppressLint
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.wrapContentWidth
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Snackbar
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel

@SuppressLint("DefaultLocale")
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(vm: ChatViewModel = viewModel()) {
    val state by vm.state.collectAsState()
    val context = LocalContext.current
    val listState = rememberLazyListState()

    // Track which picker type we're waiting for
    var pendingPickerType by remember { mutableStateOf("") } // "model", "vision", "image", "doc"

    // Single SAF launcher that routes based on pendingPickerType
    val safLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        uri?.let {
            when (pendingPickerType) {
                "model" -> {
                    val fd = context.contentResolver.openFileDescriptor(it, "r")?.detachFd()
                    if (fd != null) vm.loadModel(fd)
                }
                "vision" -> {
                    val fd = context.contentResolver.openFileDescriptor(it, "r")?.detachFd()
                    if (fd != null) vm.loadVisionModel(fd)
                }
                "image" -> vm.attachImage(context.contentResolver, it)
                "doc" -> vm.ragImportFile(context.contentResolver, it)
            }
        }
        pendingPickerType = ""
    }

    // Auto-scroll
    LaunchedEffect(state.messages.size, state.messages.lastOrNull()?.text?.length) {
        if (state.messages.isNotEmpty()) {
            listState.animateScrollToItem(state.messages.size - 1)
        }
    }

    // Load model dialog
    if (state.showLoadDialog) {
        LoadModelDialog(
            modelLoaded = state.modelLoaded,
            visionLoaded = state.visionLoaded,
            onLoadModel = {
                pendingPickerType = "model"
                vm.hideLoadDialog()
                safLauncher.launch(arrayOf("*/*"))
            },
            onLoadVision = {
                pendingPickerType = "vision"
                vm.hideLoadDialog()
                safLauncher.launch(arrayOf("*/*"))
            },
            onDismiss = { vm.hideLoadDialog() }
        )
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("Luna", style = MaterialTheme.typography.titleLarge)
                        if (state.modelLoaded) {
                            Text(
                                state.modelName,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                        }
                    }
                },
                actions = {
                    if (state.visionLoaded) {
                        Badge("VLM", Color(0xFF4CAF50))
                        Spacer(Modifier.width(4.dp))
                    }
                    if (state.ragDocs > 0) {
                        Badge(
                            if (state.ragReady) "RAG" else "RAG...",
                            if (state.ragReady) Color(0xFF2196F3) else Color(0xFFFFC107)
                        )
                        Spacer(Modifier.width(4.dp))
                    }
                    if (state.webSearchOn) {
                        Badge("Web", Color(0xFF9C27B0))
                        Spacer(Modifier.width(4.dp))
                    }
                    IconButton(onClick = { vm.showLoadDialog() }) {
                        Icon(
                            if (state.modelLoaded) Icons.Default.CheckCircle else Icons.Default.Add,
                            contentDescription = "Load Model",
                            tint = if (state.modelLoaded) Color(0xFF4CAF50)
                            else MaterialTheme.colorScheme.onSurface
                        )
                    }
                    IconButton(onClick = { vm.clearChat() }) {
                        Icon(Icons.Default.Delete, contentDescription = "Clear")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        }
    ) { padding ->
        Box(Modifier.fillMaxSize().padding(padding).imePadding()) {
            Column(Modifier.fillMaxSize()) {
                // Messages
                LazyColumn(
                    state = listState,
                    modifier = Modifier.weight(1f).fillMaxWidth(),
                    contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    if (state.messages.isEmpty()) {
                        item { WelcomeCard(state.modelLoaded) }
                    }
                    items(state.messages) { msg ->
                        MessageBubble(msg)
                    }
                }

                // Loading stage progress bar
                AnimatedVisibility(
                    visible = state.loadingStage != LoadingStage.NONE,
                    enter = fadeIn() + slideInVertically { it },
                    exit = fadeOut() + slideOutVertically { it }
                ) {
                    StageProgressBar(state.loadingStage)
                }

                // Metrics
                AnimatedVisibility(
                    visible = state.isGenerating || state.genTokens > 0,
                    enter = fadeIn(), exit = fadeOut()
                ) {
                    MetricsBar(state)
                }

                // Image preview
                state.pendingImage?.let { bmp ->
                    ImagePreview(bmp) { vm.clearImage() }
                }

                // Input area
                InputBar(
                    text = state.inputText,
                    onTextChange = { vm.updateInput(it) },
                    onSend = { vm.send() },
                    onStop = { vm.stop() },
                    isGenerating = state.isGenerating,
                    modelLoaded = state.modelLoaded,
                    visionLoaded = state.visionLoaded,
                    webSearchOn = state.webSearchOn,
                    hasImage = state.pendingImage != null,
                    onToggleWeb = { vm.toggleWebSearch() },
                    onAttachImage = {
                        pendingPickerType = "image"
                        safLauncher.launch(arrayOf("image/*"))
                    },
                    onAttachDoc = {
                        pendingPickerType = "doc"
                        safLauncher.launch(arrayOf("text/*", "application/pdf", "*/*"))
                    }
                )
            }

            // Error snackbar
            AnimatedVisibility(
                visible = state.error != null,
                modifier = Modifier.align(Alignment.BottomCenter).padding(16.dp),
                enter = fadeIn(), exit = fadeOut()
            ) {
                state.error?.let { err ->
                    Snackbar(
                        action = {
                            TextButton(onClick = { vm.dismissError() }) { Text("OK") }
                        }
                    ) { Text(err, maxLines = 2, overflow = TextOverflow.Ellipsis) }
                }
            }
        }
    }
}

// ── Components ───────────────────────────────────────────────────────────────

@Composable
private fun StageProgressBar(stage: LoadingStage) {
    val label = when (stage) {
        LoadingStage.LOADING_MODEL   -> "Loading model..."
        LoadingStage.LOADING_VISION  -> "Loading vision model..."
        LoadingStage.EMBEDDING_RAG   -> "Embedding documents..."
        LoadingStage.SEARCHING_WEB   -> "Searching the web..."
        LoadingStage.SEARCHING_DOCS  -> "Searching documents..."
        LoadingStage.ANALYZING_IMAGE -> "Analyzing image..."
        LoadingStage.GENERATING      -> "Generating..."
        LoadingStage.NONE            -> ""
    }
    val color = when (stage) {
        LoadingStage.LOADING_MODEL, LoadingStage.LOADING_VISION -> Color(0xFF2196F3)
        LoadingStage.EMBEDDING_RAG, LoadingStage.SEARCHING_DOCS -> Color(0xFF9C27B0)
        LoadingStage.SEARCHING_WEB   -> Color(0xFFFF9800)
        LoadingStage.ANALYZING_IMAGE -> Color(0xFF4CAF50)
        LoadingStage.GENERATING      -> MaterialTheme.colorScheme.primary
        LoadingStage.NONE            -> MaterialTheme.colorScheme.primary
    }

    Column(
        Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f))
            .padding(horizontal = 16.dp, vertical = 6.dp)
    ) {
        Text(
            label,
            style = MaterialTheme.typography.labelSmall,
            color = color
        )
        Spacer(Modifier.height(4.dp))
        LinearProgressIndicator(
            modifier = Modifier
                .fillMaxWidth()
                .height(3.dp)
                .clip(RoundedCornerShape(2.dp)),
            color = color,
            trackColor = color.copy(alpha = 0.15f)
        )
    }
}

@Composable
private fun Badge(label: String, color: Color) {
    Surface(
        shape = RoundedCornerShape(8.dp),
        color = color.copy(alpha = 0.15f)
    ) {
        Text(
            label,
            Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
            style = MaterialTheme.typography.labelSmall,
            color = color
        )
    }
}

@Composable
private fun WelcomeCard(modelLoaded: Boolean) {
    Column(
        Modifier.fillMaxWidth().padding(vertical = 48.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            "hey, i'm luna",
            style = MaterialTheme.typography.headlineMedium,
            color = MaterialTheme.colorScheme.primary
        )
        Spacer(Modifier.height(8.dp))
        Text(
            if (modelLoaded) "what's on your mind?"
            else "tap + to load a model",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun MessageBubble(msg: ChatMessage) {
    // System info messages — small centered italic
    if (msg.role == "system") {
        Row(
            Modifier.fillMaxWidth().padding(vertical = 2.dp),
            horizontalArrangement = Arrangement.Center
        ) {
            Text(
                text = msg.text,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )
        }
        return
    }

    val isUser = msg.role == "user"
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
    ) {
        Surface(
            shape = RoundedCornerShape(
                topStart = 16.dp, topEnd = 16.dp,
                bottomStart = if (isUser) 16.dp else 4.dp,
                bottomEnd = if (isUser) 4.dp else 16.dp
            ),
            color = if (isUser) MaterialTheme.colorScheme.primary
            else MaterialTheme.colorScheme.surfaceVariant,
            modifier = Modifier.widthIn(max = 300.dp)
        ) {
            Text(
                text = msg.text.ifEmpty { if (msg.isStreaming) "..." else "" },
                modifier = Modifier.padding(12.dp).animateContentSize(),
                color = if (isUser) MaterialTheme.colorScheme.onPrimary
                else MaterialTheme.colorScheme.onSurfaceVariant,
                style = MaterialTheme.typography.bodyMedium
            )
        }
    }
}

@SuppressLint("DefaultLocale")
@Composable
private fun MetricsBar(state: ChatUiState) {
    Row(
        Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
            .padding(horizontal = 16.dp, vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            "${state.genTokens} tok",
            style = MaterialTheme.typography.labelSmall.copy(fontFamily = FontFamily.Monospace),
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            String.format("%.1f t/s", state.tokPerSec),
            style = MaterialTheme.typography.labelSmall.copy(fontFamily = FontFamily.Monospace),
            color = if (state.tokPerSec >= 5f) Color(0xFF4CAF50) else Color(0xFFFFC107)
        )
        Text(
            "TTFT ${state.ttft}ms",
            style = MaterialTheme.typography.labelSmall.copy(fontFamily = FontFamily.Monospace),
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun ImagePreview(bmp: android.graphics.Bitmap, onRemove: () -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Image(
            bitmap = bmp.asImageBitmap(),
            contentDescription = "Attached image",
            modifier = Modifier.size(56.dp).clip(RoundedCornerShape(8.dp)),
            contentScale = ContentScale.Crop
        )
        Spacer(Modifier.width(8.dp))
        Text(
            "${bmp.width}x${bmp.height}",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(Modifier.weight(1f))
        IconButton(onClick = onRemove, modifier = Modifier.size(32.dp)) {
            Icon(Icons.Default.Close, "Remove", Modifier.size(16.dp))
        }
    }
}

@Composable
private fun InputBar(
    text: String,
    onTextChange: (String) -> Unit,
    onSend: () -> Unit,
    onStop: () -> Unit,
    isGenerating: Boolean,
    modelLoaded: Boolean,
    visionLoaded: Boolean,
    webSearchOn: Boolean,
    hasImage: Boolean,
    onToggleWeb: () -> Unit,
    onAttachImage: () -> Unit,
    onAttachDoc: () -> Unit
) {
    Surface(
        tonalElevation = 2.dp,
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(Modifier.padding(horizontal = 12.dp, vertical = 8.dp)) {
            // Feature chips row
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                FilterChip(
                    selected = webSearchOn,
                    onClick = onToggleWeb,
                    label = { Text("Web", style = MaterialTheme.typography.labelSmall) },
                    leadingIcon = {
                        Icon(Icons.Default.Search, null, Modifier.size(14.dp))
                    },
                    colors = FilterChipDefaults.filterChipColors(
                        selectedContainerColor = MaterialTheme.colorScheme.primaryContainer
                    ),
                    modifier = Modifier.height(28.dp)
                )
                AssistChip(
                    onClick = onAttachDoc,
                    label = { Text("Doc", style = MaterialTheme.typography.labelSmall) },
                    modifier = Modifier.height(28.dp)
                )
                if (visionLoaded && !hasImage) {
                    AssistChip(
                        onClick = onAttachImage,
                        label = { Text("Image", style = MaterialTheme.typography.labelSmall) },
                        modifier = Modifier.height(28.dp)
                    )
                }
            }

            Spacer(Modifier.height(6.dp))

            // Input row
            Row(
                Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedTextField(
                    value = text,
                    onValueChange = onTextChange,
                    modifier = Modifier.weight(1f),
                    placeholder = {
                        Text(
                            if (hasImage) "describe this image..."
                            else "message luna...",
                            style = MaterialTheme.typography.bodyMedium
                        )
                    },
                    shape = RoundedCornerShape(24.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = MaterialTheme.colorScheme.primary,
                        unfocusedBorderColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)
                    ),
                    maxLines = 4,
                    textStyle = MaterialTheme.typography.bodyMedium
                )
                Spacer(Modifier.width(8.dp))
                if (isGenerating) {
                    FilledIconButton(
                        onClick = onStop,
                        shape = CircleShape,
                        colors = IconButtonDefaults.filledIconButtonColors(
                            containerColor = Color(0xFFF44336)
                        )
                    ) {
                        Icon(Icons.Default.Close, "Stop", tint = Color.White)
                    }
                } else {
                    FilledIconButton(
                        onClick = onSend,
                        enabled = text.isNotBlank() && modelLoaded,
                        shape = CircleShape
                    ) {
                        Icon(Icons.AutoMirrored.Filled.Send, "Send")
                    }
                }
            }
        }
    }
}

// ── Load Model Dialog ────────────────────────────────────────────────────────

@Composable
private fun LoadModelDialog(
    modelLoaded: Boolean,
    visionLoaded: Boolean,
    onLoadModel: () -> Unit,
    onLoadVision: () -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Load Model") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedCard(
                    onClick = onLoadModel,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        Modifier.padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text("Text Model", style = MaterialTheme.typography.titleSmall)
                            Text(
                                "Load a GGUF language model",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        if (modelLoaded) {
                            Icon(
                                Icons.Default.CheckCircle, null,
                                tint = Color(0xFF4CAF50),
                                modifier = Modifier.size(20.dp)
                            )
                        }
                    }
                }

                OutlinedCard(
                    onClick = onLoadVision,
                    enabled = modelLoaded,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        Modifier.padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(
                                "Vision Model (VLM)",
                                style = MaterialTheme.typography.titleSmall,
                                color = if (modelLoaded)
                                    MaterialTheme.colorScheme.onSurface
                                else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.38f)
                            )
                            Text(
                                if (modelLoaded) "Load mmproj GGUF for image understanding"
                                else "Load text model first",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        if (visionLoaded) {
                            Icon(
                                Icons.Default.CheckCircle, null,
                                tint = Color(0xFF4CAF50),
                                modifier = Modifier.size(20.dp)
                            )
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) { Text("Close") }
        }
    )
}
