package com.dark.demon_system.ui.vlm

import android.graphics.BitmapFactory
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.compose.runtime.rememberCoroutineScope

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun VlmScreen(
    onBack: () -> Unit,
    vm: VlmViewModel = viewModel(),
) {
    val state by vm.state.collectAsState()
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var prompt by remember { mutableStateOf("Describe this image in detail.") }
    var imageBytes by remember { mutableStateOf<ByteArray?>(null) }
    var imageBitmap by remember { mutableStateOf<android.graphics.Bitmap?>(null) }

    val pickImage = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        scope.launch {
            withContext(Dispatchers.IO) {
                runCatching {
                    val bytes = context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
                    if (bytes != null) {
                        imageBytes = bytes
                        imageBitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                        // Pre-warm the ViT cache in the background so the first
                        // generate() call against this image hits VT cache and
                        // skips the ~9s vision encoder pass. Fire-and-forget.
                        vm.precomputeVisionFor(bytes)
                    }
                }
            }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("VLM Test", style = MaterialTheme.typography.titleLarge)
                        Text(
                            vm.spec.displayName,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onBack) { Text("←") }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                ),
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            ModelStatusCard(state = state, vm = vm)

            val routing by vm.routing.collectAsState()
            Text(
                routing,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                fontFamily = FontFamily.Monospace,
                modifier = Modifier.fillMaxWidth(),
            )
            val prewarm by vm.prewarmState.collectAsState()
            ImagePickerCard(
                bitmap = imageBitmap,
                prewarm = prewarm,
                onPick = {
                    pickImage.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                    )
                },
                onClear = {
                    imageBytes = null
                    imageBitmap = null
                    vm.resetPrewarm()
                },
            )

            val quality by vm.imageQuality.collectAsState()
            ImageQualityPicker(
                selected = quality,
                enabled = prewarm !is PrewarmState.InProgress &&
                          state !is VlmState.Generating,
                onSelected = { vm.setImageQuality(it) },
            )

            OutlinedTextField(
                value = prompt,
                onValueChange = { prompt = it },
                label = { Text("Prompt") },
                modifier = Modifier.fillMaxWidth(),
                minLines = 2,
                maxLines = 4,
            )

            val isReady  = state is VlmState.Ready || state is VlmState.GenerationDone
            val isBusy   = state is VlmState.Generating ||
                           state is VlmState.LoadingModel ||
                           state is VlmState.LoadingProjector ||
                           state is VlmState.Downloading
            val canRun   = isReady && imageBytes != null && prompt.isNotBlank() && !isBusy

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(
                    onClick = { vm.generate(prompt, imageBytes!!) },
                    enabled = canRun,
                    modifier = Modifier.weight(1f),
                ) { Text("Run") }

                OutlinedButton(
                    onClick = { vm.stopGeneration() },
                    enabled = state is VlmState.Generating,
                ) { Text("Stop") }
            }

            OutputCard(state = state)
        }
    }
}

@Composable
private fun ModelStatusCard(state: VlmState, vm: VlmViewModel) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant,
        ),
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Text("Model", style = MaterialTheme.typography.titleSmall)
            Spacer(Modifier.height(6.dp))

            when (state) {
                is VlmState.Downloading -> {
                    Text(
                        "Downloading ${state.fileName} (${state.fileIndex + 1}/2)",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Spacer(Modifier.height(4.dp))
                    LinearProgressIndicator(
                        progress = { state.overallPct },
                        modifier = Modifier.fillMaxWidth(),
                    )
                    Spacer(Modifier.height(4.dp))
                    Text(
                        "%s / %s @ %s/s • overall %.0f%%".format(
                            state.bytesDownloaded.humanBytes(),
                            state.totalBytes.takeIf { it > 0 }?.humanBytes() ?: "?",
                            state.bytesPerSecond.humanBytes(),
                            state.overallPct * 100,
                        ),
                        style = MaterialTheme.typography.bodySmall,
                    )
                }

                is VlmState.DownloadFailed ->
                    Text("Download failed: ${state.message}", color = MaterialTheme.colorScheme.error)

                VlmState.LoadingModel -> RowSpinner("Loading text model…")
                VlmState.LoadingProjector -> RowSpinner("Loading vision projector…")

                VlmState.Ready -> Text(
                    "Ready",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.primary,
                )

                is VlmState.Error ->
                    Text("Error: ${state.message}", color = MaterialTheme.colorScheme.error)

                else -> Text(
                    if (vm.modelDownloaded) "Downloaded — tap Load." else "Not downloaded.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }

            Spacer(Modifier.height(8.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (!vm.modelDownloaded) {
                    Button(
                        onClick = vm::startDownload,
                        enabled = state !is VlmState.Downloading,
                    ) { Text("Download") }
                }
                Button(
                    onClick = vm::loadAndPrepare,
                    enabled = vm.modelDownloaded &&
                              state !is VlmState.LoadingModel &&
                              state !is VlmState.LoadingProjector &&
                              state !is VlmState.Generating,
                ) { Text(if (vm.isLoaded && vm.isVlmLoaded) "Reload" else "Load") }
            }
        }
    }
}

@Composable
private fun RowSpinner(text: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
        Spacer(Modifier.size(8.dp))
        Text(text, style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun ImagePickerCard(
    bitmap: android.graphics.Bitmap?,
    prewarm: PrewarmState,
    onPick: () -> Unit,
    onClear: () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant,
        ),
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Text("Image", style = MaterialTheme.typography.titleSmall)
            Spacer(Modifier.height(6.dp))

            if (bitmap != null) {
                Box {
                    Image(
                        bitmap = bitmap.asImageBitmap(),
                        contentDescription = null,
                        contentScale = ContentScale.Fit,
                        modifier = Modifier
                            .fillMaxWidth()
                            .aspectRatio(bitmap.width.toFloat() / bitmap.height.toFloat())
                            .clip(RoundedCornerShape(8.dp)),
                    )
                    if (prewarm is PrewarmState.InProgress) {
                        val pct = if (prewarm.totalChunks > 0)
                            prewarm.chunkIndex.toFloat() / prewarm.totalChunks
                        else null

                        Box(
                            modifier = Modifier
                                .matchParentSize()
                                .clip(RoundedCornerShape(8.dp))
                                .background(MaterialTheme.colorScheme.scrim.copy(alpha = 0.6f)),
                            contentAlignment = Alignment.Center,
                        ) {
                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally,
                                modifier = Modifier.padding(16.dp),
                            ) {
                                if (pct != null) {
                                    CircularProgressIndicator(
                                        progress = { pct },
                                        color = MaterialTheme.colorScheme.inverseOnSurface,
                                        strokeWidth = 3.dp,
                                    )
                                } else {
                                    CircularProgressIndicator(
                                        color = MaterialTheme.colorScheme.inverseOnSurface,
                                        strokeWidth = 3.dp,
                                    )
                                }
                                Spacer(Modifier.height(10.dp))
                                Text(
                                    prewarm.stage,
                                    color = MaterialTheme.colorScheme.inverseOnSurface,
                                    style = MaterialTheme.typography.labelMedium,
                                )
                                if (prewarm.lastEncodeMs != null || prewarm.lastDecodeMs != null) {
                                    Spacer(Modifier.height(4.dp))
                                    val parts = buildList {
                                        prewarm.lastEncodeMs?.let { add("enc %.0f ms".format(it)) }
                                        prewarm.lastDecodeMs?.let { add("dec %.0f ms".format(it)) }
                                    }
                                    Text(
                                        parts.joinToString("  •  "),
                                        color = MaterialTheme.colorScheme.inverseOnSurface,
                                        style = MaterialTheme.typography.bodySmall,
                                    )
                                }
                            }
                        }
                    }
                }
                Spacer(Modifier.height(8.dp))
                PrewarmStatusLine(prewarm)
                Spacer(Modifier.height(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(
                        onClick = onPick,
                        enabled = prewarm !is PrewarmState.InProgress,
                    ) { Text("Replace") }
                    OutlinedButton(
                        onClick = onClear,
                        enabled = prewarm !is PrewarmState.InProgress,
                    ) { Text("Clear") }
                }
            } else {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(100.dp)
                        .clip(RoundedCornerShape(8.dp)),
                    contentAlignment = Alignment.Center,
                ) {
                    OutlinedButton(onClick = onPick) { Text("Pick image") }
                }
            }
        }
    }
}

@Composable
private fun PrewarmStatusLine(state: PrewarmState) {
    when (state) {
        PrewarmState.Idle -> {} // nothing
        is PrewarmState.InProgress -> Text(
            "${state.stage}  ·  first prompt will be fast",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        is PrewarmState.Done -> Text(
            if (state.cached) "⚡ Pre-warmed in ${state.durationMs} ms · ${state.totalChunks} chunks · ${state.nTokens} tok · ${state.blobBytes / 1024} KB cached"
            else              "Pre-warm finished in ${state.durationMs} ms — cache write skipped",
            style = MaterialTheme.typography.bodySmall,
            color = if (state.cached) MaterialTheme.colorScheme.primary
                    else              MaterialTheme.colorScheme.onSurfaceVariant,
        )
        is PrewarmState.Failed -> Text(
            "Pre-warm failed: ${state.message}",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.error,
        )
    }
}

@Composable
private fun OutputCard(state: VlmState) {
    val text = when (state) {
        is VlmState.Generating -> state.text.ifEmpty { "…" }
        is VlmState.GenerationDone -> state.text
        else -> ""
    }
    if (text.isEmpty() && state !is VlmState.Generating && state !is VlmState.GenerationDone) return

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(12.dp)) {
            Text("Output", style = MaterialTheme.typography.titleSmall)
            Spacer(Modifier.height(6.dp))
            Text(
                text = text,
                style = MaterialTheme.typography.bodyMedium,
                fontFamily = FontFamily.Monospace,
            )

            Spacer(Modifier.height(12.dp))
            HorizontalDivider()
            Spacer(Modifier.height(8.dp))
            MetricsGrid(state)
        }
    }
}

@Composable
private fun MetricsGrid(state: VlmState) {
    val m = when (state) {
        is VlmState.Generating -> MetricsTuple(
            state.vlmEncodeMs, state.vlmDecodeMs, state.imageTokens,
            null, null, state.vtCacheHit, state.vlmKvCacheHit,
        )
        is VlmState.GenerationDone -> MetricsTuple(
            state.vlmEncodeMs, state.vlmDecodeMs, state.imageTokens,
            state.metrics?.tokensPerSecond, state.metrics?.timeToFirstTokenMs,
            state.vtCacheHit, state.vlmKvCacheHit,
        )
        else -> MetricsTuple(null, null, null, null, null, null, null)
    }

    fun chip(hit: Boolean?): String = when (hit) {
        true  -> "⚡ cached"
        false -> "miss"
        null  -> "—"
    }

    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text("Metrics", style = MaterialTheme.typography.labelMedium)
        MetricsGridRow(
            left  = MetricsCellData("VLM-KV cache", chip(m.vlmKvHit), accent = m.vlmKvHit == true),
            right = MetricsCellData("VT cache",     chip(m.cacheHit), accent = m.cacheHit == true),
        )
        MetricsGridRow(
            left  = MetricsCellData("Encode (ViT)", m.encodeMs?.let { "%.0f ms".format(it) } ?: "—"),
            right = MetricsCellData("Decode (LLM prefill)", m.decodeMs?.let { "%.0f ms".format(it) } ?: "—"),
        )
        MetricsGridRow(
            left  = MetricsCellData("TTFT", m.ttft?.let { "%.0f ms".format(it) } ?: "—"),
            right = MetricsCellData("Throughput", m.tps?.let { "%.1f tok/s".format(it) } ?: "—"),
        )
        MetricsGridRow(
            left  = MetricsCellData("Image tokens", m.imgTokens?.toString() ?: "—"),
            right = MetricsCellData(" ", " "),
        )
    }
}

@Composable
private fun MetricsGridRow(left: MetricsCellData, right: MetricsCellData) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        MetricsCell(left,  Modifier.weight(1f))
        MetricsCell(right, Modifier.weight(1f))
    }
}

@Composable
private fun MetricsCell(data: MetricsCellData, modifier: Modifier = Modifier) {
    val bg = if (data.accent) MaterialTheme.colorScheme.primaryContainer
             else             MaterialTheme.colorScheme.surface
    val fg = if (data.accent) MaterialTheme.colorScheme.onPrimaryContainer
             else             MaterialTheme.colorScheme.onSurface
    Column(
        modifier = modifier
            .clip(RoundedCornerShape(8.dp))
            .background(bg)
            .padding(horizontal = 10.dp, vertical = 8.dp),
    ) {
        Text(
            data.label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(2.dp))
        Text(
            data.value,
            style = MaterialTheme.typography.bodyMedium,
            color = fg,
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.Medium,
        )
    }
}

private data class MetricsCellData(
    val label: String,
    val value: String,
    val accent: Boolean = false,
)

private data class MetricsTuple(
    val encodeMs: Float?,
    val decodeMs: Float?,
    val imgTokens: Int?,
    val tps: Float?,
    val ttft: Float?,
    val cacheHit: Boolean?,
    val vlmKvHit: Boolean?,
)

private fun Long.humanBytes(): String {
    val b = this
    if (b < 1024) return "$b B"
    val kb = b / 1024.0
    if (kb < 1024) return "%.0f KB".format(kb)
    val mb = kb / 1024.0
    if (mb < 1024) return "%.1f MB".format(mb)
    val gb = mb / 1024.0
    return "%.2f GB".format(gb)
}

@Composable
private fun ImageQualityPicker(
    selected: com.dark.gguf_lib.ImageQuality,
    enabled: Boolean,
    onSelected: (com.dark.gguf_lib.ImageQuality) -> Unit,
) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Text(
            "Image quality",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(4.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            com.dark.gguf_lib.ImageQuality.entries.forEach { q ->
                val isSelected = q == selected
                val container =
                    if (isSelected) MaterialTheme.colorScheme.primary
                    else            MaterialTheme.colorScheme.surface
                val content =
                    if (isSelected) MaterialTheme.colorScheme.onPrimary
                    else            MaterialTheme.colorScheme.onSurface
                Button(
                    onClick = { onSelected(q) },
                    enabled = enabled,
                    modifier = Modifier.weight(1f),
                    colors = androidx.compose.material3.ButtonDefaults.buttonColors(
                        containerColor = container,
                        contentColor = content,
                    ),
                ) {
                    Text(
                        when (q) {
                            com.dark.gguf_lib.ImageQuality.LOW    -> "LOW (384)"
                            com.dark.gguf_lib.ImageQuality.MEDIUM -> "MEDIUM (768)"
                            com.dark.gguf_lib.ImageQuality.HIGH   -> "HIGH (full)"
                        },
                        style = MaterialTheme.typography.labelSmall,
                    )
                }
            }
        }
        Spacer(Modifier.height(2.dp))
        Text(
            "Lower = faster encode + smaller cache; less detail in model perception. Changing invalidates cached entries.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}
