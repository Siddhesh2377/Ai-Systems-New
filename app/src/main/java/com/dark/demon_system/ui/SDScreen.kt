package com.dark.demon_system.ui

import androidx.compose.foundation.Image
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
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.collectAsState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.dark.ai_sd.DiffusionBackendState
import com.dark.ai_sd.DiffusionGenerationState
import com.dark.ai_sd.RuntimeSetupState
import com.dark.demon_system.data.ModelInstallState
import com.dark.demon_system.data.ModelSpec

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SDScreen(
    vm: SDViewModel = viewModel(factory = SDViewModel.Factory)
) {
    val installState by vm.installState.collectAsState()
    val runtimeState by vm.runtimeState.collectAsState()
    val backendState by vm.backendState.collectAsState()
    val generationState by vm.generationState.collectAsState()
    val isGenerating by vm.isGenerating.collectAsState()
    val form by vm.form.collectAsState()
    val socInfo by vm.socInfo.collectAsState()
    val modelLoadError by vm.modelLoadError.collectAsState()

    val isInstalled = installState is ModelInstallState.Installed || vm.isInstalled
    val isModelLoaded = backendState is DiffusionBackendState.Running

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("ai_sd Demo", style = MaterialTheme.typography.titleLarge)
                        Text(
                            "Stable Diffusion on-device",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            ModelInstallCard(
                spec = vm.modelSpec,
                state = installState,
                isAlreadyInstalled = isInstalled,
                onDownload = vm::installModel,
                onCancel = vm::cancelInstall,
                onUninstall = vm::uninstallModel,
                onResetError = vm::resetInstallError
            )

            BackendCard(
                runtimeState = runtimeState,
                backendState = backendState,
                socInfo = socInfo,
                runOnCpu = form.runOnCpu,
                onRunOnCpuChange = { vm.updateForm { it.copy(runOnCpu = !it.runOnCpu) } },
                isInstalled = isInstalled,
                isLoaded = isModelLoaded,
                preloadError = modelLoadError,
                onLoad = vm::loadModel,
                onUnload = vm::unloadModel
            )

            GenerationCard(
                form = form,
                isModelLoaded = isModelLoaded,
                isGenerating = isGenerating,
                onFormChange = vm::updateForm,
                onGenerate = vm::generate,
                onCancel = vm::cancelGeneration
            )

            ResultCard(
                state = generationState,
                onReset = vm::resetGenerationState
            )

            Spacer(Modifier.height(16.dp))
        }
    }
}

@Composable
private fun ModelInstallCard(
    spec: ModelSpec,
    state: ModelInstallState,
    isAlreadyInstalled: Boolean,
    onDownload: () -> Unit,
    onCancel: () -> Unit,
    onUninstall: () -> Unit,
    onResetError: () -> Unit
) {
    Card(
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SectionHeader("1. Model", subtitle = spec.displayName)
            Text(
                "Source: huggingface.co/xororz/sd-qnn",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                "Approx ${formatMB(spec.approxBytes)} (zip)",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            HorizontalDivider()

            when (state) {
                is ModelInstallState.Idle -> {
                    if (isAlreadyInstalled) {
                        StatusLine("Installed", success = true)
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedButton(onClick = onUninstall) { Text("Delete model") }
                        }
                    } else {
                        StatusLine("Not installed")
                        Button(onClick = onDownload) { Text("Download model") }
                    }
                }
                is ModelInstallState.Downloading -> {
                    val frac = if (state.totalBytes > 0)
                        state.bytesDownloaded.toFloat() / state.totalBytes else 0f
                    StatusLine("Downloading…")
                    LinearProgressIndicator(
                        progress = { frac.coerceIn(0f, 1f) },
                        modifier = Modifier.fillMaxWidth()
                    )
                    Text(
                        "${formatMB(state.bytesDownloaded)} / ${formatMB(state.totalBytes)}  •  " +
                            "${formatBps(state.bytesPerSecond)}",
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace
                    )
                    OutlinedButton(onClick = onCancel) { Text("Cancel") }
                }
                is ModelInstallState.Extracting -> {
                    StatusLine("Extracting…")
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    Text(
                        "Files: ${state.filesExtracted}  •  ${state.currentFile.takeLast(40)}",
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace
                    )
                }
                ModelInstallState.Finalizing -> {
                    StatusLine("Finalizing…")
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                }
                is ModelInstallState.Installed -> {
                    StatusLine("Installed", success = true)
                    Text(
                        "${state.files.size} files",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    OutlinedButton(onClick = onUninstall) { Text("Delete model") }
                }
                is ModelInstallState.Error -> {
                    StatusLine("Error", error = true)
                    Text(state.message, style = MaterialTheme.typography.bodySmall)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(onClick = { onResetError(); onDownload() }) { Text("Retry") }
                        OutlinedButton(onClick = onResetError) { Text("Dismiss") }
                    }
                }
                ModelInstallState.Cancelled -> {
                    StatusLine("Cancelled")
                    Button(onClick = onDownload) { Text("Resume download") }
                }
            }
        }
    }
}

@Composable
private fun BackendCard(
    runtimeState: RuntimeSetupState,
    backendState: DiffusionBackendState,
    socInfo: String?,
    runOnCpu: Boolean,
    onRunOnCpuChange: () -> Unit,
    isInstalled: Boolean,
    isLoaded: Boolean,
    preloadError: String?,
    onLoad: () -> Unit,
    onUnload: () -> Unit
) {
    Card(elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SectionHeader("2. Runtime + model load")

            if (socInfo != null) {
                Text(
                    socInfo,
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    "Run on CPU (MNN)",
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.weight(1f)
                )
                Switch(checked = runOnCpu, onCheckedChange = { onRunOnCpuChange() })
            }
            Text(
                if (runOnCpu) "Uses .mnn files" else "Uses QNN HTP (.bin)",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            HorizontalDivider()

            // Runtime status line
            when (val r = runtimeState) {
                RuntimeSetupState.Idle -> StatusLine("Runtime: idle")
                is RuntimeSetupState.CopyingAsset -> {
                    val frac = if (r.totalBytes > 0) r.bytesWritten.toFloat() / r.totalBytes else 0f
                    StatusLine("Runtime: copying QNN libs…")
                    LinearProgressIndicator(
                        progress = { frac.coerceIn(0f, 1f) },
                        modifier = Modifier.fillMaxWidth()
                    )
                }
                is RuntimeSetupState.Extracting -> {
                    StatusLine("Runtime: extracting (${r.filesExtracted})")
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                }
                RuntimeSetupState.CopyingSafetyChecker -> StatusLine("Runtime: safety checker…")
                RuntimeSetupState.InitializingRuntime -> StatusLine("Runtime: initializing…")
                RuntimeSetupState.Complete -> StatusLine("Runtime: ready", success = true)
                is RuntimeSetupState.Error -> StatusLine("Runtime: ${r.message}", error = true)
                is RuntimeSetupState.Downloading -> StatusLine("Runtime: downloading ${r.fileName}")
            }

            // Backend status line
            when (val b = backendState) {
                DiffusionBackendState.Idle -> StatusLine("Model: idle")
                DiffusionBackendState.Starting -> {
                    StatusLine("Model: loading…")
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                }
                DiffusionBackendState.Running -> StatusLine("Model: loaded", success = true)
                is DiffusionBackendState.Error -> StatusLine("Model: ${b.message}", error = true)
            }

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    enabled = isInstalled && !isLoaded && backendState !is DiffusionBackendState.Starting,
                    onClick = onLoad
                ) { Text(if (isLoaded) "Loaded" else "Load model") }
                if (isLoaded) {
                    OutlinedButton(onClick = onUnload) { Text("Unload") }
                }
            }
            if (!isInstalled) {
                Text(
                    "Install the model first.",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            if (preloadError != null) {
                Text(
                    preloadError,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.error
                )
            }
        }
    }
}

@Composable
private fun GenerationCard(
    form: GenerationForm,
    isModelLoaded: Boolean,
    isGenerating: Boolean,
    onFormChange: ((GenerationForm) -> GenerationForm) -> Unit,
    onGenerate: () -> Unit,
    onCancel: () -> Unit
) {
    Card(elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            SectionHeader("3. Generate")

            OutlinedTextField(
                value = form.prompt,
                onValueChange = { v -> onFormChange { it.copy(prompt = v) } },
                label = { Text("Prompt") },
                modifier = Modifier.fillMaxWidth(),
                minLines = 2,
                maxLines = 4
            )
            OutlinedTextField(
                value = form.negativePrompt,
                onValueChange = { v -> onFormChange { it.copy(negativePrompt = v) } },
                label = { Text("Negative prompt") },
                modifier = Modifier.fillMaxWidth(),
                maxLines = 3
            )

            LabeledSlider(
                label = "Steps: ${form.steps}",
                value = form.steps.toFloat(),
                range = 4f..50f,
                steps = 45,
                onChange = { v -> onFormChange { it.copy(steps = v.toInt()) } }
            )
            LabeledSlider(
                label = "CFG: ${"%.1f".format(form.cfgScale)}",
                value = form.cfgScale,
                range = 1f..15f,
                onChange = { v -> onFormChange { it.copy(cfgScale = v) } }
            )

            OutlinedTextField(
                value = form.seedText,
                onValueChange = { v -> onFormChange { it.copy(seedText = v.filter { ch -> ch.isDigit() || ch == '-' }) } },
                label = { Text("Seed (blank = random)") },
                modifier = Modifier.fillMaxWidth(),
                maxLines = 1
            )

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                listOf("dpm" to "DPM++", "euler" to "Euler", "euler_a" to "Euler-a").forEach { (k, label) ->
                    FilterChip(
                        selected = form.scheduler == k,
                        onClick = { onFormChange { it.copy(scheduler = k) } },
                        label = { Text(label) }
                    )
                }
            }

            HorizontalDivider()

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    enabled = isModelLoaded && !isGenerating,
                    onClick = onGenerate
                ) { Text(if (isGenerating) "Generating…" else "Generate") }
                if (isGenerating) {
                    OutlinedButton(onClick = onCancel) { Text("Cancel") }
                }
            }
            if (!isModelLoaded) {
                Text(
                    "Load the model first.",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
private fun ResultCard(state: DiffusionGenerationState, onReset: () -> Unit) {
    Card(elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SectionHeader("4. Result")

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(1f)
                    .clip(RoundedCornerShape(12.dp)),
                contentAlignment = Alignment.Center
            ) {
                when (state) {
                    DiffusionGenerationState.Idle -> {
                        Text(
                            "Output will appear here",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    is DiffusionGenerationState.Progress -> {
                        val bmp = state.intermediateImage
                        if (bmp != null) {
                            Image(
                                bitmap = bmp.asImageBitmap(),
                                contentDescription = null,
                                modifier = Modifier.fillMaxSize(),
                                contentScale = ContentScale.Crop
                            )
                        }
                        Box(
                            Modifier.fillMaxSize(),
                            contentAlignment = Alignment.BottomCenter
                        ) {
                            Column(
                                Modifier
                                    .fillMaxWidth()
                                    .padding(12.dp),
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                LinearProgressIndicator(
                                    progress = { state.progress.coerceIn(0f, 1f) },
                                    modifier = Modifier.fillMaxWidth()
                                )
                                Spacer(Modifier.height(4.dp))
                                Text(
                                    "Step ${state.currentStep} / ${state.totalSteps}",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    }
                    is DiffusionGenerationState.Complete -> {
                        Image(
                            bitmap = state.bitmap.asImageBitmap(),
                            contentDescription = null,
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Crop
                        )
                    }
                    is DiffusionGenerationState.Error -> {
                        Text(
                            "Error\n${state.message}",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.error,
                            textAlign = TextAlign.Center
                        )
                    }
                }
            }

            when (state) {
                is DiffusionGenerationState.Complete -> {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        AssistChip(
                            onClick = {},
                            label = { Text("seed: ${state.seed ?: "?"}") }
                        )
                        Spacer(Modifier.size(8.dp))
                        AssistChip(
                            onClick = {},
                            label = { Text("${state.width}×${state.height}") }
                        )
                        Spacer(Modifier.weight(1f))
                        OutlinedButton(onClick = onReset) { Text("Clear") }
                    }
                }
                is DiffusionGenerationState.Error -> {
                    OutlinedButton(onClick = onReset) { Text("Dismiss") }
                }
                else -> {}
            }
        }
    }
}

@Composable
private fun SectionHeader(title: String, subtitle: String? = null) {
    Column {
        Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
        if (subtitle != null) {
            Text(
                subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun StatusLine(text: String, success: Boolean = false, error: Boolean = false) {
    val color = when {
        error -> MaterialTheme.colorScheme.error
        success -> MaterialTheme.colorScheme.primary
        else -> MaterialTheme.colorScheme.onSurfaceVariant
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(
            modifier = Modifier
                .size(8.dp)
                .clip(RoundedCornerShape(50))
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .clip(RoundedCornerShape(50))
            ) {
                androidx.compose.foundation.Canvas(Modifier.fillMaxSize()) {
                    drawCircle(color)
                }
            }
        }
        Spacer(Modifier.size(8.dp))
        Text(text, style = MaterialTheme.typography.bodyMedium, color = color)
    }
}

@Composable
private fun LabeledSlider(
    label: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    steps: Int = 0,
    onChange: (Float) -> Unit
) {
    Column {
        Text(label, style = MaterialTheme.typography.labelMedium)
        Slider(
            value = value,
            onValueChange = onChange,
            valueRange = range,
            steps = steps
        )
    }
}

private fun formatMB(bytes: Long): String {
    if (bytes < 0) return "?"
    val mb = bytes / 1_048_576.0
    return "%.1f MB".format(mb)
}

private fun formatBps(bps: Long): String {
    if (bps <= 0) return "—"
    val mbps = bps / 1_048_576.0
    return if (mbps >= 1.0) "%.2f MB/s".format(mbps) else "%.1f KB/s".format(bps / 1024.0)
}
