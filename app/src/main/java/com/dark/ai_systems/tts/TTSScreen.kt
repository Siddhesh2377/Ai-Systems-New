package com.dark.ai_systems.tts

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.dark.ai_chatterbox.ChatterboxVariant

@Composable
fun TTSScreen(
    modifier: Modifier = Modifier,
    viewModel: TTSViewModel = viewModel()
) {
    val state by viewModel.uiState.collectAsState()
    val scrollState = rememberScrollState()

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        // Header
        Text(
            text = "Chatterbox TTS",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold
        )

        // Status bar
        StatusCard(state)

        // Model paths section
        SectionHeader("Model Paths")

        OutlinedTextField(
            value = state.modelDir,
            onValueChange = viewModel::updateModelDir,
            label = { Text("Model Directory") },
            placeholder = { Text("/sdcard/chatterbox") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            textStyle = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace)
        )

        OutlinedTextField(
            value = state.tokenizerPath,
            onValueChange = viewModel::updateTokenizerPath,
            label = { Text("Tokenizer Path") },
            placeholder = { Text("/sdcard/chatterbox/tokenizer.json") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            textStyle = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace)
        )

        OutlinedTextField(
            value = state.voicePresetDir,
            onValueChange = viewModel::updateVoicePresetDir,
            label = { Text("Voice Preset Directory (optional)") },
            placeholder = { Text("/sdcard/chatterbox/voice") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            textStyle = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace)
        )

        // Variant selector
        SectionHeader("Variant")

        SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
            SegmentedButton(
                selected = state.variant == ChatterboxVariant.TURBO,
                onClick = { viewModel.updateVariant(ChatterboxVariant.TURBO) },
                shape = SegmentedButtonDefaults.itemShape(index = 0, count = 2)
            ) {
                Text("Turbo (350M)")
            }
            SegmentedButton(
                selected = state.variant == ChatterboxVariant.ORIGINAL,
                onClick = { viewModel.updateVariant(ChatterboxVariant.ORIGINAL) },
                shape = SegmentedButtonDefaults.itemShape(index = 1, count = 2)
            ) {
                Text("Original (500M)")
            }
        }

        // Exaggeration slider (only for ORIGINAL)
        if (state.variant == ChatterboxVariant.ORIGINAL) {
            Text(
                text = "Exaggeration: ${String.format("%.1f", state.exaggeration)}",
                style = MaterialTheme.typography.bodyMedium
            )
            Slider(
                value = state.exaggeration,
                onValueChange = viewModel::updateExaggeration,
                valueRange = 0f..2f,
                steps = 19,
                modifier = Modifier.fillMaxWidth()
            )
        }

        // Load button
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = viewModel::loadModel,
                enabled = !state.isGenerating && state.modelDir.isNotBlank(),
                modifier = Modifier.weight(1f)
            ) {
                Text(if (state.isModelLoaded) "Reload Model" else "Load Model")
            }

            if (state.isModelLoaded && state.voicePresetDir.isNotBlank()) {
                OutlinedButton(
                    onClick = viewModel::loadVoicePreset,
                    enabled = !state.isGenerating
                ) {
                    Text("Swap Voice")
                }
            }
        }

        // Synthesis section
        SectionHeader("Synthesis")

        OutlinedTextField(
            value = state.inputText,
            onValueChange = viewModel::updateInputText,
            label = { Text("Text to synthesize") },
            modifier = Modifier
                .fillMaxWidth()
                .height(120.dp),
            maxLines = 5
        )

        // Generate / Stop / Play row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = viewModel::synthesize,
                enabled = state.isModelLoaded && !state.isGenerating && state.inputText.isNotBlank(),
                modifier = Modifier.weight(1f)
            ) {
                Text("Synthesize")
            }

            if (state.isGenerating) {
                OutlinedButton(
                    onClick = viewModel::stopGeneration,
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = MaterialTheme.colorScheme.error
                    )
                ) {
                    Text("Stop")
                }
            }
        }

        // Progress indicator
        if (state.isGenerating) {
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        }

        // Playback controls
        if (state.lastAudioSamples > 0 || state.generationTimeMs > 0) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Button(
                    onClick = viewModel::playAudio,
                    enabled = !state.isGenerating,
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Play")
                }
                OutlinedButton(
                    onClick = viewModel::stopAudio,
                    modifier = Modifier.width(80.dp)
                ) {
                    Text("Stop")
                }
            }
        }

        // Stats
        if (state.generationTimeMs > 0) {
            StatsCard(state)
        }

        Spacer(modifier = Modifier.height(32.dp))
    }
}

@Composable
private fun StatusCard(state: TTSUiState) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = when {
                state.statusMessage.startsWith("Error") -> MaterialTheme.colorScheme.errorContainer
                state.isModelLoaded && !state.isGenerating -> MaterialTheme.colorScheme.primaryContainer
                state.isGenerating -> MaterialTheme.colorScheme.tertiaryContainer
                else -> MaterialTheme.colorScheme.surfaceVariant
            }
        )
    ) {
        Text(
            text = state.statusMessage,
            modifier = Modifier.padding(12.dp),
            style = MaterialTheme.typography.bodyMedium,
            fontFamily = FontFamily.Monospace,
            fontSize = 13.sp
        )
    }
}

@Composable
private fun SectionHeader(title: String) {
    Text(
        text = title,
        style = MaterialTheme.typography.titleSmall,
        fontWeight = FontWeight.SemiBold,
        color = MaterialTheme.colorScheme.primary
    )
}

@Composable
private fun StatsCard(state: TTSUiState) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Text(
                text = "Generation Stats",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold
            )
            Spacer(modifier = Modifier.height(4.dp))
            val duration = state.lastAudioSamples / 24000f
            val rtf = if (duration > 0) state.generationTimeMs / 1000f / duration else 0f
            StatsRow("Speech tokens", "${state.lastTokenCount}")
            StatsRow("Audio samples", "${state.lastAudioSamples}")
            StatsRow("Audio duration", "${String.format("%.1f", duration)}s")
            StatsRow("Generation time", "${String.format("%.1f", state.generationTimeMs / 1000f)}s")
            StatsRow("RTF (real-time factor)", String.format("%.2fx", rtf))
        }
    }
}

@Composable
private fun StatsRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 1.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(text = label, style = MaterialTheme.typography.bodySmall)
        Text(
            text = value,
            style = MaterialTheme.typography.bodySmall,
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.Medium
        )
    }
}
