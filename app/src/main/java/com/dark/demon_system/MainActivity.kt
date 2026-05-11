package com.dark.demon_system

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.dark.demon_system.ui.SDScreen
import com.dark.demon_system.ui.theme.AiSystemsTheme
import com.dark.demon_system.ui.vlm.VlmScreen

private enum class Screen { Picker, Sd, Vlm }

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AiSystemsTheme {
                AppRoot()
            }
        }
    }
}

@Composable
private fun AppRoot() {
    var screen by remember { mutableStateOf(Screen.Picker) }
    when (screen) {
        Screen.Picker -> PickerScreen(
            onSd  = { screen = Screen.Sd },
            onVlm = { screen = Screen.Vlm },
        )
        Screen.Sd  -> SDScreen()
        Screen.Vlm -> VlmScreen(onBack = { screen = Screen.Picker })
    }
}

@Composable
private fun PickerScreen(onSd: () -> Unit, onVlm: () -> Unit) {
    Scaffold { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterVertically),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text("Ai-Systems test app")
            Button(onClick = onVlm, modifier = Modifier.fillMaxWidth()) {
                Text("VLM Test (Qwen3-VL-2B)")
            }
            Button(onClick = onSd, modifier = Modifier.fillMaxWidth()) {
                Text("Stable Diffusion Test")
            }
        }
    }
}
