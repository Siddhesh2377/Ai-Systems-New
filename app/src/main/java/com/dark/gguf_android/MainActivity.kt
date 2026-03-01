package com.dark.gguf_android

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.dark.gguf_android.ui.chat.ChatScreen
import com.dark.gguf_android.ui.theme.AiSystemsTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AiSystemsTheme {
                ChatScreen()
            }
        }
    }
}
