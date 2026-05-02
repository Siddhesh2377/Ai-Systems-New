package com.dark.demon_system

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.dark.demon_system.ui.SDScreen
import com.dark.demon_system.ui.theme.AiSystemsTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AiSystemsTheme {
                SDScreen()
            }
        }
    }
}
