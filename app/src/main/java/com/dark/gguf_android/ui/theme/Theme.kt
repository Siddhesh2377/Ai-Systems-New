package com.dark.gguf_android.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext

private val DarkColorScheme = darkColorScheme(
    primary = Teal60,
    onPrimary = Color.Black,
    primaryContainer = Teal20,
    onPrimaryContainer = Teal80,
    secondary = Gray70,
    onSecondary = Color.Black,
    secondaryContainer = Gray20,
    onSecondaryContainer = Gray90,
    tertiary = Teal40,
    background = Gray10,
    onBackground = Color(0xFFE6E1E5),
    surface = Gray15,
    onSurface = Color(0xFFE6E1E5),
    surfaceVariant = Gray20,
    onSurfaceVariant = Gray70,
    outline = Gray40,
)

private val LightColorScheme = lightColorScheme(
    primary = Teal40,
    onPrimary = Color.White,
    primaryContainer = Teal80,
    onPrimaryContainer = Teal20,
    secondary = Gray40,
    onSecondary = Color.White,
    secondaryContainer = Gray90,
    onSecondaryContainer = Gray20,
    tertiary = Teal20,
    background = Color(0xFFFFFBFE),
    onBackground = Color(0xFF1C1B1F),
    surface = Gray95,
    onSurface = Color(0xFF1C1B1F),
    surfaceVariant = Gray90,
    onSurfaceVariant = Gray40,
    outline = Gray70,
)

@Composable
fun AiSystemsTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}
