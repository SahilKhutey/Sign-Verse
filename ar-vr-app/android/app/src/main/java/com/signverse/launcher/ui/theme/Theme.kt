package com.signverse.launcher.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable


private val LightColors = lightColorScheme(
    primary = Cobalt,
    secondary = Coral,
    tertiary = Mint,
    background = Sand,
    surface = Parchment,
    onPrimary = Parchment,
    onSecondary = Ink,
    onTertiary = Ink,
    onBackground = Ink,
    onSurface = Ink,
)

private val DarkColors = darkColorScheme(
    primary = Cobalt,
    secondary = Coral,
    tertiary = Mint,
    background = Night,
    surface = NightSurface,
    onPrimary = Mist,
    onSecondary = Mist,
    onTertiary = Mist,
    onBackground = Mist,
    onSurface = Mist,
)


@Composable
fun SignVerseTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    val colors = if (darkTheme) DarkColors else LightColors

    MaterialTheme(
        colorScheme = colors,
        typography = SignVerseTypography,
        content = content,
    )
}

