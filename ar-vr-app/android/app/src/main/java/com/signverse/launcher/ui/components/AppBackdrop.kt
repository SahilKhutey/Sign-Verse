package com.signverse.launcher.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.material3.MaterialTheme


@Composable
fun AppBackdrop(content: @Composable () -> Unit) {
    val bg = MaterialTheme.colorScheme.background
    val accent = MaterialTheme.colorScheme.primary
    val warm = MaterialTheme.colorScheme.secondary

    val brush = Brush.linearGradient(
        colors = listOf(
            bg,
            bg,
            blend(accent, bg, 0.12f),
            blend(warm, bg, 0.08f),
        )
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(brush),
    ) {
        // Soft geometry to avoid a flat "default app" feel.
        Canvas(modifier = Modifier.fillMaxSize()) {
            val w = size.width
            val h = size.height
            drawCircle(
                color = accent.copy(alpha = 0.08f),
                radius = w * 0.55f,
                center = Offset(w * 0.15f, h * 0.25f),
            )
            drawCircle(
                color = warm.copy(alpha = 0.06f),
                radius = w * 0.65f,
                center = Offset(w * 0.95f, h * 0.10f),
            )
            drawCircle(
                color = accent.copy(alpha = 0.05f),
                radius = w * 0.80f,
                center = Offset(w * 0.75f, h * 1.10f),
            )
        }

        content()
    }
}

private fun blend(a: Color, b: Color, t: Float): Color {
    val clamped = t.coerceIn(0f, 1f)
    return Color(
        red = a.red * clamped + b.red * (1f - clamped),
        green = a.green * clamped + b.green * (1f - clamped),
        blue = a.blue * clamped + b.blue * (1f - clamped),
        alpha = 1f
    )
}

