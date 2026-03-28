package com.signverse.launcher.ui.components

import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp


@Composable
fun StatusPill(status: ServerStatus, modifier: Modifier = Modifier) {
    val (label, dot) = when (status) {
        ServerStatus.Unknown -> "Unknown" to MaterialTheme.colorScheme.onSurface.copy(alpha = 0.35f)
        ServerStatus.Loading -> "Checking..." to MaterialTheme.colorScheme.secondary
        is ServerStatus.Ok -> "Online" to MaterialTheme.colorScheme.tertiary
        is ServerStatus.Error -> "Offline" to MaterialTheme.colorScheme.secondary
    }

    Surface(
        modifier = modifier,
        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.9f),
        shape = RoundedCornerShape(999.dp),
        tonalElevation = 1.dp,
        shadowElevation = 0.dp,
        border = null,
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Surface(
                modifier = Modifier.size(10.dp),
                shape = CircleShape,
                color = dotColor(dot, status),
                content = {},
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(text = label, style = MaterialTheme.typography.labelLarge)
        }
    }
}

private fun dotColor(base: Color, status: ServerStatus): Color {
    return when (status) {
        is ServerStatus.Error -> base.copy(alpha = 0.95f)
        else -> base
    }
}
