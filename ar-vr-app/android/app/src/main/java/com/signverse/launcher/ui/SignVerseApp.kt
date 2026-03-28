package com.signverse.launcher.ui

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.History
import androidx.compose.material.icons.outlined.Mic
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material.icons.outlined.TextFields
import androidx.compose.material.icons.outlined.Videocam
import androidx.compose.material.icons.outlined.Visibility
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.runtime.collectAsState
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import com.signverse.launcher.data.HistoryStore
import com.signverse.launcher.data.SettingsStore
import com.signverse.launcher.ui.components.AppBackdrop
import com.signverse.launcher.ui.screens.HistoryScreen
import com.signverse.launcher.ui.screens.InterpretScreen
import com.signverse.launcher.ui.screens.LiveScreen
import com.signverse.launcher.ui.screens.SpeakScreen
import com.signverse.launcher.ui.screens.SettingsScreen
import com.signverse.launcher.ui.screens.TranslateScreen


private enum class AppTab(val label: String, val icon: ImageVector) {
    Translate("Translate", Icons.Outlined.TextFields),
    Interpret("Interpret", Icons.Outlined.Visibility),
    Speak("Speak", Icons.Outlined.Mic),
    Live("Live", Icons.Outlined.Videocam),
    History("History", Icons.Outlined.History),
    Settings("Settings", Icons.Outlined.Settings),
}


@Composable
fun SignVerseApp(
    settingsStore: SettingsStore,
    historyStore: HistoryStore,
) {
    val settings by settingsStore.settings.collectAsState()
    val history by historyStore.items.collectAsState()

    var tab by rememberSaveable { mutableStateOf(AppTab.Translate) }

    AppBackdrop {
        Scaffold(
            modifier = Modifier.fillMaxSize(),
            containerColor = MaterialTheme.colorScheme.background.copy(alpha = 0.0f),
            contentWindowInsets = WindowInsets.safeDrawing,
            bottomBar = {
                NavigationBar(
                    containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.92f),
                ) {
                    AppTab.entries.forEach { t ->
                        NavigationBarItem(
                            selected = tab == t,
                            onClick = { tab = t },
                            icon = { Icon(imageVector = t.icon, contentDescription = t.label) },
                            label = { Text(t.label) },
                        )
                    }
                }
            }
        ) { padding ->
            AnimatedContent(
                targetState = tab,
                transitionSpec = {
                    fadeIn(animationSpec = tween(durationMillis = 220)) togetherWith
                        fadeOut(animationSpec = tween(durationMillis = 220))
                },
                label = "tab-content",
                modifier = Modifier.padding(padding),
            ) { t ->
                when (t) {
                    AppTab.Translate -> TranslateScreen(
                        settings = settings,
                        historyStore = historyStore,
                    )

                    AppTab.Interpret -> InterpretScreen(
                        settings = settings,
                        historyStore = historyStore,
                    )

                    AppTab.Speak -> SpeakScreen(
                        settings = settings,
                        historyStore = historyStore,
                    )

                    AppTab.Live -> LiveScreen(
                        settings = settings,
                    )

                    AppTab.History -> HistoryScreen(
                        items = history,
                        onClear = { historyStore.clear() },
                    )

                    AppTab.Settings -> SettingsScreen(
                        settings = settings,
                        onSaveApiServer = settingsStore::updateApiServerBaseUrl,
                        onSaveApiGateway = settingsStore::updateApiGatewayBaseUrl,
                    )
                }
            }
        }
    }
}
