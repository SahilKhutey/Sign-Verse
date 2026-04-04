package com.signverse.launcher

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.signverse.launcher.data.HistoryStore
import com.signverse.launcher.data.SettingsStore
import com.signverse.launcher.ui.SignVerseApp
import com.signverse.launcher.ui.theme.SignVerseTheme


class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val settingsStore = SettingsStore(applicationContext)
        val historyStore = HistoryStore(applicationContext)

        setContent {
            SignVerseTheme {
                SignVerseApp(
                    settingsStore = settingsStore,
                    historyStore = historyStore,
                )
            }
        }
    }
}

