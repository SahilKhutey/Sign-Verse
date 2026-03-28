package com.signverse.launcher.data

import android.content.Context
import android.content.SharedPreferences
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow


class SettingsStore(context: Context) {
    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val _settings = MutableStateFlow(load())
    val settings: StateFlow<AppSettings> = _settings.asStateFlow()

    fun updateApiServerBaseUrl(raw: String) {
        val value = sanitizeBaseUrl(raw)
        prefs.edit().putString(KEY_API_SERVER, value).apply()
        _settings.value = load()
    }

    fun updateApiGatewayBaseUrl(raw: String) {
        val value = sanitizeBaseUrl(raw)
        prefs.edit().putString(KEY_API_GATEWAY, value).apply()
        _settings.value = load()
    }

    private fun load(): AppSettings {
        val apiServer = prefs.getString(KEY_API_SERVER, null)?.let(::sanitizeBaseUrl)
        val apiGateway = prefs.getString(KEY_API_GATEWAY, null)?.let(::sanitizeBaseUrl)
        return AppSettings(
            apiServerBaseUrl = apiServer ?: AppSettings().apiServerBaseUrl,
            apiGatewayBaseUrl = apiGateway ?: AppSettings().apiGatewayBaseUrl,
        )
    }

    companion object {
        private const val PREFS_NAME = "signverse_launcher_settings"
        private const val KEY_API_SERVER = "api_server_base_url"
        private const val KEY_API_GATEWAY = "api_gateway_base_url"

        fun sanitizeBaseUrl(input: String): String {
            val trimmed = input.trim()
            if (trimmed.isEmpty()) return ""
            // Avoid accidental double slashes in URL joins.
            return trimmed.removeSuffix("/")
        }
    }
}

