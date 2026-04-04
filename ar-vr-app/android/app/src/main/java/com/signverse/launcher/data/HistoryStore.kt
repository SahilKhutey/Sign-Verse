package com.signverse.launcher.data

import android.content.Context
import android.content.SharedPreferences
import org.json.JSONArray
import org.json.JSONObject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.UUID


class HistoryStore(context: Context) {
    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val _items = MutableStateFlow(load())
    val items: StateFlow<List<HistoryEntry>> = _items.asStateFlow()

    fun addTextToSign(input: String, tokens: List<String>) {
        val title = input.trim().ifEmpty { "(empty)" }
        val subtitle = if (tokens.isEmpty()) "No tokens" else tokens.joinToString(" ")
        add(
            HistoryEntry(
                id = UUID.randomUUID().toString(),
                kind = HistoryKind.TEXT_TO_SIGN,
                title = title,
                subtitle = subtitle,
                createdAtMs = System.currentTimeMillis(),
            )
        )
    }

    fun addImageToGesture(gestureId: Int?, gestureLabel: String?, dim: Int?, error: String?) {
        val title = "Image inference"
        val subtitle = when {
            error != null -> "Error: $error"
            gestureId == null -> "No gesture detected"
            gestureLabel != null -> "gesture=$gestureLabel (id=$gestureId)  dim=${dim ?: "?"}"
            else -> "gesture_id=$gestureId  dim=${dim ?: "?"}"
        }
        add(
            HistoryEntry(
                id = UUID.randomUUID().toString(),
                kind = HistoryKind.IMAGE_TO_GESTURE,
                title = title,
                subtitle = subtitle,
                createdAtMs = System.currentTimeMillis(),
            )
        )
    }

    fun addSpeechToSign(text: String?, tokens: List<String>, error: String?) {
        val title = if (!text.isNullOrBlank()) text else "Speech inference"
        val subtitle = when {
            error != null -> "Error: $error"
            tokens.isEmpty() -> "No tokens"
            else -> tokens.joinToString(" ")
        }
        add(
            HistoryEntry(
                id = UUID.randomUUID().toString(),
                kind = HistoryKind.SPEECH_TO_SIGN,
                title = title,
                subtitle = subtitle,
                createdAtMs = System.currentTimeMillis(),
            )
        )
    }

    fun clear() {
        prefs.edit().remove(KEY_HISTORY).apply()
        _items.value = emptyList()
    }

    private fun add(entry: HistoryEntry) {
        val next = (listOf(entry) + _items.value).take(MAX_ITEMS)
        save(next)
        _items.value = next
    }

    private fun load(): List<HistoryEntry> {
        val raw = prefs.getString(KEY_HISTORY, null) ?: return emptyList()
        return try {
            val arr = JSONArray(raw)
            buildList {
                for (i in 0 until arr.length()) {
                    val obj = arr.getJSONObject(i)
                    val kind = runCatching { HistoryKind.valueOf(obj.getString("kind")) }.getOrNull()
                        ?: continue
                    add(
                        HistoryEntry(
                            id = obj.optString("id", UUID.randomUUID().toString()),
                            kind = kind,
                            title = obj.optString("title", ""),
                            subtitle = obj.optString("subtitle", ""),
                            createdAtMs = obj.optLong("createdAtMs", 0L),
                        )
                    )
                }
            }
        } catch (_: Exception) {
            emptyList()
        }
    }

    private fun save(items: List<HistoryEntry>) {
        val arr = JSONArray()
        for (it in items) {
            val obj = JSONObject()
            obj.put("id", it.id)
            obj.put("kind", it.kind.name)
            obj.put("title", it.title)
            obj.put("subtitle", it.subtitle)
            obj.put("createdAtMs", it.createdAtMs)
            arr.put(obj)
        }
        prefs.edit().putString(KEY_HISTORY, arr.toString()).apply()
    }

    companion object {
        private const val PREFS_NAME = "signverse_launcher_history"
        private const val KEY_HISTORY = "history_json"
        private const val MAX_ITEMS = 50
    }
}
