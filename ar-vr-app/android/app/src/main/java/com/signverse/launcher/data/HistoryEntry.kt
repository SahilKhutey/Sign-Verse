package com.signverse.launcher.data

enum class HistoryKind {
    TEXT_TO_SIGN,
    IMAGE_TO_GESTURE,
    SPEECH_TO_SIGN,
}

data class HistoryEntry(
    val id: String,
    val kind: HistoryKind,
    val title: String,
    val subtitle: String,
    val createdAtMs: Long,
)
