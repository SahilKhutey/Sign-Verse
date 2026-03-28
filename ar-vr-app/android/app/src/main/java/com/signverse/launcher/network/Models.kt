package com.signverse.launcher.network

data class HealthInfo(
    val status: String,
    val models: List<String>,
)

data class VisionExtractResult(
    val features: FloatArray,
    val dim: Int,
)

data class SpeechToSignResult(
    val text: String,
    val signTokens: List<String>,
)

data class AnalyzeFrameResult(
    val gestureId: Int?,
    val gestureLabel: String?,
    val signTokens: List<String>,
    val dim: Int,
)
