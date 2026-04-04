package com.signverse.launcher.network

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import kotlin.math.min


object SignVerseApi {
    private const val CONNECT_TIMEOUT_MS = 8_000
    private const val READ_TIMEOUT_MS = 20_000

    suspend fun health(baseUrl: String): HealthInfo = withContext(Dispatchers.IO) {
        val url = join(baseUrl, "/health")
        val json = requestJson(url, method = "GET")
        val models = json.optJSONArray("models")?.toStringList().orEmpty()
        HealthInfo(
            status = json.optString("status", "unknown"),
            models = models,
        )
    }

    suspend fun translateTextToSign(baseUrl: String, text: String): List<String> =
        withContext(Dispatchers.IO) {
            val url = join(baseUrl, "/translate/text-to-sign")
            val payload = JSONObject().put("text", text)
            val json = requestJson(url, method = "POST", jsonBody = payload.toString())
            json.optJSONArray("tokens")?.toStringList().orEmpty()
        }

    suspend fun extractFeaturesFromImage(
        baseUrl: String,
        filename: String,
        contentType: String,
        bytes: ByteArray,
    ): VisionExtractResult = withContext(Dispatchers.IO) {
        val url = join(baseUrl, "/vision/extract")
        val boundary = "signverse_${System.currentTimeMillis()}"
        val body = MultipartBody(boundary).apply {
            addFile(fieldName = "file", filename = filename, contentType = contentType, bytes = bytes)
        }.build()

        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = READ_TIMEOUT_MS
            doOutput = true
            setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
            setRequestProperty("Accept", "application/json")
        }

        conn.outputStream.use { it.write(body) }

        val responseText = readResponseText(conn)
        if (conn.responseCode !in 200..299) {
            throw IOException("HTTP ${conn.responseCode}: $responseText")
        }

        val json = JSONObject(responseText)
        val dim = json.optInt("dim", 0)
        val feats = json.optJSONArray("features") ?: JSONArray()
        val features = FloatArray(min(feats.length(), dim.coerceAtLeast(feats.length()))) { i ->
            feats.optDouble(i, 0.0).toFloat()
        }
        VisionExtractResult(features = features, dim = dim)
    }

    suspend fun analyzeFrame(
        baseUrl: String,
        filename: String,
        contentType: String,
        bytes: ByteArray,
    ): AnalyzeFrameResult = withContext(Dispatchers.IO) {
        val url = join(baseUrl, "/analyze/frame")
        val boundary = "signverse_frame_${System.currentTimeMillis()}"
        val body = MultipartBody(boundary).apply {
            addFile(fieldName = "file", filename = filename, contentType = contentType, bytes = bytes)
        }.build()

        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = READ_TIMEOUT_MS
            doOutput = true
            setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
            setRequestProperty("Accept", "application/json")
        }

        conn.outputStream.use { it.write(body) }

        val responseText = readResponseText(conn)
        if (conn.responseCode !in 200..299) {
            throw IOException("HTTP ${conn.responseCode}: $responseText")
        }

        val json = JSONObject(responseText)
        val gestureId = json.optInt("gesture_id", -1).let { if (it >= 0) it else null }
        val gestureLabel = json.optString("gesture_label", "").takeIf { it.isNotBlank() }
        val dim = json.optInt("dim", 0)
        val tokens = json.optJSONArray("sign_tokens")?.toStringList().orEmpty()
        AnalyzeFrameResult(
            gestureId = gestureId,
            gestureLabel = gestureLabel,
            signTokens = tokens,
            dim = dim,
        )
    }

    suspend fun classifyGesture(baseUrl: String, keypoints: FloatArray): Int =
        withContext(Dispatchers.IO) {
            val url = join(baseUrl, "/gesture/classify")
            val arr = JSONArray()
            for (v in keypoints) arr.put(v.toDouble())
            val payload = JSONObject().put("keypoints", arr)
            val json = requestJson(url, method = "POST", jsonBody = payload.toString())
            json.optInt("gesture_id", -1)
        }

    suspend fun speechToSign(
        gatewayBaseUrl: String,
        filename: String,
        contentType: String,
        bytes: ByteArray,
    ): SpeechToSignResult = withContext(Dispatchers.IO) {
        val url = join(gatewayBaseUrl, "/speech-to-sign/")
        val boundary = "signverse_audio_${System.currentTimeMillis()}"
        val body = MultipartBody(boundary).apply {
            addFile(fieldName = "file", filename = filename, contentType = contentType, bytes = bytes)
        }.build()

        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = READ_TIMEOUT_MS
            doOutput = true
            setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
            setRequestProperty("Accept", "application/json")
        }

        conn.outputStream.use { it.write(body) }

        val responseText = readResponseText(conn)
        if (conn.responseCode !in 200..299) {
            throw IOException("HTTP ${conn.responseCode}: $responseText")
        }

        val json = JSONObject(responseText)
        val tokens = json.optJSONArray("sign_tokens")?.toStringList().orEmpty()
        SpeechToSignResult(
            text = json.optString("text", ""),
            signTokens = tokens,
        )
    }

    private fun join(base: String, path: String): URL {
        val b = base.trim().removeSuffix("/")
        val p = if (path.startsWith("/")) path else "/$path"
        return URL(b + p)
    }

    private fun requestJson(url: URL, method: String, jsonBody: String? = null): JSONObject {
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = READ_TIMEOUT_MS
            setRequestProperty("Accept", "application/json")
        }

        if (jsonBody != null) {
            conn.doOutput = true
            conn.setRequestProperty("Content-Type", "application/json; charset=utf-8")
            conn.outputStream.use { it.write(jsonBody.toByteArray(Charsets.UTF_8)) }
        }

        val responseText = readResponseText(conn)
        if (conn.responseCode !in 200..299) {
            throw IOException("HTTP ${conn.responseCode}: $responseText")
        }
        return JSONObject(responseText)
    }

    private fun readResponseText(conn: HttpURLConnection): String {
        val stream = try {
            if (conn.responseCode in 200..299) conn.inputStream else conn.errorStream
        } catch (_: Exception) {
            conn.errorStream
        }
        return stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }.orEmpty()
    }
}

private fun JSONArray.toStringList(): List<String> =
    buildList {
        for (i in 0 until length()) {
            add(optString(i, "").toString())
        }
    }
