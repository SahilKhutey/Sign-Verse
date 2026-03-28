package com.signverse.launcher.ui.screens

import android.content.Context
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.signverse.launcher.data.AppSettings
import com.signverse.launcher.data.HistoryStore
import com.signverse.launcher.network.SignVerseApi
import com.signverse.launcher.ui.components.TokenChipsRow
import kotlinx.coroutines.launch


@Composable
fun SpeakScreen(
    settings: AppSettings,
    historyStore: HistoryStore,
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var audioBytes by remember { mutableStateOf<ByteArray?>(null) }
    var audioLabel by remember { mutableStateOf<String?>(null) }
    var audioFilename by remember { mutableStateOf<String>("speech.wav") }
    var audioContentType by remember { mutableStateOf<String>("audio/wav") }
    var loading by remember { mutableStateOf(false) }
    var textOut by remember { mutableStateOf<String?>(null) }
    var tokens by remember { mutableStateOf<List<String>>(emptyList()) }
    var error by remember { mutableStateOf<String?>(null) }

    val picker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
        onResult = { uri: Uri? ->
            error = null
            textOut = null
            tokens = emptyList()
            if (uri == null) return@rememberLauncherForActivityResult
            val bytes = context.openBytes(uri)
            audioBytes = bytes
            val label = uri.lastPathSegment ?: "audio"
            audioLabel = label
            audioFilename = label.takeIf { it.contains(".") } ?: "speech.wav"
            audioContentType = context.contentResolver.getType(uri) ?: "application/octet-stream"
            if (bytes == null) {
                error = "Failed to load audio"
            }
        }
    )

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(18.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Text(
            text = "Speech -> Sign Tokens",
            style = MaterialTheme.typography.displayLarge,
        )
        Text(
            text = "Pick an audio file and send it to the API Gateway for speech-to-sign.",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.78f),
        )

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(14.dp)) {
                Text("API Gateway", style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = if (settings.apiGatewayBaseUrl.isBlank()) "(not set)" else settings.apiGatewayBaseUrl,
                    style = MaterialTheme.typography.bodyLarge,
                )
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "Endpoint: POST /speech-to-sign/",
                    style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.65f),
                )
            }
        }

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(14.dp)) {
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    OutlinedButton(
                        onClick = { picker.launch("audio/*") },
                        enabled = !loading,
                    ) { Text("Pick Audio") }

                    Button(
                        onClick = {
                            val bytes = audioBytes ?: return@Button
                            loading = true
                            error = null
                            textOut = null
                            tokens = emptyList()
                            scope.launch {
                                try {
                                    val res = SignVerseApi.speechToSign(
                                        gatewayBaseUrl = settings.apiGatewayBaseUrl,
                                        filename = audioFilename,
                                        contentType = audioContentType,
                                        bytes = bytes,
                                    )
                                    textOut = res.text
                                    tokens = res.signTokens
                                    historyStore.addSpeechToSign(
                                        text = res.text,
                                        tokens = res.signTokens,
                                        error = null,
                                    )
                                } catch (e: Exception) {
                                    val msg = e.message ?: "request failed"
                                    error = msg
                                    historyStore.addSpeechToSign(
                                        text = null,
                                        tokens = emptyList(),
                                        error = msg,
                                    )
                                } finally {
                                    loading = false
                                }
                            }
                        },
                        enabled = !loading && audioBytes != null && settings.apiGatewayBaseUrl.isNotBlank(),
                    ) { Text("Transcribe") }
                }

                if (audioLabel != null) {
                    Spacer(modifier = Modifier.height(10.dp))
                    Text(
                        text = "Selected: $audioLabel",
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
                    )
                }

                if (loading) {
                    Spacer(modifier = Modifier.height(12.dp))
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                }

                if (textOut != null) {
                    Spacer(modifier = Modifier.height(12.dp))
                    Text("Transcript", style = MaterialTheme.typography.headlineMedium)
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = textOut!!,
                        style = MaterialTheme.typography.bodyLarge,
                    )
                }

                if (tokens.isNotEmpty()) {
                    Spacer(modifier = Modifier.height(12.dp))
                    Text("Sign tokens", style = MaterialTheme.typography.headlineMedium)
                    Spacer(modifier = Modifier.height(8.dp))
                    TokenChipsRow(tokens = tokens, modifier = Modifier.fillMaxWidth())
                }

                if (error != null) {
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = "Error: $error",
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.secondary,
                    )
                }
            }
        }
    }
}

private fun Context.openBytes(uri: Uri): ByteArray? {
    return try {
        contentResolver.openInputStream(uri)?.use { it.readBytes() }
    } catch (_: Exception) {
        null
    }
}
