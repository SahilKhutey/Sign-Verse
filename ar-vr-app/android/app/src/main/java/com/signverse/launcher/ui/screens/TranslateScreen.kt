package com.signverse.launcher.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.signverse.launcher.data.AppSettings
import com.signverse.launcher.data.HistoryStore
import com.signverse.launcher.network.SignVerseApi
import com.signverse.launcher.ui.components.ServerStatusHeader
import com.signverse.launcher.ui.components.TokenChipsRow
import kotlinx.coroutines.launch


@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TranslateScreen(
    settings: AppSettings,
    historyStore: HistoryStore,
) {
    val scope = rememberCoroutineScope()

    var text by remember { mutableStateOf("") }
    var tokens by remember { mutableStateOf<List<String>>(emptyList()) }
    var error by remember { mutableStateOf<String?>(null) }
    var loading by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(18.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Text(
            text = "Text -> Sign Tokens",
            style = MaterialTheme.typography.displayLarge,
        )
        Text(
            text = "Convert English text into gloss-like tokens (baseline rule-based for now).",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.78f),
        )

        ServerStatusHeader(baseUrl = settings.apiServerBaseUrl)

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(14.dp)) {
                OutlinedTextField(
                    value = text,
                    onValueChange = { text = it },
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Enter text") },
                    placeholder = { Text("e.g., I am going home") },
                    singleLine = false,
                    minLines = 3,
                )
                Spacer(modifier = Modifier.height(10.dp))
                Button(
                    enabled = !loading && text.isNotBlank() && settings.apiServerBaseUrl.isNotBlank(),
                    onClick = {
                        loading = true
                        error = null
                        tokens = emptyList()
                        scope.launch {
                            try {
                                val out = SignVerseApi.translateTextToSign(
                                    baseUrl = settings.apiServerBaseUrl,
                                    text = text,
                                )
                                tokens = out
                                historyStore.addTextToSign(text, out)
                            } catch (e: Exception) {
                                error = e.message ?: "request failed"
                            } finally {
                                loading = false
                            }
                        }
                    }
                ) {
                    Text("Convert")
                }

                if (loading) {
                    Spacer(modifier = Modifier.height(12.dp))
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                }

                if (error != null) {
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = "Error: $error",
                        color = MaterialTheme.colorScheme.secondary,
                        style = MaterialTheme.typography.bodyLarge,
                    )
                }

                if (tokens.isNotEmpty()) {
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = "Tokens",
                        style = MaterialTheme.typography.headlineMedium,
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    TokenChipsRow(tokens = tokens, modifier = Modifier.fillMaxWidth())
                }
            }
        }
    }
}

