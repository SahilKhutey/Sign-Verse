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
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.signverse.launcher.data.AppSettings
import com.signverse.launcher.ui.components.ServerStatusHeader


@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    settings: AppSettings,
    onSaveApiServer: (String) -> Unit,
    onSaveApiGateway: (String) -> Unit,
) {
    var apiServerDraft by remember(settings.apiServerBaseUrl) { mutableStateOf(settings.apiServerBaseUrl) }
    var apiGatewayDraft by remember(settings.apiGatewayBaseUrl) { mutableStateOf(settings.apiGatewayBaseUrl) }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(18.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Text(
            text = "Settings",
            style = MaterialTheme.typography.displayLarge,
        )
        Text(
            text = "Configure which SignVerse server the app talks to.",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.78f),
        )

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(14.dp)) {
                Text(
                    text = "API Server (required)",
                    style = MaterialTheme.typography.headlineMedium,
                )
                Spacer(modifier = Modifier.height(8.dp))
                OutlinedTextField(
                    value = apiServerDraft,
                    onValueChange = { apiServerDraft = it },
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Base URL") },
                    placeholder = { Text("http://10.0.2.2:8000") },
                    singleLine = true,
                )
                Spacer(modifier = Modifier.height(10.dp))
                Button(
                    onClick = { onSaveApiServer(apiServerDraft) },
                    enabled = apiServerDraft.isNotBlank(),
                ) { Text("Save API Server") }
                Spacer(modifier = Modifier.height(12.dp))
                ServerStatusHeader(baseUrl = apiServerDraft)
            }
        }

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(14.dp)) {
                Text(
                    text = "API Gateway (optional)",
                    style = MaterialTheme.typography.headlineMedium,
                )
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "Used for speech-to-sign in this repo (separate FastAPI app).",
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.78f),
                )
                Spacer(modifier = Modifier.height(8.dp))
                OutlinedTextField(
                    value = apiGatewayDraft,
                    onValueChange = { apiGatewayDraft = it },
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Base URL") },
                    placeholder = { Text("http://10.0.2.2:8001") },
                    singleLine = true,
                )
                Spacer(modifier = Modifier.height(10.dp))
                Button(
                    onClick = { onSaveApiGateway(apiGatewayDraft) },
                    enabled = apiGatewayDraft.isNotBlank(),
                ) { Text("Save API Gateway") }
            }
        }

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(14.dp)) {
                Text(text = "Quick Tips", style = MaterialTheme.typography.headlineMedium)
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "Emulator -> host: use http://10.0.2.2:8000\nDevice -> PC: use your PC's LAN IP (same Wi-Fi).",
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.78f),
                )
            }
        }
    }
}

