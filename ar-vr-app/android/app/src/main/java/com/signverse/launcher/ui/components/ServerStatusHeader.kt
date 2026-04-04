package com.signverse.launcher.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.runtime.rememberCoroutineScope
import com.signverse.launcher.network.SignVerseApi
import kotlinx.coroutines.launch


@Composable
fun ServerStatusHeader(
    baseUrl: String,
    modifier: Modifier = Modifier,
) {
    val scope = rememberCoroutineScope()
    var status by remember { mutableStateOf<ServerStatus>(ServerStatus.Unknown) }

    Card(modifier = modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text(
                text = "Server",
                style = MaterialTheme.typography.titleMedium,
            )
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = if (baseUrl.isBlank()) "(not set)" else baseUrl,
                style = MaterialTheme.typography.bodyLarge,
            )
            Spacer(modifier = Modifier.height(10.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                StatusPill(status = status)

                Spacer(modifier = Modifier.width(12.dp))
                Button(
                    enabled = baseUrl.isNotBlank() && status !is ServerStatus.Loading,
                    onClick = {
                        status = ServerStatus.Loading
                        scope.launch {
                            status = try {
                                val info = SignVerseApi.health(baseUrl)
                                ServerStatus.Ok("models=${info.models.size}")
                            } catch (e: Exception) {
                                ServerStatus.Error(e.message ?: "request failed")
                            }
                        }
                    }
                ) {
                    Text("Ping")
                }
            }
        }
    }
}


sealed interface ServerStatus {
    data object Unknown : ServerStatus
    data object Loading : ServerStatus
    data class Ok(val detail: String) : ServerStatus
    data class Error(val msg: String) : ServerStatus
}

