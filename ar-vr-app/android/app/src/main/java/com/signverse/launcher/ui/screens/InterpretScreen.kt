package com.signverse.launcher.ui.screens

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
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
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.signverse.launcher.data.AppSettings
import com.signverse.launcher.data.HistoryStore
import com.signverse.launcher.network.SignVerseApi
import com.signverse.launcher.ui.components.ServerStatusHeader
import com.signverse.launcher.ui.components.TokenChipsRow
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream


@Composable
fun InterpretScreen(
    settings: AppSettings,
    historyStore: HistoryStore,
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var imageBytes by remember { mutableStateOf<ByteArray?>(null) }
    var previewBitmap by remember { mutableStateOf<androidx.compose.ui.graphics.ImageBitmap?>(null) }
    var loading by remember { mutableStateOf(false) }
    var gestureId by remember { mutableStateOf<Int?>(null) }
    var gestureLabel by remember { mutableStateOf<String?>(null) }
    var signTokens by remember { mutableStateOf<List<String>>(emptyList()) }
    var keypointDim by remember { mutableStateOf<Int?>(null) }
    var error by remember { mutableStateOf<String?>(null) }

    val picker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
        onResult = { uri: Uri? ->
            error = null
            gestureId = null
            gestureLabel = null
            signTokens = emptyList()
            keypointDim = null
            if (uri == null) return@rememberLauncherForActivityResult
            val loaded = loadAndCompressImage(context, uri)
            imageBytes = loaded?.first
            previewBitmap = loaded?.second
            if (loaded == null) {
                error = "Failed to load image"
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
            text = "Interpret (Photo)",
            style = MaterialTheme.typography.displayLarge,
        )
        Text(
            text = "Pick a frame and run cloud inference: image -> keypoints -> gesture id.",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.78f),
        )

        ServerStatusHeader(baseUrl = settings.apiServerBaseUrl)

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(14.dp)) {
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    OutlinedButton(
                        onClick = { picker.launch("image/*") },
                        enabled = !loading,
                    ) {
                        Text("Pick Photo")
                    }
                    Button(
                        onClick = {
                            val bytes = imageBytes ?: return@Button
                            loading = true
                            error = null
                            gestureId = null
                            gestureLabel = null
                            signTokens = emptyList()
                            keypointDim = null
                            scope.launch {
                                try {
                                    val res = SignVerseApi.analyzeFrame(
                                        baseUrl = settings.apiServerBaseUrl,
                                        filename = "frame.jpg",
                                        contentType = "image/jpeg",
                                        bytes = bytes,
                                    )
                                    keypointDim = res.dim
                                    gestureId = res.gestureId
                                    gestureLabel = res.gestureLabel
                                    signTokens = res.signTokens
                                    historyStore.addImageToGesture(
                                        gestureId = gestureId,
                                        gestureLabel = gestureLabel,
                                        dim = res.dim,
                                        error = null,
                                    )
                                } catch (e: Exception) {
                                    val msg = e.message ?: "request failed"
                                    error = msg
                                    historyStore.addImageToGesture(
                                        gestureId = null,
                                        gestureLabel = null,
                                        dim = null,
                                        error = msg,
                                    )
                                } finally {
                                    loading = false
                                }
                            }
                        },
                        enabled = !loading && imageBytes != null && settings.apiServerBaseUrl.isNotBlank(),
                    ) {
                        Text("Analyze")
                    }
                }

                if (loading) {
                    Spacer(modifier = Modifier.height(12.dp))
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                }

                if (previewBitmap != null) {
                    Spacer(modifier = Modifier.height(12.dp))
                    Card {
                        Image(
                            bitmap = previewBitmap!!,
                            contentDescription = "Selected image",
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(220.dp),
                            contentScale = ContentScale.Crop,
                        )
                    }
                } else {
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = "No image selected yet.",
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.72f),
                    )
                }

                if (gestureId != null) {
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = if (gestureLabel != null) "gesture = $gestureLabel" else "gesture_id = $gestureId",
                        style = MaterialTheme.typography.headlineMedium,
                    )
                    if (gestureLabel != null) {
                        Text(
                            text = "gesture_id = $gestureId",
                            style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.72f),
                        )
                    }
                    Text(
                        text = "keypoints dim = ${keypointDim ?: "?"}",
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.72f),
                    )
                    if (signTokens.isNotEmpty()) {
                        Spacer(modifier = Modifier.height(10.dp))
                        Text(
                            text = "Sign tokens",
                            style = MaterialTheme.typography.titleMedium,
                        )
                        Spacer(modifier = Modifier.height(6.dp))
                        TokenChipsRow(tokens = signTokens, modifier = Modifier.fillMaxWidth())
                    }
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

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(14.dp)) {
                Text(
                    text = "Next: Real-time Camera",
                    style = MaterialTheme.typography.headlineMedium,
                )
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "For <200ms streaming, we should either extract keypoints on-device (MediaPipe) and send them over WebSocket, or add an image-frame WebSocket on the server.",
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.78f),
                )
            }
        }
    }
}

private fun loadAndCompressImage(context: Context, uri: Uri): Pair<ByteArray, androidx.compose.ui.graphics.ImageBitmap>? {
    return try {
        val cr = context.contentResolver
        val original = cr.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) } ?: return null

        val scaled = original.scaleDown(maxSide = 960)
        val preview = scaled.asImageBitmap()

        val out = ByteArrayOutputStream()
        scaled.compress(Bitmap.CompressFormat.JPEG, 85, out)
        val bytes = out.toByteArray()
        Pair(bytes, preview)
    } catch (_: Exception) {
        null
    }
}

private fun Bitmap.scaleDown(maxSide: Int): Bitmap {
    val w = width
    val h = height
    if (w <= maxSide && h <= maxSide) return this

    val scale = if (w >= h) maxSide.toFloat() / w else maxSide.toFloat() / h
    val nw = (w * scale).toInt().coerceAtLeast(1)
    val nh = (h * scale).toInt().coerceAtLeast(1)
    return Bitmap.createScaledBitmap(this, nw, nh, true)
}
