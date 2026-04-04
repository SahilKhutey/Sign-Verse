package com.signverse.launcher.ui.screens

import android.Manifest
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
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
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.signverse.launcher.data.AppSettings
import com.signverse.launcher.network.SignVerseApi
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executor
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException


@Composable
fun LiveScreen(settings: AppSettings) {
    val context = LocalContext.current

    var hasCameraPermission by remember { mutableStateOf(false) }
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = { granted -> hasCameraPermission = granted }
    )

    var imageCapture by remember { mutableStateOf<ImageCapture?>(null) }
    var running by remember { mutableStateOf(false) }
    var loading by remember { mutableStateOf(false) }

    var gestureLabel by remember { mutableStateOf<String?>(null) }
    var gestureId by remember { mutableStateOf<Int?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    var frames by remember { mutableStateOf(0) }

    LaunchedEffect(Unit) {
        // Ask once when opening the screen.
        permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    LaunchedEffect(running, imageCapture, settings.apiServerBaseUrl) {
        val cap = imageCapture
        if (!running || cap == null) return@LaunchedEffect
        if (settings.apiServerBaseUrl.isBlank()) return@LaunchedEffect

        while (isActive && running) {
            loading = true
            error = null
            try {
                val jpeg = cap.captureJpeg(
                    executor = ContextCompat.getMainExecutor(context),
                    jpegQuality = 80,
                )
                val res = SignVerseApi.analyzeFrame(
                    baseUrl = settings.apiServerBaseUrl,
                    filename = "live.jpg",
                    contentType = "image/jpeg",
                    bytes = jpeg,
                )
                gestureId = res.gestureId
                gestureLabel = res.gestureLabel
                frames += 1
            } catch (e: Exception) {
                error = e.message ?: "capture failed"
            } finally {
                loading = false
            }

            delay(700)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(18.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Text(
            text = "Live",
            style = MaterialTheme.typography.displayLarge,
        )
        Text(
            text = "Continuous camera mode (demo). It captures a frame every ~0.7s and calls POST /analyze/frame.",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.78f),
        )

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(14.dp)) {
                Text("API Server", style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = if (settings.apiServerBaseUrl.isBlank()) "(not set)" else settings.apiServerBaseUrl,
                    style = MaterialTheme.typography.bodyLarge,
                )
            }
        }

        if (!hasCameraPermission) {
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(14.dp)) {
                    Text("Camera permission is required.", style = MaterialTheme.typography.headlineMedium)
                    Spacer(modifier = Modifier.height(10.dp))
                    Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) {
                        Text("Grant permission")
                    }
                }
            }
            return
        }

        Card(modifier = Modifier.fillMaxWidth()) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(240.dp)
            ) {
                CameraPreview(
                    modifier = Modifier.fillMaxWidth(),
                    onImageCaptureReady = { imageCapture = it },
                )
                Column(
                    modifier = Modifier
                        .align(Alignment.BottomStart)
                        .padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    val line = when {
                        gestureLabel != null -> "gesture=$gestureLabel"
                        gestureId != null -> "gesture_id=$gestureId"
                        else -> "No result yet"
                    }
                    Text(
                        text = line,
                        style = MaterialTheme.typography.headlineMedium,
                        color = MaterialTheme.colorScheme.onSurface,
                    )
                    Text(
                        text = "frames=$frames",
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
                    )
                }
            }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Button(
                onClick = { running = true },
                enabled = !running && settings.apiServerBaseUrl.isNotBlank(),
            ) { Text("Start") }
            OutlinedButton(
                onClick = { running = false },
                enabled = running,
            ) { Text("Stop") }
        }

        if (loading) {
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        }
        if (error != null) {
            Text(
                text = "Error: $error",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.secondary,
            )
        }

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(14.dp)) {
                Text("Note", style = MaterialTheme.typography.headlineMedium)
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "This is a network demo. For low latency, we should run pose extraction on-device (MediaPipe) and stream keypoints over WebSocket.",
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.78f),
                )
            }
        }
    }
}


@Composable
private fun CameraPreview(
    modifier: Modifier = Modifier,
    onImageCaptureReady: (ImageCapture) -> Unit,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val previewView = remember { PreviewView(context) }
    val executor: Executor = remember { ContextCompat.getMainExecutor(context) }

    // Keep a stable instance across recompositions.
    val capture = remember {
        ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .build()
    }

    DisposableEffect(lifecycleOwner) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener(
            {
                val cameraProvider = cameraProviderFuture.get()
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
                val selector = CameraSelector.DEFAULT_FRONT_CAMERA
                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        selector,
                        preview,
                        capture,
                    )
                    onImageCaptureReady(capture)
                } catch (_: Exception) {
                    // If binding fails, the UI will show "No result yet".
                }
            },
            executor
        )

        onDispose {
            runCatching {
                val cameraProvider = cameraProviderFuture.get()
                cameraProvider.unbindAll()
            }
        }
    }

    AndroidView(
        factory = { previewView },
        modifier = modifier,
    )
}


private suspend fun ImageCapture.captureJpeg(
    executor: Executor,
    jpegQuality: Int,
): ByteArray {
    return suspendCancellableCoroutine { cont ->
        takePicture(
            executor,
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    try {
                        val bytes = imageProxyToJpeg(image, jpegQuality)
                        cont.resume(bytes)
                    } catch (e: Exception) {
                        cont.resumeWithException(e)
                    } finally {
                        image.close()
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    cont.resumeWithException(exception)
                }
            }
        )
    }
}

private fun imageProxyToJpeg(image: ImageProxy, quality: Int): ByteArray {
    val planes = image.planes
    if (planes.size < 3) {
        throw IllegalStateException("Unexpected image format")
    }

    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    // NV21 format: YYYY... VU VU...
    val nv21 = ByteArray(ySize + uSize + vSize)
    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuv = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
    val out = ByteArrayOutputStream()
    yuv.compressToJpeg(Rect(0, 0, image.width, image.height), quality.coerceIn(50, 95), out)
    return out.toByteArray()
}
