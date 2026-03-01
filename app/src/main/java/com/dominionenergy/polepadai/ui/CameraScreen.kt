package com.dominionenergy.polepadai.ui

import android.net.Uri
import android.os.Environment
import android.util.Log
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddCircle
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.runtime.Composable
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
import java.io.File

@Composable
fun CameraScreen(
//    onImageCaptured: (Uri) -> Unit,
//    onError: (ImageCaptureException) -> Unit
) {
    // Camera state
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var lensFacing by remember { mutableStateOf(CameraSelector.LENS_FACING_BACK) }
    val cameraSelector = remember(lensFacing) {
        CameraSelector.Builder().requireLensFacing(lensFacing).build()
    }

    // Initialize CameraX use cases
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }

    // Configure image capture with high-quality settings
    val imageCapture = remember {
        ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
            .setFlashMode(ImageCapture.FLASH_MODE_AUTO)
            .build()
    }

    // Camera preview using AndroidView to integrate with Compose
    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            factory = { ctx ->
                // Configure preview view
                val previewView = PreviewView(ctx).apply {
                    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                }

                // Setup and bind use cases when camera provider is available
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()

                    // Configure preview use case
                    val preview = Preview.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                        .build()
                        .also {
                            it.setSurfaceProvider(previewView.surfaceProvider)
                        }

                    try {
                        // Unbind previous use cases before rebinding
                        cameraProvider.unbindAll()

                        // Bind camera to lifecycle
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            cameraSelector,
                            preview,
                            imageCapture
                        )
                    } catch (e: Exception) {
                        Log.e("CameraScreen", "Camera binding failed", e)
                    }
                }, ContextCompat.getMainExecutor(ctx))

                previewView
            },
            modifier = Modifier.fillMaxSize()
        )

        // Camera controls overlay
        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.Bottom,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 32.dp),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                // Camera flip button
                FloatingActionButton(
                    onClick = {
                        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK)
                            CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK
                    }
                ) {
                    Icon(Icons.Default.AddCircle, contentDescription = "Switch camera")
                }

                // Capture button
                FloatingActionButton(
                    onClick = {
                        // Create output file in cache directory
                        val photoFile = File(
                            context.cacheDir,
                            "IMG_${System.currentTimeMillis()}.jpg"
                        )

                        // Configure output options with metadata
                        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile)
                            .setMetadata(
                                ImageCapture.Metadata().apply {
                                    isReversedHorizontal = lensFacing == CameraSelector.LENS_FACING_FRONT
                                }
                            ).build()

                        // Capture the image
                        imageCapture.takePicture(
                            outputOptions,
                            ContextCompat.getMainExecutor(context),
                            object : ImageCapture.OnImageSavedCallback {
                                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
//                                    // Return the image URI to the caller
//                                    output.savedUri?.let { uri ->
//                                        onImageCaptured(uri)
//                                    } ?: onImageCaptured(Uri.fromFile(photoFile))
                                }

                                override fun onError(exception: ImageCaptureException) {
                                    Log.e("CameraScreen", "Image capture failed", exception)
                                    //onError(exception)
                                }
                            }
                        )
                    },
                    modifier = Modifier.size(72.dp)
                ) {
                    Icon(
                        Icons.Default.Refresh,
                        contentDescription = "Take photo",
                        modifier = Modifier.size(36.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.height(36.dp))
        }
    }
}
