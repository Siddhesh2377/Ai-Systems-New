package com.dark.unified_inference.image

sealed class ImageEvent {
    data class Progress(val progress: Float, val step: Int, val totalSteps: Int) : ImageEvent()
    data class IntermediateImage(val rgbData: ByteArray, val width: Int, val height: Int) : ImageEvent()
    data class Complete(val rgbData: ByteArray, val width: Int, val height: Int, val seed: Long) : ImageEvent()
    data class Error(val message: String) : ImageEvent()
}
