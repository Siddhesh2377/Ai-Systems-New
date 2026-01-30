package com.mp.ai_supertonic_tts.audio

import android.content.Context
import android.net.Uri
import com.mp.ai_supertonic_tts.SupertonicNativeLib
import com.mp.ai_supertonic_tts.models.AudioFormat
import com.mp.ai_supertonic_tts.models.SynthesisResult
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Save synthesis results to files in various audio formats.
 */
object AudioSaver {

    /**
     * Save a synthesis result to a file path.
     *
     * @param result Audio data to save
     * @param path Output file path
     * @param format Audio format
     * @param nativeLib JNI library for WAV encoding
     * @return true if saved successfully
     */
    fun save(
        result: SynthesisResult,
        path: String,
        format: AudioFormat,
        nativeLib: SupertonicNativeLib
    ): Boolean {
        return try {
            val bytes = toByteArray(result, format, nativeLib)
            FileOutputStream(File(path)).use { it.write(bytes) }
            true
        } catch (_: Exception) {
            false
        }
    }

    /**
     * Save a synthesis result to a URI (for SAF / content URIs).
     *
     * @param result Audio data to save
     * @param uri Output URI
     * @param context Android context for ContentResolver
     * @param format Audio format
     * @param nativeLib JNI library for WAV encoding
     * @return true if saved successfully
     */
    fun save(
        result: SynthesisResult,
        uri: Uri,
        context: Context,
        format: AudioFormat,
        nativeLib: SupertonicNativeLib
    ): Boolean {
        return try {
            val bytes = toByteArray(result, format, nativeLib)
            context.contentResolver.openOutputStream(uri)?.use { it.write(bytes) }
            true
        } catch (_: Exception) {
            false
        }
    }

    /**
     * Convert a synthesis result to a byte array in the specified format.
     */
    fun toByteArray(
        result: SynthesisResult,
        format: AudioFormat,
        nativeLib: SupertonicNativeLib
    ): ByteArray {
        return when (format) {
            AudioFormat.WAV_16 -> nativeLib.nativeEncodeWav16(
                result.audioData, result.sampleRate, result.channels
            )
            AudioFormat.WAV_32F -> nativeLib.nativeEncodeWav32f(
                result.audioData, result.sampleRate, result.channels
            )
            AudioFormat.PCM_16 -> nativeLib.nativeEncodePcm16(result.audioData)
            AudioFormat.PCM_32F -> {
                val buf = ByteBuffer.allocate(result.audioData.size * 4)
                buf.order(ByteOrder.LITTLE_ENDIAN)
                for (sample in result.audioData) {
                    buf.putFloat(sample)
                }
                buf.array()
            }
            AudioFormat.RAW_FLOAT -> {
                val buf = ByteBuffer.allocate(result.audioData.size * 4)
                buf.order(ByteOrder.nativeOrder())
                for (sample in result.audioData) {
                    buf.putFloat(sample)
                }
                buf.array()
            }
        }
    }
}
