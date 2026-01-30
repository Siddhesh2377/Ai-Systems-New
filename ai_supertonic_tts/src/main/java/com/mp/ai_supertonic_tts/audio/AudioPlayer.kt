package com.mp.ai_supertonic_tts.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import com.mp.ai_supertonic_tts.models.SynthesisResult

/**
 * Audio playback via Android AudioTrack.
 *
 * Supports playing float32 audio from SynthesisResult directly
 * without needing to convert to PCM first (AudioTrack supports
 * ENCODING_PCM_FLOAT natively on API 21+).
 */
class AudioPlayer {

    private var audioTrack: AudioTrack? = null

    /**
     * Play a synthesis result.
     *
     * This blocks the calling thread until playback finishes.
     * Call from a background thread or coroutine.
     */
    fun play(result: SynthesisResult) {
        stop()

        val bufferSize = AudioTrack.getMinBufferSize(
            result.sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT
        )

        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(result.sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(maxOf(bufferSize, result.audioData.size * 4))
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()

        audioTrack = track

        track.write(result.audioData, 0, result.audioData.size, AudioTrack.WRITE_BLOCKING)
        track.play()

        // Wait for playback to complete
        val durationMs = (result.audioData.size.toLong() * 1000) / result.sampleRate
        Thread.sleep(durationMs + 100) // Small buffer for playback latency

        track.stop()
    }

    /**
     * Play audio using streaming mode (for large audio or real-time use).
     *
     * Writes audio in chunks to avoid allocating a huge static buffer.
     */
    fun playStreaming(result: SynthesisResult) {
        stop()

        val bufferSize = AudioTrack.getMinBufferSize(
            result.sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT
        )

        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(result.sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(maxOf(bufferSize, 4096 * 4))
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()

        audioTrack = track
        track.play()

        // Write in chunks
        val chunkSize = 4096
        var offset = 0
        while (offset < result.audioData.size) {
            val remaining = result.audioData.size - offset
            val writeSize = minOf(chunkSize, remaining)
            track.write(result.audioData, offset, writeSize, AudioTrack.WRITE_BLOCKING)
            offset += writeSize
        }

        // Wait for remaining buffer to drain
        track.stop()
    }

    fun stop() {
        audioTrack?.let { track ->
            try {
                if (track.playState == AudioTrack.PLAYSTATE_PLAYING) {
                    track.stop()
                }
                track.release()
            } catch (_: Exception) {
                // Ignore cleanup errors
            }
        }
        audioTrack = null
    }

    fun pause() {
        audioTrack?.let { track ->
            if (track.playState == AudioTrack.PLAYSTATE_PLAYING) {
                track.pause()
            }
        }
    }

    fun resume() {
        audioTrack?.let { track ->
            if (track.playState == AudioTrack.PLAYSTATE_PAUSED) {
                track.play()
            }
        }
    }

    fun isPlaying(): Boolean {
        return audioTrack?.playState == AudioTrack.PLAYSTATE_PLAYING
    }

    fun setVolume(volume: Float) {
        audioTrack?.setVolume(volume.coerceIn(0f, 1f))
    }

    fun release() {
        stop()
    }
}
