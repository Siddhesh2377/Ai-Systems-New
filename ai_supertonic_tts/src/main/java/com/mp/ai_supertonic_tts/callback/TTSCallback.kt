package com.mp.ai_supertonic_tts.callback

import com.mp.ai_supertonic_tts.models.SynthesisResult

/**
 * Callback interface for monitoring TTS synthesis progress.
 *
 * All methods have default no-op implementations so you only need
 * to override the ones you care about.
 */
interface TTSCallback {
    /** Called when synthesis begins */
    fun onSynthesisStart(textLength: Int, chunkCount: Int) {}

    /** Called after each chunk is synthesized (for multi-chunk text) */
    fun onChunkProgress(chunkIndex: Int, totalChunks: Int) {}

    /** Called when the final audio is ready */
    fun onAudioReady(result: SynthesisResult) {}

    /** Called on error */
    fun onError(error: String) {}
}
