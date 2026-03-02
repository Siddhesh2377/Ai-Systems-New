package com.dark.ai_chatterbox

class ChatterboxNativeLib {
    // Lifecycle
    external fun nativeLoadModels(modelDir: String): Boolean
    external fun nativeLoadVoicePreset(styleDir: String): Boolean
    external fun nativeLoadTokenizer(tokenizerPath: String): Boolean
    external fun nativeRelease()
    external fun nativeIsLoaded(): Boolean
    external fun nativeIsVoiceLoaded(): Boolean

    // Synthesis
    external fun nativeSynthesize(text: String, callback: ChatterboxCallback): Boolean
    external fun nativeStop()

    // Tokenizer
    external fun nativeTokenize(text: String): LongArray?

    // Config
    external fun nativeSetRepetitionPenalty(penalty: Float)
    external fun nativeSetMaxTokens(maxTokens: Int)

    // Variant & exaggeration
    /** Set model variant: 0=TURBO, 1=ORIGINAL. Must be called BEFORE nativeLoadModels(). */
    external fun nativeSetVariant(variant: Int)
    /** Set emotion exaggeration (ORIGINAL variant only). 0.0=flat, 1.0=normal, 2.0=expressive. */
    external fun nativeSetExaggeration(exaggeration: Float)

    companion object {
        init {
            System.loadLibrary("ai_chatterbox")
        }
    }
}
