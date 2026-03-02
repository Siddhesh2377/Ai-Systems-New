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

    companion object {
        init {
            System.loadLibrary("ai_chatterbox")
        }
    }
}
