package com.dark.ai_mnn

class NativeMNNChat {
    companion object {
        // Used to load the 'ai_mnn' library on application startup.
        init {
            System.loadLibrary("ai_mnn")
        }
    }

    private external fun initNative(
        configPath: String?,
        history: List<String>?,
        mergedConfigStr: String?,
        configJsonStr: String?
    ): Long

    private external fun submitNative(
        instanceId: Long, input: String, keepHistory: Boolean, listener: GenerateProgressListener
    ): HashMap<String, Any>

    private external fun resetNative(instanceId: Long)

    private external fun getDebugInfoNative(instanceId: Long): String

    private external fun releaseNative(instanceId: Long)

    private external fun setWavformCallbackNative(
        instanceId: Long, listener: AudioDataListener?
    ): Boolean

    private external fun updateEnableAudioOutputNative(llmPtr: Long, enable: Boolean)


    private external fun updateMaxNewTokensNative(llmPtr: Long, maxNewTokens: Int)

    private external fun updateSystemPromptNative(llmPtr: Long, systemPrompt: String)

    private external fun updateAssistantPromptNative(llmPtr: Long, assistantPrompt: String)

    private external fun updateConfigNative(llmPtr: Long, configJson: String)

    private external fun getSystemPromptNative(llmPtr: Long): String?
}
