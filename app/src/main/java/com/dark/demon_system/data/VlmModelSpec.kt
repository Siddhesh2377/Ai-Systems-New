package com.dark.demon_system.data

/**
 * VLM (text + projector) model spec for HuggingFace download.
 *
 * HF resolve URL pattern:
 *   https://huggingface.co/{repo}/resolve/main/{filename}?download=true
 *
 * Both files land under context.filesDir/vlm_models/{repoSafe}/.
 */
data class VlmModelSpec(
    val displayName: String,
    val repo: String,
    val textFilename: String,
    val projFilename: String,
    val expectedTextBytes: Long,
    val expectedProjBytes: Long,
) {
    val repoSafeDir: String get() = repo.replace('/', '_')

    fun textUrl() = "https://huggingface.co/$repo/resolve/main/$textFilename?download=true"
    fun projUrl() = "https://huggingface.co/$repo/resolve/main/$projFilename?download=true"

    companion object {
        /**
         * Default mobile pick. Verified file list as of repo's `main`:
         *   Qwen3-VL-2B-Instruct-Q8_0.gguf      (1.83 GB)
         *   mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf (0.45 GB)
         * Total ~2.28 GB on disk; ~2.5-2.8 GB RAM at 4K ctx + Q8_0 KV.
         */
        val QWEN3_VL_2B = VlmModelSpec(
            displayName = "Qwen3-VL-2B-Instruct",
            repo = "ggml-org/Qwen3-VL-2B-Instruct-GGUF",
            textFilename = "Qwen3-VL-2B-Instruct-Q8_0.gguf",
            projFilename = "mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf",
            expectedTextBytes = 1_834_427_296L,
            expectedProjBytes = 445_053_056L,
        )

        /**
         * Tiny LFM2-VL test pick — under 500 MB total, ideal for fast
         * pipeline validation on bandwidth-limited connections.
         *   LFM2-VL-450M-Q8_0.gguf      (361.6 MB)
         *   mmproj-LFM2-VL-450M-Q8_0.gguf (99.1 MB)
         */
        val LFM2_VL_450M = VlmModelSpec(
            displayName = "LFM2-VL-450M",
            repo = "LiquidAI/LFM2-VL-450M-GGUF",
            textFilename = "LFM2-VL-450M-Q8_0.gguf",
            projFilename = "mmproj-LFM2-VL-450M-Q8_0.gguf",
            expectedTextBytes = 379_180_000L,    // ~361.6 MiB
            expectedProjBytes = 103_915_000L,    // ~99.1 MiB
        )
    }
}
