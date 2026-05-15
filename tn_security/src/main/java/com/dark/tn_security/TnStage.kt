package com.dark.tn_security

/**
 * Stage in an operation where an error/log occurred. Numeric values mirror
 * `tn_stage` in tn_security.h. UI uses this to say "failed during UNet step
 * 14" not "failed".
 */
enum class TnStage(val value: Int) {
    UNSPECIFIED    (0),
    INIT           (10),
    LOAD           (20),
    WARMUP         (21),

    // text generation
    TOKENIZE       (30),
    PROMPT_EVAL    (40),
    DECODE         (41),
    SAMPLE         (42),
    DETOKENIZE     (43),

    // vlm
    VLM_PROJECT    (50),
    VLM_DECODE_IMG (51),
    VLM_TOKENIZE   (52),

    // speech
    STT_DECODE     (60),
    TTS_GENERATE   (61),
    AUDIO_ACCEPT   (62),

    // diffusion
    SD_UNET        (70),
    SD_CLIP        (71),
    SD_VAE         (72),
    SD_SCHEDULER   (73),
    SD_UPSCALE     (74),
    SD_SEGMENT     (75),
    SD_INPAINT     (76),
    SD_DEPTH       (77),
    SD_STYLE       (78),

    // rag
    RAG_INGEST     (80),
    RAG_EMBED      (81),
    RAG_QUERY      (82),

    // plugin
    PLUGIN_LOAD    (90),
    PLUGIN_EXEC    (91),

    // runtime setup
    ASSET_COPY     (100),
    ASSET_EXTRACT  (101),
    ASSET_PATCH    (102);

    companion object {
        fun fromInt(v: Int): TnStage = entries.firstOrNull { it.value == v } ?: UNSPECIFIED
    }
}
