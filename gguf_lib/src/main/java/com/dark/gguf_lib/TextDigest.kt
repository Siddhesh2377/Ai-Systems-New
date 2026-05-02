package com.dark.gguf_lib

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

object TextDigest {

    data class Options(
        val targetTokens: Int = 200,
        val weightQuery: Float = 0.40f,
        val weightCentrality: Float = 0.30f,
        val weightLead: Float = 0.15f,
        val weightEntity: Float = 0.15f,
        val mmrLambda: Float = 0.7f,
        val maxSentences: Int = 80,
        val minSentenceChars: Int = 20,
        val maxSentenceChars: Int = 600,
        val textrankIterations: Int = 30,
        val textrankDamping: Float = 0.85f,
    )

    suspend fun compress(
        text: String,
        query: String? = null,
        options: Options = Options(),
    ): String = withContext(Dispatchers.Default) {
        if (text.isBlank()) return@withContext ""
        GGUFNativeLib.nativeTextDigest(
            text,
            query,
            options.targetTokens,
            options.weightQuery,
            options.weightCentrality,
            options.weightLead,
            options.weightEntity,
            options.mmrLambda,
            options.maxSentences,
            options.minSentenceChars,
            options.maxSentenceChars,
            options.textrankIterations,
            options.textrankDamping,
        ).orEmpty()
    }
}
