package com.dark.gguf_lib

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Extractive text digest — picks the highest-value sentences from a long
 * document and concatenates them, optionally biased toward a query.
 *
 * Pure-CPU, no model required. Useful for summarizing RAG chunks before
 * passing them to a generation model, or for compressing long contexts.
 */
object TextDigest {

    /**
     * Knobs for the extractive digest. Defaults are tuned for typical
     * news-style prose; for code or dialogue you may want lower [mmrLambda]
     * (more diversity) and higher [weightLead].
     *
     * @param targetTokens Soft target for the output token count.
     * @param weightQuery Weight on query-relevance. Set to 0 for unbiased summaries.
     * @param weightCentrality Weight on TextRank centrality (graph-based importance).
     * @param weightLead Weight on lead bias (sentences near the document start).
     * @param weightEntity Weight on named-entity density.
     * @param mmrLambda 0..1 — Maximum-Marginal-Relevance trade-off; lower = more diverse.
     * @param maxSentences Hard cap on the number of sentences in the output.
     * @param minSentenceChars Skip sentences shorter than this (filters list bullets, headers).
     * @param maxSentenceChars Truncate sentences longer than this.
     * @param textrankIterations Power-iteration count for centrality.
     * @param textrankDamping PageRank damping factor.
     */
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

    /**
     * Digest [text] down to roughly [Options.targetTokens] tokens.
     *
     * @param query Optional bias term — the digest will prefer sentences
     *              relevant to this query if provided.
     */
    suspend fun compress(
        text: String,
        query: String? = null,
        options: Options = Options(),
    ): String = withContext(Dispatchers.Default) {
        if (text.isBlank()) return@withContext ""
        GGUFNativeLib.nativeTextDigest(
            text, query,
            options.targetTokens,
            options.weightQuery, options.weightCentrality, options.weightLead, options.weightEntity,
            options.mmrLambda,
            options.maxSentences, options.minSentenceChars, options.maxSentenceChars,
            options.textrankIterations, options.textrankDamping,
        ).orEmpty()
    }
}
