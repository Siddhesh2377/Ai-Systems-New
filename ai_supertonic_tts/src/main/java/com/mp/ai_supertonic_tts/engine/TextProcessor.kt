package com.mp.ai_supertonic_tts.engine

import com.mp.ai_supertonic_tts.models.Language
import org.json.JSONArray
import java.io.File
import java.io.InputStream
import java.text.Normalizer

/**
 * Text preprocessing pipeline for Supertonic TTS.
 *
 * Converts raw text into Unicode vocabulary indices + attention masks
 * for the ONNX models. No external phonemizer required — operates
 * directly on Unicode code points via unicode_indexer.json.
 */
class TextProcessor(private val unicodeIndexer: LongArray) {

    /**
     * Process text into model-ready tensors.
     *
     * @param text Input text
     * @param language Target language
     * @return Pair of (textIds as LongArray, textMask as FloatArray [1, 1, seqLen])
     */
    fun process(text: String, language: Language): TextProcessResult {
        val processed = preprocessText(text, language)
        val codePoints = textToCodePoints(processed)

        val textIds = LongArray(codePoints.size) { i ->
            val cp = codePoints[i]
            if (cp < unicodeIndexer.size) unicodeIndexer[cp] else 0L
        }

        val textMask = FloatArray(codePoints.size) { 1.0f }

        return TextProcessResult(textIds, textMask)
    }

    /**
     * Full text preprocessing pipeline (matching Java reference).
     */
    private fun preprocessText(text: String, language: Language): String {
        var t = text

        // NFKD normalization
        t = Normalizer.normalize(t, Normalizer.Form.NFKD)

        // Remove emojis
        t = removeEmojis(t)

        // Replace dashes and symbols
        t = t.replace("\u2013", "-")   // en dash
            .replace("\u2011", "-")    // non-breaking hyphen
            .replace("\u2014", "-")    // em dash
            .replace("_", " ")
            .replace("\u201C", "\"")   // left double quote
            .replace("\u201D", "\"")   // right double quote
            .replace("\u2018", "'")    // left single quote
            .replace("\u2019", "'")    // right single quote
            .replace("\u00B4", "'")    // acute accent
            .replace("`", "'")
            .replace("[", " ")
            .replace("]", " ")
            .replace("|", " ")
            .replace("/", " ")
            .replace("#", " ")
            .replace("\u2192", " ")    // right arrow
            .replace("\u2190", " ")    // left arrow

        // Remove special symbols
        t = t.replace(Regex("[♥☆♡©\\\\]"), "")

        // Replace known expressions
        t = t.replace("@", " at ")
            .replace("e.g.,", "for example, ")
            .replace("i.e.,", "that is, ")

        // Fix spacing around punctuation
        t = t.replace(" ,", ",")
            .replace(" .", ".")
            .replace(" !", "!")
            .replace(" ?", "?")
            .replace(" ;", ";")
            .replace(" :", ":")
            .replace(" '", "'")

        // Remove duplicate quotes
        while ("\"\"" in t) t = t.replace("\"\"", "\"")
        while ("''" in t) t = t.replace("''", "'")
        while ("``" in t) t = t.replace("``", "`")

        // Collapse whitespace
        t = t.replace(Regex("\\s+"), " ").trim()

        // Ensure text ends with punctuation
        if (!t.matches(Regex(".*[.!?;:,'\"\\u201C\\u201D\\u2018\\u2019)\\]}…。」』】〉》›»]$"))) {
            t += "."
        }

        // Wrap with language tags
        t = "<${language.tag}>$t</${language.tag}>"

        return t
    }

    /**
     * Remove emoji code points from text.
     */
    private fun removeEmojis(text: String): String {
        val sb = StringBuilder()
        var i = 0
        while (i < text.length) {
            val cp = Character.codePointAt(text, i)
            val charCount = Character.charCount(cp)

            val isEmoji = (cp in 0x1F600..0x1F64F) ||
                    (cp in 0x1F300..0x1F5FF) ||
                    (cp in 0x1F680..0x1F6FF) ||
                    (cp in 0x1F700..0x1F77F) ||
                    (cp in 0x1F780..0x1F7FF) ||
                    (cp in 0x1F800..0x1F8FF) ||
                    (cp in 0x1F900..0x1F9FF) ||
                    (cp in 0x1FA00..0x1FA6F) ||
                    (cp in 0x1FA70..0x1FAFF) ||
                    (cp in 0x2600..0x26FF) ||
                    (cp in 0x2700..0x27BF) ||
                    (cp in 0x1F1E6..0x1F1FF)

            if (!isEmoji) {
                sb.appendCodePoint(cp)
            }
            i += charCount
        }
        return sb.toString()
    }

    /**
     * Convert text to array of Unicode code point values.
     */
    private fun textToCodePoints(text: String): IntArray {
        return text.codePoints().toArray()
    }

    companion object {
        /**
         * Load the Unicode indexer from a JSON file.
         * The file is a flat JSON array of integers mapping codepoint -> vocab index.
         */
        fun loadUnicodeIndexer(path: String): LongArray {
            val json = File(path).readText()
            return parseIndexerJson(json)
        }

        /**
         * Load the Unicode indexer from an InputStream.
         */
        fun loadUnicodeIndexerFromStream(stream: InputStream): LongArray {
            val json = stream.bufferedReader().readText()
            return parseIndexerJson(json)
        }

        private fun parseIndexerJson(json: String): LongArray {
            val arr = JSONArray(json)
            return LongArray(arr.length()) { arr.getLong(it) }
        }
    }
}

/**
 * Result of text processing: token IDs and attention mask.
 */
data class TextProcessResult(
    /** Vocabulary indices for each code point [seqLen] */
    val textIds: LongArray,
    /** Attention mask (1.0 = valid, 0.0 = padding) [seqLen] */
    val textMask: FloatArray
) {
    val sequenceLength: Int get() = textIds.size

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is TextProcessResult) return false
        return textIds.contentEquals(other.textIds) && textMask.contentEquals(other.textMask)
    }

    override fun hashCode(): Int = textIds.contentHashCode()
}
