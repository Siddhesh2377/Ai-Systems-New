package com.mp.ai_supertonic_tts.engine

import com.mp.ai_supertonic_tts.models.Language
import java.util.regex.Pattern

/**
 * Splits long text into chunks at sentence boundaries for
 * independent synthesis. Each chunk is synthesized separately
 * and concatenated with silence gaps.
 */
object TextChunker {

    private const val DEFAULT_MAX_LEN = 300
    private const val KOREAN_MAX_LEN = 120

    private val ABBREVIATIONS = arrayOf(
        "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sr.", "Jr.",
        "St.", "Ave.", "Rd.", "Blvd.", "Dept.", "Inc.", "Ltd.",
        "Co.", "Corp.", "etc.", "vs.", "i.e.", "e.g.", "Ph.D."
    )

    /**
     * Split text into chunks respecting sentence boundaries.
     *
     * @param text Input text
     * @param language Target language (Korean uses shorter max length)
     * @return List of text chunks
     */
    fun chunk(text: String, language: Language = Language.EN): List<String> {
        val maxLen = if (language == Language.KO) KOREAN_MAX_LEN else DEFAULT_MAX_LEN
        return chunkText(text.trim(), maxLen)
    }

    private fun chunkText(text: String, maxLen: Int): List<String> {
        if (text.isEmpty()) return listOf("")
        if (text.length <= maxLen) return listOf(text)

        // Split by paragraphs first
        val paragraphs = text.split(Regex("\\n\\s*\\n"))
        val chunks = mutableListOf<String>()

        for (rawPara in paragraphs) {
            val para = rawPara.trim()
            if (para.isEmpty()) continue

            if (para.length <= maxLen) {
                chunks.add(para)
                continue
            }

            // Split by sentences
            val sentences = splitSentences(para)
            val current = StringBuilder()
            var currentLen = 0

            for (rawSentence in sentences) {
                val sentence = rawSentence.trim()
                if (sentence.isEmpty()) continue

                val sentenceLen = sentence.length

                if (sentenceLen > maxLen) {
                    // Flush current buffer
                    if (current.isNotEmpty()) {
                        chunks.add(current.toString().trim())
                        current.clear()
                        currentLen = 0
                    }
                    // Split long sentence by comma
                    splitByCommaOrSpace(sentence, maxLen, chunks)
                    continue
                }

                if (currentLen + sentenceLen + 1 > maxLen && current.isNotEmpty()) {
                    chunks.add(current.toString().trim())
                    current.clear()
                    currentLen = 0
                }

                if (current.isNotEmpty()) {
                    current.append(" ")
                    currentLen++
                }
                current.append(sentence)
                currentLen += sentenceLen
            }

            if (current.isNotEmpty()) {
                chunks.add(current.toString().trim())
            }
        }

        return if (chunks.isEmpty()) listOf("") else chunks
    }

    private fun splitSentences(text: String): List<String> {
        val abbrevPattern = ABBREVIATIONS.joinToString("|") { Pattern.quote(it) }
        val pattern = Pattern.compile("(?<!(?:$abbrevPattern))(?<=[.!?])\\s+")
        return pattern.split(text).toList()
    }

    private fun splitByCommaOrSpace(text: String, maxLen: Int, chunks: MutableList<String>) {
        val parts = text.split(",")
        val current = StringBuilder()
        var currentLen = 0

        for (rawPart in parts) {
            val part = rawPart.trim()
            if (part.isEmpty()) continue

            if (part.length > maxLen) {
                // Last resort: split by space
                if (current.isNotEmpty()) {
                    chunks.add(current.toString().trim())
                    current.clear()
                    currentLen = 0
                }
                splitBySpace(part, maxLen, chunks)
                continue
            }

            if (currentLen + part.length + 2 > maxLen && current.isNotEmpty()) {
                chunks.add(current.toString().trim())
                current.clear()
                currentLen = 0
            }

            if (current.isNotEmpty()) {
                current.append(", ")
                currentLen += 2
            }
            current.append(part)
            currentLen += part.length
        }

        if (current.isNotEmpty()) {
            chunks.add(current.toString().trim())
        }
    }

    private fun splitBySpace(text: String, maxLen: Int, chunks: MutableList<String>) {
        val words = text.split(Regex("\\s+"))
        val current = StringBuilder()
        var currentLen = 0

        for (word in words) {
            if (currentLen + word.length + 1 > maxLen && current.isNotEmpty()) {
                chunks.add(current.toString().trim())
                current.clear()
                currentLen = 0
            }

            if (current.isNotEmpty()) {
                current.append(" ")
                currentLen++
            }
            current.append(word)
            currentLen += word.length
        }

        if (current.isNotEmpty()) {
            chunks.add(current.toString().trim())
        }
    }
}
