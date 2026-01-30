package com.mp.ai_supertonic_tts.models

import org.json.JSONObject
import java.io.File
import java.io.InputStream

/**
 * Voice style embeddings for controlling speaker identity.
 *
 * Each voice style contains:
 * - styleTtl: embeddings for the Text-to-Latent module [nStyleTtl][styleTtlDim]
 * - styleDp: embeddings for the Duration Predictor [nStyleDp][styleDpDim]
 *
 * Loaded from JSON files that ship with the Supertonic model (e.g. F1.json, M1.json).
 */
data class VoiceStyle(
    val name: String,
    val styleTtl: FloatArray,
    val styleTtlShape: LongArray,
    val styleDp: FloatArray,
    val styleDpShape: LongArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is VoiceStyle) return false
        return name == other.name
    }

    override fun hashCode(): Int = name.hashCode()

    override fun toString(): String = "VoiceStyle(name='$name', ttl=${styleTtlShape.contentToString()}, dp=${styleDpShape.contentToString()})"

    companion object {
        /**
         * Load a voice style from a JSON file path.
         *
         * JSON format:
         * ```json
         * {
         *   "style_ttl": { "data": [[...], ...], "dims": [1, N, D] },
         *   "style_dp":  { "data": [[...], ...], "dims": [1, M, E] }
         * }
         * ```
         */
        fun loadFromJson(path: String): VoiceStyle {
            val name = File(path).nameWithoutExtension
            val json = File(path).readText()
            return parseJson(json, name)
        }

        /**
         * Load a voice style from an InputStream.
         */
        fun loadFromInputStream(stream: InputStream, name: String): VoiceStyle {
            val json = stream.bufferedReader().readText()
            return parseJson(json, name)
        }

        private fun parseJson(json: String, name: String): VoiceStyle {
            val root = JSONObject(json)

            val ttlObj = root.getJSONObject("style_ttl")
            val ttlDimsArr = ttlObj.getJSONArray("dims")
            val ttlShape = LongArray(ttlDimsArr.length()) { ttlDimsArr.getLong(it) }
            val ttlData = ttlObj.getJSONArray("data")
            val ttlFlat = flattenNestedArray(ttlData)

            val dpObj = root.getJSONObject("style_dp")
            val dpDimsArr = dpObj.getJSONArray("dims")
            val dpShape = LongArray(dpDimsArr.length()) { dpDimsArr.getLong(it) }
            val dpData = dpObj.getJSONArray("data")
            val dpFlat = flattenNestedArray(dpData)

            return VoiceStyle(
                name = name,
                styleTtl = ttlFlat,
                styleTtlShape = ttlShape,
                styleDp = dpFlat,
                styleDpShape = dpShape
            )
        }

        /**
         * Recursively flatten a nested JSON array of numbers into a FloatArray.
         */
        private fun flattenNestedArray(arr: org.json.JSONArray): FloatArray {
            val result = mutableListOf<Float>()
            flattenRecursive(arr, result)
            return result.toFloatArray()
        }

        private fun flattenRecursive(arr: org.json.JSONArray, out: MutableList<Float>) {
            for (i in 0 until arr.length()) {
                val item = arr.get(i)
                if (item is org.json.JSONArray) {
                    flattenRecursive(item, out)
                } else {
                    out.add((item as Number).toFloat())
                }
            }
        }
    }
}
