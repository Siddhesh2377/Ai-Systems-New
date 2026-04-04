// Copyright (c) 2025 Dark Matter Labs
package com.dark.ai_sherpa

import android.content.res.AssetManager

object WaveReader {
    init { SherpaLib.init() }

    fun readFromFile(filename: String): WaveData = readWaveFromFile(filename)
    fun readFromAsset(assetManager: AssetManager, filename: String): WaveData =
        readWaveFromAsset(assetManager, filename)

    @JvmStatic private external fun readWaveFromFile(filename: String): WaveData
    @JvmStatic private external fun readWaveFromAsset(assetManager: AssetManager, filename: String): WaveData
}

object WaveWriter {
    init { SherpaLib.init() }

    fun writeToFile(filename: String, samples: FloatArray, sampleRate: Int): Boolean =
        writeWaveToFile(filename, samples, sampleRate)

    @JvmStatic private external fun writeWaveToFile(filename: String, samples: FloatArray, sampleRate: Int): Boolean
}
