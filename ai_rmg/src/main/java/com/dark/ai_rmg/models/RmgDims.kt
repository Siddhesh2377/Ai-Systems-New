package com.dark.ai_rmg.models

data class RmgDims(
    val dModel: Int,
    val nLayers: Int,
    val nHeads: Int,
    val nKvHeads: Int,
    val dHead: Int,
    val dFf: Int,
    val vocabSize: Int,
    val maxSeq: Int,
    val ropeTheta: Float,
    val rmsEps: Float,
    val ropeInterleaved: Boolean,
    val tieWordEmbeddings: Boolean
)
