package com.dark.ai_rmg

internal interface RmgTokenCallback {
    // Bound by JNI signature (I[B)Z. Return false to abort generation.
    fun onToken(tokenId: Int, bytes: ByteArray?): Boolean
}

internal object RmgNativeLib {
    init {
        System.loadLibrary("ai_rmg")
        nativeInit()
    }

    @JvmStatic external fun nativeInit()
    @JvmStatic external fun nativeSetLogLevel(level: Int)

    @JvmStatic external fun nativeOpen(path: String): Long
    @JvmStatic external fun nativeClose(handle: Long)
    @JvmStatic external fun nativeReset(handle: Long)
    @JvmStatic external fun nativeSeqPos(handle: Long): Int

    @JvmStatic external fun nativeGetDims(handle: Long): IntArray?

    @JvmStatic external fun nativeForward(
        handle: Long,
        tokenId: Int,
        logitsOut: FloatArray
    ): Boolean

    @JvmStatic external fun nativeGenerate(
        handle: Long,
        promptIds: IntArray,
        maxNew: Int,
        stopId: Int
    ): IntArray?

    @JvmStatic external fun nativeGenerateStream(
        handle: Long,
        promptIds: IntArray,
        maxNew: Int,
        stopId: Int,
        callback: RmgTokenCallback
    ): Int

    @JvmStatic external fun nativeHasTokenizer(handle: Long): Boolean
    @JvmStatic external fun nativeTokenBytes(handle: Long, tokenId: Int): ByteArray?
    @JvmStatic external fun nativeDecodeTokens(handle: Long, tokenIds: IntArray): ByteArray?
}
