package com.dark.demon_system.data

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import kotlin.coroutines.coroutineContext

/**
 * Streaming HTTP downloader with resume support.
 *
 * Writes to `<target>.part`, then renames to `target` on success. If the partial
 * file exists from a previous attempt, the next call resumes from its size using
 * a Range header. Cancellation closes the underlying connection so a blocked
 * socket read unblocks promptly.
 */
class ModelDownloader {

    data class Progress(
        val bytesDownloaded: Long,
        val totalBytes: Long,
        val bytesPerSecond: Long
    )

    suspend fun download(
        url: String,
        targetFile: File,
        bufferSize: Int = 64 * 1024,
        onProgress: suspend (Progress) -> Unit
    ): File = withContext(Dispatchers.IO) {
        if (targetFile.exists() && targetFile.length() > 0) return@withContext targetFile

        targetFile.parentFile?.mkdirs()
        val partFile = File(targetFile.parentFile, targetFile.name + ".part")
        val resumeFrom = if (partFile.exists()) partFile.length() else 0L

        coroutineScope {
            val conn = (URL(url).openConnection() as HttpURLConnection).apply {
                connectTimeout = 30_000
                readTimeout = 30_000
                instanceFollowRedirects = true
                requestMethod = "GET"
                if (resumeFrom > 0) {
                    setRequestProperty("Range", "bytes=$resumeFrom-")
                }
            }

            // Watchdog: closes the connection on cancellation so blocked reads unblock.
            val watchdog = launch {
                try {
                    awaitCancellation()
                } finally {
                    runCatching { conn.disconnect() }
                }
            }

            try {
                val code = conn.responseCode
                val resuming = code == HttpURLConnection.HTTP_PARTIAL && resumeFrom > 0
                if (code != HttpURLConnection.HTTP_OK && code != HttpURLConnection.HTTP_PARTIAL) {
                    throw IOException("HTTP $code: ${conn.responseMessage}")
                }

                if (resumeFrom > 0 && code == HttpURLConnection.HTTP_OK) {
                    // Server ignored Range — start over.
                    partFile.delete()
                }

                val contentLength = conn.contentLengthLong.takeIf { it >= 0 } ?: -1L
                val totalBytes = if (resuming && contentLength > 0) {
                    resumeFrom + contentLength
                } else if (contentLength > 0) {
                    contentLength
                } else -1L

                val startBytes = if (resuming) resumeFrom else 0L
                var bytesSoFar = startBytes
                val startNs = System.nanoTime()
                var lastEmitNs = 0L
                var lastEmitBytes = startBytes

                conn.inputStream.use { input ->
                    RandomAccessFile(partFile, "rw").use { raf ->
                        if (resuming) raf.seek(resumeFrom) else raf.setLength(0)

                        val buf = ByteArray(bufferSize)
                        while (true) {
                            coroutineContext.ensureActive()
                            val n = try {
                                input.read(buf)
                            } catch (e: IOException) {
                                if (coroutineContext[Job]?.isCancelled == true) {
                                    throw CancellationException("Download cancelled")
                                }
                                throw e
                            }
                            if (n < 0) break
                            raf.write(buf, 0, n)
                            bytesSoFar += n

                            val nowNs = System.nanoTime()
                            if (nowNs - lastEmitNs > 100_000_000L) { // 100ms
                                val elapsedSec = (nowNs - startNs) / 1_000_000_000.0
                                val avgBps = if (elapsedSec > 0) {
                                    ((bytesSoFar - startBytes) / elapsedSec).toLong()
                                } else 0L
                                onProgress(Progress(bytesSoFar, totalBytes, avgBps))
                                lastEmitNs = nowNs
                                lastEmitBytes = bytesSoFar
                            }
                        }
                    }
                }

                onProgress(Progress(bytesSoFar, totalBytes, 0))

                if (totalBytes > 0 && bytesSoFar < totalBytes) {
                    throw IOException("Truncated download: $bytesSoFar / $totalBytes bytes")
                }

                if (!partFile.renameTo(targetFile)) {
                    partFile.copyTo(targetFile, overwrite = true)
                    partFile.delete()
                }
                targetFile
            } finally {
                watchdog.cancel()
                runCatching { conn.disconnect() }
            }
        }
    }
}
