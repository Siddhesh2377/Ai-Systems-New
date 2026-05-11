package com.dark.demon_system.data

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.util.zip.ZipInputStream
import kotlin.coroutines.coroutineContext

/**
 * Streaming ZIP extractor with progress reporting and path-traversal protection.
 *
 * Strips a single shared root directory if all entries are nested inside it
 * (matches the convention used by the xororz QNN bundles).
 */
class ZipExtractor {

    data class Progress(
        val filesExtracted: Int,
        val currentFile: String,
        val totalUncompressedBytes: Long,
        val uncompressedBytesSoFar: Long
    )

    suspend fun extract(
        zipFile: File,
        targetDir: File,
        bufferSize: Int = 64 * 1024,
        onProgress: suspend (Progress) -> Unit
    ): File = withContext(Dispatchers.IO) {
        if (!zipFile.exists()) throw IOException("Zip not found: ${zipFile.absolutePath}")
        targetDir.mkdirs()

        val rootName = detectSharedRoot(zipFile)
        val canonicalTarget = targetDir.canonicalPath
        var filesExtracted = 0
        var bytesSoFar = 0L

        zipFile.inputStream().buffered(bufferSize).use { fis ->
            ZipInputStream(fis).use { zis ->
                val buf = ByteArray(bufferSize)
                while (true) {
                    coroutineContext.ensureActive()
                    val entry = zis.nextEntry ?: break
                    val rawName = entry.name
                    val rel = if (rootName != null && rawName.startsWith("$rootName/")) {
                        rawName.substring(rootName.length + 1)
                    } else rawName

                    if (rel.isEmpty()) {
                        zis.closeEntry()
                        continue
                    }

                    val out = File(targetDir, rel)
                    if (!out.canonicalPath.startsWith(canonicalTarget + File.separator) &&
                        out.canonicalPath != canonicalTarget) {
                        throw SecurityException("Path traversal in zip entry: $rawName")
                    }

                    if (entry.isDirectory) {
                        out.mkdirs()
                        zis.closeEntry()
                        continue
                    }

                    out.parentFile?.mkdirs()
                    out.outputStream().use { fos ->
                        while (true) {
                            coroutineContext.ensureActive()
                            val n = zis.read(buf)
                            if (n < 0) break
                            fos.write(buf, 0, n)
                            bytesSoFar += n
                        }
                    }

                    filesExtracted++
                    onProgress(Progress(filesExtracted, rel, -1L, bytesSoFar))
                    zis.closeEntry()
                }
            }
        }

        // Final progress emit.
        onProgress(Progress(filesExtracted, "", -1L, bytesSoFar))
        targetDir
    }

    private fun detectSharedRoot(zipFile: File): String? {
        var root: String? = null
        zipFile.inputStream().buffered().use { fis ->
            ZipInputStream(fis).use { zis ->
                while (true) {
                    val e = zis.nextEntry ?: break
                    val name = e.name
                    val firstSlash = name.indexOf('/')
                    val candidate = if (firstSlash >= 0) name.substring(0, firstSlash) else null
                    if (candidate == null) {
                        // A top-level file means there's no single shared root.
                        zis.closeEntry()
                        return null
                    }
                    if (root == null) root = candidate
                    else if (root != candidate) {
                        zis.closeEntry()
                        return null
                    }
                    zis.closeEntry()
                }
            }
        }
        return root
    }
}
