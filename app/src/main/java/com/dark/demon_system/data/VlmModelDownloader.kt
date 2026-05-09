package com.dark.demon_system.data

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Downloads a [VlmModelSpec]'s text + projector pair into
 * `context.filesDir/vlm_models/{repoSafeDir}/` and emits unified progress.
 *
 * Uses [ModelDownloader] under the hood — resumable via Range header, writes
 * to `.part` then renames on completion. Skips files that already exist.
 *
 * Built on `channelFlow` so progress emissions can come from arbitrary
 * downstream coroutine contexts (the inner downloader switches to
 * Dispatchers.IO via withContext, which would otherwise violate the
 * `flow { }` context-preservation invariant).
 */
class VlmModelDownloader(private val context: Context) {

    sealed interface Event {
        data class FileProgress(
            val fileIndex: Int,           // 0 = text, 1 = projector
            val fileName: String,
            val bytesDownloaded: Long,
            val totalBytes: Long,
            val bytesPerSecond: Long,
            val overallPct: Float,        // 0.0 .. 1.0 across both files
        ) : Event

        data class Done(val textPath: String, val projPath: String) : Event
        data class Failed(val message: String) : Event
    }

    fun targetDir(spec: VlmModelSpec): File =
        File(context.filesDir, "vlm_models/${spec.repoSafeDir}").apply { mkdirs() }

    fun textFile(spec: VlmModelSpec): File = File(targetDir(spec), spec.textFilename)
    fun projFile(spec: VlmModelSpec): File = File(targetDir(spec), spec.projFilename)

    fun isFullyDownloaded(spec: VlmModelSpec): Boolean {
        val t = textFile(spec)
        val p = projFile(spec)
        return t.exists() && t.length() > 0 && p.exists() && p.length() > 0
    }

    fun download(spec: VlmModelSpec): Flow<Event> = channelFlow {
        val downloader = ModelDownloader()
        val text = textFile(spec)
        val proj = projFile(spec)
        val totalExpected = spec.expectedTextBytes + spec.expectedProjBytes

        try {
            withContext(Dispatchers.IO) {
                // File 0: text
                if (!(text.exists() && text.length() > 0)) {
                    downloader.download(spec.textUrl(), text) { p ->
                        val total = if (p.totalBytes > 0) p.totalBytes else spec.expectedTextBytes
                        val overall = p.bytesDownloaded.toFloat() / totalExpected
                        trySend(
                            Event.FileProgress(
                                fileIndex = 0,
                                fileName = spec.textFilename,
                                bytesDownloaded = p.bytesDownloaded,
                                totalBytes = total,
                                bytesPerSecond = p.bytesPerSecond,
                                overallPct = overall.coerceIn(0f, 1f),
                            )
                        )
                    }
                }

                // File 1: projector
                if (!(proj.exists() && proj.length() > 0)) {
                    downloader.download(spec.projUrl(), proj) { p ->
                        val total = if (p.totalBytes > 0) p.totalBytes else spec.expectedProjBytes
                        val overall =
                            (spec.expectedTextBytes + p.bytesDownloaded).toFloat() / totalExpected
                        trySend(
                            Event.FileProgress(
                                fileIndex = 1,
                                fileName = spec.projFilename,
                                bytesDownloaded = p.bytesDownloaded,
                                totalBytes = total,
                                bytesPerSecond = p.bytesPerSecond,
                                overallPct = overall.coerceIn(0f, 1f),
                            )
                        )
                    }
                }
            }

            send(Event.Done(textPath = text.absolutePath, projPath = proj.absolutePath))
        } catch (t: Throwable) {
            send(Event.Failed(t.message ?: t::class.java.simpleName))
        }

        awaitClose { /* downloader cancellation is handled via parent scope cancel */ }
    }
}
