package com.dark.gguf_lib

/**
 * Document type detected by the native ingester. Values must stay in sync
 * with `rag_ingest_kind_t` in `rag_ingest.h`.
 */
enum class DocKind(val nativeValue: Int, val label: String) {
    Unknown(0, "Unknown"),
    Text(1, "Text"),
    Html(2, "HTML"),
    Pdf(3, "PDF"),
    Docx(4, "DOCX"),
    Epub(5, "EPUB"),
    Odt(6, "ODT"),
    Pptx(7, "PPTX"),
    Xlsx(8, "XLSX"),
    Rtf(9, "RTF");

    val isSupported: Boolean get() = this != Unknown

    companion object {
        fun fromNative(v: Int): DocKind = entries.firstOrNull { it.nativeValue == v } ?: Unknown
    }
}
