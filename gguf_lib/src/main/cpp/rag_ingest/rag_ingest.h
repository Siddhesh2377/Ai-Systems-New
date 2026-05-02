#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    RAG_INGEST_OK                = 0,
    RAG_INGEST_ERR_UNSUPPORTED   = -1,
    RAG_INGEST_ERR_PARSE         = -2,
    RAG_INGEST_ERR_EMPTY         = -3,
    RAG_INGEST_ERR_OOM           = -4,
    RAG_INGEST_ERR_INTERNAL      = -5,
};

enum rag_doc_kind {
    RAG_DOC_UNKNOWN = 0,
    RAG_DOC_TEXT,
    RAG_DOC_HTML,
    RAG_DOC_PDF,
    RAG_DOC_DOCX,
    RAG_DOC_EPUB,
    RAG_DOC_ODT,
    RAG_DOC_PPTX,
    RAG_DOC_XLSX,
    RAG_DOC_RTF,
};

enum rag_doc_kind rag_ingest_detect_kind(
    const uint8_t* bytes, size_t len,
    const char* mime_hint,
    const char* name_hint);

int rag_ingest_extract(
    const uint8_t* bytes, size_t len,
    const char* mime_hint,
    const char* name_hint,
    char** out_text);

void rag_ingest_free_string(char* s);

const char* rag_ingest_kind_name(enum rag_doc_kind k);

#ifdef __cplusplus
}
#endif
