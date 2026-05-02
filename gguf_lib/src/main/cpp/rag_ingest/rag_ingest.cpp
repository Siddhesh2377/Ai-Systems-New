#include "rag_ingest.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>

#include <android/log.h>
#define INGEST_TAG "RagIngest"
#define IGLOGI(...) __android_log_print(ANDROID_LOG_INFO,  INGEST_TAG, __VA_ARGS__)
#define IGLOGW(...) __android_log_print(ANDROID_LOG_WARN,  INGEST_TAG, __VA_ARGS__)
#define IGLOGE(...) __android_log_print(ANDROID_LOG_ERROR, INGEST_TAG, __VA_ARGS__)

int rag_ingest_extract_html(const uint8_t* bytes, size_t len, std::string& out);
int rag_ingest_extract_docx(const uint8_t* bytes, size_t len, std::string& out);
int rag_ingest_extract_epub(const uint8_t* bytes, size_t len, std::string& out);
int rag_ingest_extract_odt (const uint8_t* bytes, size_t len, std::string& out);
int rag_ingest_extract_pptx(const uint8_t* bytes, size_t len, std::string& out);
int rag_ingest_extract_xlsx(const uint8_t* bytes, size_t len, std::string& out);
int rag_ingest_extract_pdf (const uint8_t* bytes, size_t len, std::string& out);

namespace {

constexpr size_t kMaxInputBytes = 64u * 1024u * 1024u;

std::string to_lower(const char* s) {
    std::string r;
    if (!s) return r;
    r.reserve(strlen(s));
    for (const char* p = s; *p; ++p) r.push_back((char) std::tolower((unsigned char) *p));
    return r;
}

bool ends_with(const std::string& s, const char* suffix) {
    size_t sl = strlen(suffix);
    return s.size() >= sl && memcmp(s.data() + s.size() - sl, suffix, sl) == 0;
}

bool has_zip_magic(const uint8_t* b, size_t n) {
    return n >= 4 && b[0] == 'P' && b[1] == 'K' && (b[2] == 0x03 || b[2] == 0x05 || b[2] == 0x07);
}
bool has_pdf_magic(const uint8_t* b, size_t n) {
    return n >= 5 && b[0] == '%' && b[1] == 'P' && b[2] == 'D' && b[3] == 'F' && b[4] == '-';
}
bool has_rtf_magic(const uint8_t* b, size_t n) {
    return n >= 5 && b[0] == '{' && b[1] == '\\' && b[2] == 'r' && b[3] == 't' && b[4] == 'f';
}

bool looks_like_html(const uint8_t* b, size_t n) {
    size_t scan = n < 1024 ? n : 1024;
    for (size_t i = 0; i + 4 < scan; ++i) {
        if (b[i] == '<') {
            char c = (char) std::tolower(b[i + 1]);
            if (c == 'h' || c == 'b' || c == 'p' || c == '!' || c == 'd') return true;
        }
    }
    return false;
}

rag_doc_kind kind_from_mime(const std::string& mime) {
    if (mime.empty()) return RAG_DOC_UNKNOWN;
    if (mime == "application/pdf") return RAG_DOC_PDF;
    if (mime == "text/html" || mime == "application/xhtml+xml") return RAG_DOC_HTML;
    if (mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document") return RAG_DOC_DOCX;
    if (mime == "application/vnd.openxmlformats-officedocument.presentationml.presentation") return RAG_DOC_PPTX;
    if (mime == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet") return RAG_DOC_XLSX;
    if (mime == "application/epub+zip") return RAG_DOC_EPUB;
    if (mime == "application/vnd.oasis.opendocument.text") return RAG_DOC_ODT;
    if (mime == "application/rtf" || mime == "text/rtf") return RAG_DOC_RTF;
    if (mime.rfind("text/", 0) == 0) return RAG_DOC_TEXT;
    if (mime == "application/json" ||
        mime == "application/xml"  ||
        mime == "application/csv"  ||
        mime == "application/x-yaml") return RAG_DOC_TEXT;
    return RAG_DOC_UNKNOWN;
}

rag_doc_kind kind_from_name(const std::string& name) {
    if (name.empty()) return RAG_DOC_UNKNOWN;
    if (ends_with(name, ".pdf"))  return RAG_DOC_PDF;
    if (ends_with(name, ".docx")) return RAG_DOC_DOCX;
    if (ends_with(name, ".epub")) return RAG_DOC_EPUB;
    if (ends_with(name, ".odt"))  return RAG_DOC_ODT;
    if (ends_with(name, ".pptx")) return RAG_DOC_PPTX;
    if (ends_with(name, ".xlsx")) return RAG_DOC_XLSX;
    if (ends_with(name, ".rtf"))  return RAG_DOC_RTF;
    if (ends_with(name, ".html") || ends_with(name, ".htm") || ends_with(name, ".xhtml"))
        return RAG_DOC_HTML;
    if (ends_with(name, ".txt") || ends_with(name, ".md")   ||
        ends_with(name, ".csv") || ends_with(name, ".tsv")  ||
        ends_with(name, ".log") || ends_with(name, ".json") ||
        ends_with(name, ".yaml")|| ends_with(name, ".yml")  ||
        ends_with(name, ".toml")|| ends_with(name, ".ini")  ||
        ends_with(name, ".xml") || ends_with(name, ".srt")  ||
        ends_with(name, ".vtt"))
        return RAG_DOC_TEXT;
    return RAG_DOC_UNKNOWN;
}

rag_doc_kind kind_from_magic(const uint8_t* b, size_t n) {
    if (has_pdf_magic(b, n)) return RAG_DOC_PDF;
    if (has_rtf_magic(b, n)) return RAG_DOC_RTF;
    if (has_zip_magic(b, n)) return RAG_DOC_DOCX;
    if (looks_like_html(b, n)) return RAG_DOC_HTML;
    return RAG_DOC_UNKNOWN;
}

int utf16le_to_utf8(const uint8_t* src, size_t src_bytes, std::string& out) {
    out.clear();
    out.reserve(src_bytes);
    size_t i = 0;
    while (i + 1 < src_bytes) {
        uint32_t cp = (uint32_t) src[i] | ((uint32_t) src[i + 1] << 8);
        i += 2;
        if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < src_bytes) {
            uint32_t low = (uint32_t) src[i] | ((uint32_t) src[i + 1] << 8);
            if (low >= 0xDC00 && low <= 0xDFFF) {
                cp = 0x10000 + (((cp - 0xD800) << 10) | (low - 0xDC00));
                i += 2;
            }
        }
        if (cp < 0x80) {
            out.push_back((char) cp);
        } else if (cp < 0x800) {
            out.push_back((char) (0xC0 | (cp >> 6)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back((char) (0xE0 | (cp >> 12)));
            out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        } else {
            out.push_back((char) (0xF0 | (cp >> 18)));
            out.push_back((char) (0x80 | ((cp >> 12) & 0x3F)));
            out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        }
    }
    return RAG_INGEST_OK;
}

void drop_control_bytes(std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = (unsigned char) s[i];
        if (c == '\n' || c == '\r' || c == '\t' || c >= 0x20) out.push_back((char) c);
    }
    s.swap(out);
}

int extract_text_like(const uint8_t* bytes, size_t len, std::string& out) {
    if (len == 0) return RAG_INGEST_ERR_EMPTY;

    if (len >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF) {
        bytes += 3; len -= 3;
    } else if (len >= 2 && bytes[0] == 0xFF && bytes[1] == 0xFE) {
        return utf16le_to_utf8(bytes + 2, len - 2, out);
    } else if (len >= 2 && bytes[0] == 0xFE && bytes[1] == 0xFF) {
        std::string swapped;
        swapped.resize(len - 2);
        for (size_t i = 0; i + 1 < len - 2; i += 2) {
            swapped[i]     = (char) bytes[2 + i + 1];
            swapped[i + 1] = (char) bytes[2 + i];
        }
        return utf16le_to_utf8((const uint8_t*) swapped.data(), swapped.size(), out);
    }

    out.assign((const char*) bytes, len);
    drop_control_bytes(out);
    return RAG_INGEST_OK;
}

int extract_rtf(const uint8_t* bytes, size_t len, std::string& out) {
    out.clear();
    out.reserve(len);
    int brace_depth = 0;
    bool skip_group = false;
    int skip_depth = 0;

    for (size_t i = 0; i < len; ++i) {
        char c = (char) bytes[i];
        if (c == '\\') {
            if (i + 1 < len && (bytes[i + 1] == '\\' || bytes[i + 1] == '{' || bytes[i + 1] == '}')) {
                if (!skip_group) out.push_back((char) bytes[i + 1]);
                ++i;
                continue;
            }
            std::string word;
            size_t j = i + 1;
            while (j < len && std::isalpha(bytes[j])) { word.push_back((char) bytes[j]); ++j; }
            int param = 0; bool had_param = false; bool neg = false;
            if (j < len && bytes[j] == '-') { neg = true; ++j; }
            while (j < len && std::isdigit(bytes[j])) { had_param = true; param = param * 10 + (bytes[j] - '0'); ++j; }
            if (j < len && bytes[j] == ' ') ++j;

            if (word == "par" || word == "line" || word == "sect") out.push_back('\n');
            else if (word == "tab") out.push_back('\t');
            else if (word == "fonttbl" || word == "colortbl" || word == "stylesheet" ||
                     word == "info"    || word == "pict"     || word == "header"     ||
                     word == "footer"  || word == "object") {
                skip_group = true;
                skip_depth = brace_depth;
            } else if (word == "u" && had_param) {
                int cp = neg ? -param : param;
                if (cp < 0) cp += 0x10000;
                if (!skip_group) {
                    if (cp < 0x80) out.push_back((char) cp);
                    else if (cp < 0x800) {
                        out.push_back((char) (0xC0 | (cp >> 6)));
                        out.push_back((char) (0x80 | (cp & 0x3F)));
                    } else {
                        out.push_back((char) (0xE0 | (cp >> 12)));
                        out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
                        out.push_back((char) (0x80 | (cp & 0x3F)));
                    }
                }
            }
            i = j - 1;
        } else if (c == '{') {
            ++brace_depth;
        } else if (c == '}') {
            if (skip_group && brace_depth == skip_depth + 1) skip_group = false;
            --brace_depth;
        } else if (!skip_group) {
            if (c == '\r' || c == '\n') { /* soft breaks in RTF are not paragraph breaks */ }
            else out.push_back(c);
        }
    }

    drop_control_bytes(out);
    return out.empty() ? RAG_INGEST_ERR_EMPTY : RAG_INGEST_OK;
}

char* steal_to_cstr(std::string& s) {
    char* out = (char*) std::malloc(s.size() + 1);
    if (!out) return nullptr;
    std::memcpy(out, s.data(), s.size());
    out[s.size()] = '\0';
    return out;
}

}

extern "C" {

enum rag_doc_kind rag_ingest_detect_kind(
        const uint8_t* bytes, size_t len,
        const char* mime_hint, const char* name_hint) {
    rag_doc_kind k = kind_from_mime(to_lower(mime_hint));
    if (k != RAG_DOC_UNKNOWN && k != RAG_DOC_DOCX) return k;

    rag_doc_kind k2 = kind_from_name(to_lower(name_hint));
    if (k2 != RAG_DOC_UNKNOWN) return k2;

    if (k != RAG_DOC_UNKNOWN) return k;

    if (bytes && len > 0) return kind_from_magic(bytes, len);
    return RAG_DOC_UNKNOWN;
}

int rag_ingest_extract(
        const uint8_t* bytes, size_t len,
        const char* mime_hint, const char* name_hint,
        char** out_text) {
    if (!out_text) return RAG_INGEST_ERR_INTERNAL;
    *out_text = nullptr;

    if (!bytes || len == 0) return RAG_INGEST_ERR_EMPTY;
    if (len > kMaxInputBytes) {
        IGLOGW("document too large: %zu bytes (max %zu)", len, kMaxInputBytes);
        return RAG_INGEST_ERR_PARSE;
    }

    rag_doc_kind kind = rag_ingest_detect_kind(bytes, len, mime_hint, name_hint);
    IGLOGI("ingest kind=%s size=%zu", rag_ingest_kind_name(kind), len);

    std::string text;
    int rc = RAG_INGEST_ERR_UNSUPPORTED;
    switch (kind) {
        case RAG_DOC_TEXT:    rc = extract_text_like(bytes, len, text); break;
        case RAG_DOC_HTML:    rc = rag_ingest_extract_html(bytes, len, text); break;
        case RAG_DOC_RTF:     rc = extract_rtf(bytes, len, text); break;
        case RAG_DOC_PDF:     rc = rag_ingest_extract_pdf(bytes, len, text); break;
        case RAG_DOC_DOCX:    rc = rag_ingest_extract_docx(bytes, len, text); break;
        case RAG_DOC_EPUB:    rc = rag_ingest_extract_epub(bytes, len, text); break;
        case RAG_DOC_ODT:     rc = rag_ingest_extract_odt (bytes, len, text); break;
        case RAG_DOC_PPTX:    rc = rag_ingest_extract_pptx(bytes, len, text); break;
        case RAG_DOC_XLSX:    rc = rag_ingest_extract_xlsx(bytes, len, text); break;
        case RAG_DOC_UNKNOWN:
        default: {
            if (looks_like_html(bytes, len)) {
                rc = rag_ingest_extract_html(bytes, len, text);
            } else {
                rc = extract_text_like(bytes, len, text);
            }
            break;
        }
    }

    if (rc != RAG_INGEST_OK) {
        IGLOGW("ingest failed: rc=%d kind=%s", rc, rag_ingest_kind_name(kind));
        return rc;
    }
    if (text.empty()) return RAG_INGEST_ERR_EMPTY;

    *out_text = steal_to_cstr(text);
    return *out_text ? RAG_INGEST_OK : RAG_INGEST_ERR_OOM;
}

void rag_ingest_free_string(char* s) { std::free(s); }

const char* rag_ingest_kind_name(enum rag_doc_kind k) {
    switch (k) {
        case RAG_DOC_TEXT: return "text";
        case RAG_DOC_HTML: return "html";
        case RAG_DOC_PDF:  return "pdf";
        case RAG_DOC_DOCX: return "docx";
        case RAG_DOC_EPUB: return "epub";
        case RAG_DOC_ODT:  return "odt";
        case RAG_DOC_PPTX: return "pptx";
        case RAG_DOC_XLSX: return "xlsx";
        case RAG_DOC_RTF:  return "rtf";
        case RAG_DOC_UNKNOWN:
        default:           return "unknown";
    }
}

}
