#include "rag_ingest.h"

#include "third_party/pdfium/include/fpdfview.h"
#include "third_party/pdfium/include/fpdf_text.h"

#include <mutex>
#include <string>
#include <vector>

namespace {

std::once_flag g_pdfium_init_once;
bool g_pdfium_init_ok = false;

void init_pdfium_once() {
    std::call_once(g_pdfium_init_once, []() {
        FPDF_LIBRARY_CONFIG cfg{};
        cfg.version = 2;
        cfg.m_pUserFontPaths = nullptr;
        cfg.m_pIsolate = nullptr;
        cfg.m_v8EmbedderSlot = 0;
        FPDF_InitLibraryWithConfig(&cfg);
        g_pdfium_init_ok = true;
    });
}

void append_utf16le_to_utf8(const unsigned short* src, int len, std::string& out) {
    for (int i = 0; i < len; ++i) {
        uint32_t cp = src[i];
        if (cp == 0) break;
        if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < len) {
            uint32_t low = src[i + 1];
            if (low >= 0xDC00 && low <= 0xDFFF) {
                cp = 0x10000 + (((cp - 0xD800) << 10) | (low - 0xDC00));
                ++i;
            }
        }
        if (cp < 0x80) out.push_back((char) cp);
        else if (cp < 0x800) {
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
}

void normalize_spaces(std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool prev_space = true;
    bool prev_nl = true;
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = (unsigned char) s[i];
        if (c == '\n') {
            if (!prev_nl) { out.push_back('\n'); prev_nl = true; prev_space = true; }
        } else if (c == '\r' || c == '\t' || c == ' ') {
            if (!prev_space) { out.push_back(' '); prev_space = true; }
        } else if (c < 0x20) {
            continue;
        } else {
            out.push_back((char) c);
            prev_space = false;
            prev_nl = false;
        }
    }
    while (!out.empty() && (out.back() == ' ' || out.back() == '\n')) out.pop_back();
    s.swap(out);
}

}

int rag_ingest_extract_pdf(const uint8_t* bytes, size_t len, std::string& out) {
    init_pdfium_once();
    if (!g_pdfium_init_ok) return RAG_INGEST_ERR_INTERNAL;

    FPDF_DOCUMENT doc = FPDF_LoadMemDocument(bytes, (int) len, nullptr);
    if (!doc) {
        unsigned long err = FPDF_GetLastError();
        if (err == FPDF_ERR_PASSWORD) return RAG_INGEST_ERR_UNSUPPORTED;
        return RAG_INGEST_ERR_PARSE;
    }

    int n_pages = FPDF_GetPageCount(doc);
    if (n_pages <= 0) { FPDF_CloseDocument(doc); return RAG_INGEST_ERR_EMPTY; }

    out.clear();
    out.reserve((size_t) n_pages * 1024);

    std::vector<unsigned short> buf;

    for (int p = 0; p < n_pages; ++p) {
        FPDF_PAGE page = FPDF_LoadPage(doc, p);
        if (!page) continue;
        FPDF_TEXTPAGE tp = FPDFText_LoadPage(page);
        if (!tp) { FPDF_ClosePage(page); continue; }

        int n_chars = FPDFText_CountChars(tp);
        if (n_chars > 0) {
            buf.resize((size_t) n_chars + 1);
            int got = FPDFText_GetText(tp, 0, n_chars, buf.data());
            if (got > 0) append_utf16le_to_utf8(buf.data(), got, out);
        }
        if (!out.empty() && out.back() != '\n') out.push_back('\n');

        FPDFText_ClosePage(tp);
        FPDF_ClosePage(page);
    }

    FPDF_CloseDocument(doc);

    normalize_spaces(out);
    return out.empty() ? RAG_INGEST_ERR_EMPTY : RAG_INGEST_OK;
}
