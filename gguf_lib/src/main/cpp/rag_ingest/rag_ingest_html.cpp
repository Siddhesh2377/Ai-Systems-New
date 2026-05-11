#include "rag_ingest.h"

#include <cctype>
#include <cstring>
#include <string>
#include <unordered_map>

namespace {

const std::unordered_map<std::string, const char*> kEntities = {
    {"amp","&"}, {"lt","<"}, {"gt",">"}, {"quot","\""}, {"apos","'"},
    {"nbsp"," "}, {"copy","(c)"}, {"reg","(r)"}, {"trade","(tm)"},
    {"hellip","..."}, {"mdash","—"}, {"ndash","–"},
    {"lsquo","'"}, {"rsquo","'"}, {"ldquo","\""}, {"rdquo","\""},
    {"bull","•"}, {"middot","·"},
};

bool eq_ci(const char* a, size_t alen, const char* b) {
    size_t blen = std::strlen(b);
    if (alen != blen) return false;
    for (size_t i = 0; i < alen; ++i)
        if (std::tolower((unsigned char) a[i]) != std::tolower((unsigned char) b[i])) return false;
    return true;
}

bool starts_with_ci(const uint8_t* b, size_t n, size_t i, const char* s) {
    size_t sl = std::strlen(s);
    if (i + sl > n) return false;
    for (size_t k = 0; k < sl; ++k)
        if (std::tolower((unsigned char) b[i + k]) != std::tolower((unsigned char) s[k])) return false;
    return true;
}

void append_cp(std::string& out, uint32_t cp) {
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

void decode_entity(const char* name, size_t len, std::string& out) {
    if (len == 0) { out.append("&"); return; }
    if (name[0] == '#') {
        uint32_t cp = 0;
        if (len >= 2 && (name[1] == 'x' || name[1] == 'X')) {
            for (size_t i = 2; i < len; ++i) {
                char c = name[i];
                if (c >= '0' && c <= '9') cp = cp * 16 + (c - '0');
                else if (c >= 'a' && c <= 'f') cp = cp * 16 + 10 + (c - 'a');
                else if (c >= 'A' && c <= 'F') cp = cp * 16 + 10 + (c - 'A');
                else { out.append("&"); out.append(name, len); out.append(";"); return; }
            }
        } else {
            for (size_t i = 1; i < len; ++i) {
                char c = name[i];
                if (c < '0' || c > '9') { out.append("&"); out.append(name, len); out.append(";"); return; }
                cp = cp * 10 + (c - '0');
            }
        }
        if (cp == 0 || cp > 0x10FFFF) return;
        append_cp(out, cp);
        return;
    }
    auto it = kEntities.find(std::string(name, len));
    if (it != kEntities.end()) out.append(it->second);
    else { out.append("&"); out.append(name, len); out.append(";"); }
}

bool is_block_tag(const char* tag, size_t len) {
    return eq_ci(tag, len, "p") || eq_ci(tag, len, "div") ||
           eq_ci(tag, len, "br") || eq_ci(tag, len, "li") ||
           eq_ci(tag, len, "tr") || eq_ci(tag, len, "td") || eq_ci(tag, len, "th") ||
           eq_ci(tag, len, "h1") || eq_ci(tag, len, "h2") || eq_ci(tag, len, "h3") ||
           eq_ci(tag, len, "h4") || eq_ci(tag, len, "h5") || eq_ci(tag, len, "h6") ||
           eq_ci(tag, len, "section") || eq_ci(tag, len, "article") ||
           eq_ci(tag, len, "header")  || eq_ci(tag, len, "footer") ||
           eq_ci(tag, len, "nav")     || eq_ci(tag, len, "main") ||
           eq_ci(tag, len, "aside")   || eq_ci(tag, len, "blockquote") ||
           eq_ci(tag, len, "pre")     || eq_ci(tag, len, "ul") || eq_ci(tag, len, "ol") ||
           eq_ci(tag, len, "figure")  || eq_ci(tag, len, "figcaption") ||
           eq_ci(tag, len, "table");
}

bool is_skip_tag(const char* tag, size_t len) {
    return eq_ci(tag, len, "script") || eq_ci(tag, len, "style") ||
           eq_ci(tag, len, "svg")    || eq_ci(tag, len, "noscript") ||
           eq_ci(tag, len, "template");
}

void collapse_whitespace(std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool prev_nl = true;
    bool prev_space = true;
    for (char c : s) {
        if (c == '\n') {
            if (!prev_nl) { out.push_back('\n'); prev_nl = true; prev_space = true; }
        } else if (c == ' ' || c == '\t' || c == '\r') {
            if (!prev_space) { out.push_back(' '); prev_space = true; }
        } else {
            out.push_back(c);
            prev_nl = false; prev_space = false;
        }
    }
    while (!out.empty() && (out.back() == ' ' || out.back() == '\n')) out.pop_back();
    s.swap(out);
}

}

int rag_ingest_extract_html(const uint8_t* bytes, size_t len, std::string& out) {
    out.clear();
    out.reserve(len);

    size_t i = 0;
    if (len >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF) i = 3;

    while (i < len) {
        uint8_t c = bytes[i];

        if (c == '<') {
            if (i + 3 < len && bytes[i + 1] == '!' && bytes[i + 2] == '-' && bytes[i + 3] == '-') {
                size_t j = i + 4;
                while (j + 2 < len &&
                       !(bytes[j] == '-' && bytes[j + 1] == '-' && bytes[j + 2] == '>')) ++j;
                i = (j + 3 < len) ? j + 3 : len;
                continue;
            }
            if (i + 1 < len && bytes[i + 1] == '!') {
                size_t j = i + 2;
                while (j < len && bytes[j] != '>') ++j;
                i = (j < len) ? j + 1 : len;
                continue;
            }

            bool closing = (i + 1 < len && bytes[i + 1] == '/');
            size_t name_start = i + (closing ? 2 : 1);
            size_t name_end = name_start;
            while (name_end < len) {
                char nc = (char) bytes[name_end];
                if (nc == ' ' || nc == '\t' || nc == '\r' || nc == '\n' ||
                    nc == '>' || nc == '/') break;
                ++name_end;
            }
            size_t tag_len = name_end - name_start;
            const char* tag = (const char*) &bytes[name_start];

            if (!closing && is_skip_tag(tag, tag_len)) {
                size_t j = name_end;
                std::string close_tag = "</";
                close_tag.append(tag, tag_len);
                while (j < len && !starts_with_ci(bytes, len, j, close_tag.c_str())) ++j;
                if (j < len) {
                    size_t k = j + close_tag.size();
                    while (k < len && bytes[k] != '>') ++k;
                    i = (k < len) ? k + 1 : len;
                } else {
                    i = len;
                }
                continue;
            }

            size_t j = name_end;
            while (j < len && bytes[j] != '>') ++j;
            i = (j < len) ? j + 1 : len;

            if (is_block_tag(tag, tag_len)) {
                if (!out.empty() && out.back() != '\n') out.push_back('\n');
            } else {
                if (!out.empty() && out.back() != ' ' && out.back() != '\n') out.push_back(' ');
            }
            continue;
        }

        if (c == '&') {
            size_t j = i + 1;
            while (j < len && bytes[j] != ';' && (j - i) < 10 &&
                   (std::isalnum((unsigned char) bytes[j]) || bytes[j] == '#')) ++j;
            if (j < len && bytes[j] == ';') {
                decode_entity((const char*) &bytes[i + 1], j - i - 1, out);
                i = j + 1;
                continue;
            }
            out.push_back('&');
            ++i;
            continue;
        }

        out.push_back((char) c);
        ++i;
    }

    collapse_whitespace(out);
    return out.empty() ? RAG_INGEST_ERR_EMPTY : RAG_INGEST_OK;
}
