#include "rag_ingest.h"

#include "third_party/miniz/miniz.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>

int rag_ingest_extract_html(const uint8_t* bytes, size_t len, std::string& out);

namespace {

struct Archive {
    mz_zip_archive zip{};
    bool opened = false;
    ~Archive() { if (opened) mz_zip_reader_end(&zip); }
};

bool open_zip(Archive& a, const uint8_t* bytes, size_t len) {
    std::memset(&a.zip, 0, sizeof(a.zip));
    if (!mz_zip_reader_init_mem(&a.zip, bytes, len, 0)) return false;
    a.opened = true;
    return true;
}

bool read_entry(Archive& a, const char* path, std::string& content) {
    int idx = mz_zip_reader_locate_file(&a.zip, path, nullptr, 0);
    if (idx < 0) return false;
    mz_zip_archive_file_stat st;
    if (!mz_zip_reader_file_stat(&a.zip, idx, &st)) return false;
    if (st.m_uncomp_size == 0) { content.clear(); return true; }
    content.resize((size_t) st.m_uncomp_size);
    if (!mz_zip_reader_extract_to_mem(&a.zip, idx, content.data(), content.size(), 0)) return false;
    return true;
}

bool entry_exists(Archive& a, const char* path) {
    return mz_zip_reader_locate_file(&a.zip, path, nullptr, 0) >= 0;
}

std::vector<std::string> list_entries(Archive& a) {
    std::vector<std::string> names;
    mz_uint count = mz_zip_reader_get_num_files(&a.zip);
    names.reserve(count);
    for (mz_uint i = 0; i < count; ++i) {
        char name[512];
        mz_uint got = mz_zip_reader_get_filename(&a.zip, i, name, sizeof(name));
        if (got > 0) names.emplace_back(name);
    }
    return names;
}

bool ends_with_ci(const std::string& s, const char* suffix) {
    size_t sl = std::strlen(suffix);
    if (s.size() < sl) return false;
    for (size_t i = 0; i < sl; ++i) {
        if (std::tolower((unsigned char) s[s.size() - sl + i]) !=
            std::tolower((unsigned char) suffix[i])) return false;
    }
    return true;
}

bool starts_with(const std::string& s, const char* prefix) {
    size_t pl = std::strlen(prefix);
    return s.size() >= pl && std::memcmp(s.data(), prefix, pl) == 0;
}

void decode_xml_entities(std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ) {
        if (s[i] == '&') {
            size_t j = i + 1;
            while (j < s.size() && s[j] != ';' && (j - i) < 10) ++j;
            if (j < s.size() && s[j] == ';') {
                std::string name = s.substr(i + 1, j - i - 1);
                if      (name == "amp")  out.push_back('&');
                else if (name == "lt")   out.push_back('<');
                else if (name == "gt")   out.push_back('>');
                else if (name == "quot") out.push_back('"');
                else if (name == "apos") out.push_back('\'');
                else if (!name.empty() && name[0] == '#') {
                    uint32_t cp = 0;
                    bool ok = true;
                    if (name.size() >= 2 && (name[1] == 'x' || name[1] == 'X')) {
                        for (size_t k = 2; k < name.size() && ok; ++k) {
                            char c = name[k];
                            if (c >= '0' && c <= '9') cp = cp * 16 + (c - '0');
                            else if (c >= 'a' && c <= 'f') cp = cp * 16 + 10 + (c - 'a');
                            else if (c >= 'A' && c <= 'F') cp = cp * 16 + 10 + (c - 'A');
                            else ok = false;
                        }
                    } else {
                        for (size_t k = 1; k < name.size() && ok; ++k) {
                            char c = name[k];
                            if (c < '0' || c > '9') ok = false;
                            else cp = cp * 10 + (c - '0');
                        }
                    }
                    if (ok && cp > 0 && cp <= 0x10FFFF) {
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
                } else {
                    out.append(s, i, j - i + 1);
                }
                i = j + 1;
                continue;
            }
        }
        out.push_back(s[i++]);
    }
    s.swap(out);
}

void extract_between_tags(const std::string& xml, const std::string& open, const std::string& close,
                          std::string& out, const char* separator_after) {
    size_t pos = 0;
    while (pos < xml.size()) {
        size_t start = xml.find(open, pos);
        if (start == std::string::npos) break;
        size_t gt = xml.find('>', start);
        if (gt == std::string::npos) break;
        size_t content_start = gt + 1;
        size_t close_pos = xml.find(close, content_start);
        if (close_pos == std::string::npos) break;
        out.append(xml, content_start, close_pos - content_start);
        if (separator_after) out.append(separator_after);
        pos = close_pos + close.size();
    }
}

void scan_paragraphs_and_text(const std::string& xml,
                              const char* text_open, const char* text_close,
                              const char* para_marker,
                              std::string& out) {
    size_t i = 0;
    while (i < xml.size()) {
        if (para_marker && i + std::strlen(para_marker) <= xml.size() &&
            std::memcmp(xml.data() + i, para_marker, std::strlen(para_marker)) == 0) {
            if (!out.empty() && out.back() != '\n') out.push_back('\n');
            i += std::strlen(para_marker);
            continue;
        }
        size_t open_len = std::strlen(text_open);
        if (i + open_len < xml.size() &&
            std::memcmp(xml.data() + i, text_open, open_len) == 0) {
            char next = xml[i + open_len];
            if (next == ' ' || next == '>' || next == '\t' || next == '\r' || next == '\n' || next == '/') {
                if (next == '/') { i += open_len + 1; continue; }
                size_t tag_end = xml.find('>', i + open_len);
                if (tag_end == std::string::npos) break;
                if (tag_end > 0 && xml[tag_end - 1] == '/') {
                    i = tag_end + 1;
                    continue;
                }
                size_t content_start = tag_end + 1;
                size_t close_pos = xml.find(text_close, content_start);
                if (close_pos == std::string::npos) break;
                out.append(xml, content_start, close_pos - content_start);
                i = close_pos + std::strlen(text_close);
                continue;
            }
        }
        ++i;
    }
}

void trim_trailing_whitespace(std::string& s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\n' || s.back() == '\r' || s.back() == '\t'))
        s.pop_back();
}

}

int rag_ingest_extract_docx(const uint8_t* bytes, size_t len, std::string& out) {
    Archive a;
    if (!open_zip(a, bytes, len)) return RAG_INGEST_ERR_PARSE;

    std::string xml;
    if (!read_entry(a, "word/document.xml", xml)) return RAG_INGEST_ERR_PARSE;

    out.clear();
    out.reserve(xml.size() / 2);
    scan_paragraphs_and_text(xml, "<w:t", "</w:t>", "<w:p", out);
    decode_xml_entities(out);
    trim_trailing_whitespace(out);
    return out.empty() ? RAG_INGEST_ERR_EMPTY : RAG_INGEST_OK;
}

int rag_ingest_extract_odt(const uint8_t* bytes, size_t len, std::string& out) {
    Archive a;
    if (!open_zip(a, bytes, len)) return RAG_INGEST_ERR_PARSE;

    std::string xml;
    if (!read_entry(a, "content.xml", xml)) return RAG_INGEST_ERR_PARSE;

    out.clear();
    out.reserve(xml.size() / 2);

    size_t i = 0;
    while (i < xml.size()) {
        if (xml.compare(i, 9, "<text:p") == 0 || xml.compare(i, 9, "<text:h") == 0) {
            if (!out.empty() && out.back() != '\n') out.push_back('\n');
            size_t gt = xml.find('>', i);
            if (gt == std::string::npos) break;
            i = gt + 1;
            continue;
        }
        if (xml.compare(i, 10, "<text:tab/") == 0) { out.push_back('\t'); i += 10; continue; }
        if (xml.compare(i, 15, "<text:line-break") == 0) {
            out.push_back('\n');
            size_t gt = xml.find('>', i); if (gt == std::string::npos) break;
            i = gt + 1;
            continue;
        }
        if (xml[i] == '<') {
            size_t gt = xml.find('>', i);
            if (gt == std::string::npos) break;
            i = gt + 1;
            continue;
        }
        out.push_back(xml[i++]);
    }

    decode_xml_entities(out);
    trim_trailing_whitespace(out);
    return out.empty() ? RAG_INGEST_ERR_EMPTY : RAG_INGEST_OK;
}

int rag_ingest_extract_pptx(const uint8_t* bytes, size_t len, std::string& out) {
    Archive a;
    if (!open_zip(a, bytes, len)) return RAG_INGEST_ERR_PARSE;

    out.clear();
    auto entries = list_entries(a);

    std::vector<std::string> slides;
    for (auto& n : entries) {
        if (starts_with(n, "ppt/slides/") && ends_with_ci(n, ".xml")) slides.push_back(n);
    }
    std::sort(slides.begin(), slides.end());

    for (auto& n : slides) {
        std::string xml;
        if (!read_entry(a, n.c_str(), xml)) continue;
        std::string slide_text;
        scan_paragraphs_and_text(xml, "<a:t", "</a:t>", "<a:p", slide_text);
        if (!slide_text.empty()) {
            if (!out.empty() && out.back() != '\n') out.push_back('\n');
            out.append(slide_text);
            out.push_back('\n');
        }
    }

    decode_xml_entities(out);
    trim_trailing_whitespace(out);
    return out.empty() ? RAG_INGEST_ERR_EMPTY : RAG_INGEST_OK;
}

int rag_ingest_extract_xlsx(const uint8_t* bytes, size_t len, std::string& out) {
    Archive a;
    if (!open_zip(a, bytes, len)) return RAG_INGEST_ERR_PARSE;

    out.clear();

    std::string shared;
    if (read_entry(a, "xl/sharedStrings.xml", shared) && !shared.empty()) {
        std::string text;
        extract_between_tags(shared, "<t", "</t>", text, "\n");
        out.append(text);
    }

    auto entries = list_entries(a);
    std::vector<std::string> sheets;
    for (auto& n : entries) {
        if (starts_with(n, "xl/worksheets/") && ends_with_ci(n, ".xml")) sheets.push_back(n);
    }
    std::sort(sheets.begin(), sheets.end());

    for (auto& n : sheets) {
        std::string xml;
        if (!read_entry(a, n.c_str(), xml)) continue;
        std::string vals;
        extract_between_tags(xml, "<is><t", "</t></is>", vals, "\n");
        if (!vals.empty()) out.append(vals);
    }

    decode_xml_entities(out);
    trim_trailing_whitespace(out);
    return out.empty() ? RAG_INGEST_ERR_EMPTY : RAG_INGEST_OK;
}

int rag_ingest_extract_epub(const uint8_t* bytes, size_t len, std::string& out) {
    Archive a;
    if (!open_zip(a, bytes, len)) return RAG_INGEST_ERR_PARSE;

    out.clear();
    auto entries = list_entries(a);

    std::string opf_path;
    for (auto& n : entries) {
        if (ends_with_ci(n, ".opf")) { opf_path = n; break; }
    }

    std::vector<std::string> spine_order;
    std::string base_dir;
    if (!opf_path.empty()) {
        std::string opf;
        if (read_entry(a, opf_path.c_str(), opf)) {
            size_t slash = opf_path.find_last_of('/');
            if (slash != std::string::npos) base_dir = opf_path.substr(0, slash + 1);

            std::vector<std::pair<std::string,std::string>> manifest;
            size_t pos = 0;
            while ((pos = opf.find("<item ", pos)) != std::string::npos) {
                size_t end = opf.find('>', pos);
                if (end == std::string::npos) break;
                std::string item = opf.substr(pos, end - pos);
                auto attr = [&](const char* k) -> std::string {
                    std::string key = std::string(k) + "=\"";
                    size_t a = item.find(key);
                    if (a == std::string::npos) return {};
                    a += key.size();
                    size_t b = item.find('"', a);
                    if (b == std::string::npos) return {};
                    return item.substr(a, b - a);
                };
                std::string id = attr("id");
                std::string href = attr("href");
                if (!id.empty() && !href.empty()) manifest.emplace_back(id, href);
                pos = end;
            }

            size_t sp = opf.find("<spine");
            if (sp != std::string::npos) {
                size_t sp_end = opf.find("</spine>", sp);
                if (sp_end != std::string::npos) {
                    std::string spine = opf.substr(sp, sp_end - sp);
                    size_t p = 0;
                    while ((p = spine.find("idref=\"", p)) != std::string::npos) {
                        p += 7;
                        size_t q = spine.find('"', p);
                        if (q == std::string::npos) break;
                        std::string idref = spine.substr(p, q - p);
                        for (auto& m : manifest)
                            if (m.first == idref) { spine_order.push_back(base_dir + m.second); break; }
                        p = q;
                    }
                }
            }
        }
    }

    if (spine_order.empty()) {
        for (auto& n : entries) {
            if (ends_with_ci(n, ".xhtml") || ends_with_ci(n, ".html") || ends_with_ci(n, ".htm"))
                spine_order.push_back(n);
        }
        std::sort(spine_order.begin(), spine_order.end());
    }

    for (auto& path : spine_order) {
        std::string html;
        if (!read_entry(a, path.c_str(), html)) continue;
        std::string chunk;
        rag_ingest_extract_html((const uint8_t*) html.data(), html.size(), chunk);
        if (!chunk.empty()) {
            if (!out.empty() && out.back() != '\n') out.push_back('\n');
            out.append(chunk);
            out.push_back('\n');
        }
    }

    trim_trailing_whitespace(out);
    return out.empty() ? RAG_INGEST_ERR_EMPTY : RAG_INGEST_OK;
}
