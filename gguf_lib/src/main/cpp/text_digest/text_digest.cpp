#include "text_digest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace text_digest {

namespace {

const std::unordered_set<std::string>& stopwords() {
    static const std::unordered_set<std::string> s = {
        "a","an","the","and","or","but","if","then","else","so","because","as","of","at","by",
        "for","with","about","against","between","into","through","during","before","after",
        "above","below","to","from","up","down","in","out","on","off","over","under","again",
        "further","once","here","there","when","where","why","how","all","any","both","each",
        "few","more","most","other","some","such","no","nor","not","only","own","same","than",
        "too","very","can","will","just","don","should","now","is","am","are","was","were","be",
        "been","being","have","has","had","having","do","does","did","doing","this","that",
        "these","those","i","me","my","myself","we","our","ours","ourselves","you","your","yours",
        "yourself","yourselves","he","him","his","himself","she","her","hers","herself","it",
        "its","itself","they","them","their","theirs","themselves","what","which","who","whom",
        "whose","also","may","might","must","shall","would","could","one","two","three","new",
        "use","used","using","via"
    };
    return s;
}

const std::unordered_set<std::string>& abbreviations() {
    static const std::unordered_set<std::string> s = {
        "mr","mrs","ms","dr","prof","sr","jr","st","mt","ft","gen","rev",
        "etc","vs","e.g","i.e","cf","u.s","u.k","u.n","fig","figs","p","pp","ch","sec",
        "no","vol","ed","eds","trans","approx","incl","excl"
    };
    return s;
}

bool is_ascii_alpha(uint8_t c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool is_ascii_digit(uint8_t c) { return c >= '0' && c <= '9'; }
bool is_ascii_upper(uint8_t c) { return c >= 'A' && c <= 'Z'; }
bool is_token_byte(uint8_t c) {
    return is_ascii_alpha(c) || is_ascii_digit(c) || c >= 0x80 || c == '_';
}
bool is_sentence_end(uint8_t c) { return c == '.' || c == '!' || c == '?'; }
bool is_break_punct(uint8_t c) { return c == ';' || c == ':' || c == '\n'; }

uint8_t ascii_lower(uint8_t c) {
    if (c >= 'A' && c <= 'Z') return c + ('a' - 'A');
    return c;
}

std::string to_ascii_lower(const std::string& s) {
    std::string r;
    r.reserve(s.size());
    for (uint8_t c : s) r.push_back((char)ascii_lower(c));
    return r;
}

std::vector<std::string> tokenize(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    cur.reserve(16);
    for (size_t i = 0; i < s.size(); ++i) {
        uint8_t c = (uint8_t)s[i];
        if (is_token_byte(c)) {
            cur.push_back((char)c);
        } else {
            if (!cur.empty()) { out.push_back(std::move(cur)); cur.clear(); }
        }
    }
    if (!cur.empty()) out.push_back(std::move(cur));
    return out;
}

std::vector<std::string> filter_tokens(const std::vector<std::string>& toks) {
    const auto& sw = stopwords();
    std::vector<std::string> out;
    out.reserve(toks.size());
    for (const auto& raw : toks) {
        if (raw.size() < 2 || raw.size() > 40) continue;
        std::string lo = to_ascii_lower(raw);
        if (sw.count(lo)) continue;
        bool any_alpha = false;
        for (uint8_t c : lo) {
            if (is_ascii_alpha(c) || c >= 0x80) { any_alpha = true; break; }
        }
        if (!any_alpha) continue;
        out.push_back(std::move(lo));
    }
    return out;
}

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b) {
        uint8_t c = (uint8_t)s[a];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') a++;
        else break;
    }
    while (b > a) {
        uint8_t c = (uint8_t)s[b-1];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') b--;
        else break;
    }
    return s.substr(a, b - a);
}

std::string previous_word(const std::string& text, size_t end_pos) {
    size_t i = end_pos;
    while (i > 0 && !is_token_byte((uint8_t)text[i-1])) i--;
    size_t e = i;
    while (i > 0 && is_token_byte((uint8_t)text[i-1])) i--;
    return text.substr(i, e - i);
}

bool looks_like_abbreviation(const std::string& text, size_t period_pos) {
    std::string w = to_ascii_lower(previous_word(text, period_pos));
    if (w.empty()) return false;
    if (abbreviations().count(w)) return true;
    if (w.size() <= 2 && period_pos + 1 < text.size()) {
        uint8_t c0 = (uint8_t)text[period_pos + 1];
        if (c0 == ' ' || c0 == '\t') {
            size_t k = period_pos + 1;
            while (k < text.size() && (text[k] == ' ' || text[k] == '\t')) k++;
            if (k < text.size() && is_ascii_alpha((uint8_t)text[k]) && !is_ascii_upper((uint8_t)text[k])) {
                return true;
            }
        }
    }
    return false;
}

bool is_decimal_period(const std::string& text, size_t i) {
    if (i == 0 || i + 1 >= text.size()) return false;
    return is_ascii_digit((uint8_t)text[i-1]) && is_ascii_digit((uint8_t)text[i+1]);
}

bool followed_by_sentence_start(const std::string& text, size_t i) {
    size_t k = i + 1;
    while (k < text.size() && (text[k] == ' ' || text[k] == '\t' || text[k] == '\n' || text[k] == '\r')) k++;
    if (k >= text.size()) return true;
    uint8_t c = (uint8_t)text[k];
    if (is_ascii_upper(c)) return true;
    if (c == '"' || c == '\'' || c == '(' || c == '[' || c >= 0x80) return true;
    return false;
}

std::vector<std::string> split_sentences(const std::string& text, int min_chars, int max_chars) {
    std::vector<std::string> out;
    if (text.empty()) return out;

    size_t start = 0;
    size_t n = text.size();

    auto push_range = [&](size_t a, size_t b) {
        std::string piece = trim(text.substr(a, b - a));
        if (piece.empty()) return;
        while ((int)piece.size() > max_chars) {
            size_t cut = (size_t)max_chars;
            size_t bk = piece.rfind(' ', cut);
            if (bk == std::string::npos || bk < (size_t)(max_chars / 2)) bk = cut;
            std::string head = trim(piece.substr(0, bk));
            if (!head.empty() && (int)head.size() >= min_chars) out.push_back(head);
            piece = trim(piece.substr(bk));
        }
        if ((int)piece.size() >= min_chars) out.push_back(piece);
    };

    for (size_t i = 0; i < n; ++i) {
        uint8_t c = (uint8_t)text[i];

        if (c == '\n' && i + 1 < n && text[i+1] == '\n') {
            push_range(start, i);
            size_t k = i;
            while (k < n && (text[k] == '\n' || text[k] == '\r' || text[k] == ' ' || text[k] == '\t')) k++;
            start = k;
            i = k - 1;
            continue;
        }

        if (is_break_punct(c)) {
            push_range(start, i);
            start = i + 1;
            continue;
        }

        if (!is_sentence_end(c)) continue;

        size_t j = i;
        while (j + 1 < n && is_sentence_end((uint8_t)text[j+1])) j++;

        if (c == '.') {
            if (is_decimal_period(text, i)) { i = j; continue; }
            if (looks_like_abbreviation(text, i)) { i = j; continue; }
        }

        if (!followed_by_sentence_start(text, j)) { i = j; continue; }

        push_range(start, j + 1);
        start = j + 1;
        i = j;
    }

    if (start < n) push_range(start, n);
    return out;
}

struct Sparse {
    std::vector<std::pair<int, float>> entries;
};

void sparse_l2_normalize(Sparse& v) {
    double s = 0.0;
    for (auto& p : v.entries) s += (double)p.second * (double)p.second;
    if (s <= 0.0) return;
    float inv = (float)(1.0 / std::sqrt(s));
    for (auto& p : v.entries) p.second *= inv;
}

float sparse_cosine(const Sparse& a, const Sparse& b) {
    if (a.entries.empty() || b.entries.empty()) return 0.f;
    size_t i = 0, j = 0;
    double s = 0.0;
    while (i < a.entries.size() && j < b.entries.size()) {
        int ai = a.entries[i].first;
        int bj = b.entries[j].first;
        if (ai == bj) {
            s += (double)a.entries[i].second * (double)b.entries[j].second;
            i++; j++;
        } else if (ai < bj) i++;
        else j++;
    }
    return (float)s;
}

struct Vocab {
    std::unordered_map<std::string, int> id;
    std::vector<int> df;
    int N = 0;
};

void build_vocab(Vocab& v, const std::vector<std::vector<std::string>>& sent_tokens) {
    v.N = (int)sent_tokens.size();
    for (const auto& toks : sent_tokens) {
        std::unordered_set<std::string> uniq(toks.begin(), toks.end());
        for (const auto& t : uniq) {
            auto it = v.id.find(t);
            if (it == v.id.end()) {
                int newId = (int)v.id.size();
                v.id.emplace(t, newId);
                v.df.push_back(1);
            } else {
                v.df[it->second]++;
            }
        }
    }
}

Sparse to_tfidf(const std::vector<std::string>& toks, const Vocab& v) {
    Sparse out;
    if (toks.empty()) return out;
    std::unordered_map<int, int> tf;
    for (const auto& t : toks) {
        auto it = v.id.find(t);
        if (it == v.id.end()) continue;
        tf[it->second]++;
    }
    out.entries.reserve(tf.size());
    for (auto& kv : tf) {
        int term_id = kv.first;
        float tf_w = 1.0f + std::log((float)kv.second);
        float idf = std::log(((float)v.N + 1.0f) / ((float)v.df[term_id] + 1.0f)) + 1.0f;
        out.entries.emplace_back(term_id, tf_w * idf);
    }
    std::sort(out.entries.begin(), out.entries.end(),
              [](const std::pair<int,float>& a, const std::pair<int,float>& b){ return a.first < b.first; });
    return out;
}

std::vector<float> textrank(const std::vector<std::vector<float>>& sim,
                            int iters, float damping) {
    int n = (int)sim.size();
    std::vector<float> r(n, 1.0f / std::max(1, n));
    if (n <= 1) return r;

    std::vector<float> row_sum(n, 0.f);
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) if (i != j) s += sim[i][j];
        row_sum[i] = (float)s;
    }

    std::vector<float> next(n, 0.f);
    float teleport = (1.0f - damping) / (float)n;

    for (int it = 0; it < iters; ++it) {
        for (int i = 0; i < n; ++i) {
            double acc = 0.0;
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                if (row_sum[j] <= 0.f) continue;
                acc += (double)(sim[j][i] / row_sum[j]) * (double)r[j];
            }
            next[i] = teleport + damping * (float)acc;
        }
        double diff = 0.0;
        for (int i = 0; i < n; ++i) {
            diff += std::fabs((double)next[i] - (double)r[i]);
            r[i] = next[i];
        }
        if (diff < 1e-4) break;
    }
    return r;
}

float ne_density(const std::string& sent) {
    int total = 0;
    int caps = 0;
    bool in_word = false;
    bool word_starts_caps = false;
    bool word_has_lower = false;
    size_t word_len = 0;
    bool first_word = true;

    auto flush = [&]() {
        if (in_word) {
            total++;
            if (word_starts_caps && word_has_lower && word_len > 1 && !first_word) caps++;
            first_word = false;
        }
        in_word = false;
        word_starts_caps = false;
        word_has_lower = false;
        word_len = 0;
    };

    for (size_t i = 0; i < sent.size(); ++i) {
        uint8_t c = (uint8_t)sent[i];
        if (is_token_byte(c)) {
            if (!in_word) {
                in_word = true;
                word_starts_caps = is_ascii_upper(c);
                word_has_lower = false;
                word_len = 0;
            } else if (is_ascii_alpha(c) && !is_ascii_upper(c)) {
                word_has_lower = true;
            }
            word_len++;
        } else {
            flush();
        }
    }
    flush();

    if (total == 0) return 0.f;
    return (float)caps / (float)total;
}

void normalize_to_max1(std::vector<float>& v) {
    float mx = 0.f;
    for (float x : v) if (x > mx) mx = x;
    if (mx <= 0.f) return;
    for (auto& x : v) x /= mx;
}

int approx_tokens(const std::string& s) {
    return ((int)s.size() + 3) / 4;
}

std::vector<int> mmr_select(const std::vector<float>& score,
                            const std::vector<std::vector<float>>& sim,
                            const std::vector<std::string>& sentences,
                            int target_tokens,
                            float lambda) {
    int n = (int)score.size();
    std::vector<int> picked;
    if (n == 0) return picked;

    std::vector<char> taken(n, 0);
    int budget = target_tokens;

    while ((int)picked.size() < n) {
        int best = -1;
        float best_score = -1e30f;
        for (int i = 0; i < n; ++i) {
            if (taken[i]) continue;
            float redundancy = 0.f;
            for (int j : picked) {
                float s = sim[i][j];
                if (s > redundancy) redundancy = s;
            }
            float mmr = lambda * score[i] - (1.0f - lambda) * redundancy;
            if (mmr > best_score) { best_score = mmr; best = i; }
        }
        if (best < 0) break;
        int t = approx_tokens(sentences[best]);
        if (!picked.empty() && t > budget) {
            taken[best] = 1;
            continue;
        }
        picked.push_back(best);
        taken[best] = 1;
        budget -= t;
        if (budget <= 0) break;
    }
    return picked;
}

}

std::string compress(const std::string& text, const std::string& query, const Options& opts) {
    if (text.empty()) return "";

    auto sentences = split_sentences(text, opts.min_sentence_chars, opts.max_sentence_chars);
    if (sentences.empty()) return trim(text);
    if ((int)sentences.size() == 1) return sentences.front();

    if ((int)sentences.size() > opts.max_sentences) {
        sentences.resize(opts.max_sentences);
    }

    int total_tokens = 0;
    for (const auto& s : sentences) total_tokens += approx_tokens(s);
    if (total_tokens <= opts.target_tokens) {
        std::string out;
        for (const auto& s : sentences) {
            if (!out.empty()) out += " ";
            out += s;
        }
        return out;
    }

    int N = (int)sentences.size();

    std::vector<std::vector<std::string>> sent_tokens(N);
    for (int i = 0; i < N; ++i) {
        sent_tokens[i] = filter_tokens(tokenize(sentences[i]));
    }

    Vocab vocab;
    build_vocab(vocab, sent_tokens);

    std::vector<Sparse> sent_vecs(N);
    for (int i = 0; i < N; ++i) {
        sent_vecs[i] = to_tfidf(sent_tokens[i], vocab);
        sparse_l2_normalize(sent_vecs[i]);
    }

    std::vector<std::vector<float>> sim(N, std::vector<float>(N, 0.f));
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            float s = sparse_cosine(sent_vecs[i], sent_vecs[j]);
            sim[i][j] = s;
            sim[j][i] = s;
        }
    }

    auto centrality = textrank(sim, opts.textrank_iterations, opts.textrank_damping);

    std::vector<float> q_rel(N, 0.f);
    auto q_toks = filter_tokens(tokenize(query));
    bool has_query = !q_toks.empty();
    if (has_query) {
        Sparse q_vec = to_tfidf(q_toks, vocab);
        sparse_l2_normalize(q_vec);
        for (int i = 0; i < N; ++i) q_rel[i] = sparse_cosine(sent_vecs[i], q_vec);
    }

    std::vector<float> lead(N, 0.f);
    for (int i = 0; i < N; ++i) lead[i] = 1.0f / (1.0f + (float)i);

    std::vector<float> ne(N, 0.f);
    for (int i = 0; i < N; ++i) ne[i] = ne_density(sentences[i]);

    normalize_to_max1(centrality);
    normalize_to_max1(q_rel);
    normalize_to_max1(lead);
    normalize_to_max1(ne);

    float w_q = has_query ? opts.w_query : 0.f;
    float w_c = has_query ? opts.w_centrality : (opts.w_centrality + opts.w_query);

    std::vector<float> score(N, 0.f);
    for (int i = 0; i < N; ++i) {
        score[i] = w_q * q_rel[i]
                 + w_c * centrality[i]
                 + opts.w_lead * lead[i]
                 + opts.w_entity * ne[i];
    }

    auto picked = mmr_select(score, sim, sentences, opts.target_tokens, opts.mmr_lambda);
    if (picked.empty()) return sentences.front();

    std::sort(picked.begin(), picked.end());

    std::string out;
    for (int idx : picked) {
        if (!out.empty()) out += " ";
        out += sentences[idx];
    }
    return out;
}

}
