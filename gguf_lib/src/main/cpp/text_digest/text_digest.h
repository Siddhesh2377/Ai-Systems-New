#pragma once

#include <string>

namespace text_digest {

struct Options {
    int target_tokens = 200;
    float w_query = 0.40f;
    float w_centrality = 0.30f;
    float w_lead = 0.15f;
    float w_entity = 0.15f;
    float mmr_lambda = 0.7f;
    int max_sentences = 80;
    int min_sentence_chars = 20;
    int max_sentence_chars = 600;
    int textrank_iterations = 30;
    float textrank_damping = 0.85f;
};

std::string compress(const std::string& text,
                     const std::string& query,
                     const Options& opts);

}
