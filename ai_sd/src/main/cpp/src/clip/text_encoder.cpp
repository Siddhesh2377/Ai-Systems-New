/**
 * Text encoder implementation — prompt processing and embedding preparation.
 *
 * Extracted from diffusion_pipeline.cpp (Phase 1.3).
 * Reads globals (tokenizer, promptProcessor, token_emb, pos_emb, etc.)
 * via pipeline_globals.h during migration.
 */

#include "text_encoder.h"
#include "../pipeline/pipeline_globals.h"
#include "../pipeline/prompt_processor.h"
#include "../utils/config.h"
#include "../utils/float_conversion.h"

#include "tokenizers_cpp.h"

ProcessedPrompt processWeightedPrompt(const std::string& prompt_text,
                                       int max_len) {
    ProcessedPrompt result;

    auto tokens = promptProcessor.process(prompt_text);

    // embedding (77 x 768)
    std::vector<float> embeddings(max_len * 768, 0.0f);
    std::vector<int> ids;
    std::vector<float> weights;

    int current_pos = 1;
    ids.push_back(49406);  // BOS token

    for (const auto& token : tokens) {
        if (current_pos >= max_len - 1) break;

        if (token.is_embedding) {
            int emb_size = token.embedding_data.size();
            int emb_tokens = emb_size / 768;

            int pad_id = (text_embedding_size == 1024) ? 0 : 49407;
            for (int i = 0; i < emb_tokens && current_pos < max_len - 1; i++) {
                ids.push_back(pad_id);
                for (int j = 0; j < 768; j++) {
                    embeddings[current_pos * 768 + j] =
                        token.embedding_data[i * 768 + j] * token.weight;
                }
                weights.push_back(token.weight);
                current_pos++;
            }
        } else {
            // tokenize
            std::vector<int> token_ids = tokenizer->Encode(token.text);

            for (int tid : token_ids) {
                if (current_pos >= max_len - 1) break;
                ids.push_back(tid);

                if (current_pos < max_len) {
                    weights.push_back(token.weight);
                }
                current_pos++;
            }
        }
    }

    while (ids.size() < (size_t)max_len) {
        ids.push_back(49407);  // PAD/EOS token
        weights.push_back(1.0f);
    }

    if (ids.size() > (size_t)max_len) {
        ids.resize(max_len);
    }

    result.ids = ids;

    if (use_clip_v2 && !token_emb.empty() && !pos_emb.empty()) {
        for (int i = 0; i < max_len; i++) {
            int token_id = ids[i];
            float weight = (i < (int)weights.size()) ? weights[i] : 1.0f;

            bool has_emb = false;
            for (int j = 0; j < 768; j++) {
                if (embeddings[i * 768 + j] != 0.0f) {
                    has_emb = true;
                    break;
                }
            }

            if (!has_emb) {
                for (int j = 0; j < 768; j++) {
                    float token_val = fp16_to_fp32(token_emb[token_id * 768 + j]);
                    embeddings[i * 768 + j] = token_val * weight + pos_emb[i * 768 + j];
                }
            } else {
                for (int j = 0; j < 768; j++) {
                    embeddings[i * 768 + j] += pos_emb[i * 768 + j];
                }
            }
        }
    }

    result.weighted_embeddings = embeddings;
    return result;
}

ProcessedPromptPair processPromptPair(const std::string& positive,
                                       const std::string& negative,
                                       int max_len) {
    ProcessedPromptPair result;

    auto pos_result = processWeightedPrompt(positive, max_len);
    auto neg_result = processWeightedPrompt(negative, max_len);

    result.ids.reserve(2 * max_len);
    result.ids.insert(result.ids.end(), neg_result.ids.begin(),
                      neg_result.ids.end());
    result.ids.insert(result.ids.end(), pos_result.ids.begin(),
                      pos_result.ids.end());

    result.negative_embeddings = neg_result.weighted_embeddings;
    result.positive_embeddings = pos_result.weighted_embeddings;

    return result;
}
