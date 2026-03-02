#pragma once

/**
 * Text encoder — prompt tokenization, weighting, and embedding preparation.
 *
 * Extracted from diffusion_pipeline.cpp Phase 1.3.
 * Uses globals via pipeline_globals.h during migration.
 */

#include <string>
#include <vector>

struct ProcessedPrompt {
    std::vector<int> ids;                    // CLIP token IDs (77)
    std::vector<float> weighted_embeddings;  // CLIP V2 embeddings (77*768)
};

struct ProcessedPromptPair {
    std::vector<int> ids;                    // concatenated neg+pos IDs (2*77)
    std::vector<float> negative_embeddings;  // (77*768)
    std::vector<float> positive_embeddings;  // (77*768)
};

/// Tokenize and weight a single prompt, optionally computing CLIP V2 embeddings.
ProcessedPrompt processWeightedPrompt(const std::string& prompt_text,
                                       int max_len = 77);

/// Process a positive/negative prompt pair for CFG.
ProcessedPromptPair processPromptPair(const std::string& positive,
                                       const std::string& negative,
                                       int max_len = 77);
