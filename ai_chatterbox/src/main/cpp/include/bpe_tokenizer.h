#pragma once

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <regex>
#include <cstdint>
#include <climits>

// JSON library (nlohmann/json)
#include <nlohmann/json.hpp>

/**
 * Byte-level BPE Tokenizer for GPT-2 style models
 * Pure C++ implementation compatible with HuggingFace tokenizers
 *
 * Ported from DDATT/Chatterbox-turbo-cpp
 */
class BPETokenizer {
private:
    // Vocabulary: token -> ID
    std::unordered_map<std::string, int64_t> vocab;

    // Reverse vocabulary: ID -> token
    std::unordered_map<int64_t, std::string> id_to_token;

    // BPE merges: pairs of tokens to merge
    std::vector<std::pair<std::string, std::string>> merges;

    // Rank of each merge (lower = applied earlier)
    std::map<std::pair<std::string, std::string>, int> bpe_ranks;

    // Added tokens (special tokens like [chuckle], [laugh], etc.)
    std::unordered_map<std::string, int64_t> added_tokens;

    // Byte encoder/decoder for GPT-2 byte-level encoding
    std::unordered_map<uint8_t, char32_t> byte_encoder;
    std::unordered_map<char32_t, uint8_t> byte_decoder;

    // Cache for BPE operations
    std::unordered_map<std::string, std::string> cache;

    // Pre-tokenization pattern (GPT-2 style)
    std::regex pattern;

    // Special token IDs
    int64_t bos_token_id = 50256;
    int64_t eos_token_id = 50256;
    int64_t pad_token_id = 50256;
    int64_t unk_token_id = 50256;

    /**
     * Initialize byte-to-unicode mapping (GPT-2 style)
     */
    void initBytesToUnicode();

    /**
     * Get all adjacent pairs from a word
     */
    std::set<std::pair<std::string, std::string>> getPairs(
        const std::vector<std::string>& word) const;

    /**
     * Apply BPE algorithm to a token
     */
    std::string bpe(const std::string& token);

    /**
     * Convert UTF-8 string to UTF-32 for proper character handling
     */
    std::u32string utf8ToUtf32(const std::string& str) const;

    /**
     * Convert UTF-32 string back to UTF-8
     */
    std::string utf32ToUtf8(const std::u32string& str) const;

    /**
     * Split text by added tokens (special tokens)
     */
    std::vector<std::string> splitByAddedTokens(const std::string& text) const;

public:
    /**
     * Constructor
     */
    BPETokenizer();

    /**
     * Load tokenizer from tokenizer.json file
     */
    bool loadFromFile(const std::string& filepath);

    /**
     * Encode text to token IDs
     *
     * @param text Input text to encode
     * @param add_special_tokens If true, adds 2x EOS tokens at end
     * @return Vector of token IDs
     */
    std::vector<int64_t> encode(const std::string& text,
                                 bool add_special_tokens = true);

    /**
     * Decode token IDs to text
     *
     * @param token_ids Vector of token IDs
     * @param skip_special_tokens If true, skip special tokens in output
     * @return Decoded text
     */
    std::string decode(const std::vector<int64_t>& token_ids,
                      bool skip_special_tokens = true) const;

    /**
     * Get vocabulary size
     */
    size_t vocabSize() const { return vocab.size(); }

    /**
     * Get number of merges
     */
    size_t mergeCount() const { return merges.size(); }

    /**
     * Get number of added tokens
     */
    size_t addedTokenCount() const { return added_tokens.size(); }
};
