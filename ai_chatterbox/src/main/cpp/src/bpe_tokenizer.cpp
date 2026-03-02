#include "bpe_tokenizer.h"
#include <codecvt>
#include <locale>

#include <android/log.h>

#define LOG_TAG "ChatterboxTTS"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using json = nlohmann::json;

BPETokenizer::BPETokenizer() {
    // Initialize byte encoder/decoder
    initBytesToUnicode();

    // GPT-2 pre-tokenization pattern
    pattern = std::regex(
        R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+)"
    );
}

void BPETokenizer::initBytesToUnicode() {
    // GPT-2 bytes-to-unicode mapping
    // This avoids mapping to whitespace/control characters
    std::vector<int> bs;

    // Printable ASCII range
    for (int i = '!'; i <= '~'; i++) bs.push_back(i);
    for (int i = 0xA1; i <= 0xAC; i++) bs.push_back(i);
    for (int i = 0xAE; i <= 0xFF; i++) bs.push_back(i);

    std::vector<int> cs = bs;
    int n = 0;

    // Map remaining bytes to unused unicode
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    // Create encoder/decoder maps
    for (size_t i = 0; i < bs.size(); i++) {
        byte_encoder[static_cast<uint8_t>(bs[i])] = static_cast<char32_t>(cs[i]);
        byte_decoder[static_cast<char32_t>(cs[i])] = static_cast<uint8_t>(bs[i]);
    }
}

bool BPETokenizer::loadFromFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        LOGE("Cannot open tokenizer file: %s", filepath.c_str());
        return false;
    }

    try {
        json config;
        file >> config;

        // Load vocabulary
        if (config.contains("model") && config["model"].contains("vocab")) {
            for (auto& [token, id] : config["model"]["vocab"].items()) {
                vocab[token] = id.get<int64_t>();
                id_to_token[id.get<int64_t>()] = token;
            }
        }

        // Load added tokens (special tokens)
        if (config.contains("added_tokens")) {
            for (auto& token_info : config["added_tokens"]) {
                std::string content = token_info["content"].get<std::string>();
                int64_t id = token_info["id"].get<int64_t>();
                added_tokens[content] = id;
                // Also add to vocab
                vocab[content] = id;
                id_to_token[id] = content;
            }
        }

        // Load merges
        if (config.contains("model") && config["model"].contains("merges")) {
            auto& merges_data = config["model"]["merges"];
            for (auto& merge : merges_data) {
                if (merge.is_array() && merge.size() == 2) {
                    std::string first = merge[0].get<std::string>();
                    std::string second = merge[1].get<std::string>();
                    merges.push_back({first, second});
                } else if (merge.is_string()) {
                    std::string merge_str = merge.get<std::string>();
                    size_t space_pos = merge_str.find(' ');
                    if (space_pos != std::string::npos) {
                        std::string first = merge_str.substr(0, space_pos);
                        std::string second = merge_str.substr(space_pos + 1);
                        merges.push_back({first, second});
                    }
                }
            }
        }

        // Build merge ranks
        for (size_t i = 0; i < merges.size(); i++) {
            bpe_ranks[merges[i]] = i;
        }

        LOGI("Loaded tokenizer: %zu tokens, %zu merges, %zu added tokens",
             vocab.size(), merges.size(), added_tokens.size());

        return true;

    } catch (const std::exception& e) {
        LOGE("Error loading tokenizer: %s", e.what());
        return false;
    }
}

std::set<std::pair<std::string, std::string>> BPETokenizer::getPairs(
    const std::vector<std::string>& word) const {

    std::set<std::pair<std::string, std::string>> pairs;

    if (word.size() < 2) return pairs;

    for (size_t i = 0; i < word.size() - 1; i++) {
        pairs.insert({word[i], word[i + 1]});
    }

    return pairs;
}

std::string BPETokenizer::bpe(const std::string& token) {
    // Check cache
    if (cache.find(token) != cache.end()) {
        return cache[token];
    }

    // Start with individual UTF-8 characters (not bytes!)
    std::vector<std::string> word;
    std::u32string u32token = utf8ToUtf32(token);
    for (char32_t c : u32token) {
        std::u32string u32char(1, c);
        word.push_back(utf32ToUtf8(u32char));
    }

    if (word.size() == 1) {
        return token;
    }

    while (true) {
        auto pairs = getPairs(word);
        if (pairs.empty()) break;

        // Find pair with minimum rank
        std::pair<std::string, std::string> bigram;
        int min_rank = INT_MAX;
        bool found = false;

        for (const auto& pair : pairs) {
            auto it = bpe_ranks.find(pair);
            if (it != bpe_ranks.end() && it->second < min_rank) {
                min_rank = it->second;
                bigram = pair;
                found = true;
            }
        }

        if (!found) break;

        // Merge the bigram
        std::string first = bigram.first;
        std::string second = bigram.second;
        std::vector<std::string> new_word;

        size_t i = 0;
        while (i < word.size()) {
            // Find next occurrence of first
            auto it = std::find(word.begin() + i, word.end(), first);
            if (it == word.end()) {
                // Copy remaining elements
                new_word.insert(new_word.end(), word.begin() + i, word.end());
                break;
            }

            // Copy elements before the match
            new_word.insert(new_word.end(), word.begin() + i, it);
            i = it - word.begin();

            // Check if we can merge
            if (i < word.size() - 1 && word[i] == first && word[i + 1] == second) {
                new_word.push_back(first + second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i++;
            }
        }

        word = new_word;
        if (word.size() == 1) break;
    }

    // Join with spaces
    std::string result;
    for (size_t i = 0; i < word.size(); i++) {
        result += word[i];
        if (i < word.size() - 1) result += " ";
    }

    cache[token] = result;
    return result;
}

std::u32string BPETokenizer::utf8ToUtf32(const std::string& str) const {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    return converter.from_bytes(str);
}

std::string BPETokenizer::utf32ToUtf8(const std::u32string& str) const {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    return converter.to_bytes(str);
}

std::vector<std::string> BPETokenizer::splitByAddedTokens(const std::string& text) const {
    std::vector<std::string> parts = {text};
    std::vector<bool> is_special(1, false);

    // Sort added tokens by length (longest first) for proper matching
    std::vector<std::string> sorted_tokens;
    for (const auto& pair : added_tokens) {
        sorted_tokens.push_back(pair.first);
    }
    std::sort(sorted_tokens.begin(), sorted_tokens.end(),
              [](const std::string& a, const std::string& b) {
                  return a.length() > b.length();
              });

    // Split by each added token
    for (const auto& token : sorted_tokens) {
        std::vector<std::string> new_parts;
        std::vector<bool> new_is_special;

        for (size_t i = 0; i < parts.size(); i++) {
            if (is_special[i]) {
                // Already a special token, keep as-is
                new_parts.push_back(parts[i]);
                new_is_special.push_back(true);
            } else {
                // Split this part by the token
                std::string part = parts[i];
                size_t pos = 0;
                size_t found = part.find(token, pos);

                if (found == std::string::npos) {
                    // Token not found, keep part as-is
                    new_parts.push_back(part);
                    new_is_special.push_back(false);
                } else {
                    // Split by token
                    while (found != std::string::npos) {
                        // Add text before token
                        if (found > pos) {
                            new_parts.push_back(part.substr(pos, found - pos));
                            new_is_special.push_back(false);
                        }

                        // Add the token itself
                        new_parts.push_back(token);
                        new_is_special.push_back(true);

                        pos = found + token.length();
                        found = part.find(token, pos);
                    }

                    // Add remaining text
                    if (pos < part.length()) {
                        new_parts.push_back(part.substr(pos));
                        new_is_special.push_back(false);
                    }
                }
            }
        }

        parts = new_parts;
        is_special = new_is_special;
    }

    // Mark which indices are special tokens
    std::vector<std::string> result;
    for (size_t i = 0; i < parts.size(); i++) {
        if (is_special[i]) {
            result.push_back("\x01" + parts[i]); // Mark with special prefix
        } else {
            result.push_back(parts[i]);
        }
    }

    return result;
}

std::vector<int64_t> BPETokenizer::encode(const std::string& text,
                                          bool add_special_tokens) {
    // Clear cache for this encoding session
    cache.clear();

    // Split by added tokens
    auto text_parts = splitByAddedTokens(text);

    std::vector<std::string> bpe_tokens;

    // Process each part
    for (const auto& part : text_parts) {
        if (part.empty()) continue;

        // Check if this is a special token (marked with \x01)
        if (part[0] == '\x01') {
            // This is an added token
            std::string token = part.substr(1);
            bpe_tokens.push_back(token);
        } else {
            // Normal text - apply regex splitting and BPE
            std::sregex_iterator iter(part.begin(), part.end(), pattern);
            std::sregex_iterator end;

            for (; iter != end; ++iter) {
                std::string token = iter->str();

                // Convert to byte-level representation
                std::string token_bytes;
                for (unsigned char c : token) {
                    auto it = byte_encoder.find(c);
                    if (it != byte_encoder.end()) {
                        std::u32string u32str(1, it->second);
                        token_bytes += utf32ToUtf8(u32str);
                    }
                }

                // Apply BPE
                std::string bpe_output = bpe(token_bytes);

                // Split by spaces and add tokens
                std::istringstream iss(bpe_output);
                std::string subtoken;
                while (iss >> subtoken) {
                    bpe_tokens.push_back(subtoken);
                }
            }
        }
    }

    // Convert tokens to IDs
    std::vector<int64_t> token_ids;
    for (const auto& token : bpe_tokens) {
        auto it = vocab.find(token);
        if (it != vocab.end()) {
            token_ids.push_back(it->second);
        } else {
            token_ids.push_back(unk_token_id);
        }
    }

    // Add special tokens (2x EOS at end)
    if (add_special_tokens) {
        token_ids.push_back(eos_token_id);
        token_ids.push_back(eos_token_id);
    }

    return token_ids;
}

std::string BPETokenizer::decode(const std::vector<int64_t>& token_ids,
                                 bool skip_special_tokens) const {
    // Convert IDs to tokens
    std::vector<std::string> tokens;
    for (int64_t id : token_ids) {
        if (skip_special_tokens && id == eos_token_id) {
            continue;
        }

        auto it = id_to_token.find(id);
        if (it != id_to_token.end()) {
            tokens.push_back(it->second);
        }
    }

    // Join tokens
    std::string text;
    for (const auto& token : tokens) {
        text += token;
    }

    // Decode from byte-level representation
    std::u32string u32text = utf8ToUtf32(text);
    std::vector<uint8_t> byte_array;

    for (char32_t c : u32text) {
        auto it = byte_decoder.find(c);
        if (it != byte_decoder.end()) {
            byte_array.push_back(it->second);
        }
    }

    // Convert bytes to string
    std::string result(byte_array.begin(), byte_array.end());

    return result;
}
