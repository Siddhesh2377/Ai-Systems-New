/**
 * Diffusion Pipeline - Ported from xororz/local-dream main.cpp
 *
 * HTTP server code removed, progress reported via callbacks.
 * CLI argument parsing replaced with direct config struct.
 * Base64 encoding removed (raw byte arrays passed via JNI).
 *
 * Original: https://github.com/nicenote3r0t/local-dream
 */

#include "diffusion_pipeline.h"
#include "../utils/logger.h"

#include <chrono>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <atomic>

// Model headers (ported from xororz)
#include "../utils/config.h"
#include "../pipeline/schedulers/dpm_solver.h"
#include "../pipeline/schedulers/euler_ancestral.h"
#include "../utils/float_conversion.h"
#include "../utils/laplacian_blend.h"
#include "../pipeline/prompt_processor.h"
#include "../model/qnn_model.h"
#include "../utils/sd_utils.h"
#include "../utils/safetensor_to_mnn.h"
#include "../pipeline/schedulers/scheduler.h"

// QNN Headers
#include "DynamicLoadUtil.hpp"
#include "Logger.hpp"
#include "PAL/DynamicLoading.hpp"
#include "QnnSampleAppUtils.hpp"

// External Libraries (httplib removed - no longer needed)
#include "json.hpp"
#include "tokenizers_cpp.h"

// MNN
#include <MNN/MNNDefine.h>
#include <MNN/Interpreter.hpp>

// Xtensor
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xeval.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xoperation.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

#include "zstd.h"

// --- Diagnostic: compute min/max/mean of a float buffer ---
static void log_tensor_stats(const char* label, const float* data, int count) {
  if (!data || count <= 0) {
    SD_LOG_INFO("[DIAG] %s: null or empty", label);
    return;
  }
  float mn = data[0], mx = data[0];
  double sum = 0.0;
  int nan_count = 0, inf_count = 0;
  for (int i = 0; i < count; i++) {
    if (std::isnan(data[i])) { nan_count++; continue; }
    if (std::isinf(data[i])) { inf_count++; continue; }
    if (data[i] < mn) mn = data[i];
    if (data[i] > mx) mx = data[i];
    sum += data[i];
  }
  float mean = static_cast<float>(sum / (count - nan_count - inf_count));
  SD_LOG_INFO("[DIAG] %s: count=%d min=%.8f max=%.8f mean=%.8f nan=%d inf=%d first4=[%.8f %.8f %.8f %.8f]",
              label, count, mn, mx, mean, nan_count, inf_count,
              data[0], count > 1 ? data[1] : 0.f,
              count > 2 ? data[2] : 0.f, count > 3 ? data[3] : 0.f);
}

// --- Diagnostic: compute L2 difference and max abs diff between two buffers ---
static void log_tensor_diff(const char* label, const float* a, const float* b, int count) {
  if (!a || !b || count <= 0) return;
  double sum_sq = 0.0;
  float max_diff = 0.0f;
  int max_diff_idx = 0;
  for (int i = 0; i < count; i++) {
    float d = a[i] - b[i];
    sum_sq += d * d;
    if (std::abs(d) > max_diff) { max_diff = std::abs(d); max_diff_idx = i; }
  }
  float rmse = static_cast<float>(std::sqrt(sum_sq / count));
  SD_LOG_INFO("[DIAG] %s: rmse=%.8f max_abs_diff=%.8f@idx=%d  a[0..3]=[%.8f %.8f %.8f %.8f] b[0..3]=[%.8f %.8f %.8f %.8f]",
              label, rmse, max_diff, max_diff_idx,
              a[0], count > 1 ? a[1] : 0.f, count > 2 ? a[2] : 0.f, count > 3 ? a[3] : 0.f,
              b[0], count > 1 ? b[1] : 0.f, count > 2 ? b[2] : 0.f, count > 3 ? b[3] : 0.f);
}

// --- Helper: Convert NCHW float [-1,1] to interleaved RGB uint8 [0,255] ---
// Explicit loop avoids xtensor transpose iterator layout ambiguity.
static std::vector<uint8_t> nchw_to_rgb_bytes(const float* nchw_data,
                                                int channels, int height,
                                                int width) {
  const int pixel_count = height * width;
  std::vector<uint8_t> rgb(pixel_count * channels);
  for (int c = 0; c < channels; c++) {
    const float* ch_ptr = nchw_data + c * pixel_count;
    for (int i = 0; i < pixel_count; i++) {
      float val = ((ch_ptr[i] + 1.0f) * 0.5f) * 255.0f;
      if (val < 0.0f) val = 0.0f;
      if (val > 255.0f) val = 255.0f;
      rgb[i * channels + c] = static_cast<uint8_t>(val);
    }
  }
  return rgb;
}

// Port and listen_address removed (no HTTP server)
bool ponyv55 = false;
bool use_mnn = false;
bool use_safety_checker = false;
bool use_mnn_clip = false;
bool use_clip_v2 = false;
bool upscaler_mode = false;
float nsfw_threshold = 0.5f;
std::string clipPath, unetPath, vaeDecoderPath, vaeEncoderPath,
    safetyCheckerPath, tokenizerPath, patchPath, modelDir, upscalerPath;
std::vector<float> pos_emb;
std::vector<uint16_t> token_emb;  // Stored as FP16 to save memory
std::shared_ptr<tokenizers::Tokenizer> tokenizer;
PromptProcessor promptProcessor;
std::unique_ptr<QnnModel> clipApp = nullptr;
std::unique_ptr<QnnModel> unetApp = nullptr;
std::unique_ptr<QnnModel> vaeDecoderApp = nullptr;
std::unique_ptr<QnnModel> vaeEncoderApp = nullptr;
std::unique_ptr<QnnModel> upscalerApp = nullptr;
MNN::Interpreter *clipInterpreter = nullptr;
MNN::Interpreter *unetInterpreter = nullptr;
MNN::Interpreter *vaeDecoderInterpreter = nullptr;
MNN::Interpreter *vaeEncoderInterpreter = nullptr;
MNN::Interpreter *safetyCheckerInterpreter = nullptr;

// MNN Session Pointers
MNN::Session *clipSession = nullptr;
MNN::Session *unetSession = nullptr;
MNN::Session *vaeDecoderSession = nullptr;
MNN::Session *vaeEncoderSession = nullptr;
MNN::Session *safetyCheckerSession = nullptr;

std::string prompt;
std::string negative_prompt;
int steps;
float cfg;
unsigned seed;
std::string scheduler_type;
std::vector<float> img_data;
std::vector<float> mask_data;
std::vector<float> mask_data_full;
float denoise_strength;
bool request_img2img;
bool request_has_mask;
bool use_opencl;

bool cvt_model = false;
bool show_diffusion_process = false;
int show_diffusion_stride = 1;

struct PatchedModelBuffer {
  std::shared_ptr<uint8_t> buffer;
  uint64_t size;

  PatchedModelBuffer() : buffer(nullptr), size(0) {}

  PatchedModelBuffer(uint8_t *buf, uint64_t sz)
      : buffer(buf, std::default_delete<uint8_t[]>()), size(sz) {}

  void reset() {
    buffer.reset();
    size = 0;
  }
};

std::unique_ptr<PatchedModelBuffer> g_unetPatchedBuffer;
std::string model_dir;
bool clip_skip_2 = false;

// QNN function pointers and backend path for dynamic model loading
QnnFunctionPointers g_qnnSystemFuncs;
std::string g_backendPathCmd;

// Global function to create QNN models dynamically
std::unique_ptr<QnnModel> createQnnModel(const std::string &modelPath,
                                         const std::string &modelName) {
  using namespace qnn::tools;
  QnnFunctionPointers funcs = g_qnnSystemFuncs;
  void *backendHandle = nullptr;
  void *modelHandle = nullptr;
  dynamicloadutil::StatusCode drvStatus =
      dynamicloadutil::getQnnFunctionPointers(g_backendPathCmd, modelPath,
                                              &funcs, &backendHandle, false,
                                              &modelHandle);
  if (drvStatus != dynamicloadutil::StatusCode::SUCCESS) {
    QNN_ERROR("Failed get QNN func ptrs for %s.", modelName.c_str());
    return nullptr;
  }
  std::string inputListPaths, opPackagePaths, outputPath, saveBinaryName;
  bool debug = false;
  bool dumpOutputs = false;
  iotensor::OutputDataType outputDataType =
      iotensor::OutputDataType::FLOAT_ONLY;
  iotensor::InputDataType inputDataType = iotensor::InputDataType::FLOAT;
  sample_app::ProfilingLevel profilingLevel = ProfilingLevel::OFF;
  return std::make_unique<QnnModel>(
      funcs, inputListPaths, opPackagePaths, backendHandle, outputPath, debug,
      outputDataType, inputDataType, profilingLevel, dumpOutputs, modelPath,
      saveBinaryName);
}

namespace qnn {
namespace tools {
namespace sample_app {

std::vector<char> readFileForPatch(const std::string &filePath) {
  std::ifstream file(filePath, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filePath);
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  if (size > 0) {
    if (!file.read(buffer.data(), size)) {
      throw std::runtime_error("Failed to read file: " + filePath);
    }
  }
  return buffer;
}

void writeFileForPatch(const std::string &filePath,
                       const std::vector<char> &data) {
  std::ofstream file(filePath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + filePath);
  }
  if (!data.empty()) {
    if (!file.write(data.data(), data.size())) {
      throw std::runtime_error("Failed to write file: " + filePath);
    }
  }
}

int applyZstdPatch(const std::string &oldFilePath,
                   const std::string &patchFilePath,
                   const std::string &newFilePath) {
  try {
    std::vector<char> oldFileBuffer = readFileForPatch(oldFilePath);
    QNN_INFO("Read old file (%s): %zu bytes.", oldFilePath.c_str(),
             oldFileBuffer.size());

    std::vector<char> patchFileBuffer = readFileForPatch(patchFilePath);
    QNN_INFO("Read patch file (%s): %zu bytes.", patchFilePath.c_str(),
             patchFileBuffer.size());

    if (patchFileBuffer.empty()) {
      throw std::runtime_error("Patch file (" + patchFilePath +
                               ") is empty or could not be read.");
    }

    unsigned long long const decompressedSize = ZSTD_getFrameContentSize(
        patchFileBuffer.data(), patchFileBuffer.size());

    if (decompressedSize == ZSTD_CONTENTSIZE_ERROR) {
      throw std::runtime_error("Patch file (" + patchFilePath +
                               ") is not a valid zstd frame.");
    }
    if (decompressedSize == ZSTD_CONTENTSIZE_UNKNOWN) {
      throw std::runtime_error(
          "Decompressed size is unknown. Cannot proceed with this simple "
          "implementation.");
    }

    std::vector<char> newFileBuffer;
    if (decompressedSize > 0) {
      newFileBuffer.resize(decompressedSize);
    } else {
      writeFileForPatch(newFilePath, newFileBuffer);
      QNN_INFO(
          "Successfully applied patch (resulting in an empty file). New file "
          "saved to: %s",
          newFilePath.c_str());
      return 0;
    }

    ZSTD_DCtx *const dctx = ZSTD_createDCtx();
    if (dctx == nullptr) {
      throw std::runtime_error("ZSTD_createDCtx() failed!");
    }

    size_t const actualDecompressedSize = ZSTD_decompress_usingDict(
        dctx, newFileBuffer.data(), newFileBuffer.size(),
        patchFileBuffer.data(), patchFileBuffer.size(), oldFileBuffer.data(),
        oldFileBuffer.size());

    ZSTD_freeDCtx(dctx);

    if (ZSTD_isError(actualDecompressedSize)) {
      throw std::runtime_error(
          "ZSTD_decompress_usingDict() failed: " +
          std::string(ZSTD_getErrorName(actualDecompressedSize)));
    }

    if (actualDecompressedSize != decompressedSize) {
      if (actualDecompressedSize < newFileBuffer.size()) {
        newFileBuffer.resize(actualDecompressedSize);
      }
    }

    QNN_INFO("Decompressed %zu bytes into new file buffer.",
             actualDecompressedSize);

    writeFileForPatch(newFilePath, newFileBuffer);
    QNN_INFO("Successfully applied patch. New file saved to: %s",
             newFilePath.c_str());

  } catch (const std::exception &e) {
    QNN_ERROR("Error applying patch: %s", e.what());
    return 1;
  }
  return 0;
}

std::unique_ptr<PatchedModelBuffer> applyZstdPatchToBuffer(
    const std::string &oldFilePath, const std::string &patchFilePath) {
  try {
    std::vector<char> oldFileBuffer = readFileForPatch(oldFilePath);
    QNN_INFO("Read old file (%s): %zu bytes.", oldFilePath.c_str(),
             oldFileBuffer.size());

    std::vector<char> patchFileBuffer = readFileForPatch(patchFilePath);
    QNN_INFO("Read patch file (%s): %zu bytes.", patchFilePath.c_str(),
             patchFileBuffer.size());

    if (patchFileBuffer.empty()) {
      throw std::runtime_error("Patch file (" + patchFilePath +
                               ") is empty or could not be read.");
    }

    unsigned long long const decompressedSize = ZSTD_getFrameContentSize(
        patchFileBuffer.data(), patchFileBuffer.size());

    if (decompressedSize == ZSTD_CONTENTSIZE_ERROR) {
      throw std::runtime_error("Patch file (" + patchFilePath +
                               ") is not a valid zstd frame.");
    }
    if (decompressedSize == ZSTD_CONTENTSIZE_UNKNOWN) {
      throw std::runtime_error(
          "Decompressed size is unknown. Cannot proceed with this simple "
          "implementation.");
    }

    if (decompressedSize == 0) {
      QNN_ERROR("Patch resulted in empty buffer.");
      return nullptr;
    }

    uint8_t *newBuffer = new uint8_t[decompressedSize];

    ZSTD_DCtx *const dctx = ZSTD_createDCtx();
    if (dctx == nullptr) {
      delete[] newBuffer;
      throw std::runtime_error("ZSTD_createDCtx() failed!");
    }

    size_t const actualDecompressedSize = ZSTD_decompress_usingDict(
        dctx, newBuffer, decompressedSize, patchFileBuffer.data(),
        patchFileBuffer.size(), oldFileBuffer.data(), oldFileBuffer.size());

    ZSTD_freeDCtx(dctx);

    if (ZSTD_isError(actualDecompressedSize)) {
      delete[] newBuffer;
      throw std::runtime_error(
          "ZSTD_decompress_usingDict() failed: " +
          std::string(ZSTD_getErrorName(actualDecompressedSize)));
    }

    QNN_INFO("Successfully applied patch to buffer. Decompressed %zu bytes.",
             actualDecompressedSize);

    return std::make_unique<PatchedModelBuffer>(newBuffer,
                                                actualDecompressedSize);

  } catch (const std::exception &e) {
    QNN_ERROR("Error applying patch to buffer: %s", e.what());
    return nullptr;
  }
}

// QnnModel Initialization
template <typename AppType>
int initializeQnnApp(const std::string &modelName,
                     std::unique_ptr<AppType> &app,
                     const uint8_t *buffer = nullptr, uint64_t bufferSize = 0) {
  if (!app) return EXIT_FAILURE;

  if (buffer && bufferSize > 0) {
    QNN_INFO("Initializing QNN App from Buffer: %s (size: %llu bytes)",
             modelName.c_str(), bufferSize);
  } else {
    QNN_INFO("Initializing QNN App from Cache: %s", modelName.c_str());
  }

  if (StatusCode::SUCCESS != app->initialize())
    return app->reportError(modelName + " Init failure");
  if (StatusCode::SUCCESS != app->initializeBackend())
    return app->reportError(modelName + " Backend Init failure");
  auto devPropStat = app->isDevicePropertySupported();
  if (StatusCode::FAILURE != devPropStat) {
    if (StatusCode::SUCCESS != app->createDevice())
      return app->reportError(modelName + " Device Creation failure");
  }
  if (StatusCode::SUCCESS != app->initializeProfiling())
    return app->reportError(modelName + " Profiling Init failure");
  if (StatusCode::SUCCESS != app->registerOpPackages())
    return app->reportError(modelName + " Register Op Packages failure");

  if (buffer && bufferSize > 0) {
    if (StatusCode::SUCCESS != app->createFromBuffer(buffer, bufferSize))
      return app->reportError(modelName + " Create From Buffer failure");
  } else {
    if (StatusCode::SUCCESS != app->createFromBinary())
      return app->reportError(modelName + " Create From Binary failure");
  }

  if (StatusCode::SUCCESS != app->enablePerformaceMode())
    return app->reportError(modelName + " Enable Performance Mode failure");

  if (buffer && bufferSize > 0) {
    QNN_INFO("QNN App Initialized from Buffer: %s", modelName.c_str());
  } else {
    QNN_INFO("QNN App Initialized from Cache: %s", modelName.c_str());
  }
  return EXIT_SUCCESS;
}

}  // namespace sample_app
}  // namespace tools
}  // namespace qnn

// --- Text Processing ---
struct ProcessedPrompt {
  std::vector<int> ids;                    // CLIP
  std::vector<float> weighted_embeddings;  // CLIP V2 (77*768)
};

ProcessedPrompt processWeightedPrompt(const std::string &prompt_text,
                                      int max_len = 77) {
  ProcessedPrompt result;

  auto tokens = promptProcessor.process(prompt_text);

  // embedding (77 x 768)
  std::vector<float> embeddings(max_len * 768, 0.0f);
  std::vector<int> ids;
  std::vector<float> weights;

  int current_pos = 1;
  ids.push_back(49406);  // BOS token

  for (const auto &token : tokens) {
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

  while (ids.size() < max_len) {
    ids.push_back(49407);  // PAD/EOS token
    weights.push_back(1.0f);
  }

  if (ids.size() > max_len) {
    ids.resize(max_len);
  }

  result.ids = ids;

  if (use_clip_v2 && !token_emb.empty() && !pos_emb.empty()) {
    for (int i = 0; i < max_len; i++) {
      int token_id = ids[i];
      float weight = (i < weights.size()) ? weights[i] : 1.0f;

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

struct ProcessedPromptPair {
  std::vector<int> ids;                    // old (2*77)
  std::vector<float> negative_embeddings;  // new embedding (77*768)
  std::vector<float> positive_embeddings;  // new embedding (77*768)
};

ProcessedPromptPair processPromptPair(const std::string &positive,
                                      const std::string &negative,
                                      int max_len = 77) {
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
xt::xarray<float> blend_vae_encoder_tiles(
    const std::vector<std::pair<xt::xarray<float>, xt::xarray<float>>>
        &tiles_mean_std,
    const std::vector<std::pair<int, int>> &positions, int latent_h,
    int latent_w, int tile_size, int overlap_x, int overlap_y) {
  if (tiles_mean_std.empty()) {
    throw std::runtime_error(
        "Tile list cannot be empty for VAE encoder blending.");
  }

  std::vector<int> accumulated_shape = {1, 4, latent_h, latent_w};
  xt::xarray<float> accumulated_mean = xt::zeros<float>(accumulated_shape);
  xt::xarray<float> accumulated_std = xt::zeros<float>(accumulated_shape);
  xt::xarray<float> weight_map = xt::zeros<float>({latent_h, latent_w});

  int fade_size_x = overlap_x / 2;
  int fade_size_y = overlap_y / 2;

  for (size_t idx = 0; idx < tiles_mean_std.size(); ++idx) {
    int x = positions[idx].first;
    int y = positions[idx].second;

    xt::xarray<float> tile_weight = xt::ones<float>({tile_size, tile_size});

    if (fade_size_y > 0) {
      if (y > 0) {
        for (int i = 0; i < fade_size_y; ++i) {
          float alpha = (float)(i + 1) / fade_size_y;
          xt::view(tile_weight, i, xt::all()) *= alpha;
        }
      }
      if (y + tile_size < latent_h) {
        for (int i = 0; i < fade_size_y; ++i) {
          float alpha = (float)(i + 1) / fade_size_y;
          xt::view(tile_weight, tile_size - 1 - i, xt::all()) *= alpha;
        }
      }
    }

    if (fade_size_x > 0) {
      if (x > 0) {
        for (int i = 0; i < fade_size_x; ++i) {
          float alpha = (float)(i + 1) / fade_size_x;
          xt::view(tile_weight, xt::all(), i) *= alpha;
        }
      }
      if (x + tile_size < latent_w) {
        for (int i = 0; i < fade_size_x; ++i) {
          float alpha = (float)(i + 1) / fade_size_x;
          xt::view(tile_weight, xt::all(), tile_size - 1 - i) *= alpha;
        }
      }
    }

    const auto &mean_tile =
        tiles_mean_std[idx].first;  // (1, 4, tile_size, tile_size)
    const auto &std_tile =
        tiles_mean_std[idx].second;  // (1, 4, tile_size, tile_size)

    for (int c = 0; c < 4; ++c) {
      auto acc_mean_slice =
          xt::view(accumulated_mean, 0, c, xt::range(y, y + tile_size),
                   xt::range(x, x + tile_size));
      auto mean_slice = xt::view(mean_tile, 0, c, xt::all(), xt::all());
      acc_mean_slice += mean_slice * tile_weight;

      auto acc_std_slice =
          xt::view(accumulated_std, 0, c, xt::range(y, y + tile_size),
                   xt::range(x, x + tile_size));
      auto std_slice = xt::view(std_tile, 0, c, xt::all(), xt::all());
      acc_std_slice += std_slice * tile_weight;
    }

    auto weight_slice = xt::view(weight_map, xt::range(y, y + tile_size),
                                 xt::range(x, x + tile_size));
    weight_slice += tile_weight;
  }

  weight_map = xt::maximum(weight_map, 1e-8f);
  xt::xarray<float> weight_expanded =
      xt::reshape_view(weight_map, {1, 1, latent_h, latent_w});

  xt::xarray<float> final_mean = accumulated_mean / weight_expanded;
  xt::xarray<float> final_std = accumulated_std / weight_expanded;

  xt::xarray<float> noise =
      xt::random::randn<float>({1, 4, latent_h, latent_w});
  xt::xarray<float> latent = xt::eval(final_mean + final_std * noise);

  return latent;
}
xt::xarray<float> blend_vae_output_tiles(
    const std::vector<xt::xarray<float>> &tiles,
    const std::vector<std::pair<int, int>> &positions, int output_h,
    int output_w, int tile_size, int overlap_x, int overlap_y) {
  if (tiles.empty()) {
    throw std::runtime_error(
        "Tile list cannot be empty for VAE output blending.");
  }

  std::vector<int> accumulated_shape = {1, 3, output_h, output_w};
  xt::xarray<float> accumulated = xt::zeros<float>(accumulated_shape);
  xt::xarray<float> weight_map = xt::zeros<float>({output_h, output_w});

  int fade_size_x = overlap_x / 2;
  int fade_size_y = overlap_y / 2;

  for (size_t idx = 0; idx < tiles.size(); ++idx) {
    int x = positions[idx].first;
    int y = positions[idx].second;

    xt::xarray<float> tile_weight = xt::ones<float>({tile_size, tile_size});

    if (fade_size_y > 0) {
      if (y > 0) {
        for (int i = 0; i < fade_size_y; ++i) {
          float alpha = (float)(i + 1) / fade_size_y;
          xt::view(tile_weight, i, xt::all()) *= alpha;
        }
      }
      if (y + tile_size < output_h) {
        for (int i = 0; i < fade_size_y; ++i) {
          float alpha = (float)(i + 1) / fade_size_y;
          xt::view(tile_weight, tile_size - 1 - i, xt::all()) *= alpha;
        }
      }
    }

    if (fade_size_x > 0) {
      if (x > 0) {
        for (int i = 0; i < fade_size_x; ++i) {
          float alpha = (float)(i + 1) / fade_size_x;
          xt::view(tile_weight, xt::all(), i) *= alpha;
        }
      }
      if (x + tile_size < output_w) {
        for (int i = 0; i < fade_size_x; ++i) {
          float alpha = (float)(i + 1) / fade_size_x;
          xt::view(tile_weight, xt::all(), tile_size - 1 - i) *= alpha;
        }
      }
    }

    for (int c = 0; c < 3; ++c) {
      auto acc_slice = xt::view(accumulated, 0, c, xt::range(y, y + tile_size),
                                xt::range(x, x + tile_size));
      auto tile_slice = xt::view(tiles[idx], 0, c, xt::all(), xt::all());
      acc_slice += tile_slice * tile_weight;
    }

    auto weight_slice = xt::view(weight_map, xt::range(y, y + tile_size),
                                 xt::range(x, x + tile_size));
    weight_slice += tile_weight;
  }

  weight_map = xt::maximum(weight_map, 1e-8f);
  xt::xarray<float> weight_expanded =
      xt::reshape_view(weight_map, {1, 1, output_h, output_w});

  return accumulated / weight_expanded;
}

// --- Upscaler Tiling ---
std::vector<int> calculate_tile_positions(int dimension, int tile_size,
                                          int min_overlap) {
  if (dimension <= tile_size) {
    return {0};
  }

  int num_tiles = 1;
  int effective_tile_size = tile_size - min_overlap;
  if (dimension > tile_size) {
    num_tiles +=
        (dimension - tile_size + effective_tile_size - 1) / effective_tile_size;
  }

  std::vector<int> positions;
  positions.reserve(num_tiles);
  positions.push_back(0);

  if (num_tiles == 1) {
    return positions;
  }

  int total_distance = dimension - tile_size;
  int num_strides = num_tiles - 1;

  int base_stride = total_distance / num_strides;
  int remainder = total_distance % num_strides;

  int current_pos = 0;
  for (int i = 0; i < num_strides; ++i) {
    int stride = base_stride + (i < remainder ? 1 : 0);
    current_pos += stride;
    positions.push_back(current_pos);
  }

  positions.back() = dimension - tile_size;

  return positions;
}

xt::xarray<uint8_t> upscaleImageWithModel(
    const std::vector<uint8_t> &input_image, int width, int height,
    std::unique_ptr<QnnModel> &upscaler) {
  if (!upscaler) {
    throw std::runtime_error("Upscaler model not provided");
  }

  const int tile_size = 192;
  const int output_tile_size = 768;
  const int min_overlap = 12;
  const float scale_factor = 4.0f;

  auto x_coords = calculate_tile_positions(width, tile_size, min_overlap);
  auto y_coords = calculate_tile_positions(height, tile_size, min_overlap);
  int num_tiles_w = x_coords.size();
  int num_tiles_h = y_coords.size();

  int output_width = width * scale_factor;
  int output_height = height * scale_factor;

  QNN_INFO("Upscaling %dx%d to %dx%d using %dx%d tiles (variable overlap)",
           width, height, output_width, output_height, num_tiles_w,
           num_tiles_h);

  std::vector<int> input_shape = {1, height, width, 3};
  xt::xarray<uint8_t> input_hwc_u8 = xt::adapt(input_image, input_shape);
  xt::xarray<float> input_hwc_f32 = xt::cast<float>(input_hwc_u8) / 255.0f;
  xt::xarray<float> input_chw =
      xt::transpose(input_hwc_f32, {0, 3, 1, 2});  // (1, 3, H, W)

  std::vector<int> output_shape = {1, 3, output_height, output_width};
  xt::xarray<float> accumulated_output = xt::zeros<float>(output_shape);
  xt::xarray<float> weight_map =
      xt::zeros<float>({output_height, output_width});

  int output_overlap = min_overlap * scale_factor;
  int fade_size = output_overlap / 2;
  xt::xarray<float> tile_weight =
      xt::ones<float>({output_tile_size, output_tile_size});

  if (fade_size > 0) {
    for (int i = 0; i < fade_size; ++i) {
      float alpha = static_cast<float>(i + 1) / fade_size;
      xt::view(tile_weight, i, xt::all()) *= alpha;
      xt::view(tile_weight, output_tile_size - 1 - i, xt::all()) *= alpha;
      xt::view(tile_weight, xt::all(), i) *= alpha;
      xt::view(tile_weight, xt::all(), output_tile_size - 1 - i) *= alpha;
    }
  }

  int tile_count = 0;
  for (int y : y_coords) {
    for (int x : x_coords) {
      xt::xarray<float> input_tile =
          xt::view(input_chw, 0, xt::all(), xt::range(y, y + tile_size),
                   xt::range(x, x + tile_size));

      std::vector<float> tile_input_vec(input_tile.begin(), input_tile.end());
      std::vector<float> tile_output_vec(1 * 3 * output_tile_size *
                                         output_tile_size);

      if (StatusCode::SUCCESS !=
          upscaler->executeUpscalerGraphs(tile_input_vec.data(),
                                          tile_output_vec.data())) {
        throw std::runtime_error("Upscaler execution failed for tile");
      }

      std::vector<int> tile_output_shape = {1, 3, output_tile_size,
                                            output_tile_size};
      xt::xarray<float> output_tile =
          xt::adapt(tile_output_vec, tile_output_shape);

      int out_x = x * scale_factor;
      int out_y = y * scale_factor;

      for (int c = 0; c < 3; ++c) {
        auto acc_slice = xt::view(accumulated_output, 0, c,
                                  xt::range(out_y, out_y + output_tile_size),
                                  xt::range(out_x, out_x + output_tile_size));
        auto tile_slice = xt::view(output_tile, 0, c, xt::all(), xt::all());
        acc_slice += tile_slice * tile_weight;
      }

      auto weight_slice =
          xt::view(weight_map, xt::range(out_y, out_y + output_tile_size),
                   xt::range(out_x, out_x + output_tile_size));
      weight_slice += tile_weight;

      tile_count++;
      std::cout << "Processed tile " << tile_count << "/"
                << (num_tiles_w * num_tiles_h) << std::endl;
    }
  }

  weight_map = xt::maximum(weight_map, 1e-8f);
  xt::xarray<float> weight_expanded =
      xt::reshape_view(weight_map, {1, 1, output_height, output_width});

  xt::xarray<float> normalized_output = accumulated_output / weight_expanded;

  auto output_hwc = xt::transpose(normalized_output, {0, 2, 3, 1});
  auto output_clamped = xt::clip(output_hwc, 0.0f, 1.0f);
  auto output_normalized = output_clamped * 255.0f;
  xt::xarray<uint8_t> output_uint8 = xt::cast<uint8_t>(output_normalized);

  return output_uint8;
}

// --- VAE Tiling Helper ---
// Calculate tile positions and overlaps for VAE encoder/decoder
// Returns: {pixel_positions, latent_positions, pixel_overlap_x,
// pixel_overlap_y, latent_overlap_x, latent_overlap_y}
std::tuple<std::vector<std::pair<int, int>>, std::vector<std::pair<int, int>>,
           int, int, int, int>
calculate_vae_tile_positions(int pixel_width, int pixel_height) {
  const int vae_tile_size = 512;        // Fixed VAE tile size in pixel space
  const int vae_latent_tile_size = 64;  // Fixed VAE tile size in latent space
  const int min_latent_overlap = 16;    // Minimum overlap in latent space
  const int scale_factor = 8;           // VAE scale: 512/64 = 8

  // Calculate positions for width and height separately
  auto pixel_x_coords = calculate_tile_positions(
      pixel_width, vae_tile_size, min_latent_overlap * scale_factor);
  auto pixel_y_coords = calculate_tile_positions(
      pixel_height, vae_tile_size, min_latent_overlap * scale_factor);

  // Calculate corresponding latent positions
  std::vector<int> latent_x_coords;
  std::vector<int> latent_y_coords;
  for (int px : pixel_x_coords) {
    latent_x_coords.push_back(px / scale_factor);
  }
  for (int py : pixel_y_coords) {
    latent_y_coords.push_back(py / scale_factor);
  }

  // Create position pairs
  std::vector<std::pair<int, int>> pixel_positions;
  std::vector<std::pair<int, int>> latent_positions;

  for (int py : pixel_y_coords) {
    for (int px : pixel_x_coords) {
      pixel_positions.push_back({px, py});
    }
  }

  for (int ly : latent_y_coords) {
    for (int lx : latent_x_coords) {
      latent_positions.push_back({lx, ly});
    }
  }

  // Calculate actual overlaps based on tile positions
  int pixel_overlap_x = 0;
  int latent_overlap_x = 0;
  int pixel_overlap_y = 0;
  int latent_overlap_y = 0;

  if (pixel_x_coords.size() > 1) {
    pixel_overlap_x = vae_tile_size - (pixel_x_coords[1] - pixel_x_coords[0]);
    latent_overlap_x =
        vae_latent_tile_size - (latent_x_coords[1] - latent_x_coords[0]);
  }

  if (pixel_y_coords.size() > 1) {
    pixel_overlap_y = vae_tile_size - (pixel_y_coords[1] - pixel_y_coords[0]);
    latent_overlap_y =
        vae_latent_tile_size - (latent_y_coords[1] - latent_y_coords[0]);
  }

  return {pixel_positions, latent_positions, pixel_overlap_x,
          pixel_overlap_y, latent_overlap_x, latent_overlap_y};
}

// Upscale image using MNN model
xt::xarray<uint8_t> upscaleImageWithMNN(const std::vector<uint8_t> &input_image,
                                        int width, int height,
                                        const std::string &model_path,
                                        bool use_opencl) {
  const int tile_size = 192;
  const int output_tile_size = 768;
  const int min_overlap = 12;
  const float scale_factor = 4.0f;

  auto interpreter = std::shared_ptr<MNN::Interpreter>(
      MNN::Interpreter::createFromFile(model_path.c_str()));
  if (!interpreter) {
    throw std::runtime_error("Failed to create MNN interpreter from: " +
                             model_path);
  }

  MNN::ScheduleConfig config;
  MNN::BackendConfig backendConfig;
  if (use_opencl) {
    auto cache_file = model_path + ".mnnc";
    interpreter->setCacheFile(cache_file.c_str());
    config.type = MNN_FORWARD_OPENCL;
    config.mode = MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_FAST;
    backendConfig.precision = MNN::BackendConfig::Precision_Low;
  } else {
    config.type = MNN_FORWARD_CPU;
    config.numThread = 4;
    backendConfig.memory = MNN::BackendConfig::Memory_Low;
  }
  backendConfig.power = MNN::BackendConfig::Power_High;
  config.backendConfig = &backendConfig;

  auto session = interpreter->createSession(config);
  if (!session) {
    throw std::runtime_error("Failed to create MNN session");
  }

  auto x_coords = calculate_tile_positions(width, tile_size, min_overlap);
  auto y_coords = calculate_tile_positions(height, tile_size, min_overlap);
  int num_tiles_w = x_coords.size();
  int num_tiles_h = y_coords.size();

  int output_width = width * scale_factor;
  int output_height = height * scale_factor;

  QNN_INFO("Upscaling %dx%d to %dx%d using MNN (%s), %dx%d tiles", width,
           height, output_width, output_height, use_opencl ? "OpenCL" : "CPU",
           num_tiles_w, num_tiles_h);

  std::vector<int> input_shape = {1, height, width, 3};
  xt::xarray<uint8_t> input_hwc_u8 = xt::adapt(input_image, input_shape);
  xt::xarray<float> input_hwc_f32 = xt::cast<float>(input_hwc_u8) / 255.0f;
  xt::xarray<float> input_chw =
      xt::transpose(input_hwc_f32, {0, 3, 1, 2});  // (1, 3, H, W)

  std::vector<int> output_shape = {1, 3, output_height, output_width};
  xt::xarray<float> accumulated_output = xt::zeros<float>(output_shape);
  xt::xarray<float> weight_map =
      xt::zeros<float>({output_height, output_width});

  int output_overlap = min_overlap * scale_factor;
  int fade_size = output_overlap / 2;
  xt::xarray<float> tile_weight =
      xt::ones<float>({output_tile_size, output_tile_size});

  if (fade_size > 0) {
    for (int i = 0; i < fade_size; ++i) {
      float alpha = static_cast<float>(i + 1) / fade_size;
      xt::view(tile_weight, i, xt::all()) *= alpha;
      xt::view(tile_weight, output_tile_size - 1 - i, xt::all()) *= alpha;
      xt::view(tile_weight, xt::all(), i) *= alpha;
      xt::view(tile_weight, xt::all(), output_tile_size - 1 - i) *= alpha;
    }
  }

  // Get input and output tensors
  auto input_tensor = interpreter->getSessionInput(session, nullptr);
  auto output_tensor = interpreter->getSessionOutput(session, nullptr);

  int tile_count = 0;
  for (int y : y_coords) {
    for (int x : x_coords) {
      xt::xarray<float> input_tile =
          xt::view(input_chw, 0, xt::all(), xt::range(y, y + tile_size),
                   xt::range(x, x + tile_size));

      // Prepare input tensor
      std::vector<int> dims = {1, 3, tile_size, tile_size};
      interpreter->resizeTensor(input_tensor, dims);
      interpreter->resizeSession(session);

      auto host_tensor = MNN::Tensor::create<float>(
          dims, const_cast<float *>(input_tile.data()), MNN::Tensor::CAFFE);
      input_tensor->copyFromHostTensor(host_tensor);
      delete host_tensor;

      // Run inference
      if (interpreter->runSession(session) != 0) {
        throw std::runtime_error("MNN inference failed for tile");
      }

      // Get output
      auto output_host =
          MNN::Tensor::create<float>({1, 3, output_tile_size, output_tile_size},
                                     nullptr, MNN::Tensor::CAFFE);
      output_tensor->copyToHostTensor(output_host);

      std::vector<int> tile_output_shape = {1, 3, output_tile_size,
                                            output_tile_size};
      xt::xarray<float> output_tile = xt::adapt(
          output_host->host<float>(), output_tile_size * output_tile_size * 3,
          xt::no_ownership(), tile_output_shape);

      int out_x = x * scale_factor;
      int out_y = y * scale_factor;

      for (int c = 0; c < 3; ++c) {
        auto acc_slice = xt::view(accumulated_output, 0, c,
                                  xt::range(out_y, out_y + output_tile_size),
                                  xt::range(out_x, out_x + output_tile_size));
        auto tile_slice = xt::view(output_tile, 0, c, xt::all(), xt::all());
        acc_slice += tile_slice * tile_weight;
      }

      auto weight_slice =
          xt::view(weight_map, xt::range(out_y, out_y + output_tile_size),
                   xt::range(out_x, out_x + output_tile_size));
      weight_slice += tile_weight;

      delete output_host;

      tile_count++;
      std::cout << "Processed tile " << tile_count << "/"
                << (num_tiles_w * num_tiles_h) << std::endl;
    }
  }

  weight_map = xt::maximum(weight_map, 1e-8f);
  xt::xarray<float> weight_expanded =
      xt::reshape_view(weight_map, {1, 1, output_height, output_width});

  xt::xarray<float> normalized_output = accumulated_output / weight_expanded;

  auto output_hwc = xt::transpose(normalized_output, {0, 2, 3, 1});
  auto output_clamped = xt::clip(output_hwc, 0.0f, 1.0f);
  auto output_normalized = output_clamped * 255.0f;
  xt::xarray<uint8_t> output_uint8 = xt::cast<uint8_t>(output_normalized);

  return output_uint8;
}

// --- Image Generation ---
GenerationResult generateImage(
    std::function<void(int step, int total_steps,
                       const std::vector<uint8_t> &image_data)>
        progress_callback) {
  using namespace qnn::tools::sample_app;
  if (prompt.empty()) throw std::invalid_argument("Global prompt empty");
  if (use_safety_checker && !safetyCheckerInterpreter)
    throw std::runtime_error("SafetyChecker missing");
  if (!use_mnn) {
    if (!use_mnn_clip && !clipApp) throw std::runtime_error("QNN CLIP missing");
    if (use_mnn_clip && !clipInterpreter)
      throw std::runtime_error("MNN CLIP missing(hybrid)");
    if (!unetApp) throw std::runtime_error("QNN UNET missing");
    if (!vaeDecoderApp) throw std::runtime_error("QNN VAE Dec missing");
    if (request_img2img && !vaeEncoderApp)
      throw std::runtime_error("QNN VAE Enc missing");
  }
  if (request_img2img && img_data.size() != 3 * output_width * output_height)
    throw std::invalid_argument("Invalid global img_data");
  if (request_has_mask &&
      (mask_data.size() != 4 * sample_width * sample_height ||
       mask_data_full.size() != 3 * output_width * output_height))
    throw std::invalid_argument("Invalid global mask_data*");

  try {
    auto start_time = std::chrono::high_resolution_clock::now();
    int first_step_time_ms = 0;
    int total_run_steps = steps + (request_img2img ? 1 : 0) + 2;
    int current_step = 0;
    const int batch_size = 2;

    // --- CLIP ---
    ProcessedPromptPair processed =
        processPromptPair(prompt, negative_prompt, 77);

    std::vector<int> clip_input_ids = processed.ids;  // old (2*77)
    auto parsed_input_text = tokenizer->Decode(clip_input_ids);
    SD_LOG_INFO("Parsed Input Text: %s", parsed_input_text.c_str());
    std::vector<float> text_embedding_float(batch_size * 77 *
                                            text_embedding_size);
    auto clip_start = std::chrono::high_resolution_clock::now();
    int32_t *input_ids_ptr = clip_input_ids.data();
    float *embed_ptr = text_embedding_float.data();

    // Log CLIP input IDs for both prompts
    SD_LOG_INFO("[DIAG] CLIP neg IDs (first 8): %d %d %d %d %d %d %d %d",
                input_ids_ptr[0], input_ids_ptr[1], input_ids_ptr[2], input_ids_ptr[3],
                input_ids_ptr[4], input_ids_ptr[5], input_ids_ptr[6], input_ids_ptr[7]);
    SD_LOG_INFO("[DIAG] CLIP pos IDs (first 8): %d %d %d %d %d %d %d %d",
                input_ids_ptr[77], input_ids_ptr[78], input_ids_ptr[79], input_ids_ptr[80],
                input_ids_ptr[81], input_ids_ptr[82], input_ids_ptr[83], input_ids_ptr[84]);
    SD_LOG_INFO("[DIAG] use_mnn=%d use_mnn_clip=%d use_clip_v2=%d",
                (int)use_mnn, (int)use_mnn_clip, (int)use_clip_v2);

    if (use_mnn || use_mnn_clip) {
      MNN::Interpreter *currentClipInterpreter = nullptr;
      MNN::Session *currentClipSession = nullptr;
      bool dynamicCreated = false;

      if (use_mnn_clip) {
        currentClipInterpreter = clipInterpreter;
        currentClipSession = clipSession;
        if (!currentClipInterpreter)
          throw std::runtime_error(
              "Global clipInterpreter (hybrid) not initialized!");
      } else {
        currentClipInterpreter =
            MNN::Interpreter::createFromFile(clipPath.c_str());
        if (!currentClipInterpreter)
          throw std::runtime_error(
              "Failed to create temporary MNN CLIP interpreter!");
        dynamicCreated = true;
      }

      bool sessionCreated = false;
      if (!currentClipSession) {
        MNN::ScheduleConfig cfg_clip;
        cfg_clip.type = MNN_FORWARD_CPU;
        cfg_clip.numThread = 4;
        MNN::BackendConfig bkCfg_clip;
        bkCfg_clip.memory = MNN::BackendConfig::Memory_Low;
        bkCfg_clip.power = MNN::BackendConfig::Power_High;
        cfg_clip.backendConfig = &bkCfg_clip;
        currentClipSession = currentClipInterpreter->createSession(cfg_clip);
        if (!currentClipSession)
          throw std::runtime_error(
              "Failed to create temporary MNN CLIP session!");
        sessionCreated = true;
      }

      if (use_clip_v2) {
        auto input = currentClipInterpreter->getSessionInput(currentClipSession,
                                                             "input_embedding");
        currentClipInterpreter->resizeTensor(input, {1, 77, 768});
        currentClipInterpreter->resizeSession(currentClipSession);

        if (dynamicCreated) currentClipInterpreter->releaseModel();

        memcpy(input->host<float>(), processed.negative_embeddings.data(),
               77 * 768 * sizeof(float));
        currentClipInterpreter->runSession(currentClipSession);
        auto out = currentClipInterpreter->getSessionOutput(
            currentClipSession, "last_hidden_state");
        memcpy(embed_ptr, out->host<float>(),
               77 * text_embedding_size * sizeof(float));

        memcpy(input->host<float>(), processed.positive_embeddings.data(),
               77 * 768 * sizeof(float));
        currentClipInterpreter->runSession(currentClipSession);
        out = currentClipInterpreter->getSessionOutput(currentClipSession,
                                                       "last_hidden_state");
        memcpy(embed_ptr + 77 * text_embedding_size, out->host<float>(),
               77 * text_embedding_size * sizeof(float));

        if (sessionCreated)
          currentClipInterpreter->releaseSession(currentClipSession);
        if (dynamicCreated) delete currentClipInterpreter;

      } else {
        auto input = currentClipInterpreter->getSessionInput(currentClipSession,
                                                             "input_ids");
        currentClipInterpreter->resizeTensor(input, {1, 77});
        currentClipInterpreter->resizeSession(currentClipSession);

        if (dynamicCreated) currentClipInterpreter->releaseModel();

        memcpy(input->host<int>(), input_ids_ptr, 77 * sizeof(int32_t));
        currentClipInterpreter->runSession(currentClipSession);
        auto out = currentClipInterpreter->getSessionOutput(
            currentClipSession, "last_hidden_state");
        memcpy(embed_ptr, out->host<float>(),
               77 * text_embedding_size * sizeof(float));

        memcpy(input->host<int>(), input_ids_ptr + 77, 77 * sizeof(int32_t));
        currentClipInterpreter->runSession(currentClipSession);
        out = currentClipInterpreter->getSessionOutput(currentClipSession,
                                                       "last_hidden_state");
        memcpy(embed_ptr + 77 * text_embedding_size, out->host<float>(),
               77 * text_embedding_size * sizeof(float));

        if (sessionCreated)
          currentClipInterpreter->releaseSession(currentClipSession);
        if (dynamicCreated) delete currentClipInterpreter;
      }
    } else {
      if (!clipApp) throw std::runtime_error("Global clipApp not initialized!");
      if (StatusCode::SUCCESS !=
          clipApp->executeClipGraphs(input_ids_ptr, embed_ptr))
        throw std::runtime_error("QNN CLIP exec failed (neg)");
      if (StatusCode::SUCCESS !=
          clipApp->executeClipGraphs(input_ids_ptr + 77,
                                     embed_ptr + 77 * text_embedding_size))
        throw std::runtime_error("QNN CLIP exec failed (pos)");
    }

    auto clip_end = std::chrono::high_resolution_clock::now();
    SD_LOG_INFO("CLIP dur: %dms",
                (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                    clip_end - clip_start).count());
    log_tensor_stats("clip_embed_neg", text_embedding_float.data(),
                     77 * text_embedding_size);
    log_tensor_stats("clip_embed_pos", text_embedding_float.data() + 77 * text_embedding_size,
                     77 * text_embedding_size);
    log_tensor_diff("clip_neg_vs_pos",
                    text_embedding_float.data(),
                    text_embedding_float.data() + 77 * text_embedding_size,
                    77 * text_embedding_size);
    current_step++;
    progress_callback(current_step, total_run_steps, {});

    // --- Scheduler & Latents ---
    std::unique_ptr<Scheduler> scheduler;
    if (scheduler_type == "euler_a" || scheduler_type == "eulera") {
      scheduler = std::make_unique<EulerAncestralDiscreteScheduler>(
          1000, 0.00085f, 0.012f, "scaled_linear", "epsilon", "leading");
    } else {
      // Default to DPM solver
      scheduler = std::make_unique<DPMSolverMultistepScheduler>(
          1000, 0.00085f, 0.012f, "scaled_linear", 2, "epsilon", "leading");
    }
    if (ponyv55) scheduler->set_prediction_type("v_prediction");
    scheduler->set_timesteps(steps);
    xt::xarray<float> timesteps = scheduler->get_timesteps();
    std::vector<int> shape = {1, 4, sample_height, sample_width};
    std::vector<int> shape_batch2 = {batch_size, 4, sample_height,
                                     sample_width};
    xt::random::seed(seed);
    xt::xarray<float> latents = xt::random::randn<float>(shape);
    xt::xarray<float> latents_noise = xt::random::randn<float>(shape);

    // Scale initial latents by init_noise_sigma (required for Euler schedulers)
    float init_noise_sigma = scheduler->get_init_noise_sigma();
    latents = latents * init_noise_sigma;

    SD_LOG_INFO("[DIAG] Scheduler: type=%s steps=%d pony=%d init_noise_sigma=%.6f",
                scheduler_type.c_str(), steps, (int)ponyv55, init_noise_sigma);
    SD_LOG_INFO("[DIAG] Latent shape: [1, 4, %d, %d], batch_size=%d",
                sample_height, sample_width, batch_size);
    SD_LOG_INFO("[DIAG] Timesteps: first=%.1f last=%.1f count=%zu",
                timesteps.size() > 0 ? timesteps(0) : -1.f,
                timesteps.size() > 0 ? timesteps(timesteps.size() - 1) : -1.f,
                timesteps.size());
    log_tensor_stats("initial_latents", latents.data(),
                     static_cast<int>(latents.size()));

    xt::xarray<float> original_latents, original_image, mask, mask_full;
    int start_step = 0;

    // --- Img2Img / VAE Encode ---
    if (request_img2img) {
      auto vae_enc_start = std::chrono::high_resolution_clock::now();
      std::vector<int> img_shape = {1, 3, output_height, output_width};
      original_image = xt::adapt(img_data, img_shape);

      bool need_vae_enc_tiling = ((output_width > 512 || output_height > 512) &&
                                  !use_mnn && vaeEncoderApp);

      xt::xarray<float> img_lat_scaled;

      if (!need_vae_enc_tiling) {
        std::vector<float> vae_enc_mean(1 * 4 * sample_width * sample_height);
        std::vector<float> vae_enc_std(1 * 4 * sample_width * sample_height);

        if (use_mnn) {
          MNN::Interpreter *currentVaeEncoderInterpreter =
              MNN::Interpreter::createFromFile(vaeEncoderPath.c_str());
          if (!currentVaeEncoderInterpreter)
            throw std::runtime_error("Failed MNN VAE Enc create");

          MNN::ScheduleConfig cfg_vae_enc;
          MNN::BackendConfig bkCfg_vae_enc;
          if (use_opencl) {
            auto cache_file = modelDir + "/vae_enc_cache.mnnc." +
                              std::to_string(output_width);
            currentVaeEncoderInterpreter->setCacheFile(cache_file.c_str());
            cfg_vae_enc.type = MNN_FORWARD_OPENCL;
            cfg_vae_enc.mode = MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_FAST;
            bkCfg_vae_enc.precision = MNN::BackendConfig::Precision_Low;
          } else {
            cfg_vae_enc.type = MNN_FORWARD_CPU;
            cfg_vae_enc.numThread = 4;
            bkCfg_vae_enc.memory = MNN::BackendConfig::Memory_Low;
          }
          bkCfg_vae_enc.power = MNN::BackendConfig::Power_High;
          cfg_vae_enc.backendConfig = &bkCfg_vae_enc;

          MNN::Session *currentVaeEncSession =
              currentVaeEncoderInterpreter->createSession(cfg_vae_enc);
          if (!currentVaeEncSession)
            throw std::runtime_error("Failed create temp MNN VAE Enc session!");

          auto input = currentVaeEncoderInterpreter->getSessionInput(
              currentVaeEncSession, "input");
          currentVaeEncoderInterpreter->resizeTensor(
              input, {1, 3, output_height, output_width});
          currentVaeEncoderInterpreter->resizeSession(currentVaeEncSession);
          if (use_opencl) {
            currentVaeEncoderInterpreter->updateCacheFile(currentVaeEncSession);
          }
          currentVaeEncoderInterpreter->releaseModel();

          auto input_nchw_tensor = new MNN::Tensor(input, MNN::Tensor::CAFFE);
          auto mean_t = currentVaeEncoderInterpreter->getSessionOutput(
              currentVaeEncSession, "mean");
          auto std_t = currentVaeEncoderInterpreter->getSessionOutput(
              currentVaeEncSession, "std");
          auto mean_nchw_tensor = new MNN::Tensor(mean_t, MNN::Tensor::CAFFE);
          auto std_nchw_tensor = new MNN::Tensor(std_t, MNN::Tensor::CAFFE);

          memcpy(input_nchw_tensor->host<float>(), img_data.data(),
                 img_data.size() * sizeof(float));
          input->copyFromHostTensor(input_nchw_tensor);
          currentVaeEncoderInterpreter->runSession(currentVaeEncSession);

          mean_t->copyToHostTensor(mean_nchw_tensor);
          std_t->copyToHostTensor(std_nchw_tensor);
          memcpy(vae_enc_mean.data(), mean_nchw_tensor->host<float>(),
                 vae_enc_mean.size() * sizeof(float));
          memcpy(vae_enc_std.data(), std_nchw_tensor->host<float>(),
                 vae_enc_std.size() * sizeof(float));

          delete input_nchw_tensor;
          delete mean_nchw_tensor;
          delete std_nchw_tensor;

          currentVaeEncoderInterpreter->releaseSession(currentVaeEncSession);
          delete currentVaeEncoderInterpreter;
        } else {
          if (!vaeEncoderApp)
            throw std::runtime_error("Global vaeEncoderApp not init!");
          if (StatusCode::SUCCESS !=
              vaeEncoderApp->executeVaeEncoderGraphs(
                  img_data.data(), vae_enc_mean.data(), vae_enc_std.data()))
            throw std::runtime_error("QNN VAE enc exec failed");
        }

        auto mean = xt::adapt(vae_enc_mean, shape);
        auto std_dev = xt::adapt(vae_enc_std, shape);
        xt::xarray<float> noise_0 = xt::random::randn<float>(shape);
        xt::xarray<float> img_lat = xt::eval(mean + std_dev * noise_0);
        img_lat_scaled = xt::eval(0.18215 * img_lat);

      } else {
        std::cout << "Using VAE encoder tiling for " << output_width << "x"
                  << output_height << " input..." << std::endl;

        const int vae_enc_tile_size = 512;
        const int vae_enc_latent_tile_size = 64;

        // Use generic tile position calculator
        auto [img_positions, latent_positions, img_overlap_x, img_overlap_y,
              latent_overlap_x, latent_overlap_y] =
            calculate_vae_tile_positions(output_width, output_height);

        int num_tiles = img_positions.size();
        std::cout << "VAE encoder will use " << num_tiles
                  << " tiles with overlap " << img_overlap_x << "x"
                  << img_overlap_y << "px (latent: " << latent_overlap_x << "x"
                  << latent_overlap_y << ")" << std::endl;

        int original_output_width = output_width;
        int original_output_height = output_height;
        int original_sample_width = sample_width;
        int original_sample_height = sample_height;

        output_width = vae_enc_tile_size;
        output_height = vae_enc_tile_size;
        sample_width = vae_enc_latent_tile_size;
        sample_height = vae_enc_latent_tile_size;

        std::vector<std::pair<xt::xarray<float>, xt::xarray<float>>>
            encoded_tiles_mean_std;
        encoded_tiles_mean_std.reserve(img_positions.size());

        for (size_t i = 0; i < img_positions.size(); ++i) {
          auto img_pos = img_positions[i];
          xt::xarray<float> img_tile = xt::view(
              original_image, 0, xt::all(),
              xt::range(img_pos.second, img_pos.second + vae_enc_tile_size),
              xt::range(img_pos.first, img_pos.first + vae_enc_tile_size));

          std::vector<float> tile_img_vec(img_tile.begin(), img_tile.end());
          std::vector<float> tile_mean_vec(1 * 4 * vae_enc_latent_tile_size *
                                           vae_enc_latent_tile_size);
          std::vector<float> tile_std_vec(1 * 4 * vae_enc_latent_tile_size *
                                          vae_enc_latent_tile_size);

          if (!vaeEncoderApp)
            throw std::runtime_error("Global vaeEncoderApp not init!");

          if (StatusCode::SUCCESS !=
              vaeEncoderApp->executeVaeEncoderGraphs(tile_img_vec.data(),
                                                     tile_mean_vec.data(),
                                                     tile_std_vec.data()))
            throw std::runtime_error("QNN VAE enc exec failed for tile");

          std::vector<int> tile_shape = {1, 4, vae_enc_latent_tile_size,
                                         vae_enc_latent_tile_size};
          encoded_tiles_mean_std.push_back(
              {xt::adapt(tile_mean_vec, tile_shape),
               xt::adapt(tile_std_vec, tile_shape)});
          std::cout << "Processed VAE encoder tile " << i + 1 << "/"
                    << img_positions.size() << std::endl;
        }

        output_width = original_output_width;
        output_height = original_output_height;
        sample_width = original_sample_width;
        sample_height = original_sample_height;

        xt::xarray<float> img_lat = blend_vae_encoder_tiles(
            encoded_tiles_mean_std, latent_positions, sample_height,
            sample_width, vae_enc_latent_tile_size, latent_overlap_x,
            latent_overlap_y);

        img_lat_scaled = xt::eval(0.18215 * img_lat);

        std::cout << "VAE encoder tiling completed: "
                  << encoded_tiles_mean_std.size()
                  << " tiles processed and blended" << std::endl;
      }

      auto vae_enc_end = std::chrono::high_resolution_clock::now();
      std::cout << "VAE Enc dur: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       vae_enc_end - vae_enc_start)
                       .count()
                << "ms\n";

      original_latents = img_lat_scaled;
      start_step = steps * (1.0f - denoise_strength);
      total_run_steps -= start_step;
      scheduler->set_begin_index(start_step);
      xt::xarray<int> t = {(int)(timesteps(start_step))};
      latents = scheduler->add_noise(original_latents, latents_noise, t);

      if (request_has_mask) {
        mask = xt::adapt(mask_data, {1, 4, sample_height, sample_width});
        mask_full =
            xt::adapt(mask_data_full, {1, 3, output_height, output_width});
      }

      current_step++;
      progress_callback(current_step, total_run_steps, {});
    }  // --- UNET Denoising Loop ---
    int single_latent_size = 1 * 4 * sample_width * sample_height;

    MNN::Interpreter *currentUnetInterpreter = nullptr;
    MNN::Session *currentUnetSession = nullptr;

    if (use_mnn) {
      currentUnetInterpreter =
          MNN::Interpreter::createFromFile(unetPath.c_str());
      if (!currentUnetInterpreter)
        throw std::runtime_error(
            "Failed to create temporary MNN UNET interpreter!");

      MNN::ScheduleConfig cfg_unet;
      MNN::BackendConfig bkCfg_unet;
      if (use_opencl) {
        auto cache_file =
            modelDir + "/unet_cache.mnnc." + std::to_string(output_width);
        currentUnetInterpreter->setCacheFile(cache_file.c_str());
        cfg_unet.type = MNN_FORWARD_OPENCL;
        cfg_unet.mode = MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_FAST;
        bkCfg_unet.precision = MNN::BackendConfig::Precision_Low;
      } else {
        cfg_unet.type = MNN_FORWARD_CPU;
        cfg_unet.numThread = 4;
        bkCfg_unet.memory = MNN::BackendConfig::Memory_Low;
      }
      bkCfg_unet.power = MNN::BackendConfig::Power_High;
      cfg_unet.backendConfig = &bkCfg_unet;

      currentUnetSession = currentUnetInterpreter->createSession(cfg_unet);
      if (!currentUnetSession)
        throw std::runtime_error(
            "Failed to create temporary MNN UNET session!");

      auto samp =
          currentUnetInterpreter->getSessionInput(currentUnetSession, "sample");
      auto ts = currentUnetInterpreter->getSessionInput(currentUnetSession,
                                                        "timestep");
      auto enc = currentUnetInterpreter->getSessionInput(
          currentUnetSession, "encoder_hidden_states");

      currentUnetInterpreter->resizeTensor(
          samp, {batch_size, 4, sample_height, sample_width});
      currentUnetInterpreter->resizeTensor(ts, {1});
      currentUnetInterpreter->resizeTensor(
          enc, {batch_size, 77, text_embedding_size});
      currentUnetInterpreter->resizeSession(currentUnetSession);
      if (use_opencl) {
        currentUnetInterpreter->updateCacheFile(currentUnetSession);
      }

      currentUnetInterpreter->releaseModel();
    }

    for (int i = start_step; i < timesteps.size(); ++i) {
      SD_LOG_INFO("[DIAG] Step %d/%zu: show_process=%d use_mnn=%d stride=%d cond=(%d && %d && %d)",
                  i, timesteps.size(), (int)show_diffusion_process, (int)use_mnn, show_diffusion_stride,
                  (int)show_diffusion_process, (int)(!use_mnn),
                  (int)((i - start_step) % show_diffusion_stride == 0));
      if (show_diffusion_process && !use_mnn &&
          (i - start_step) % show_diffusion_stride == 0) {
        SD_LOG_INFO("[DIAG] Preview decode attempt at step %d", i);
        try {
          // Decode current latents for preview
          xt::xarray<float> preview_latents =
              xt::eval((1.0 / 0.18215) * latents);

          xt::xarray<float> pixels;
          bool preview_success = false;

          if (output_width > 512 || output_height > 512) {
            // Use tiling for QNN large resolution preview
            auto [output_positions, latent_positions, overlap_x, overlap_y,
                  latent_overlap_x, latent_overlap_y] =
                calculate_vae_tile_positions(output_width, output_height);

            const int vae_tile_size = 512;
            const int vae_latent_tile_size = 64;

            int original_output_width = output_width;
            int original_output_height = output_height;
            int original_sample_width = sample_width;
            int original_sample_height = sample_height;

            output_width = vae_tile_size;
            output_height = vae_tile_size;
            sample_width = vae_latent_tile_size;
            sample_height = vae_latent_tile_size;

            std::vector<xt::xarray<float>> decoded_tiles;
            decoded_tiles.reserve(latent_positions.size());

            bool tile_success = true;
            for (size_t tile_idx = 0; tile_idx < latent_positions.size();
                 ++tile_idx) {
              auto lat_pos = latent_positions[tile_idx];
              // Extract latent tile
              xt::xarray<float> latent_tile =
                  xt::view(preview_latents, 0, xt::all(),
                           xt::range(lat_pos.second,
                                     lat_pos.second + vae_latent_tile_size),
                           xt::range(lat_pos.first,
                                     lat_pos.first + vae_latent_tile_size));

              std::vector<float> tile_latent_vec(latent_tile.begin(),
                                                 latent_tile.end());
              xt::xarray<float> tile_output =
                  xt::zeros<float>({1, 3, vae_tile_size, vae_tile_size});

              if (StatusCode::SUCCESS !=
                  vaeDecoderApp->executeVaeDecoderGraphs(tile_latent_vec.data(),
                                                         tile_output.data())) {
                tile_success = false;
                break;
              }

              decoded_tiles.push_back(std::move(tile_output));
            }

            output_width = original_output_width;
            output_height = original_output_height;
            sample_width = original_sample_width;
            sample_height = original_sample_height;

            if (tile_success) {
              pixels = blend_vae_output_tiles(
                  decoded_tiles, output_positions, output_height, output_width,
                  vae_tile_size, overlap_x, overlap_y);
              preview_success = true;
            }
          } else {
            // Single inference for QNN <= 512
            std::vector<float> vae_dec_in_vec(preview_latents.begin(),
                                              preview_latents.end());
            std::vector<float> vae_dec_out_pixels(1 * 3 * output_width *
                                                  output_height);
            if (StatusCode::SUCCESS ==
                vaeDecoderApp->executeVaeDecoderGraphs(
                    vae_dec_in_vec.data(), vae_dec_out_pixels.data())) {
              std::vector<int> pixel_shape = {1, 3, output_height,
                                              output_width};
              pixels = xt::adapt(vae_dec_out_pixels, pixel_shape);
              preview_success = true;
            }
          }

          if (preview_success) {
            SD_LOG_INFO("[DIAG] Preview VAE decode succeeded at step %d", i);
            // Direct CHW→RGB conversion (no xtensor transpose, no base64)
            std::vector<uint8_t> out_data = nchw_to_rgb_bytes(
                pixels.data(), 3, output_height, output_width);
            SD_LOG_INFO("[DIAG] Preview image: %zu bytes, first RGB=(%d,%d,%d)",
                        out_data.size(),
                        out_data.size() >= 3 ? out_data[0] : -1,
                        out_data.size() >= 3 ? out_data[1] : -1,
                        out_data.size() >= 3 ? out_data[2] : -1);
            progress_callback(current_step, total_run_steps, out_data);
          } else {
            SD_LOG_WARN("[DIAG] Preview VAE decode FAILED at step %d (returned failure)", i);
            progress_callback(current_step, total_run_steps, {});
          }
        } catch (const std::exception &e) {
          SD_LOG_WARN("[DIAG] Preview decode EXCEPTION at step %d: %s", i, e.what());
          progress_callback(current_step, total_run_steps, {});
        }
      } else {
        progress_callback(current_step, total_run_steps, {});
      }

      auto step_start_time = std::chrono::high_resolution_clock::now();

      // Scale model input (required for Euler schedulers)
      float current_ts = timesteps(i);
      xt::xarray<float> latents_scaled =
          scheduler->scale_model_input(latents, current_ts);

      std::vector<float> latents_in_vec;
      latents_in_vec.reserve(batch_size * single_latent_size);
      latents_in_vec.insert(latents_in_vec.end(), latents_scaled.begin(),
                            latents_scaled.end());
      latents_in_vec.insert(latents_in_vec.end(), latents_scaled.begin(),
                            latents_scaled.end());
      std::vector<float> unet_out_latents(batch_size * single_latent_size);

      if (use_mnn) {
        auto samp = currentUnetInterpreter->getSessionInput(currentUnetSession,
                                                            "sample");
        auto ts = currentUnetInterpreter->getSessionInput(currentUnetSession,
                                                          "timestep");
        auto enc = currentUnetInterpreter->getSessionInput(
            currentUnetSession, "encoder_hidden_states");

        int current_ts_int = (int)(current_ts);

        auto samp_nchw_tensor = new MNN::Tensor(samp, MNN::Tensor::CAFFE);
        auto ts_nchw_tensor = new MNN::Tensor(ts, MNN::Tensor::CAFFE);
        auto enc_nchw_tensor = new MNN::Tensor(enc, MNN::Tensor::CAFFE);

        // Copy both batches (negative and positive) at once
        memcpy(samp_nchw_tensor->host<float>(), latents_in_vec.data(),
               latents_in_vec.size() * sizeof(float));
        memcpy(ts_nchw_tensor->host<int>(), &current_ts_int, sizeof(int));
        memcpy(enc_nchw_tensor->host<float>(), text_embedding_float.data(),
               text_embedding_float.size() * sizeof(float));

        samp->copyFromHostTensor(samp_nchw_tensor);
        ts->copyFromHostTensor(ts_nchw_tensor);
        enc->copyFromHostTensor(enc_nchw_tensor);

        // Single batch inference for both negative and positive conditions
        currentUnetInterpreter->runSession(currentUnetSession);

        auto output = currentUnetInterpreter->getSessionOutput(
            currentUnetSession, "out_sample");
        output->copyToHostTensor(samp_nchw_tensor);
        memcpy(unet_out_latents.data(), samp_nchw_tensor->host<float>(),
               unet_out_latents.size() * sizeof(float));

        delete samp_nchw_tensor;
        delete ts_nchw_tensor;
        delete enc_nchw_tensor;
      } else {
        if (!unetApp)
          throw std::runtime_error("Global unetApp not initialized!");

        float *latents_in_ptr = latents_in_vec.data();
        float *embed_ptr = text_embedding_float.data();
        float *latents_out_ptr = unet_out_latents.data();

        if (StatusCode::SUCCESS !=
            unetApp->executeUnetGraphs(latents_in_ptr,
                                       static_cast<int>(current_ts), embed_ptr,
                                       latents_out_ptr))
          throw std::runtime_error("QNN UNET exec failed (uncond)");

        if (StatusCode::SUCCESS !=
            unetApp->executeUnetGraphs(latents_in_ptr + single_latent_size,
                                       static_cast<int>(current_ts),
                                       embed_ptr + 77 * text_embedding_size,
                                       latents_out_ptr + single_latent_size))
          throw std::runtime_error("QNN UNET exec failed (cond)");
      }

      auto step_end_time = std::chrono::high_resolution_clock::now();
      auto step_dur = std::chrono::duration_cast<std::chrono::milliseconds>(
          step_end_time - step_start_time);

      if (i == start_step) first_step_time_ms = step_dur.count();
      SD_LOG_INFO("[DIAG] UNET step %d dur: %dms", i, (int)step_dur.count());

      // Log raw UNet output for first and last step
      if (i == start_step || i == (int)timesteps.size() - 1) {
        log_tensor_stats("unet_out_raw", unet_out_latents.data(),
                         static_cast<int>(unet_out_latents.size()));
      }

      xt::xarray<float> noise_pred_batch =
          xt::adapt(unet_out_latents, shape_batch2);
      xt::xarray<float> uncond = xt::view(noise_pred_batch, 0);
      xt::xarray<float> txt = xt::view(noise_pred_batch, 1);

      // Log CFG computation details at first step
      if (i == start_step) {
        log_tensor_stats("unet_uncond", uncond.data(), static_cast<int>(uncond.size()));
        log_tensor_stats("unet_cond", txt.data(), static_cast<int>(txt.size()));
        log_tensor_diff("uncond_vs_cond", uncond.data(), txt.data(), static_cast<int>(uncond.size()));
        SD_LOG_INFO("[DIAG] cfg_scale=%.4f", cfg);
      }

      xt::xarray<float> noise_pred = uncond + cfg * (txt - uncond);
      noise_pred = xt::eval(noise_pred);

      if (i == start_step) {
        log_tensor_stats("noise_pred_after_cfg", noise_pred.data(), static_cast<int>(noise_pred.size()));
      }

      // Save pre-step latents for comparison
      std::vector<float> pre_step_latents(latents.data(), latents.data() + latents.size());

      SD_LOG_INFO("[DIAG] Pre-scheduler step %d: step_idx=%zu sigma=%.8f timestep=%.0f",
                  i, scheduler->get_step_index(), scheduler->get_current_sigma(), timesteps(i));

      auto step_result = scheduler->step(noise_pred, timesteps(i), latents);

      if (i == start_step) {
        log_tensor_stats("scheduler_prev_sample", step_result.prev_sample.data(),
                         static_cast<int>(step_result.prev_sample.size()));
      }

      latents = step_result.prev_sample;
      latents = xt::eval(latents);  // Force contiguous evaluation

      // Check if latents actually changed
      log_tensor_diff(("latent_change_step_" + std::to_string(i)).c_str(),
                      pre_step_latents.data(), latents.data(),
                      static_cast<int>(latents.size()));

      // Early NaN detection
      {
        bool has_nan = false;
        for (int k = 0; k < std::min(1000, static_cast<int>(latents.size())); k++) {
          if (std::isnan(latents.data()[k])) { has_nan = true; break; }
        }
        if (has_nan) {
          SD_LOG_ERROR("[DIAG] NaN DETECTED in latents at step %d! Dumping context:", i);
          log_tensor_stats("nan_latents", latents.data(), static_cast<int>(latents.size()));
          log_tensor_stats("nan_noise_pred", noise_pred.data(), static_cast<int>(noise_pred.size()));
          log_tensor_stats("nan_pre_step", pre_step_latents.data(), static_cast<int>(pre_step_latents.size()));
        }
      }

      if (request_has_mask) {
        xt::xarray<int> t_xt = {(int)(timesteps(i))};
        xt::xarray<float> orig_noised =
            scheduler->add_noise(original_latents, latents_noise, t_xt);
        latents = xt::eval(orig_noised * (1.0f - mask) + latents * mask);
      }

      // Log latent stats at first, mid, and last step
      if (i == start_step || i == (int)timesteps.size() / 2 || i == (int)timesteps.size() - 1) {
        log_tensor_stats(("latents_after_step_" + std::to_string(i)).c_str(),
                         latents.data(), static_cast<int>(latents.size()));
      }

      current_step++;
    }

    if (use_mnn) {
      if (currentUnetSession)
        currentUnetInterpreter->releaseSession(currentUnetSession);
      if (currentUnetInterpreter) delete currentUnetInterpreter;
    }

    // --- VAE Decode ---
    auto vae_dec_start = std::chrono::high_resolution_clock::now();

    bool need_vae_tiling =
        ((output_width > 512 || output_height > 512) && !use_mnn);
    if (need_vae_tiling) {
      SD_LOG_INFO("Using VAE decoder tiling for %dx%d output", output_width, output_height);
    }

    // Log latents BEFORE VAE scaling
    log_tensor_stats("latents_pre_vae_scale", latents.data(),
                     static_cast<int>(latents.size()));

    latents = xt::eval((1.0 / 0.18215) * latents);

    // Log latents AFTER VAE scaling (this is what goes into the VAE decoder)
    log_tensor_stats("latents_vae_input", latents.data(),
                     static_cast<int>(latents.size()));

    xt::xarray<float> pixels;

    if (!need_vae_tiling) {
      std::vector<float> vae_dec_in_vec(latents.begin(), latents.end());
      std::vector<float> vae_dec_out_pixels(1 * 3 * output_width *
                                            output_height);

      if (use_mnn) {
        MNN::Interpreter *currentVaeDecoderInterpreter =
            MNN::Interpreter::createFromFile(vaeDecoderPath.c_str());

        if (!currentVaeDecoderInterpreter)
          throw std::runtime_error(
              "Failed to create temporary MNN VAE Decoder interpreter!");

        MNN::ScheduleConfig cfg_vae;
        MNN::BackendConfig bkCfg_vae;
        if (use_opencl) {
          auto cache_file =
              modelDir + "/vae_dec_cache.mnnc." + std::to_string(output_width);
          currentVaeDecoderInterpreter->setCacheFile(cache_file.c_str());
          cfg_vae.type = MNN_FORWARD_OPENCL;
          cfg_vae.mode = MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_FAST;
          bkCfg_vae.precision = MNN::BackendConfig::Precision_Low;
        } else {
          cfg_vae.type = MNN_FORWARD_CPU;
          cfg_vae.numThread = 4;
          bkCfg_vae.memory = MNN::BackendConfig::Memory_Low;
        }
        bkCfg_vae.power = MNN::BackendConfig::Power_High;
        cfg_vae.backendConfig = &bkCfg_vae;

        MNN::Session *currentVaeDecSession =
            currentVaeDecoderInterpreter->createSession(cfg_vae);

        if (!currentVaeDecSession)
          throw std::runtime_error("Failed create temp MNN VAE Dec session!");

        auto input = currentVaeDecoderInterpreter->getSessionInput(
            currentVaeDecSession, "latent_sample");

        currentVaeDecoderInterpreter->resizeTensor(
            input, {1, 4, sample_height, sample_width});
        currentVaeDecoderInterpreter->resizeSession(currentVaeDecSession);
        if (use_opencl) {
          currentVaeDecoderInterpreter->updateCacheFile(currentVaeDecSession);
        }

        currentVaeDecoderInterpreter->releaseModel();

        auto input_nchw_tensor = new MNN::Tensor(input, MNN::Tensor::CAFFE);
        auto output = currentVaeDecoderInterpreter->getSessionOutput(
            currentVaeDecSession, "sample");
        auto output_nchw_tensor = new MNN::Tensor(output, MNN::Tensor::CAFFE);

        memcpy(input_nchw_tensor->host<float>(), vae_dec_in_vec.data(),
               vae_dec_in_vec.size() * sizeof(float));
        input->copyFromHostTensor(input_nchw_tensor);

        currentVaeDecoderInterpreter->runSession(currentVaeDecSession);

        output->copyToHostTensor(output_nchw_tensor);
        memcpy(vae_dec_out_pixels.data(), output_nchw_tensor->host<float>(),
               vae_dec_out_pixels.size() * sizeof(float));

        SD_LOG_INFO("[DIAG] MNN VAE decoder completed");
        log_tensor_stats("mnn_vae_dec_output", vae_dec_out_pixels.data(),
                         static_cast<int>(vae_dec_out_pixels.size()));

        delete input_nchw_tensor;
        delete output_nchw_tensor;

        currentVaeDecoderInterpreter->releaseSession(currentVaeDecSession);
        delete currentVaeDecoderInterpreter;
      } else {
        if (!vaeDecoderApp)
          throw std::runtime_error("Global vaeDecoderApp not init!");

        SD_LOG_INFO("[DIAG] Calling QNN VAE decoder: input=%d floats, output=%d floats",
                    (int)vae_dec_in_vec.size(), (int)vae_dec_out_pixels.size());
        log_tensor_stats("vae_dec_input", vae_dec_in_vec.data(),
                         static_cast<int>(vae_dec_in_vec.size()));

        if (StatusCode::SUCCESS !=
            vaeDecoderApp->executeVaeDecoderGraphs(vae_dec_in_vec.data(),
                                                   vae_dec_out_pixels.data()))
          throw std::runtime_error("QNN VAE dec exec failed");

        SD_LOG_INFO("[DIAG] QNN VAE decoder completed successfully");
        log_tensor_stats("vae_dec_output_raw", vae_dec_out_pixels.data(),
                         static_cast<int>(vae_dec_out_pixels.size()));

        // Log per-channel stats
        int ch_size = output_width * output_height;
        log_tensor_stats("vae_out_ch0_R", vae_dec_out_pixels.data(), ch_size);
        log_tensor_stats("vae_out_ch1_G", vae_dec_out_pixels.data() + ch_size, ch_size);
        log_tensor_stats("vae_out_ch2_B", vae_dec_out_pixels.data() + 2 * ch_size, ch_size);
      }

      std::vector<int> pixel_shape = {1, 3, output_height, output_width};
      pixels = xt::adapt(vae_dec_out_pixels, pixel_shape);

    } else {
      const int vae_tile_size = 512;
      const int vae_latent_tile_size = 64;

      // Use generic tile position calculator
      auto [output_positions, latent_positions, overlap_x, overlap_y,
            latent_overlap_x, latent_overlap_y] =
          calculate_vae_tile_positions(output_width, output_height);

      int num_tiles = output_positions.size();
      std::cout << "VAE decoder will use " << num_tiles
                << " tiles with overlap " << overlap_x << "x" << overlap_y
                << "px (latent: " << latent_overlap_x << "x" << latent_overlap_y
                << ")" << std::endl;

      int original_output_width = output_width;
      int original_output_height = output_height;
      int original_sample_width = sample_width;
      int original_sample_height = sample_height;

      output_width = vae_tile_size;
      output_height = vae_tile_size;
      sample_width = vae_latent_tile_size;
      sample_height = vae_latent_tile_size;

      std::vector<xt::xarray<float>> decoded_tiles;
      decoded_tiles.reserve(latent_positions.size());

      for (size_t i = 0; i < latent_positions.size(); ++i) {
        auto lat_pos = latent_positions[i];
        xt::xarray<float> latent_tile = xt::view(
            latents, 0, xt::all(),
            xt::range(lat_pos.second, lat_pos.second + vae_latent_tile_size),
            xt::range(lat_pos.first, lat_pos.first + vae_latent_tile_size));

        std::vector<float> tile_latent_vec(latent_tile.begin(),
                                           latent_tile.end());
        xt::xarray<float> tile_output =
            xt::zeros<float>({1, 3, vae_tile_size, vae_tile_size});

        if (!vaeDecoderApp)
          throw std::runtime_error("Global vaeDecoderApp not init!");

        if (StatusCode::SUCCESS !=
            vaeDecoderApp->executeVaeDecoderGraphs(tile_latent_vec.data(),
                                                   tile_output.data()))
          throw std::runtime_error("QNN VAE dec exec failed for tile");

        decoded_tiles.push_back(std::move(tile_output));

        std::cout << "Processed VAE tile " << i + 1 << "/"
                  << latent_positions.size() << std::endl;
      }

      output_width = original_output_width;
      output_height = original_output_height;
      sample_width = original_sample_width;
      sample_height = original_sample_height;

      pixels = blend_vae_output_tiles(decoded_tiles, output_positions,
                                      output_height, output_width,
                                      vae_tile_size, overlap_x, overlap_y);

      SD_LOG_INFO("VAE tiling completed: %zu tiles processed and blended",
                decoded_tiles.size());
    }

    auto vae_dec_end = std::chrono::high_resolution_clock::now();
    SD_LOG_INFO("VAE Dec dur: %dms",
                (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                    vae_dec_end - vae_dec_start).count());

    // --- Post-process Image ---
    // Log pixel stats before post-processing
    log_tensor_stats("pixels_pre_postprocess", pixels.data(),
                     static_cast<int>(pixels.size()));

    if (request_has_mask) {
      auto orig_img_view = xt::view(original_image, 0);  // (3, H, W)
      auto gen_img_view = xt::view(pixels, 0);           // (3, H, W)
      auto mask_view = xt::view(mask_full, 0);           // (1, H, W)

      auto blended =
          laplacianPyramidBlend(orig_img_view, gen_img_view, mask_view);
      pixels = xt::reshape_view(blended, {1, 3, output_height, output_width});
    }
    // Ensure pixels is contiguous before converting
    xt::xarray<float> pixels_eval = xt::eval(pixels);

    SD_LOG_INFO("[DIAG] pixels_eval shape: ndim=%zu size=%zu, output=%dx%d",
                pixels_eval.dimension(), pixels_eval.size(), output_width, output_height);
    log_tensor_stats("pixels_eval_final", pixels_eval.data(),
                     static_cast<int>(pixels_eval.size()));

    std::vector<uint8_t> out_data = nchw_to_rgb_bytes(
        pixels_eval.data(), 3, output_height, output_width);

    // Log first few pixels for debugging
    if (out_data.size() >= 12) {
      SD_LOG_INFO("[DIAG] Final RGB first 4 pixels: "
               "(%d,%d,%d) (%d,%d,%d) (%d,%d,%d) (%d,%d,%d) total=%zu bytes",
               out_data[0], out_data[1], out_data[2],
               out_data[3], out_data[4], out_data[5],
               out_data[6], out_data[7], out_data[8],
               out_data[9], out_data[10], out_data[11],
               out_data.size());
    }

    // --- Safety Checker ---
    if (use_safety_checker) {
      auto safety_start = std::chrono::high_resolution_clock::now();
      float score = 0.0f;

      if (safety_check(out_data, output_width, output_height, score,
                       safetyCheckerInterpreter, safetyCheckerSession)) {
        std::cout << "NSFW Score: " << score << std::endl;
        if (score > nsfw_threshold) {
          QNN_WARN("NSFW detected (%.2f>%.2f).", score, nsfw_threshold);
          std::fill(out_data.begin(), out_data.end(), 255);
        }
      } else {
        QNN_WARN("Safety check failed.");
      }

      auto safety_end = std::chrono::high_resolution_clock::now();
      std::cout << "Safety check dur: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       safety_end - safety_start)
                       .count()
                << "ms\n";
    }

    current_step++;
    progress_callback(current_step, total_run_steps, {});
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          end_time - start_time)
                          .count();

    return GenerationResult{out_data,
                            output_width,
                            output_height,
                            3,
                            static_cast<int>(total_time),
                            first_step_time_ms};
  } catch (const std::exception &e) {
    QNN_ERROR("Image generation error: %s", e.what());
    throw;
  }
}

// =============================================================================
// sd_pipeline namespace - JNI-callable functions replacing HTTP server
// =============================================================================

namespace sd_pipeline {

bool initialize_models(const SDModelConfig& config) {
  using namespace qnn::tools;

  if (!qnn::log::initializeLogging()) {
    QNN_ERROR("Failed to initialize QNN logging");
    return false;
  }

  // Set globals from config (replacing CLI argument parsing)
  clipPath = config.clipPath;
  unetPath = config.unetPath;
  vaeDecoderPath = config.vaeDecoderPath;
  vaeEncoderPath = config.vaeEncoderPath;
  tokenizerPath = config.tokenizerPath;
  safetyCheckerPath = config.safetyCheckerPath;
  ponyv55 = config.isPony;
  use_mnn = config.runOnCpu;
  use_mnn_clip = config.useCpuClip;
  use_safety_checker = config.useSafetyChecker;
  nsfw_threshold = config.nsfwThreshold;
  text_embedding_size = config.textEmbeddingSize;
  modelDir = config.modelDir;

  // Check for clip_v2 variant
  if (clipPath.length() >= 8 &&
      clipPath.substr(clipPath.length() - 8) == "clip.mnn") {
    std::filesystem::path clipPathObj(clipPath);
    std::filesystem::path parentDir = clipPathObj.parent_path();
    std::filesystem::path v2Path = parentDir / "clip_v2.mnn";

    if (std::filesystem::exists(v2Path)) {
      QNN_INFO("Found clip_v2.mnn, upgrading to v2 CLIP");
      clipPath = v2Path.string();
      use_clip_v2 = true;

      std::filesystem::path posEmbPath = parentDir / "pos_emb.bin";
      std::filesystem::path tokenEmbPath = parentDir / "token_emb.bin";

      if (!std::filesystem::exists(posEmbPath)) {
        QNN_ERROR("pos_emb.bin not found: %s", posEmbPath.string().c_str());
        return false;
      }
      if (!std::filesystem::exists(tokenEmbPath)) {
        QNN_ERROR("token_emb.bin not found: %s", tokenEmbPath.string().c_str());
        return false;
      }

      std::ifstream posFile(posEmbPath, std::ios::binary);
      posFile.seekg(0, std::ios::end);
      size_t posSize = posFile.tellg() / sizeof(float);
      posFile.seekg(0, std::ios::beg);
      pos_emb.resize(posSize);
      posFile.read(reinterpret_cast<char*>(pos_emb.data()), posSize * sizeof(float));
      posFile.close();

      std::ifstream tokenFile(tokenEmbPath, std::ios::binary);
      tokenFile.seekg(0, std::ios::end);
      size_t fileSize = tokenFile.tellg();
      tokenFile.seekg(0, std::ios::beg);

      const size_t SIZE_THRESHOLD = 100 * 1024 * 1024;
      if (fileSize > SIZE_THRESHOLD) {
        size_t tokenSize = fileSize / sizeof(float);
        std::vector<float> tempBuffer(tokenSize);
        tokenFile.read(reinterpret_cast<char*>(tempBuffer.data()), fileSize);
        token_emb.resize(tokenSize);
        for (size_t i = 0; i < tokenSize; i++) {
          token_emb[i] = fp32_to_fp16(tempBuffer[i]);
        }
      } else {
        size_t tokenSize = fileSize / sizeof(uint16_t);
        token_emb.resize(tokenSize);
        tokenFile.read(reinterpret_cast<char*>(token_emb.data()), fileSize);
      }
      tokenFile.close();
    }
  }

  // Load tokenizer
  try {
    auto blob = LoadBytesFromFile(tokenizerPath);
    tokenizer = tokenizers::Tokenizer::FromBlobJSON(blob);
    if (!tokenizer) throw std::runtime_error("Tokenizer creation failed.");
  } catch (const std::exception& e) {
    QNN_ERROR("Failed to load tokenizer: %s", e.what());
    return false;
  }

  // Load embeddings
  if (!modelDir.empty()) {
    std::filesystem::path modelPath(modelDir);
    std::filesystem::path embeddingsPath =
        modelPath.parent_path().parent_path() / "embeddings";
    if (std::filesystem::exists(embeddingsPath)) {
      try {
        promptProcessor.loadEmbeddings(embeddingsPath.string());
        QNN_INFO("Loaded %zu embeddings from %s",
                 promptProcessor.getEmbeddingCount(),
                 embeddingsPath.string().c_str());
      } catch (const std::exception& e) {
        QNN_WARN("Failed to load embeddings: %s", e.what());
      }
    }
  }

  // Setup MNN sessions
  MNN::ScheduleConfig cfg_common;
  cfg_common.type = MNN_FORWARD_CPU;
  cfg_common.numThread = 1;
  MNN::BackendConfig bkCfg_common;
  bkCfg_common.memory = MNN::BackendConfig::Memory_Low;
  bkCfg_common.power = MNN::BackendConfig::Power_High;
  cfg_common.backendConfig = &bkCfg_common;
  MNN::ScheduleConfig cfg_mnn_clip = cfg_common;
  cfg_mnn_clip.numThread = 4;

  // Safety checker
  if (use_safety_checker && !safetyCheckerPath.empty()) {
    safetyCheckerInterpreter =
        MNN::Interpreter::createFromFile(safetyCheckerPath.c_str());
    if (!safetyCheckerInterpreter) {
      QNN_ERROR("Failed to load safety checker: %s", safetyCheckerPath.c_str());
      return false;
    }
    safetyCheckerSession = safetyCheckerInterpreter->createSession(cfg_common);
    if (safetyCheckerSession) {
      auto input = safetyCheckerInterpreter->getSessionInput(safetyCheckerSession, nullptr);
      safetyCheckerInterpreter->resizeTensor(input, {1, 224, 224, 3});
      safetyCheckerInterpreter->resizeSession(safetyCheckerSession);
      safetyCheckerInterpreter->releaseModel();
    }
  }

  // MNN CLIP (for CPU or hybrid mode)
  if (use_mnn_clip) {
    clipInterpreter = MNN::Interpreter::createFromFile(clipPath.c_str());
    if (!clipInterpreter) {
      QNN_ERROR("Failed to load MNN CLIP: %s", clipPath.c_str());
      return false;
    }
    clipSession = clipInterpreter->createSession(cfg_mnn_clip);
    if (clipSession) {
      if (use_clip_v2) {
        auto input = clipInterpreter->getSessionInput(clipSession, "input_embedding");
        clipInterpreter->resizeTensor(input, {1, 77, 768});
      } else {
        auto input = clipInterpreter->getSessionInput(clipSession, "input_ids");
        clipInterpreter->resizeTensor(input, {1, 77});
      }
      clipInterpreter->resizeSession(clipSession);
      clipInterpreter->releaseModel();
    }
  }

  // QNN models
  if (!use_mnn) {
    if (config.qnnSystemLibPath.empty() || config.qnnBackendPath.empty()) {
      QNN_ERROR("QNN system library and backend paths required for GPU mode");
      return false;
    }

    g_backendPathCmd = config.qnnBackendPath;
    dynamicloadutil::StatusCode sysStatus =
        dynamicloadutil::getQnnSystemFunctionPointers(config.qnnSystemLibPath,
                                                      &g_qnnSystemFuncs);
    if (sysStatus != dynamicloadutil::StatusCode::SUCCESS) {
      QNN_ERROR("Failed to get QNN system function pointers");
      return false;
    }

    // Apply patch to unet if needed
    if (!config.patchPath.empty()) {
      QNN_INFO("Applying patch to unet model in memory...");
      g_unetPatchedBuffer = applyZstdPatchToBuffer(unetPath, config.patchPath);
      if (!g_unetPatchedBuffer) {
        QNN_ERROR("Failed to apply patch to unet model buffer");
        return false;
      }
    }

    // Create QNN models
    if (!use_mnn_clip) {
      clipApp = createQnnModel(clipPath, "clip");
      if (!clipApp) { QNN_ERROR("Failed to create QNN CLIP model"); return false; }
    }

    unetApp = createQnnModel(unetPath, "unet");
    if (!unetApp) { QNN_ERROR("Failed to create QNN UNET model"); return false; }

    vaeDecoderApp = createQnnModel(vaeDecoderPath, "vae_decoder");
    if (!vaeDecoderApp) { QNN_ERROR("Failed to create QNN VAE Decoder"); return false; }

    if (!vaeEncoderPath.empty()) {
      vaeEncoderApp = createQnnModel(vaeEncoderPath, "vae_encoder");
      if (!vaeEncoderApp) QNN_WARN("Failed to create QNN VAE Encoder");
    }

    // Initialize QNN apps
    int status = EXIT_SUCCESS;
    if (!use_mnn_clip && clipApp) {
      status = sample_app::initializeQnnApp("CLIP", clipApp);
      if (status != EXIT_SUCCESS) return false;
    }
    if (unetApp) {
      if (g_unetPatchedBuffer && g_unetPatchedBuffer->buffer) {
        status = sample_app::initializeQnnApp(
            "UNET", unetApp, g_unetPatchedBuffer->buffer.get(),
            g_unetPatchedBuffer->size);
      } else {
        status = sample_app::initializeQnnApp("UNET", unetApp);
      }
      if (status != EXIT_SUCCESS) return false;
      if (g_unetPatchedBuffer) {
        QNN_INFO("Releasing unet patch buffer to free memory");
        g_unetPatchedBuffer.reset();
      }
    }
    if (vaeDecoderApp) {
      status = sample_app::initializeQnnApp("VAEDecoder", vaeDecoderApp);
      if (status != EXIT_SUCCESS) return false;
    }
    if (vaeEncoderApp) {
      status = sample_app::initializeQnnApp("VAEEncoder", vaeEncoderApp);
      if (status != EXIT_SUCCESS) return false;
    }
  }

  QNN_INFO("All models initialized successfully");
  return true;
}

SDGenerationResult run_generation(const SDGenerateParams& params,
                                  SDProgressCallback progressCb,
                                  std::atomic<bool>& stopFlag) {
  // Set global state from params (xororz's generateImage reads globals)
  prompt = params.prompt;
  negative_prompt = params.negativePrompt;
  steps = params.steps;
  cfg = params.cfgScale;
  seed = params.seed;
  scheduler_type = params.scheduler;
  use_opencl = params.useOpenCL;
  show_diffusion_process = params.showDiffusionProcess;
  show_diffusion_stride = params.showDiffusionStride;
  denoise_strength = params.denoiseStrength;
  output_width = params.width;
  output_height = params.height;
  sample_width = params.width / 8;
  sample_height = params.height / 8;

  // Handle img2img data
  request_img2img = params.isImg2Img;
  request_has_mask = params.hasMask;
  img_data = params.inputImage;
  mask_data = params.mask;
  mask_data_full = params.maskFull;

  // Generate seed if 0
  if (seed == 0) {
    seed = static_cast<unsigned>(hashSeed(
        std::chrono::system_clock::now().time_since_epoch().count()));
  }

  QNN_INFO("Generating: prompt='%s', steps=%d, size=%dx%d, seed=%u",
           prompt.c_str(), steps, output_width, output_height, seed);

  // Bridge callback: raw bytes directly (no base64 roundtrip) + stop flag check
  auto progress_bridge = [&progressCb, &stopFlag](int step, int total_steps,
                                                    const std::vector<uint8_t>& image_data) {
    if (stopFlag.load()) {
      throw std::runtime_error("Generation cancelled");
    }
    if (progressCb) {
      if (!image_data.empty()) {
        progressCb(step, total_steps, image_data.data(),
                    static_cast<int>(image_data.size()),
                    output_width, output_height);
      } else {
        progressCb(step, total_steps, nullptr, 0, 0, 0);
      }
    }
  };

  GenerationResult result = generateImage(progress_bridge);

  // Convert to our result type
  SDGenerationResult sdResult;
  sdResult.imageData = std::move(result.image_data);
  sdResult.width = result.width;
  sdResult.height = result.height;
  sdResult.seed = seed;  // Actual seed used (not the input 0)
  sdResult.channels = result.channels;
  sdResult.generationTimeMs = result.generation_time_ms;
  sdResult.firstStepTimeMs = result.first_step_time_ms;
  return sdResult;
}

bool apply_lora(const std::string& path, float weight) {
  // LoRA application uses the SafeTensor2MNN utilities
  // This is model-format dependent and will be implemented
  // when the full LoRA pipeline is tested
  QNN_INFO("LoRA apply requested: %s (weight=%.2f)", path.c_str(), weight);
  // TODO: Implement LoRA application using generateMNNModels with lora params
  return true;
}

void clear_lora() {
  QNN_INFO("LoRA clear requested");
  // TODO: Reload base weights
}

void cleanup() {
  QNN_INFO("Cleaning up pipeline resources");

  if (clipSession && clipInterpreter) {
    clipInterpreter->releaseSession(clipSession);
    clipSession = nullptr;
  }
  if (unetSession && unetInterpreter) {
    unetInterpreter->releaseSession(unetSession);
    unetSession = nullptr;
  }
  if (safetyCheckerSession && safetyCheckerInterpreter) {
    safetyCheckerInterpreter->releaseSession(safetyCheckerSession);
    safetyCheckerSession = nullptr;
  }

  delete clipInterpreter;     clipInterpreter = nullptr;
  delete unetInterpreter;     unetInterpreter = nullptr;
  delete vaeDecoderInterpreter; vaeDecoderInterpreter = nullptr;
  delete vaeEncoderInterpreter; vaeEncoderInterpreter = nullptr;
  delete safetyCheckerInterpreter; safetyCheckerInterpreter = nullptr;

  clipApp.reset();
  unetApp.reset();
  vaeDecoderApp.reset();
  vaeEncoderApp.reset();
  upscalerApp.reset();

  tokenizer.reset();
  g_unetPatchedBuffer.reset();
  pos_emb.clear();
  token_emb.clear();

  QNN_INFO("Pipeline resources cleaned up");
}

std::string get_info() {
  nlohmann::json info;
  info["backend"] = use_mnn ? "MNN (CPU)" : "QNN (NPU)";
  info["clip"] = clipPath;
  info["unet"] = unetPath;
  info["vae_decoder"] = vaeDecoderPath;
  info["vae_encoder"] = vaeEncoderPath;
  info["text_embedding_size"] = text_embedding_size;
  info["pony"] = ponyv55;
  info["safety_checker"] = use_safety_checker;
  info["cpu_clip"] = use_mnn_clip;
  return info.dump();
}

} // namespace sd_pipeline