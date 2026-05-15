/**
 * Pipeline Orchestrator — Core inference pipeline.
 *
 * Moved from pipeline/diffusion_pipeline.cpp (Phase 1.10-1.11).
 * Calls extracted modules: text_encoder, unet_runner, vae_codec,
 * upscaler, safety_checker, scheduler_factory.
 *
 * Original port: xororz/local-dream
 */

#define TN_MODULE TN_MODULE_AI_SD
#define TN_TAG    "ai_sd"
#include <tn_security/tn_security_macros.h>

#include "../pipeline/diffusion_pipeline.h"
#include "pipeline_context.h"
#include "../utils/sd_logger.h"

#include <chrono>
#include <functional>
// iostream removed — using sd_logger.h
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <atomic>
#include <unordered_map>
#include <deque>
#include <mutex>

// Model headers (ported from xororz)
#include "../utils/config.h"
#include "../pipeline/schedulers/scheduler_factory.h"
#include "../utils/float_conversion.h"
#include "../utils/image_convert.h"
#include "../utils/laplacian_blend.h"
#include "../pipeline/prompt_processor.h"
#include "../model/qnn_model.h"
#include "../utils/sd_utils.h"
#include "../utils/safetensor_to_mnn.h"

// QNN Headers (DynamicLoadUtil + PAL moved to model_loader.cpp)
#include "Logger.hpp"
#include "QnnSampleAppUtils.hpp"

// External Libraries (httplib removed - no longer needed)
#include "json.hpp"
#include "tokenizers_cpp.h"

// Extracted modules (Phase 1.3+)
#include "../clip/text_encoder.h"
#include "../safety/safety_checker.h"
#include "../unet/unet_runner.h"
#include "../upscaler/upscaler.h"
#include "../vae/vae_codec.h"

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

// zstd.h moved to model_loader.cpp

#ifdef SD_ENABLE_DIAGNOSTICS
// --- Diagnostic: compute min/max/mean of a float buffer ---
static void log_tensor_stats(const char* label, const float* data, int count) {
  if (!data || count <= 0) {
    SD_LOG_TRACE("[DIAG] %s: null or empty", label);
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
  SD_LOG_TRACE("[DIAG] %s: count=%d min=%.8f max=%.8f mean=%.8f nan=%d inf=%d first4=[%.8f %.8f %.8f %.8f]",
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
  SD_LOG_TRACE("[DIAG] %s: rmse=%.8f max_abs_diff=%.8f@idx=%d  a[0..3]=[%.8f %.8f %.8f %.8f] b[0..3]=[%.8f %.8f %.8f %.8f]",
               label, rmse, max_diff, max_diff_idx,
               a[0], count > 1 ? a[1] : 0.f, count > 2 ? a[2] : 0.f, count > 3 ? a[3] : 0.f,
               b[0], count > 1 ? b[1] : 0.f, count > 2 ? b[2] : 0.f, count > 3 ? b[3] : 0.f);
}
#endif // SD_ENABLE_DIAGNOSTICS

// nchw_to_rgb_bytes moved to utils/image_convert.h (Phase 1.9)

// ============================================================================
// Global variable DEFINITIONS — declared extern in pipeline_globals.h
// Migration artifact: removed as modules switch to PipelineContext&
// ============================================================================
#include "../pipeline/pipeline_globals.h"

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
std::vector<uint16_t> token_emb;
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

// g_unetPatchedBuffer, g_qnnSystemFuncs, g_backendPathCmd moved to model_loader.cpp
std::string model_dir;
bool clip_skip_2 = false;

// Perf 7: Persistent MNN sessions across generations
static UNetRunner g_unet_runner;
// VAE session dimension tracking (0 = not initialized)
static int g_vae_dec_sample_w = 0, g_vae_dec_sample_h = 0;
static int g_vae_enc_out_w = 0, g_vae_enc_out_h = 0;

// ============================================================================
// CLIP output cache (Perf — skip re-encoding identical prompts).
// Same (prompt, negative_prompt, embedding_dim, clip_v2) produces identical
// embeddings on a given model. Cache holds at most CLIP_CACHE_MAX entries
// in FIFO order; cleared on model load/release.
// ============================================================================
namespace {
struct ClipCacheKey {
    std::string prompt;
    std::string negative_prompt;
    int embedding_dim;
    bool use_clip_v2;
    bool operator==(const ClipCacheKey& o) const {
        return prompt == o.prompt && negative_prompt == o.negative_prompt
            && embedding_dim == o.embedding_dim && use_clip_v2 == o.use_clip_v2;
    }
};
struct ClipCacheKeyHash {
    size_t operator()(const ClipCacheKey& k) const noexcept {
        size_t h = std::hash<std::string>{}(k.prompt);
        auto mix = [&](size_t v) { h ^= v + 0x9e3779b9u + (h << 6) + (h >> 2); };
        mix(std::hash<std::string>{}(k.negative_prompt));
        mix(static_cast<size_t>(k.embedding_dim));
        mix(static_cast<size_t>(k.use_clip_v2));
        return h;
    }
};
constexpr size_t CLIP_CACHE_MAX = 8;
std::unordered_map<ClipCacheKey, std::vector<float>, ClipCacheKeyHash> g_clip_cache;
std::deque<ClipCacheKey> g_clip_cache_order;
std::mutex g_clip_cache_mtx;
} // namespace

namespace sd_pipeline {
void clear_clip_cache() {
    std::lock_guard<std::mutex> lock(g_clip_cache_mtx);
    g_clip_cache.clear();
    g_clip_cache_order.clear();
}
} // namespace sd_pipeline

// createQnnModel() moved to loader/model_loader.cpp (Phase 1.2)
// ZSTD patch functions + initializeQnnApp moved to loader/model_loader.cpp (Phase 1.2)
// processWeightedPrompt/processPromptPair moved to clip/text_encoder.cpp (Phase 1.3)
// blend_vae_encoder_tiles, blend_vae_output_tiles, calculate_tile_positions,
// calculate_vae_tile_positions moved to vae/vae_codec.cpp (Phase 1.6)

// upscaleImageWithModel, upscaleImageWithMNN moved to upscaler/upscaler.cpp (Phase 1.7)

// --- Image Generation ---
GenerationResult generateImage(
    PipelineContext& ctx,
    std::function<void(int step, int total_steps,
                       const std::vector<uint8_t> &image_data)>
        progress_callback) {
  using namespace qnn::tools::sample_app;

  // Local shadows of config.h globals — reads from PipelineContext (Phase 1.12)
  int sample_width = ctx.gen.sample_w;
  int sample_height = ctx.gen.sample_h;
  int output_width = ctx.gen.width;
  int output_height = ctx.gen.height;
  int text_embedding_size = ctx.config.text_embedding_size;
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

    if (unetApp) unetApp->clearCachedEmbeddings();

    ctx.buffers.resize_for_generation(sample_width, sample_height,
                                      output_width, output_height);

    // --- CLIP ---
    std::vector<float> text_embedding_float(batch_size * 77 *
                                            text_embedding_size);

    // Cache lookup: same (prompt, neg, dim, v2) → identical embedding on a
    // given model. Skip the entire CLIP inference on hit.
    ClipCacheKey cache_key{prompt, negative_prompt, text_embedding_size, use_clip_v2};
    bool clip_cache_hit = false;
    {
        std::lock_guard<std::mutex> lock(g_clip_cache_mtx);
        auto it = g_clip_cache.find(cache_key);
        if (it != g_clip_cache.end() &&
            it->second.size() == text_embedding_float.size()) {
            std::copy(it->second.begin(), it->second.end(),
                      text_embedding_float.begin());
            clip_cache_hit = true;
        }
    }

    if (clip_cache_hit) {
        SD_LOG_INFO("CLIP cache HIT — skipping inference");
    }

    ProcessedPromptPair processed;
    if (!clip_cache_hit) {
        processed = processPromptPair(prompt, negative_prompt, 77);
    }
    std::vector<int> clip_input_ids = clip_cache_hit ? std::vector<int>{}
                                                     : processed.ids;
    if (!clip_cache_hit) {
        auto parsed_input_text = tokenizer->Decode(clip_input_ids);
        SD_LOG_INFO("Parsed Input Text: %s", parsed_input_text.c_str());
    }
    auto clip_start = std::chrono::high_resolution_clock::now();
    int32_t *input_ids_ptr = clip_cache_hit ? nullptr : clip_input_ids.data();
    float *embed_ptr = text_embedding_float.data();

    // Log CLIP input IDs for both prompts
#ifdef SD_ENABLE_DIAGNOSTICS
    SD_LOG_INFO("[DIAG] CLIP neg IDs (first 8): %d %d %d %d %d %d %d %d",
                input_ids_ptr[0], input_ids_ptr[1], input_ids_ptr[2], input_ids_ptr[3],
                input_ids_ptr[4], input_ids_ptr[5], input_ids_ptr[6], input_ids_ptr[7]);
    SD_LOG_INFO("[DIAG] CLIP pos IDs (first 8): %d %d %d %d %d %d %d %d",
                input_ids_ptr[77], input_ids_ptr[78], input_ids_ptr[79], input_ids_ptr[80],
                input_ids_ptr[81], input_ids_ptr[82], input_ids_ptr[83], input_ids_ptr[84]);
    SD_LOG_INFO("[DIAG] use_mnn=%d use_mnn_clip=%d use_clip_v2=%d",
                (int)use_mnn, (int)use_mnn_clip, (int)use_clip_v2);
#endif

    if (clip_cache_hit) {
      // Skip both MNN/QNN branches entirely.
    } else if (use_mnn || use_mnn_clip) {
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

      // RAII cleanup lambda — prevents interpreter/session leak on exception
      auto clipCleanup = [&]() {
        if (sessionCreated && currentClipSession)
          currentClipInterpreter->releaseSession(currentClipSession);
        if (dynamicCreated && currentClipInterpreter)
          delete currentClipInterpreter;
      };

      try {
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
      }
      } catch (...) {
        clipCleanup();
        throw;
      }
      clipCleanup();
    } else {
      if (!clipApp) {
        TN_ERR(TN_CODE_NOT_READY, TN_STAGE_SD_CLIP,
               "Global clipApp not initialized");
        throw std::runtime_error("Global clipApp not initialized!");
      }
      if (StatusCode::SUCCESS !=
          clipApp->executeClipGraphs(input_ids_ptr, embed_ptr)) {
        TN_ERR(TN_CODE_DECODE_FAIL, TN_STAGE_SD_CLIP,
               "QNN CLIP exec failed (neg)");
        throw std::runtime_error("QNN CLIP exec failed (neg)");
      }
      if (StatusCode::SUCCESS !=
          clipApp->executeClipGraphs(input_ids_ptr + 77,
                                     embed_ptr + 77 * text_embedding_size)) {
        TN_ERR(TN_CODE_DECODE_FAIL, TN_STAGE_SD_CLIP,
               "QNN CLIP exec failed (pos)");
        throw std::runtime_error("QNN CLIP exec failed (pos)");
      }
    }

    auto clip_end = std::chrono::high_resolution_clock::now();
    if (!clip_cache_hit) {
        SD_LOG_INFO("CLIP dur: %dms",
                    (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                        clip_end - clip_start).count());

        // Store result for next identical (prompt, neg) pair. Bounded FIFO; the
        // cache is cleared on every model load/release.
        std::lock_guard<std::mutex> lock(g_clip_cache_mtx);
        if (g_clip_cache.find(cache_key) == g_clip_cache.end()) {
            if (g_clip_cache_order.size() >= CLIP_CACHE_MAX) {
                g_clip_cache.erase(g_clip_cache_order.front());
                g_clip_cache_order.pop_front();
            }
            g_clip_cache.emplace(cache_key, text_embedding_float);
            g_clip_cache_order.push_back(cache_key);
        }
    }
#ifdef SD_ENABLE_DIAGNOSTICS
    log_tensor_stats("clip_embed_neg", text_embedding_float.data(),
                     77 * text_embedding_size);
    log_tensor_stats("clip_embed_pos", text_embedding_float.data() + 77 * text_embedding_size,
                     77 * text_embedding_size);
    log_tensor_diff("clip_neg_vs_pos",
                    text_embedding_float.data(),
                    text_embedding_float.data() + 77 * text_embedding_size,
                    77 * text_embedding_size);
#endif
    current_step++;
    progress_callback(current_step, total_run_steps, {});

    // Phase 4.2: Sequential model loading — release CLIP after encoding to reduce
    // peak memory. Embeddings are stored in text_embedding_float and constant
    // throughout denoising. MNN CLIP interpreter + session are no longer needed.
    if (use_mnn && clipInterpreter && clipSession) {
        clipInterpreter->releaseSession(clipSession);
        clipSession = nullptr;
        delete clipInterpreter;
        clipInterpreter = nullptr;
        SD_LOG_INFO("[LOAD] CLIP interpreter released (sequential loading)");
    }

    // --- Scheduler & Latents ---
    auto scheduler = createScheduler(scheduler_type, ponyv55);
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

#ifdef SD_ENABLE_DIAGNOSTICS
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
#endif

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
          // Perf 7: Reuse VAE encoder session across generations
          if (!vaeEncoderInterpreter || !vaeEncoderSession ||
              g_vae_enc_out_w != output_width || g_vae_enc_out_h != output_height) {
            // Release old session if dimensions changed
            if (vaeEncoderSession && vaeEncoderInterpreter) {
              vaeEncoderInterpreter->releaseSession(vaeEncoderSession);
              vaeEncoderSession = nullptr;
            }
            if (vaeEncoderInterpreter) {
              delete vaeEncoderInterpreter;
              vaeEncoderInterpreter = nullptr;
            }

            vaeEncoderInterpreter =
                MNN::Interpreter::createFromFile(vaeEncoderPath.c_str());
            if (!vaeEncoderInterpreter)
              throw std::runtime_error("Failed MNN VAE Enc create");

            MNN::ScheduleConfig cfg_vae_enc;
            MNN::BackendConfig bkCfg_vae_enc;
            if (use_opencl) {
              auto cache_file = modelDir + "/vae_enc_cache.mnnc." +
                                std::to_string(output_width);
              vaeEncoderInterpreter->setCacheFile(cache_file.c_str());
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

            vaeEncoderSession = vaeEncoderInterpreter->createSession(cfg_vae_enc);
            if (!vaeEncoderSession)
              throw std::runtime_error("Failed create MNN VAE Enc session!");

            auto input = vaeEncoderInterpreter->getSessionInput(
                vaeEncoderSession, "input");
            vaeEncoderInterpreter->resizeTensor(
                input, {1, 3, output_height, output_width});
            vaeEncoderInterpreter->resizeSession(vaeEncoderSession);
            if (use_opencl) {
              vaeEncoderInterpreter->updateCacheFile(vaeEncoderSession);
            }
            vaeEncoderInterpreter->releaseModel();

            g_vae_enc_out_w = output_width;
            g_vae_enc_out_h = output_height;
            SD_LOG_INFO("[VAE] Created persistent encoder session %dx%d", output_width, output_height);
          } else {
            SD_LOG_DEBUG("[VAE] Reusing encoder session %dx%d", output_width, output_height);
          }

          auto input = vaeEncoderInterpreter->getSessionInput(
              vaeEncoderSession, "input");
          auto mean_t = vaeEncoderInterpreter->getSessionOutput(
              vaeEncoderSession, "mean");
          auto std_t = vaeEncoderInterpreter->getSessionOutput(
              vaeEncoderSession, "std");

          std::unique_ptr<MNN::Tensor> input_nchw_tensor(new MNN::Tensor(input, MNN::Tensor::CAFFE));
          std::unique_ptr<MNN::Tensor> mean_nchw_tensor(new MNN::Tensor(mean_t, MNN::Tensor::CAFFE));
          std::unique_ptr<MNN::Tensor> std_nchw_tensor(new MNN::Tensor(std_t, MNN::Tensor::CAFFE));

          memcpy(input_nchw_tensor->host<float>(), img_data.data(),
                 img_data.size() * sizeof(float));
          input->copyFromHostTensor(input_nchw_tensor.get());
          vaeEncoderInterpreter->runSession(vaeEncoderSession);

          mean_t->copyToHostTensor(mean_nchw_tensor.get());
          std_t->copyToHostTensor(std_nchw_tensor.get());
          memcpy(vae_enc_mean.data(), mean_nchw_tensor->host<float>(),
                 vae_enc_mean.size() * sizeof(float));
          memcpy(vae_enc_std.data(), std_nchw_tensor->host<float>(),
                 vae_enc_std.size() * sizeof(float));
          // Perf 7: session NOT released here — persists for next generation
        } else {
          if (!vaeEncoderApp) {
            TN_ERR(TN_CODE_NOT_READY, TN_STAGE_SD_VAE,
                   "Global vaeEncoderApp not initialized");
            throw std::runtime_error("Global vaeEncoderApp not init!");
          }
          if (StatusCode::SUCCESS !=
              vaeEncoderApp->executeVaeEncoderGraphs(
                  img_data.data(), vae_enc_mean.data(), vae_enc_std.data())) {
            TN_ERR(TN_CODE_DECODE_FAIL, TN_STAGE_SD_VAE,
                   "QNN VAE encode exec failed (non-tiled)");
            throw std::runtime_error("QNN VAE enc exec failed");
          }
        }

        auto mean = xt::adapt(vae_enc_mean, shape);
        auto std_dev = xt::adapt(vae_enc_std, shape);
        xt::xarray<float> noise_0 = xt::random::randn<float>(shape);
        xt::xarray<float> img_lat = xt::eval(mean + std_dev * noise_0);
        img_lat_scaled = xt::eval(0.18215 * img_lat);

      } else {
        SD_LOG_DEBUG("[VAE] Using VAE encoder tiling for %dx%d input", output_width, output_height);

        const int vae_enc_tile_size = 512;
        const int vae_enc_latent_tile_size = 64;

        // Use generic tile position calculator
        auto [img_positions, latent_positions, img_overlap_x, img_overlap_y,
              latent_overlap_x, latent_overlap_y] =
            calculate_vae_tile_positions(output_width, output_height);

        int num_tiles = img_positions.size();
        SD_LOG_DEBUG("[VAE] VAE encoder: %d tiles, overlap %dx%dpx (latent: %dx%d)",
                     num_tiles, img_overlap_x, img_overlap_y, latent_overlap_x, latent_overlap_y);

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

          if (!vaeEncoderApp) {
            TN_ERR(TN_CODE_NOT_READY, TN_STAGE_SD_VAE,
                   "Global vaeEncoderApp not initialized (tile path)");
            throw std::runtime_error("Global vaeEncoderApp not init!");
          }

          if (StatusCode::SUCCESS !=
              vaeEncoderApp->executeVaeEncoderGraphs(tile_img_vec.data(),
                                                     tile_mean_vec.data(),
                                                     tile_std_vec.data())) {
            TN_ERR(TN_CODE_DECODE_FAIL, TN_STAGE_SD_VAE,
                   "QNN VAE encode exec failed for tile %zu", i);
            throw std::runtime_error("QNN VAE enc exec failed for tile");
          }

          std::vector<int> tile_shape = {1, 4, vae_enc_latent_tile_size,
                                         vae_enc_latent_tile_size};
          encoded_tiles_mean_std.push_back(
              {xt::adapt(tile_mean_vec, tile_shape),
               xt::adapt(tile_std_vec, tile_shape)});
          SD_LOG_DEBUG("[VAE] Processed VAE encoder tile %zu/%zu", i + 1, img_positions.size());
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

        SD_LOG_DEBUG("[VAE] VAE encoder tiling completed: %zu tiles blended", encoded_tiles_mean_std.size());
      }

      auto vae_enc_end = std::chrono::high_resolution_clock::now();
      SD_LOG_INFO("[VAE] Encoder %ldms",
                  (long)std::chrono::duration_cast<std::chrono::milliseconds>(
                      vae_enc_end - vae_enc_start).count());

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

    // UNet runner (Phase 1.5): encapsulates MNN session lifecycle + QNN dispatch
    // Perf 7: reuse session across generations if params match
    g_unet_runner.initIfNeeded(use_mnn, use_opencl, unetPath, modelDir,
                               batch_size, sample_height, sample_width, text_embedding_size);

    for (int i = start_step; i < timesteps.size(); ++i) {
#ifdef SD_ENABLE_DIAGNOSTICS
      SD_LOG_INFO("[DIAG] Step %d/%zu: show_process=%d use_mnn=%d stride=%d cond=(%d && %d && %d)",
                  i, timesteps.size(), (int)show_diffusion_process, (int)use_mnn, show_diffusion_stride,
                  (int)show_diffusion_process, (int)(!use_mnn),
                  (int)((i - start_step) % show_diffusion_stride == 0));
#endif
      if (show_diffusion_process && !use_mnn &&
          (i - start_step) % show_diffusion_stride == 0) {
#ifdef SD_ENABLE_DIAGNOSTICS
        SD_LOG_INFO("[DIAG] Preview decode attempt at step %d", i);
#endif
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
#ifdef SD_ENABLE_DIAGNOSTICS
            SD_LOG_INFO("[DIAG] Preview VAE decode succeeded at step %d", i);
#endif
            // Direct CHW→RGB conversion (no xtensor transpose, no base64)
            std::vector<uint8_t> out_data = nchw_to_rgb_bytes(
                pixels.data(), 3, output_height, output_width);
#ifdef SD_ENABLE_DIAGNOSTICS
            SD_LOG_INFO("[DIAG] Preview image: %zu bytes, first RGB=(%d,%d,%d)",
                        out_data.size(),
                        out_data.size() >= 3 ? out_data[0] : -1,
                        out_data.size() >= 3 ? out_data[1] : -1,
                        out_data.size() >= 3 ? out_data[2] : -1);
#endif
            progress_callback(current_step, total_run_steps, out_data);
          } else {
#ifdef SD_ENABLE_DIAGNOSTICS
            SD_LOG_WARN("[DIAG] Preview VAE decode FAILED at step %d (returned failure)", i);
#endif
            progress_callback(current_step, total_run_steps, {});
          }
        } catch (const std::exception &e) {
#ifdef SD_ENABLE_DIAGNOSTICS
          SD_LOG_WARN("[DIAG] Preview decode EXCEPTION at step %d: %s", i, e.what());
#endif
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

      // Perf 1: Reuse pre-allocated buffers (no per-step heap alloc)
      float* latents_in_ptr = ctx.buffers.latents_in.data();
      std::copy(latents_scaled.begin(), latents_scaled.end(), latents_in_ptr);
      std::copy(latents_scaled.begin(), latents_scaled.end(),
                latents_in_ptr + single_latent_size);
      float* unet_out_ptr = ctx.buffers.unet_out.data();

      // UNetRunner dispatches to MNN or QNN internally; cfg lets the runner
      // skip the uncond pass on QNN when cfg == 1 (huge win on 1-4 step LCM).
      g_unet_runner.step(latents_in_ptr, static_cast<int>(current_ts),
                       text_embedding_float.data(), unet_out_ptr, cfg);

      auto step_end_time = std::chrono::high_resolution_clock::now();
      auto step_dur = std::chrono::duration_cast<std::chrono::milliseconds>(
          step_end_time - step_start_time);

      if (i == start_step) first_step_time_ms = step_dur.count();
#ifdef SD_ENABLE_DIAGNOSTICS
      SD_LOG_INFO("[DIAG] UNET step %d dur: %dms", i, (int)step_dur.count());

      // Log raw UNet output for first and last step
      if (i == start_step || i == (int)timesteps.size() - 1) {
        log_tensor_stats("unet_out_raw", unet_out_ptr,
                         batch_size * single_latent_size);
      }
#endif

      xt::xarray<float> noise_pred_batch =
          xt::adapt(unet_out_ptr, batch_size * single_latent_size, xt::no_ownership(), shape_batch2);
      xt::xarray<float> uncond = xt::view(noise_pred_batch, 0);
      xt::xarray<float> txt = xt::view(noise_pred_batch, 1);

      // Log CFG computation details at first step
#ifdef SD_ENABLE_DIAGNOSTICS
      if (i == start_step) {
        log_tensor_stats("unet_uncond", uncond.data(), static_cast<int>(uncond.size()));
        log_tensor_stats("unet_cond", txt.data(), static_cast<int>(txt.size()));
        log_tensor_diff("uncond_vs_cond", uncond.data(), txt.data(), static_cast<int>(uncond.size()));
        SD_LOG_INFO("[DIAG] cfg_scale=%.4f", cfg);
      }
#endif

      // Perf 5: Plain float loop CFG — eliminates xt::eval heap alloc
      float* np = ctx.buffers.noise_pred.data();
      const float* uc = uncond.data();
      const float* tx = txt.data();
      for (int k = 0; k < single_latent_size; ++k) {
          np[k] = uc[k] + cfg * (tx[k] - uc[k]);
      }
      xt::xarray<float> noise_pred =
          xt::adapt(np, single_latent_size, xt::no_ownership(), shape);

#ifdef SD_ENABLE_DIAGNOSTICS
      if (i == start_step) {
        log_tensor_stats("noise_pred_after_cfg", noise_pred.data(), static_cast<int>(noise_pred.size()));
      }

      // Save pre-step latents for comparison
      std::vector<float> pre_step_latents(latents.data(), latents.data() + latents.size());

      SD_LOG_INFO("[DIAG] Pre-scheduler step %d: step_idx=%zu sigma=%.8f timestep=%.0f",
                  i, scheduler->get_step_index(), scheduler->get_current_sigma(), timesteps(i));
#endif

      auto step_result = scheduler->step(noise_pred, timesteps(i), latents);

#ifdef SD_ENABLE_DIAGNOSTICS
      if (i == start_step) {
        log_tensor_stats("scheduler_prev_sample", step_result.prev_sample.data(),
                         static_cast<int>(step_result.prev_sample.size()));
      }
#endif

      latents = step_result.prev_sample;  // xarray assignment already materializes

#ifdef SD_ENABLE_DIAGNOSTICS
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
#endif

      if (request_has_mask) {
        xt::xarray<int> t_xt = {(int)(timesteps(i))};
        xt::xarray<float> orig_noised =
            scheduler->add_noise(original_latents, latents_noise, t_xt);
        latents = xt::eval(orig_noised * (1.0f - mask) + latents * mask);
      }

#ifdef SD_ENABLE_DIAGNOSTICS
      // Log latent stats at first, mid, and last step
      if (i == start_step || i == (int)timesteps.size() / 2 || i == (int)timesteps.size() - 1) {
        log_tensor_stats(("latents_after_step_" + std::to_string(i)).c_str(),
                         latents.data(), static_cast<int>(latents.size()));
      }
#endif

      current_step++;
    }

    // Perf 7: UNet session persists across generations (g_unet_runner not cleaned up here)

    // --- VAE Decode ---
    auto vae_dec_start = std::chrono::high_resolution_clock::now();

    bool need_vae_tiling =
        ((output_width > 512 || output_height > 512) && !use_mnn);
    if (need_vae_tiling) {
      SD_LOG_INFO("Using VAE decoder tiling for %dx%d output", output_width, output_height);
    }

#ifdef SD_ENABLE_DIAGNOSTICS
    // Log latents BEFORE VAE scaling
    log_tensor_stats("latents_pre_vae_scale", latents.data(),
                     static_cast<int>(latents.size()));
#endif

    latents = xt::eval((1.0 / 0.18215) * latents);

#ifdef SD_ENABLE_DIAGNOSTICS
    // Log latents AFTER VAE scaling (this is what goes into the VAE decoder)
    log_tensor_stats("latents_vae_input", latents.data(),
                     static_cast<int>(latents.size()));
#endif

    xt::xarray<float> pixels;

    if (!need_vae_tiling) {
      std::vector<float> vae_dec_in_vec(latents.begin(), latents.end());
      std::vector<float> vae_dec_out_pixels(1 * 3 * output_width *
                                            output_height);

      if (use_mnn) {
        // Perf 7: Reuse VAE decoder session across generations
        if (!vaeDecoderInterpreter || !vaeDecoderSession ||
            g_vae_dec_sample_w != sample_width || g_vae_dec_sample_h != sample_height) {
          // Release old session if dimensions changed
          if (vaeDecoderSession && vaeDecoderInterpreter) {
            vaeDecoderInterpreter->releaseSession(vaeDecoderSession);
            vaeDecoderSession = nullptr;
          }
          if (vaeDecoderInterpreter) {
            delete vaeDecoderInterpreter;
            vaeDecoderInterpreter = nullptr;
          }

          vaeDecoderInterpreter =
              MNN::Interpreter::createFromFile(vaeDecoderPath.c_str());
          if (!vaeDecoderInterpreter)
            throw std::runtime_error("Failed to create MNN VAE Decoder interpreter!");

          MNN::ScheduleConfig cfg_vae;
          MNN::BackendConfig bkCfg_vae;
          if (use_opencl) {
            auto cache_file =
                modelDir + "/vae_dec_cache.mnnc." + std::to_string(output_width);
            vaeDecoderInterpreter->setCacheFile(cache_file.c_str());
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

          vaeDecoderSession = vaeDecoderInterpreter->createSession(cfg_vae);
          if (!vaeDecoderSession)
            throw std::runtime_error("Failed create MNN VAE Dec session!");

          auto input = vaeDecoderInterpreter->getSessionInput(
              vaeDecoderSession, "latent_sample");
          vaeDecoderInterpreter->resizeTensor(
              input, {1, 4, sample_height, sample_width});
          vaeDecoderInterpreter->resizeSession(vaeDecoderSession);
          if (use_opencl) {
            vaeDecoderInterpreter->updateCacheFile(vaeDecoderSession);
          }
          vaeDecoderInterpreter->releaseModel();

          g_vae_dec_sample_w = sample_width;
          g_vae_dec_sample_h = sample_height;
          SD_LOG_INFO("[VAE] Created persistent decoder session %dx%d", sample_width, sample_height);
        } else {
          SD_LOG_DEBUG("[VAE] Reusing decoder session %dx%d", sample_width, sample_height);
        }

        auto input = vaeDecoderInterpreter->getSessionInput(
            vaeDecoderSession, "latent_sample");
        auto output = vaeDecoderInterpreter->getSessionOutput(
            vaeDecoderSession, "sample");

        std::unique_ptr<MNN::Tensor> input_nchw_tensor(new MNN::Tensor(input, MNN::Tensor::CAFFE));
        std::unique_ptr<MNN::Tensor> output_nchw_tensor(new MNN::Tensor(output, MNN::Tensor::CAFFE));

        memcpy(input_nchw_tensor->host<float>(), vae_dec_in_vec.data(),
               vae_dec_in_vec.size() * sizeof(float));
        input->copyFromHostTensor(input_nchw_tensor.get());

        vaeDecoderInterpreter->runSession(vaeDecoderSession);

        output->copyToHostTensor(output_nchw_tensor.get());
        memcpy(vae_dec_out_pixels.data(), output_nchw_tensor->host<float>(),
               vae_dec_out_pixels.size() * sizeof(float));

#ifdef SD_ENABLE_DIAGNOSTICS
        SD_LOG_INFO("[DIAG] MNN VAE decoder completed");
        log_tensor_stats("mnn_vae_dec_output", vae_dec_out_pixels.data(),
                         static_cast<int>(vae_dec_out_pixels.size()));
#endif
        // Perf 7: session NOT released here — persists for next generation
      } else {
        if (!vaeDecoderApp) {
          TN_ERR(TN_CODE_NOT_READY, TN_STAGE_SD_VAE,
                 "Global vaeDecoderApp not initialized");
          throw std::runtime_error("Global vaeDecoderApp not init!");
        }

#ifdef SD_ENABLE_DIAGNOSTICS
        SD_LOG_INFO("[DIAG] Calling QNN VAE decoder: input=%d floats, output=%d floats",
                    (int)vae_dec_in_vec.size(), (int)vae_dec_out_pixels.size());
        log_tensor_stats("vae_dec_input", vae_dec_in_vec.data(),
                         static_cast<int>(vae_dec_in_vec.size()));
#endif

        if (StatusCode::SUCCESS !=
            vaeDecoderApp->executeVaeDecoderGraphs(vae_dec_in_vec.data(),
                                                   vae_dec_out_pixels.data())) {
          TN_ERR(TN_CODE_DECODE_FAIL, TN_STAGE_SD_VAE,
                 "QNN VAE decode exec failed (non-tiled)");
          throw std::runtime_error("QNN VAE dec exec failed");
        }

#ifdef SD_ENABLE_DIAGNOSTICS
        SD_LOG_INFO("[DIAG] QNN VAE decoder completed successfully");
        log_tensor_stats("vae_dec_output_raw", vae_dec_out_pixels.data(),
                         static_cast<int>(vae_dec_out_pixels.size()));

        // Log per-channel stats
        int ch_size = output_width * output_height;
        log_tensor_stats("vae_out_ch0_R", vae_dec_out_pixels.data(), ch_size);
        log_tensor_stats("vae_out_ch1_G", vae_dec_out_pixels.data() + ch_size, ch_size);
        log_tensor_stats("vae_out_ch2_B", vae_dec_out_pixels.data() + 2 * ch_size, ch_size);
#endif
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
      SD_LOG_DEBUG("[VAE] VAE decoder: %d tiles, overlap %dx%dpx (latent: %dx%d)",
                   num_tiles, overlap_x, overlap_y, latent_overlap_x, latent_overlap_y);

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

        if (!vaeDecoderApp) {
          TN_ERR(TN_CODE_NOT_READY, TN_STAGE_SD_VAE,
                 "Global vaeDecoderApp not initialized (tile path)");
          throw std::runtime_error("Global vaeDecoderApp not init!");
        }

        if (StatusCode::SUCCESS !=
            vaeDecoderApp->executeVaeDecoderGraphs(tile_latent_vec.data(),
                                                   tile_output.data())) {
          TN_ERR(TN_CODE_DECODE_FAIL, TN_STAGE_SD_VAE,
                 "QNN VAE decode exec failed for tile %zu", i);
          throw std::runtime_error("QNN VAE dec exec failed for tile");
        }

        decoded_tiles.push_back(std::move(tile_output));

        SD_LOG_DEBUG("[VAE] Processed VAE decoder tile %zu/%zu", i + 1, latent_positions.size());
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
#ifdef SD_ENABLE_DIAGNOSTICS
    // Log pixel stats before post-processing
    log_tensor_stats("pixels_pre_postprocess", pixels.data(),
                     static_cast<int>(pixels.size()));
#endif

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

#ifdef SD_ENABLE_DIAGNOSTICS
    SD_LOG_INFO("[DIAG] pixels_eval shape: ndim=%zu size=%zu, output=%dx%d",
                pixels_eval.dimension(), pixels_eval.size(), output_width, output_height);
    log_tensor_stats("pixels_eval_final", pixels_eval.data(),
                     static_cast<int>(pixels_eval.size()));
#endif

    std::vector<uint8_t> out_data = nchw_to_rgb_bytes(
        pixels_eval.data(), 3, output_height, output_width);

#ifdef SD_ENABLE_DIAGNOSTICS
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
#endif

    // --- Safety Checker ---
    if (use_safety_checker) {
      auto safety_start = std::chrono::high_resolution_clock::now();
      float score = 0.0f;

      if (safety_check(out_data, output_width, output_height, score,
                       safetyCheckerInterpreter, safetyCheckerSession)) {
        SD_LOG_DEBUG("[SAFETY] NSFW Score: %.4f", score);
        if (score > nsfw_threshold) {
          QNN_WARN("NSFW detected (%.2f>%.2f).", score, nsfw_threshold);
          std::fill(out_data.begin(), out_data.end(), 255);
        }
      } else {
        QNN_WARN("Safety check failed.");
      }

      auto safety_end = std::chrono::high_resolution_clock::now();
      SD_LOG_DEBUG("[SAFETY] Safety check %ldms",
                   (long)std::chrono::duration_cast<std::chrono::milliseconds>(
                       safety_end - safety_start).count());
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
    // Re-raise so the JNI layer maps to on_error / SDCallback.onError; the
    // structured error was already emitted by whatever stage threw (CLIP /
    // UNET / VAE / scheduler). This is just the per-generation rollup line.
    TN_ERR(TN_CODE_DECODE_FAIL, TN_STAGE_UNSPECIFIED,
           "Image generation error: %s", e.what());
    QNN_ERROR("Image generation error: %s", e.what());
    throw;
  }
}

#include "../lora/lora_engine.h"

// =============================================================================
// sd_pipeline namespace - JNI-callable functions replacing HTTP server
// =============================================================================

namespace sd_pipeline {

// initialize_models() moved to loader/model_loader.cpp (Phase 1.2)

SDGenerationResult run_generation(PipelineContext& ctx,
                                  const SDGenerateParams& params,
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

  // Populate PipelineContext::gen (Phase 1.12 — migration toward ctx)
  ctx.gen.prompt = params.prompt;
  ctx.gen.negative_prompt = params.negativePrompt;
  ctx.gen.steps = params.steps;
  ctx.gen.cfg = params.cfgScale;
  ctx.gen.seed = params.seed;
  ctx.gen.scheduler_type = params.scheduler;
  ctx.gen.use_opencl = params.useOpenCL;
  ctx.gen.show_process = params.showDiffusionProcess;
  ctx.gen.show_stride = params.showDiffusionStride;
  ctx.gen.denoise_strength = params.denoiseStrength;
  ctx.gen.width = params.width;
  ctx.gen.height = params.height;
  ctx.gen.sample_w = params.width / 8;
  ctx.gen.sample_h = params.height / 8;
  ctx.gen.img2img = params.isImg2Img;
  ctx.gen.has_mask = params.hasMask;

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
  ctx.gen.seed = seed;  // Update ctx with actual seed

  QNN_INFO("Generating: prompt='%s', steps=%d, size=%dx%d, seed=%u",
           prompt.c_str(), steps, output_width, output_height, seed);

  // Bridge callback: raw bytes directly (no base64 roundtrip) + stop flag check
  auto progress_bridge = [&progressCb, &stopFlag](int step, int total_steps,
                                                    const std::vector<uint8_t>& image_data) {
    if (stopFlag.load()) {
      TN_CANCEL("user requested stop during diffusion loop");
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

  GenerationResult result = generateImage(ctx, progress_bridge);

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

// ============================================================================
// LoRA — runtime weight patching (MNN-only)
// ============================================================================
static LoRAEngine g_lora_engine;

bool apply_lora(const std::string& path, float weight) {
  if (!use_mnn) {
    SD_LOG_ERROR("[LORA] LoRA requires MNN/CPU mode — QNN uses pre-compiled contexts");
    return false;
  }

  if (!g_lora_engine.apply(path, weight, modelDir, use_clip_v2)) {
    return false;
  }

  // Recreate sessions to pick up modified weights
  recreateClipSession();
  recreateUNetSession();
  return true;
}

void clear_lora() {
  if (!g_lora_engine.has_active()) return;

  g_lora_engine.clear(modelDir, use_clip_v2);

  // Recreate sessions with restored original weights
  recreateClipSession();
  recreateUNetSession();
}

// cleanup() moved to loader/model_loader.cpp (Phase 1.2)

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

// Perf 7: cleanup persistent MNN sessions
void cleanup_persistent_sessions() {
    g_unet_runner.cleanup();
    g_vae_dec_sample_w = 0;
    g_vae_dec_sample_h = 0;
    g_vae_enc_out_w = 0;
    g_vae_enc_out_h = 0;
    g_lora_engine.reset();
}

} // namespace sd_pipeline

// ============================================================================
// PipelineContext::release() — needs full MNN types, so lives here
// ============================================================================

void PipelineContext::release() {
    // MNN: release sessions before interpreters (session borrows interpreter)
    auto& m = models;
    if (m.mnn_clip_session && m.mnn_clip) {
        m.mnn_clip->releaseSession(m.mnn_clip_session);
        m.mnn_clip_session = nullptr;
    }
    if (m.mnn_unet_session && m.mnn_unet) {
        m.mnn_unet->releaseSession(m.mnn_unet_session);
        m.mnn_unet_session = nullptr;
    }
    if (m.mnn_vae_dec_session && m.mnn_vae_dec) {
        m.mnn_vae_dec->releaseSession(m.mnn_vae_dec_session);
        m.mnn_vae_dec_session = nullptr;
    }
    if (m.mnn_vae_enc_session && m.mnn_vae_enc) {
        m.mnn_vae_enc->releaseSession(m.mnn_vae_enc_session);
        m.mnn_vae_enc_session = nullptr;
    }
    if (m.mnn_safety_session && m.mnn_safety) {
        m.mnn_safety->releaseSession(m.mnn_safety_session);
        m.mnn_safety_session = nullptr;
    }

    // MNN: destroy interpreters (owns loaded model data)
    delete m.mnn_clip;     m.mnn_clip = nullptr;
    delete m.mnn_unet;     m.mnn_unet = nullptr;
    delete m.mnn_vae_dec;  m.mnn_vae_dec = nullptr;
    delete m.mnn_vae_enc;  m.mnn_vae_enc = nullptr;
    delete m.mnn_safety;   m.mnn_safety = nullptr;

    // QNN + tokenizer + prompt processor: nullify (globals own during migration)
    m.clip = nullptr;
    m.unet = nullptr;
    m.vae_decoder = nullptr;
    m.vae_encoder = nullptr;
    m.upscaler = nullptr;
    m.tokenizer = nullptr;
    m.prompt_processor = nullptr;

    // Embeddings + patched buffer
    m.pos_emb.clear();
    m.token_emb.clear();
    m.unet_patched.reset();

    // Free generation buffers
    buffers.clear();
    gen.clear();
}