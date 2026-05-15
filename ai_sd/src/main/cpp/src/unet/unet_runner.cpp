/**
 * UNet Runner implementation — QNN and MNN single-step inference.
 *
 * Extracted from diffusion_pipeline.cpp (Phase 1.5).
 * Reads globals (unetApp) via pipeline_globals.h during migration.
 */

#define TN_MODULE TN_MODULE_AI_SD
#define TN_TAG    "ai_sd"
#include <tn_security/tn_security_macros.h>

#include "unet_runner.h"
#include "../pipeline/pipeline_globals.h"
#include "../model/qnn_model.h"
#include "../utils/sd_logger.h"

#include <MNN/MNNDefine.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <stdexcept>
#include <string>
#include <cstring>
#include <cmath>

UNetRunner::~UNetRunner() {
    cleanup();
}

bool UNetRunner::initIfNeeded(bool use_mnn, bool use_opencl,
                              const std::string& unet_path, const std::string& model_dir,
                              int batch_size, int sample_h, int sample_w, int text_emb_size) {
    // Perf 7: Reuse existing session if all params match
    if (initialized_ && use_mnn_ == use_mnn && use_opencl_ == use_opencl &&
        unet_path_ == unet_path && batch_size_ == batch_size &&
        sample_h_ == sample_h && sample_w_ == sample_w &&
        text_emb_size_ == text_emb_size) {
        SD_LOG_DEBUG("[UNET] Reusing existing MNN session (same params)");
        return true;
    }
    init(use_mnn, use_opencl, unet_path, model_dir, batch_size, sample_h, sample_w, text_emb_size);
    return false;
}

void UNetRunner::init(bool use_mnn, bool use_opencl,
                      const std::string& unet_path, const std::string& model_dir,
                      int batch_size, int sample_h, int sample_w, int text_emb_size) {
    cleanup();  // Release any previous state

    use_mnn_ = use_mnn;
    use_opencl_ = use_opencl;
    unet_path_ = unet_path;
    batch_size_ = batch_size;
    sample_h_ = sample_h;
    sample_w_ = sample_w;
    text_emb_size_ = text_emb_size;
    single_latent_size_ = 1 * 4 * sample_w * sample_h;

    if (use_mnn_) {
        mnn_interpreter_ = MNN::Interpreter::createFromFile(unet_path.c_str());
        if (!mnn_interpreter_) {
            TN_ERR(TN_CODE_MNN_INIT_FAIL, TN_STAGE_LOAD,
                   "Failed to create MNN UNET interpreter from: %s", unet_path.c_str());
            throw std::runtime_error("Failed to create MNN UNET interpreter from: " + unet_path);
        }

        MNN::ScheduleConfig cfg_unet;
        MNN::BackendConfig bkCfg_unet;
        if (use_opencl) {
            auto cache_file =
                model_dir + "/unet_cache.mnnc." + std::to_string(sample_w * 8);
            mnn_interpreter_->setCacheFile(cache_file.c_str());
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

        mnn_session_ = mnn_interpreter_->createSession(cfg_unet);
        if (!mnn_session_) {
            TN_ERR(TN_CODE_MNN_INIT_FAIL, TN_STAGE_LOAD,
                   "Failed to create MNN UNET session (use_opencl=%d)",
                   (int)use_opencl);
            throw std::runtime_error("Failed to create MNN UNET session");
        }

        auto samp = mnn_interpreter_->getSessionInput(mnn_session_, "sample");
        auto ts = mnn_interpreter_->getSessionInput(mnn_session_, "timestep");
        auto enc = mnn_interpreter_->getSessionInput(mnn_session_, "encoder_hidden_states");

        mnn_interpreter_->resizeTensor(samp, {batch_size, 4, sample_h, sample_w});
        mnn_interpreter_->resizeTensor(ts, {1});
        mnn_interpreter_->resizeTensor(enc, {batch_size, 77, text_emb_size});
        mnn_interpreter_->resizeSession(mnn_session_);
        if (use_opencl) {
            mnn_interpreter_->updateCacheFile(mnn_session_);
        }

        mnn_interpreter_->releaseModel();
    } else {
        if (!unetApp) {
            TN_ERR(TN_CODE_NOT_READY, TN_STAGE_INIT,
                   "Global unetApp not initialized — QNN UNET load skipped or failed");
            throw std::runtime_error("Global unetApp not initialized");
        }
    }

    initialized_ = true;
    SD_LOG_DEBUG("[UNET] Runner initialized: mnn=%d opencl=%d batch=%d latent=%dx%d emb=%d",
                 use_mnn_, (int)use_opencl, batch_size_, sample_w_, sample_h_, text_emb_size_);
}

void UNetRunner::step(const float* latents_in, int timestep,
                      const float* text_embeddings, float* unet_out,
                      float cfg_scale) {
    if (!initialized_) {
        TN_ERR(TN_CODE_NOT_READY, TN_STAGE_SD_UNET,
               "UNetRunner::step() called before init");
        throw std::runtime_error("UNetRunner not initialized");
    }

    int total_latent_size = batch_size_ * single_latent_size_;
    int total_embed_size = batch_size_ * 77 * text_emb_size_;
    // cfg=1 -> the CPU-side CFG combiner reduces to just `tx`; the uncond pass is
    // a wasted NPU/CPU inference. Skip it on QNN where the two passes are
    // sequential `executeUnetGraphs` calls.
    const bool skip_uncond = std::fabs(cfg_scale - 1.0f) < 1e-4f;

    if (use_mnn_) {
        auto samp = mnn_interpreter_->getSessionInput(mnn_session_, "sample");
        auto ts = mnn_interpreter_->getSessionInput(mnn_session_, "timestep");
        auto enc = mnn_interpreter_->getSessionInput(mnn_session_, "encoder_hidden_states");

        // Perf 6: Create host tensors from source data directly (no double memcpy)
        std::unique_ptr<MNN::Tensor> samp_host(MNN::Tensor::create(
            samp->shape(), samp->getType(),
            const_cast<float*>(latents_in), MNN::Tensor::CAFFE));
        int ts_int = timestep;
        std::unique_ptr<MNN::Tensor> ts_host(MNN::Tensor::create(
            ts->shape(), ts->getType(), &ts_int, MNN::Tensor::CAFFE));
        std::unique_ptr<MNN::Tensor> enc_host(MNN::Tensor::create(
            enc->shape(), enc->getType(),
            const_cast<float*>(text_embeddings), MNN::Tensor::CAFFE));

        samp->copyFromHostTensor(samp_host.get());
        ts->copyFromHostTensor(ts_host.get());
        enc->copyFromHostTensor(enc_host.get());

        mnn_interpreter_->runSession(mnn_session_);

        auto output = mnn_interpreter_->getSessionOutput(mnn_session_, "out_sample");
        output->copyToHostTensor(samp_host.get());
        memcpy(unet_out, samp_host->host<float>(),
               total_latent_size * sizeof(float));
    } else {
        // QNN path: two separate calls (uncond + cond), or one if cfg==1.
        float* latents_out_ptr = unet_out;
        const float* embed_ptr = text_embeddings;

        // Always run cond. We write it to the cond slot first so the uncond
        // mirror is correct in the cfg==1 fast path.
        if (StatusCode::SUCCESS !=
            unetApp->executeUnetGraphs(
                const_cast<float*>(latents_in) + single_latent_size_,
                timestep,
                const_cast<float*>(embed_ptr) + 77 * text_emb_size_,
                latents_out_ptr + single_latent_size_)) {
            TN_ERR(TN_CODE_DECODE_FAIL, TN_STAGE_SD_UNET,
                   "QNN UNET exec failed (cond) at timestep=%d", timestep);
            throw std::runtime_error("QNN UNET exec failed (cond)");
        }

        if (skip_uncond) {
            // Mirror cond into the uncond slot so the CPU-side CFG combiner
            // produces tx as expected: uc + 1.0 * (tx - uc) == tx for any uc.
            std::memcpy(latents_out_ptr, latents_out_ptr + single_latent_size_,
                        single_latent_size_ * sizeof(float));
        } else {
            if (StatusCode::SUCCESS !=
                unetApp->executeUnetGraphs(
                    const_cast<float*>(latents_in),
                    timestep, const_cast<float*>(embed_ptr),
                    latents_out_ptr)) {
                TN_ERR(TN_CODE_DECODE_FAIL, TN_STAGE_SD_UNET,
                       "QNN UNET exec failed (uncond) at timestep=%d", timestep);
                throw std::runtime_error("QNN UNET exec failed (uncond)");
            }
        }
    }
}

void UNetRunner::cleanup() {
    if (mnn_session_ && mnn_interpreter_) {
        mnn_interpreter_->releaseSession(mnn_session_);
        mnn_session_ = nullptr;
    }
    if (mnn_interpreter_) {
        delete mnn_interpreter_;
        mnn_interpreter_ = nullptr;
    }
    initialized_ = false;
}
