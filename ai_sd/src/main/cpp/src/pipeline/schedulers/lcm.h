// LCMScheduler — Latent Consistency Model scheduler.
// Port of diffusers `LCMScheduler` (consistency model distillation),
// designed for 1-8 inference steps. Distinct from Euler/DPM in that there is
// no ancestral noise added at the final step and intermediate noise uses the
// alpha_prod_t_prev term, not a sigma-derived one.
//
// References:
//   - https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lcm.py
//   - "Latent Consistency Models", Luo et al., 2023.
#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

#include "scheduler.h"

class LCMScheduler : public Scheduler {
 public:
  LCMScheduler(int num_train_timesteps,
               float beta_start,
               float beta_end,
               const std::string& beta_schedule,
               const std::string& prediction_type,
               int original_inference_steps = 50,
               float timestep_scaling = 10.0f,
               float sigma_data = 0.5f)
      : num_train_timesteps_(num_train_timesteps),
        beta_start_(beta_start),
        beta_end_(beta_end),
        beta_schedule_(beta_schedule),
        prediction_type_(prediction_type),
        original_inference_steps_(original_inference_steps),
        timestep_scaling_(timestep_scaling),
        sigma_data_(sigma_data) {
    // Build betas/alphas the same way as Euler-A.
    xt::xarray<double> betas_d;
    if (beta_schedule == "linear") {
      betas_d = xt::linspace<double>(beta_start_, beta_end_, num_train_timesteps);
    } else if (beta_schedule == "scaled_linear") {
      double bs = std::sqrt(static_cast<double>(beta_start_));
      double be = std::sqrt(static_cast<double>(beta_end_));
      betas_d = xt::pow(xt::linspace<double>(bs, be, num_train_timesteps), 2.0);
    } else {
      throw std::runtime_error(beta_schedule + " not supported by LCMScheduler");
    }
    xt::xarray<double> alphas_d = 1.0 - betas_d;
    xt::xarray<double> alphas_cumprod_d = xt::cumprod(alphas_d);
    alphas_cumprod_ = xt::cast<float>(alphas_cumprod_d);

    // Default timesteps array (training-time, descending). Replaced by set_timesteps.
    auto ts_vec = std::vector<float>(num_train_timesteps);
    for (int i = 0; i < num_train_timesteps; ++i) {
      ts_vec[i] = static_cast<float>(num_train_timesteps - 1 - i);
    }
    timesteps_ = xt::adapt(ts_vec);

    num_inference_steps_ = std::nullopt;
    step_index_ = std::nullopt;
    begin_index_ = std::nullopt;
  }

  // LCM picks a strided subset of an "original_inference_steps" schedule. For
  // the typical case (original=50, requested=4), this gives timesteps roughly
  // [999, 749, 499, 249] which empirically work well for distilled LCM models.
  void set_timesteps(int num_inference_steps) override {
    if (num_inference_steps > original_inference_steps_) {
      // LCM only supports up to its training-time inference grid.
      num_inference_steps = original_inference_steps_;
    }
    num_inference_steps_ = num_inference_steps;

    int k = num_train_timesteps_ / original_inference_steps_;
    // LCM original timesteps: every k-th from num_train_timesteps_-1 down to 0.
    std::vector<int> lcm_origin_timesteps(original_inference_steps_);
    for (int i = 0; i < original_inference_steps_; ++i) {
      lcm_origin_timesteps[i] = num_train_timesteps_ - 1 - i * k;
    }

    // Stride through that grid to pick num_inference_steps timesteps.
    int skipping_step = original_inference_steps_ / num_inference_steps;
    std::vector<float> ts_vec;
    ts_vec.reserve(num_inference_steps);
    for (int i = 0; i < num_inference_steps; ++i) {
      int idx = i * skipping_step;
      if (idx >= original_inference_steps_) idx = original_inference_steps_ - 1;
      ts_vec.push_back(static_cast<float>(lcm_origin_timesteps[idx]));
    }
    timesteps_ = xt::adapt(ts_vec);

    step_index_ = std::nullopt;
    begin_index_ = std::nullopt;
  }

  // LCM does not require pre-scaling the model input.
  xt::xarray<float> scale_model_input(const xt::xarray<float>& sample,
                                      int /*timestep*/) override {
    return sample;
  }

  SchedulerOutput step(const xt::xarray<float>& model_output, int timestep,
                       const xt::xarray<float>& sample) override {
    if (!num_inference_steps_.has_value()) {
      throw std::runtime_error("LCM: set_timesteps must be called first");
    }
    if (!step_index_.has_value()) {
      init_step_index(timestep);
    }
    int idx = step_index_.value();

    // alpha_prod_t and alpha_prod_t_prev (next timestep in the schedule).
    int t = static_cast<int>(timesteps_(idx));
    int t_prev_idx = idx + 1;
    int t_prev = (t_prev_idx < static_cast<int>(timesteps_.size()))
                     ? static_cast<int>(timesteps_(t_prev_idx))
                     : -1;

    float alpha_prod_t = alphas_cumprod_(t);
    float alpha_prod_t_prev =
        (t_prev >= 0) ? alphas_cumprod_(t_prev) : 1.0f;

    float beta_prod_t = 1.0f - alpha_prod_t;
    float beta_prod_t_prev = 1.0f - alpha_prod_t_prev;

    // Boundary conditions for the consistency function.
    float scaled_t = static_cast<float>(t) * timestep_scaling_;
    float c_skip = (sigma_data_ * sigma_data_) /
                   (scaled_t * scaled_t + sigma_data_ * sigma_data_);
    float c_out = scaled_t /
                  std::sqrt(scaled_t * scaled_t + sigma_data_ * sigma_data_);

    // Predicted x_0.
    xt::xarray<float> pred_original_sample;
    if (prediction_type_ == "epsilon") {
      pred_original_sample =
          (sample - std::sqrt(beta_prod_t) * model_output) /
          std::sqrt(alpha_prod_t);
    } else if (prediction_type_ == "v_prediction") {
      pred_original_sample = std::sqrt(alpha_prod_t) * sample -
                             std::sqrt(beta_prod_t) * model_output;
    } else if (prediction_type_ == "sample") {
      pred_original_sample = model_output;
    } else {
      throw std::runtime_error(prediction_type_ +
                               " not implemented for LCMScheduler");
    }

    // Consistency function: blend pred_x0 with sample at the boundary.
    xt::xarray<float> denoised = c_out * pred_original_sample + c_skip * sample;

    xt::xarray<float> prev_sample;
    bool is_last = (idx + 1 >= static_cast<int>(timesteps_.size()));
    if (!is_last) {
      // Re-noise to alpha_prod_t_prev with fresh Gaussian.
      xt::xarray<float> noise = xt::random::randn<float>(
          model_output.shape(), 0.0f, 1.0f,
          xt::random::get_default_random_engine());
      prev_sample = std::sqrt(alpha_prod_t_prev) * denoised +
                    std::sqrt(beta_prod_t_prev) * noise;
    } else {
      prev_sample = denoised;
    }

    step_index_ = idx + 1;
    return {prev_sample, pred_original_sample};
  }

  xt::xarray<float> add_noise(const xt::xarray<float>& original_samples,
                              const xt::xarray<float>& noise,
                              const xt::xarray<int>& timesteps) const override {
    // Standard q(x_t | x_0) = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise.
    if (timesteps.size() == 0) return original_samples;
    int t = timesteps(0);
    if (t < 0) t = 0;
    if (t >= num_train_timesteps_) t = num_train_timesteps_ - 1;
    float ap = alphas_cumprod_(t);
    return std::sqrt(ap) * original_samples + std::sqrt(1.0f - ap) * noise;
  }

  void set_begin_index(int begin_index) override { begin_index_ = begin_index; }

  void set_prediction_type(const std::string& prediction_type) override {
    prediction_type_ = prediction_type;
  }

  const xt::xarray<float>& get_timesteps() const override { return timesteps_; }
  size_t get_step_index() const override { return step_index_.value_or(0); }

  // LCM doesn't expose sigmas the way Euler does; return a derived value so the
  // pipeline orchestrator's logging still works.
  float get_current_sigma() const override {
    int t = step_index_.has_value()
                ? static_cast<int>(timesteps_(step_index_.value()))
                : static_cast<int>(timesteps_(0));
    if (t < 0 || t >= num_train_timesteps_) return 0.0f;
    float ap = alphas_cumprod_(t);
    return std::sqrt((1.0f - ap) / ap);
  }

  // For LCM, latents are scaled by sqrt(alphas_cumprod[t_max]) implicitly via
  // the consistency function, so init_noise_sigma is just 1.0.
  float get_init_noise_sigma() const override { return 1.0f; }

 private:
  int num_train_timesteps_;
  float beta_start_;
  float beta_end_;
  std::string beta_schedule_;
  std::string prediction_type_;
  int original_inference_steps_;
  float timestep_scaling_;
  float sigma_data_;

  xt::xarray<float> alphas_cumprod_;
  xt::xarray<float> timesteps_;

  std::optional<int> num_inference_steps_;
  std::optional<int> step_index_;
  std::optional<int> begin_index_;

  void init_step_index(int timestep) {
    if (begin_index_.has_value()) {
      step_index_ = begin_index_.value();
      return;
    }
    for (size_t i = 0; i < timesteps_.size(); ++i) {
      if (static_cast<int>(timesteps_(i)) == timestep) {
        step_index_ = static_cast<int>(i);
        return;
      }
    }
    step_index_ = 0;
  }
};
