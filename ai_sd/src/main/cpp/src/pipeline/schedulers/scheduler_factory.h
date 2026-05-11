#pragma once

/**
 * Scheduler factory — creates the appropriate scheduler from a type string.
 *
 * Extracted from diffusion_pipeline.cpp (Phase 1.4).
 * Header-only: no .cpp needed.
 */

#include <memory>
#include <string>

#include "scheduler.h"
#include "dpm_solver.h"
#include "euler_ancestral.h"
#include "lcm.h"

/// Create a scheduler by type name. Sets v_prediction if pony model.
inline std::unique_ptr<Scheduler> createScheduler(const std::string& type,
                                                   bool pony) {
    std::unique_ptr<Scheduler> scheduler;
    if (type == "euler_a" || type == "eulera") {
        scheduler = std::make_unique<EulerAncestralDiscreteScheduler>(
            1000, 0.00085f, 0.012f, "scaled_linear", "epsilon", "leading");
    } else if (type == "lcm") {
        // LCM converges in 1-8 steps; pair with cfg_scale ~= 1.0 for best results.
        scheduler = std::make_unique<LCMScheduler>(
            1000, 0.00085f, 0.012f, "scaled_linear", "epsilon");
    } else {
        // Default to DPM solver with Karras sigmas. Karras alone is the
        // win — at the same step count it places more samples at the
        // important "middle" noise levels and DPM++ 2M converges to a
        // similar-quality image in ~10 steps that the linear schedule
        // needed 20-28 for. We keep the legacy "zero" final-sigma
        // trailing and disable lambda_min_clipped here: those two were
        // SDXL-tuned defaults from HF diffusers and are not appropriate
        // for SD 1.5 with epsilon prediction (they shift sigma_min and
        // produced visibly noisy output the first time we shipped them).
        auto dpm = std::make_unique<DPMSolverMultistepScheduler>(
            1000, 0.00085f, 0.012f, "scaled_linear", 2, "epsilon", "leading");
        dpm->set_use_karras_sigmas(true, /*rho=*/7.0f);
        scheduler = std::move(dpm);
    }
    if (pony) scheduler->set_prediction_type("v_prediction");
    return scheduler;
}
