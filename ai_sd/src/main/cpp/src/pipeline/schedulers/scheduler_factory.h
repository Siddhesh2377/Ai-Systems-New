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
        // Default to DPM solver
        scheduler = std::make_unique<DPMSolverMultistepScheduler>(
            1000, 0.00085f, 0.012f, "scaled_linear", 2, "epsilon", "leading");
    }
    if (pony) scheduler->set_prediction_type("v_prediction");
    return scheduler;
}
