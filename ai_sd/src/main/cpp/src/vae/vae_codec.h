#pragma once

/**
 * VAE codec — tiling utilities for VAE encode/decode and tile blending.
 *
 * Extracted from diffusion_pipeline.cpp (Phase 1.6).
 * Pure utility functions with no global state dependencies.
 */

#include <tuple>
#include <utility>
#include <vector>

#include <xtensor/xarray.hpp>

/// Calculate evenly-spaced tile positions along a dimension.
std::vector<int> calculate_tile_positions(int dimension, int tile_size,
                                          int min_overlap);

/// Calculate tile positions for VAE encoder/decoder (pixel + latent space).
/// Returns: {pixel_positions, latent_positions, pixel_overlap_x,
///           pixel_overlap_y, latent_overlap_x, latent_overlap_y}
std::tuple<std::vector<std::pair<int, int>>, std::vector<std::pair<int, int>>,
           int, int, int, int>
calculate_vae_tile_positions(int pixel_width, int pixel_height);

/// Blend encoded VAE tiles (mean + std pairs) with overlap weighting.
xt::xarray<float> blend_vae_encoder_tiles(
    const std::vector<std::pair<xt::xarray<float>, xt::xarray<float>>>& tiles_mean_std,
    const std::vector<std::pair<int, int>>& positions,
    int latent_h, int latent_w, int tile_size,
    int overlap_x, int overlap_y);

/// Blend decoded VAE output tiles with overlap weighting.
xt::xarray<float> blend_vae_output_tiles(
    const std::vector<xt::xarray<float>>& tiles,
    const std::vector<std::pair<int, int>>& positions,
    int output_h, int output_w, int tile_size,
    int overlap_x, int overlap_y);
