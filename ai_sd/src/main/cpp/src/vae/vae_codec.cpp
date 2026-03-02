/**
 * VAE codec implementation — tiling and blending utilities.
 *
 * Extracted from diffusion_pipeline.cpp (Phase 1.6).
 * No global state dependencies — pure math on xtensor arrays.
 */

#include "vae_codec.h"

#include <stdexcept>

#include <xtensor/xbuilder.hpp>
#include <xtensor/xeval.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

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

std::tuple<std::vector<std::pair<int, int>>, std::vector<std::pair<int, int>>,
           int, int, int, int>
calculate_vae_tile_positions(int pixel_width, int pixel_height) {
    const int vae_tile_size = 512;
    const int vae_latent_tile_size = 64;
    const int min_latent_overlap = 16;
    const int scale_factor = 8;

    auto pixel_x_coords = calculate_tile_positions(
        pixel_width, vae_tile_size, min_latent_overlap * scale_factor);
    auto pixel_y_coords = calculate_tile_positions(
        pixel_height, vae_tile_size, min_latent_overlap * scale_factor);

    std::vector<int> latent_x_coords;
    std::vector<int> latent_y_coords;
    for (int px : pixel_x_coords) {
        latent_x_coords.push_back(px / scale_factor);
    }
    for (int py : pixel_y_coords) {
        latent_y_coords.push_back(py / scale_factor);
    }

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

xt::xarray<float> blend_vae_encoder_tiles(
    const std::vector<std::pair<xt::xarray<float>, xt::xarray<float>>>& tiles_mean_std,
    const std::vector<std::pair<int, int>>& positions,
    int latent_h, int latent_w, int tile_size,
    int overlap_x, int overlap_y) {
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

        const auto& mean_tile = tiles_mean_std[idx].first;
        const auto& std_tile = tiles_mean_std[idx].second;

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
    // xt::eval forces contiguous evaluation of the expression
    xt::xarray<float> latent = final_mean + final_std * noise;

    return latent;
}

xt::xarray<float> blend_vae_output_tiles(
    const std::vector<xt::xarray<float>>& tiles,
    const std::vector<std::pair<int, int>>& positions,
    int output_h, int output_w, int tile_size,
    int overlap_x, int overlap_y) {
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
