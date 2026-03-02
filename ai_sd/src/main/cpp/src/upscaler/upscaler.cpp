/**
 * Upscaler implementation — 4x super-resolution with tiled inference.
 *
 * Extracted from diffusion_pipeline.cpp (Phase 1.7).
 * Uses calculate_tile_positions() from vae_codec for tile layout.
 */

#include "upscaler.h"
#include "../vae/vae_codec.h"
#include "../model/qnn_model.h"
#include "../utils/sd_logger.h"

#include <MNN/MNNDefine.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor/xadapt.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>

xt::xarray<uint8_t> upscaleImageWithModel(
    const std::vector<uint8_t>& input_image, int width, int height,
    std::unique_ptr<QnnModel>& upscaler) {
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

  SD_LOG_INFO("Upscaling %dx%d to %dx%d using %dx%d tiles (variable overlap)",
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

  // Perf 11: Pre-allocate tile buffers once, reuse across tiles
  int tile_input_elems = 1 * 3 * tile_size * tile_size;
  int tile_output_elems = 1 * 3 * output_tile_size * output_tile_size;
  std::vector<float> tile_input_vec(tile_input_elems);
  std::vector<float> tile_output_vec(tile_output_elems);
  std::vector<int> tile_output_shape = {1, 3, output_tile_size,
                                        output_tile_size};

  int tile_count = 0;
  for (int y : y_coords) {
    for (int x : x_coords) {
      xt::xarray<float> input_tile =
          xt::view(input_chw, 0, xt::all(), xt::range(y, y + tile_size),
                   xt::range(x, x + tile_size));

      std::copy(input_tile.begin(), input_tile.end(), tile_input_vec.begin());

      if (StatusCode::SUCCESS !=
          upscaler->executeUpscalerGraphs(tile_input_vec.data(),
                                          tile_output_vec.data())) {
        throw std::runtime_error("Upscaler execution failed for tile");
      }

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
      SD_LOG_DEBUG("[TILE] Processed tile %d/%d", tile_count,
                   num_tiles_w * num_tiles_h);
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

xt::xarray<uint8_t> upscaleImageWithMNN(
    const std::vector<uint8_t>& input_image, int width, int height,
    const std::string& model_path, bool use_opencl) {
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

  SD_LOG_INFO("Upscaling %dx%d to %dx%d using MNN (%s), %dx%d tiles", width,
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

  // Perf 11: Pre-allocate tile tensors and shapes outside loop
  std::vector<int> dims = {1, 3, tile_size, tile_size};
  std::vector<int> mnn_tile_output_shape = {1, 3, output_tile_size,
                                            output_tile_size};
  // Resize once (all tiles are same size)
  interpreter->resizeTensor(input_tensor, dims);
  interpreter->resizeSession(session);
  // Pre-allocate output host tensor
  std::unique_ptr<MNN::Tensor> output_host(
      MNN::Tensor::create<float>(mnn_tile_output_shape,
                                 nullptr, MNN::Tensor::CAFFE));

  int tile_count = 0;
  for (int y : y_coords) {
    for (int x : x_coords) {
      xt::xarray<float> input_tile =
          xt::view(input_chw, 0, xt::all(), xt::range(y, y + tile_size),
                   xt::range(x, x + tile_size));

      // Create host tensor from source data directly (Perf 6)
      std::unique_ptr<MNN::Tensor> host_tensor(MNN::Tensor::create<float>(
          dims, const_cast<float*>(input_tile.data()), MNN::Tensor::CAFFE));
      input_tensor->copyFromHostTensor(host_tensor.get());

      // Run inference
      if (interpreter->runSession(session) != 0) {
        throw std::runtime_error("MNN inference failed for tile");
      }

      // Get output into pre-allocated tensor
      output_tensor->copyToHostTensor(output_host.get());

      xt::xarray<float> output_tile = xt::adapt(
          output_host->host<float>(), output_tile_size * output_tile_size * 3,
          xt::no_ownership(), mnn_tile_output_shape);
      // Note: output_tile uses no_ownership so output_host must outlive it

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
      SD_LOG_DEBUG("[TILE] Processed tile %d/%d", tile_count,
                   num_tiles_w * num_tiles_h);
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
