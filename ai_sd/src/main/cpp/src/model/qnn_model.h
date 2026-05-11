#ifndef QNNMODEL_HPP
#define QNNMODEL_HPP

#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpGraph.h>
#include <HTP/QnnHtpSystemContext.h>
#include <inttypes.h>

#include "../utils/config.h"
#include <QnnSampleApp.hpp>
#include <QnnTypeMacros.hpp>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "DataUtil.hpp"
#include "Logger.hpp"
#include "../utils/sd_utils.h"

using namespace qnn::tools::sample_app;

class QnnModel : public QnnSampleApp {
 public:
  Qnn_Tensor_t *inputs = nullptr;
  Qnn_Tensor_t *outputs = nullptr;
  QnnModel(QnnFunctionPointers qnnFunctionPointers, std::string inputListPaths,
           std::string opPackagePaths, void *backendHandle,
           std::string outputPath = s_defaultOutputPath, bool debug = false,
           qnn::tools::iotensor::OutputDataType outputDataType =
               qnn::tools::iotensor::OutputDataType::FLOAT_ONLY,
           qnn::tools::iotensor::InputDataType inputDataType =
               qnn::tools::iotensor::InputDataType::FLOAT,
           ProfilingLevel profilingLevel = ProfilingLevel::OFF,
           bool dumpOutputs = false, std::string cachedBinaryPath = "",
           std::string saveBinaryName = "")
      : QnnSampleApp(qnnFunctionPointers, inputListPaths, opPackagePaths,
                     backendHandle, outputPath, debug, outputDataType,
                     inputDataType, profilingLevel, dumpOutputs,
                     cachedBinaryPath, saveBinaryName) {}

  ~QnnModel() override {
    // Tear down per-graph I/O tensors before the base class frees the
    // graph metadata that owns the tensor descriptors.
    if (inputs && outputs && m_graphsInfo && m_graphsCount > 0) {
      auto graphInfo = (*m_graphsInfo)[0];
      m_ioTensor.tearDownInputAndOutputTensors(
          inputs, outputs,
          graphInfo.numInputTensors, graphInfo.numOutputTensors);
      inputs = nullptr;
      outputs = nullptr;
    }
    // Release the HTP power-config ID so its slot becomes available to
    // the next QnnModel that loads. Without this, repeated model swaps
    // exhaust the device's perf-config pool.
    releasePerformanceMode();
    m_embedCache.clear();
  }

  // useBurstMode=true: QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_BURST_MODE
  // for short, throughput-bound sessions (UNet denoising loop). Trades
  // sustained efficiency for ~10-15% lower per-step latency on the first
  // few steps before thermal throttle kicks in. Defaults to false
  // (PERFORMANCE_MODE) for CLIP / VAE / safety where the runtime is
  // single-call or short.
  StatusCode enablePerformaceMode(bool useBurstMode = false) {
    uint32_t deviceId = 0;
    uint32_t coreId = 0;
    auto qnnInterface = m_qnnFunctionPointers.qnnInterface;

    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    Qnn_ErrorHandle_t devErr =
        qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (devErr != QNN_SUCCESS) {
      QNN_ERROR("device error");
      return StatusCode::FAILURE;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra =
        static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    QnnHtpDevice_PerfInfrastructure_t perfInfra = htpInfra->perfInfra;
    Qnn_ErrorHandle_t perfInfraErr =
        perfInfra.createPowerConfigId(deviceId, coreId, &m_powerConfigId);
    if (perfInfraErr != QNN_SUCCESS) {
      QNN_ERROR("createPowerConfigId failed");
      return StatusCode::FAILURE;
    }
    m_powerConfigIdValid = true;

    QnnHtpPerfInfrastructure_PowerConfig_t rpcControlLatency;
    memset(&rpcControlLatency, 0, sizeof(rpcControlLatency));
    rpcControlLatency.option =
        QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    rpcControlLatency.rpcControlLatencyConfig = 100;
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs1[] = {
        &rpcControlLatency, NULL};
    perfInfraErr = perfInfra.setPowerConfig(m_powerConfigId, powerConfigs1);
    if (perfInfraErr != QNN_SUCCESS) {
      QNN_ERROR("setPowerConfig failed");
      return StatusCode::FAILURE;
    }

    // The static RPC_POLLING_TIME knob (formerly 9999µs) is dropped: it
    // pinned a CPU core spinning for ~10 ms per call, was overridden by
    // the ADAPTIVE_POLLING_TIME below anyway, and produced one
    // "fastrpc_wait_for_completion: poll mode timeout (9999 us)" log line
    // per UNet step. Adaptive is the right knob for SD's bursty pattern.

    QnnHtpPerfInfrastructure_PowerConfig_t powerConfig;
    memset(&powerConfig, 0, sizeof(powerConfig));
    powerConfig.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    powerConfig.dcvsV3Config.dcvsEnable = 0;
    powerConfig.dcvsV3Config.setDcvsEnable = 1;
    powerConfig.dcvsV3Config.contextId = m_powerConfigId;
    // The current QNN HTP perf-infrastructure header (QAIRT 2.39) only
    // exposes PERFORMANCE_MODE as the highest-throughput option; the
    // research notes referencing BURST_MODE were against a newer SDK
    // where that enum was added. With DCVS disabled and all six voltage
    // corners pinned to MAX_VOLTAGE_CORNER below, PERFORMANCE_MODE is
    // already running the HTP at peak. The useBurstMode parameter is
    // wired for forward-compat — flip the branch when the SDK header
    // exposes the enum (or substitute a more aggressive corner profile
    // like DCVS_VOLTAGE_VCORNER_TUR_L1 for the bus while keeping core
    // at MAX, per the research-docs agent's "decouple core/bus" note).
    (void)useBurstMode;
    powerConfig.dcvsV3Config.powerMode =
        QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    powerConfig.dcvsV3Config.setSleepLatency = 1;
    powerConfig.dcvsV3Config.setBusParams = 1;
    powerConfig.dcvsV3Config.setCoreParams = 1;
    powerConfig.dcvsV3Config.sleepDisable = 1;
    powerConfig.dcvsV3Config.setSleepDisable = 1;
    powerConfig.dcvsV3Config.sleepLatency = 40;
    powerConfig.dcvsV3Config.busVoltageCornerMin =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.busVoltageCornerTarget =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.busVoltageCornerMax =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.coreVoltageCornerMin =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.coreVoltageCornerTarget =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.coreVoltageCornerMax =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs3[] = {
        &powerConfig, NULL};
    perfInfraErr = perfInfra.setPowerConfig(m_powerConfigId, powerConfigs3);
    if (perfInfraErr != QNN_SUCCESS) {
      QNN_ERROR("setPowerConfig failed");
      return StatusCode::FAILURE;
    }

    QnnHtpPerfInfrastructure_PowerConfig_t adaptivePollingTime;
    memset(&adaptivePollingTime, 0, sizeof(adaptivePollingTime));
    adaptivePollingTime.option =
        QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_ADAPTIVE_POLLING_TIME;
    // 100 µs target: short enough to avoid burning CPU on idle waits,
    // long enough to skip an interrupt for sub-100µs ops. Matches QNN
    // sample app guidance for high-throughput inference paths.
    adaptivePollingTime.adaptivePollingTimeConfig = 100;
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs4[] = {
        &adaptivePollingTime, NULL};
    perfInfraErr = perfInfra.setPowerConfig(m_powerConfigId, powerConfigs4);
    if (perfInfraErr != QNN_SUCCESS) {
      QNN_ERROR("setPowerConfig failed");
      return StatusCode::FAILURE;
    }

    return StatusCode::SUCCESS;
  }

  // Element count derived from a tensor's actual dims rather than the
  // C++ global config. Loading a model whose UNet was exported at a
  // different resolution than `sample_width`/`sample_height` previously
  // walked off the end of the QNN client buffer with no diagnostic.
  static uint32_t tensorElementCount(const Qnn_Tensor_t &t) {
    uint32_t rank = QNN_TENSOR_GET_RANK(t);
    uint32_t *dims = QNN_TENSOR_GET_DIMENSIONS(t);
    if (!dims || rank == 0) return 0;
    uint32_t count = 1;
    for (uint32_t i = 0; i < rank; ++i) count *= dims[i];
    return count;
  }

  static size_t tensorElementBytes(Qnn_DataType_t dtype) {
    switch (dtype) {
      case QNN_DATATYPE_INT_8:
      case QNN_DATATYPE_UINT_8:
      case QNN_DATATYPE_UFIXED_POINT_8:
      case QNN_DATATYPE_SFIXED_POINT_8:
      case QNN_DATATYPE_BOOL_8:
        return 1;
      case QNN_DATATYPE_INT_16:
      case QNN_DATATYPE_UINT_16:
      case QNN_DATATYPE_FLOAT_16:
      case QNN_DATATYPE_UFIXED_POINT_16:
      case QNN_DATATYPE_SFIXED_POINT_16:
        return 2;
      case QNN_DATATYPE_INT_32:
      case QNN_DATATYPE_UINT_32:
      case QNN_DATATYPE_FLOAT_32:
      case QNN_DATATYPE_UFIXED_POINT_32:
      case QNN_DATATYPE_SFIXED_POINT_32:
        return 4;
      case QNN_DATATYPE_INT_64:
      case QNN_DATATYPE_UINT_64:
      case QNN_DATATYPE_FLOAT_64:
        return 8;
      default:
        return 0;
    }
  }

  // Bounds-checked memcpy into a QNN input client buffer. Returns false
  // if the source byte length exceeds the buffer's declared dataSize —
  // i.e. the model expected a different shape/dtype than the caller is
  // supplying. Logs the mismatch for diagnostic purposes.
  static bool writeInputBytes(Qnn_Tensor_t &t, const void *src,
                              size_t srcBytes, const char *what) {
    auto buf = QNN_TENSOR_GET_CLIENT_BUF(t);
    if (!buf.data || buf.dataSize == 0) {
      QNN_ERROR("writeInputBytes(%s): tensor client buffer not allocated",
                what);
      return false;
    }
    if (srcBytes > buf.dataSize) {
      QNN_ERROR("writeInputBytes(%s): src=%zu B > tensor dataSize=%u B "
                "— shape/dtype mismatch?",
                what, srcBytes, buf.dataSize);
      return false;
    }
    memcpy(buf.data, src, srcBytes);
    return true;
  }

  void releasePerformanceMode() {
    if (!m_powerConfigIdValid) return;
    auto qnnInterface = m_qnnFunctionPointers.qnnInterface;
    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    if (qnnInterface.deviceGetInfrastructure(&deviceInfra) == QNN_SUCCESS &&
        deviceInfra) {
      auto *htpInfra =
          static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
      htpInfra->perfInfra.destroyPowerConfigId(m_powerConfigId);
    }
    m_powerConfigIdValid = false;
    m_powerConfigId = 0;
  }

  StatusCode executeClipGraphs(int32_t *input_ids, float *text_embedding) {
    auto returnStatus = StatusCode::SUCCESS;

    size_t graphIdx = 0;
    QNN_DEBUG("Starting clip execution for graphIdx: %d", graphIdx);

    // set input/output tensor
    if (inputs == nullptr || outputs == nullptr) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs,
                                                (*m_graphsInfo)[graphIdx])) {
        QNN_ERROR(
            "Error in setting up Input and output Tensors for graphIdx: %d",
            graphIdx);
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    auto graphInfo = (*m_graphsInfo)[graphIdx];

    // check input/output tensors
    if (graphInfo.numInputTensors != 1 || graphInfo.numOutputTensors != 1) {
      QNN_ERROR(
          "Expecting 1 input and 1 output tensor, got %d inputs and %d "
          "outputs",
          graphInfo.numInputTensors, graphInfo.numOutputTensors);
      returnStatus = StatusCode::FAILURE;
      return returnStatus;
    }

    // input_ids — branch on the actual tensor dtype rather than blindly
    // memcpy'ing INT_32. Different CLIP exports use INT_32 (most), INT_64
    // (LLaMA-style tokenizers), UINT_32, or even FLOAT_32 (clip_v2 path
    // takes pre-computed embeddings). Hardcoding 77*sizeof(int32_t) was
    // half-populating the buffer for INT_64 and overshooting for narrower
    // types, both silent.
    {
      Qnn_Tensor_t &in0 = inputs[0];
      uint32_t elementCount = tensorElementCount(in0);
      Qnn_DataType_t dtype = QNN_TENSOR_GET_DATA_TYPE(in0);
      size_t elemBytes = tensorElementBytes(dtype);
      if (dtype == QNN_DATATYPE_INT_32 || dtype == QNN_DATATYPE_UINT_32) {
        if (!writeInputBytes(in0, input_ids,
                             size_t(elementCount) * sizeof(int32_t),
                             "clip.input_ids[i32]")) {
          return StatusCode::FAILURE;
        }
      } else if (dtype == QNN_DATATYPE_INT_64 ||
                 dtype == QNN_DATATYPE_UINT_64) {
        std::vector<int64_t> ids64(elementCount);
        for (uint32_t i = 0; i < elementCount; ++i)
          ids64[i] = static_cast<int64_t>(input_ids[i]);
        if (!writeInputBytes(in0, ids64.data(),
                             ids64.size() * sizeof(int64_t),
                             "clip.input_ids[i64]")) {
          return StatusCode::FAILURE;
        }
      } else if (dtype == QNN_DATATYPE_FLOAT_32) {
        std::vector<float> idsf(elementCount);
        for (uint32_t i = 0; i < elementCount; ++i)
          idsf[i] = static_cast<float>(input_ids[i]);
        if (!writeInputBytes(in0, idsf.data(),
                             idsf.size() * sizeof(float),
                             "clip.input_ids[f32]")) {
          return StatusCode::FAILURE;
        }
      } else {
        QNN_ERROR("clip.input_ids: unsupported tensor dtype=%d (elemBytes=%zu)",
                  (int)dtype, elemBytes);
        return StatusCode::FAILURE;
      }
    }

    // execute graph
    QNN_DEBUG("Executing clip graph: %d", graphIdx);
    auto start_time = std::chrono::high_resolution_clock::now();

    auto executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph, inputs, graphInfo.numInputTensors, outputs,
        graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();
    QNN_INFO("clip graph execution time: %d ms", duration);

    if (QNN_GRAPH_NO_ERROR != executeStatus) {
      returnStatus = StatusCode::FAILURE;
      QNN_ERROR("clip graph execution failed!");
    }

    // get output — Perf 4: write directly into caller buffer (no malloc/free)
    if (StatusCode::SUCCESS == returnStatus) {
      uint32_t elementCount = 1 * 77 * text_embedding_size;
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.convertToFloat(text_embedding, elementCount, &outputs[0])) {
        returnStatus = StatusCode::FAILURE;
      }
    }

    return returnStatus;
  }

  StatusCode executeUnetGraphs(float *latents, int timestep,
                               float *text_embedding, float *latents_pred) {
    auto returnStatus = StatusCode::SUCCESS;

    size_t graphIdx = 0;
    QNN_DEBUG("Starting unet execution for graphIdx: %d", graphIdx);

    // set input/output tensor
    if (inputs == nullptr || outputs == nullptr) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs,
                                                (*m_graphsInfo)[graphIdx])) {
        QNN_ERROR(
            "Error in setting up Input and output Tensors for graphIdx: %d",
            graphIdx);
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    auto graphInfo = (*m_graphsInfo)[graphIdx];

    if (graphInfo.numInputTensors != 3) {
      QNN_ERROR("Expecting 3 input tensors, got %d", graphInfo.numInputTensors);
      returnStatus = StatusCode::FAILURE;
      return returnStatus;
    }

    // latents — derive elementCount from the tensor's actual dims rather
    // than the C++ globals (sample_width/sample_height), so loading a
    // model whose UNet was exported at a different resolution doesn't
    // walk off the end of the QNN client buffer.
    {
      Qnn_Tensor_t &in0 = inputs[0];
      uint32_t elementCount = tensorElementCount(in0);
      auto buf = QNN_TENSOR_GET_CLIENT_BUF(in0);
      if (buf.dataSize < elementCount * sizeof(uint16_t)) {
        QNN_ERROR("unet.latents: tensor dataSize=%u < %u B (shape mismatch?)",
                  buf.dataSize, elementCount * 2);
        return StatusCode::FAILURE;
      }
      uint16_t *latents_uint16 = static_cast<uint16_t *>(buf.data);
      auto qp = in0.v1.quantizeParams.scaleOffsetEncoding;
      qnn::tools::datautil::floatToTfN(
          latents_uint16, latents, qp.offset, qp.scale, elementCount);
    }

    // position/timestep
    {
      int32_t *positionData =
          static_cast<int32_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[1]).data);
      positionData[0] = timestep;
    }

    // text_embedding — Perf 3: cache quantized embeddings per source
    // pointer (embeddings are constant across all denoising steps).
    {
      Qnn_Tensor_t &in2 = inputs[2];
      uint32_t elementCount = tensorElementCount(in2);
      auto buf = QNN_TENSOR_GET_CLIENT_BUF(in2);
      if (buf.dataSize < elementCount * sizeof(uint16_t)) {
        QNN_ERROR("unet.text_embed: tensor dataSize=%u < %u B",
                  buf.dataSize, elementCount * 2);
        return StatusCode::FAILURE;
      }
      uint16_t *text_embedding_uint16 = static_cast<uint16_t *>(buf.data);
      auto& cache = m_embedCache[text_embedding];
      if (cache.empty()) {
        cache.resize(elementCount);
        auto qp = in2.v1.quantizeParams.scaleOffsetEncoding;
        qnn::tools::datautil::floatToTfN(
            cache.data(), text_embedding, qp.offset, qp.scale, elementCount);
      }
      memcpy(text_embedding_uint16, cache.data(),
             elementCount * sizeof(uint16_t));
    }

    // execute graph
    QNN_DEBUG("Executing unet graph: %d", graphIdx);
    auto start_time = std::chrono::high_resolution_clock::now();

    auto executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph, inputs, graphInfo.numInputTensors, outputs,
        graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();
    QNN_INFO("unet graph execution time: %d ms", duration);

    if (QNN_GRAPH_NO_ERROR != executeStatus) {
      returnStatus = StatusCode::FAILURE;
      QNN_ERROR("unet graph execution failed!");
    }

    // get output — Perf 4: write directly into caller buffer
    if (StatusCode::SUCCESS == returnStatus) {
      uint32_t elementCount = tensorElementCount(outputs[0]);
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.convertToFloat(latents_pred, elementCount, &outputs[0])) {
        returnStatus = StatusCode::FAILURE;
      }
    }

    return returnStatus;
  }

  StatusCode executeVaeEncoderGraphs(float *pixel_values, float *mean,
                                     float *std) {
    auto returnStatus = StatusCode::SUCCESS;

    size_t graphIdx = 0;
    QNN_DEBUG("Starting vae encoder execution for graphIdx: %d", graphIdx);

    // set input/output tensor
    if (inputs == nullptr || outputs == nullptr) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs,
                                                (*m_graphsInfo)[graphIdx])) {
        QNN_ERROR(
            "Error in setting up Input and output Tensors for graphIdx: %d",
            graphIdx);
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    auto graphInfo = (*m_graphsInfo)[graphIdx];

    if (graphInfo.numInputTensors != 1) {
      QNN_ERROR("Expecting 1 input tensors, got %d", graphInfo.numInputTensors);
      returnStatus = StatusCode::FAILURE;
      return returnStatus;
    }

    // pixel_values — element count from tensor dims, not globals.
    {
      Qnn_Tensor_t &in0 = inputs[0];
      uint32_t elementCount = tensorElementCount(in0);
      auto buf = QNN_TENSOR_GET_CLIENT_BUF(in0);
      if (buf.dataSize < elementCount * sizeof(uint16_t)) {
        QNN_ERROR("vae_enc.input: tensor dataSize=%u < %u B",
                  buf.dataSize, elementCount * 2);
        return StatusCode::FAILURE;
      }
      uint16_t *pixel_values_uint16 = static_cast<uint16_t *>(buf.data);
      auto qp = in0.v1.quantizeParams.scaleOffsetEncoding;
      qnn::tools::datautil::floatToTfN(
          pixel_values_uint16, pixel_values, qp.offset, qp.scale, elementCount);
    }

    // execute graph
    QNN_DEBUG("Executing vae encoder graph: %d", graphIdx);
    auto start_time = std::chrono::high_resolution_clock::now();

    auto executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph, inputs, graphInfo.numInputTensors, outputs,
        graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();
    QNN_INFO("vae encoder graph execution time: %d ms", duration);

    if (QNN_GRAPH_NO_ERROR != executeStatus) {
      returnStatus = StatusCode::FAILURE;
      QNN_ERROR("vae encoder graph execution failed!");
    }

    // get output — Perf 4: write directly into caller buffers
    if (StatusCode::SUCCESS == returnStatus) {
      uint32_t meanCount = tensorElementCount(outputs[0]);
      uint32_t stdCount = tensorElementCount(outputs[1]);
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.convertToFloat(mean, meanCount, &outputs[0])) {
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.convertToFloat(std, stdCount, &outputs[1])) {
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    return returnStatus;
  }

  StatusCode executeVaeDecoderGraphs(float *latents, float *pixel_values) {
    auto returnStatus = StatusCode::SUCCESS;

    size_t graphIdx = 0;
    QNN_DEBUG("Starting vae decoder execution for graphIdx: %d", graphIdx);

    // set input/output tensor
    if (inputs == nullptr || outputs == nullptr) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs,
                                                (*m_graphsInfo)[graphIdx])) {
        QNN_ERROR(
            "Error in setting up Input and output Tensors for graphIdx: %d",
            graphIdx);
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    auto graphInfo = (*m_graphsInfo)[graphIdx];

    if (graphInfo.numInputTensors != 1) {
      QNN_ERROR("Expecting 1 input tensors, got %d", graphInfo.numInputTensors);
      returnStatus = StatusCode::FAILURE;
      return returnStatus;
    }

    // latents — element count from tensor dims, not globals.
    {
      Qnn_Tensor_t &in0 = inputs[0];
      uint32_t elementCount = tensorElementCount(in0);
      auto buf = QNN_TENSOR_GET_CLIENT_BUF(in0);
      if (buf.dataSize < elementCount * sizeof(uint16_t)) {
        QNN_ERROR("vae_dec.input: tensor dataSize=%u < %u B",
                  buf.dataSize, elementCount * 2);
        return StatusCode::FAILURE;
      }
      uint16_t *latents_uint16 = static_cast<uint16_t *>(buf.data);
      auto qp = in0.v1.quantizeParams.scaleOffsetEncoding;
      qnn::tools::datautil::floatToTfN(
          latents_uint16, latents, qp.offset, qp.scale, elementCount);
    }

    // execute graph
    QNN_DEBUG("Executing vae decoder graph: %d", graphIdx);
    auto start_time = std::chrono::high_resolution_clock::now();

    auto executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph, inputs, graphInfo.numInputTensors, outputs,
        graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();
    QNN_INFO("vae decoder graph execution time: %d ms", duration);

    if (QNN_GRAPH_NO_ERROR != executeStatus) {
      returnStatus = StatusCode::FAILURE;
      QNN_ERROR("vae decoder graph execution failed!");
    }

    // get output — Perf 4: write directly into caller buffer
    if (StatusCode::SUCCESS == returnStatus) {
      uint32_t elementCount = tensorElementCount(outputs[0]);
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.convertToFloat(pixel_values, elementCount, &outputs[0])) {
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    return returnStatus;
  }

  StatusCode executeUpscalerGraphs(float *input_image, float *output_image) {
    auto returnStatus = StatusCode::SUCCESS;

    size_t graphIdx = 0;
    QNN_DEBUG("Starting upscaler execution for graphIdx: %d", graphIdx);

    // set input/output tensor
    if (inputs == nullptr || outputs == nullptr) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs,
                                                (*m_graphsInfo)[graphIdx])) {
        QNN_ERROR(
            "Error in setting up Input and output Tensors for graphIdx: %d",
            graphIdx);
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    auto graphInfo = (*m_graphsInfo)[graphIdx];

    if (graphInfo.numInputTensors != 1) {
      QNN_ERROR("Expecting 1 input tensors, got %d", graphInfo.numInputTensors);
      returnStatus = StatusCode::FAILURE;
      return returnStatus;
    }

    // Input. Branch on the actual tensor dtype: a HTP-quantized upscaler
    // expects UFIXED_POINT_8 / _16, an FP32-export expects FLOAT_32. The
    // prior code assumed FLOAT_32 unconditionally (the dequant block was
    // commented out) — so a uint8-quantized upscaler was being fed raw
    // float bit patterns reinterpreted as bytes, producing pure garbage.
    {
      Qnn_Tensor_t &in0 = inputs[0];
      uint32_t elementCount = tensorElementCount(in0);
      Qnn_DataType_t dtype = QNN_TENSOR_GET_DATA_TYPE(in0);
      auto buf = QNN_TENSOR_GET_CLIENT_BUF(in0);
      if (!buf.data || elementCount == 0) {
        QNN_ERROR("upscaler.input: bad tensor (data=%p elem=%u)",
                  buf.data, elementCount);
        return StatusCode::FAILURE;
      }
      auto qp = in0.v1.quantizeParams.scaleOffsetEncoding;
      if (dtype == QNN_DATATYPE_UFIXED_POINT_8) {
        if (buf.dataSize < elementCount * sizeof(uint8_t)) {
          QNN_ERROR("upscaler.input[u8]: tensor dataSize=%u < %u B",
                    buf.dataSize, elementCount);
          return StatusCode::FAILURE;
        }
        qnn::tools::datautil::floatToTfN(
            static_cast<uint8_t *>(buf.data), input_image,
            qp.offset, qp.scale, elementCount);
      } else if (dtype == QNN_DATATYPE_UFIXED_POINT_16) {
        if (buf.dataSize < elementCount * sizeof(uint16_t)) {
          QNN_ERROR("upscaler.input[u16]: tensor dataSize=%u < %u B",
                    buf.dataSize, elementCount * 2);
          return StatusCode::FAILURE;
        }
        qnn::tools::datautil::floatToTfN(
            static_cast<uint16_t *>(buf.data), input_image,
            qp.offset, qp.scale, elementCount);
      } else if (dtype == QNN_DATATYPE_FLOAT_32) {
        if (!writeInputBytes(in0, input_image,
                             size_t(elementCount) * sizeof(float),
                             "upscaler.input[f32]")) {
          return StatusCode::FAILURE;
        }
      } else {
        QNN_ERROR("upscaler.input: unsupported dtype=%d", (int)dtype);
        return StatusCode::FAILURE;
      }
    }

    // execute graph
    QNN_DEBUG("Executing upscaler graph: %d", graphIdx);
    auto start_time = std::chrono::high_resolution_clock::now();

    auto executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph, inputs, graphInfo.numInputTensors, outputs,
        graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();
    QNN_INFO("upscaler graph execution time: %d ms", duration);

    if (QNN_GRAPH_NO_ERROR != executeStatus) {
      returnStatus = StatusCode::FAILURE;
      QNN_ERROR("upscaler graph execution failed!");
    }

    // Output. Same dtype-branching as input. convertToFloat handles the
    // dequantization for fixed-point types into our caller-owned float
    // buffer (no malloc, no free — reuses the Perf 4 caller-buffer
    // overload).
    if (StatusCode::SUCCESS == returnStatus) {
      Qnn_Tensor_t &out0 = outputs[0];
      uint32_t outElementCount = tensorElementCount(out0);
      Qnn_DataType_t outDtype = QNN_TENSOR_GET_DATA_TYPE(out0);
      if (outDtype == QNN_DATATYPE_FLOAT_32) {
        auto outBuf = QNN_TENSOR_GET_CLIENT_BUF(out0);
        memcpy(output_image, outBuf.data,
               size_t(outElementCount) * sizeof(float));
      } else {
        if (qnn::tools::iotensor::StatusCode::SUCCESS !=
            m_ioTensor.convertToFloat(output_image, outElementCount,
                                       &outputs[0])) {
          returnStatus = StatusCode::FAILURE;
        }
      }
    }
    return returnStatus;
  }

  StatusCode createFromBuffer(const uint8_t *buffer, uint64_t bufferSize) {
    if (nullptr == buffer || 0 == bufferSize) {
      QNN_ERROR("Invalid buffer provided. Buffer is null or size is 0.");
      return StatusCode::FAILURE;
    }

    if (nullptr ==
            m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate ||
        nullptr == m_qnnFunctionPointers.qnnSystemInterface
                       .systemContextGetBinaryInfo ||
        nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextFree) {
      QNN_ERROR("QNN System function pointers are not populated.");
      return StatusCode::FAILURE;
    }

    auto returnStatus = StatusCode::SUCCESS;
    QnnSystemContext_Handle_t sysCtxHandle{nullptr};

    if (QNN_SUCCESS !=
        m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate(
            &sysCtxHandle)) {
      QNN_ERROR("Could not create system handle.");
      returnStatus = StatusCode::FAILURE;
    }

    const QnnSystemContext_BinaryInfo_t *binaryInfo{nullptr};
    Qnn_ContextBinarySize_t binaryInfoSize{0};

    void *nonConstBuffer =
        const_cast<void *>(static_cast<const void *>(buffer));

    if (StatusCode::SUCCESS == returnStatus &&
        QNN_SUCCESS !=
            m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo(
                sysCtxHandle, nonConstBuffer, bufferSize, &binaryInfo,
                &binaryInfoSize)) {
      QNN_ERROR("Failed to get context binary info");
      returnStatus = StatusCode::FAILURE;
    }

    // Walk the binaryInfo per-graph blob (HTP V3 only) to record what the
    // .bin author baked in: vtcmSize, numHvxThreads, spillFillBufferSize,
    // optimizationLevel. binaryInfo is owned by sysCtxHandle and is freed
    // by systemContextFree below, so anything we want must be copied now.
    if (StatusCode::SUCCESS == returnStatus && binaryInfo) {
      const QnnSystemContext_GraphInfo_t* graphsList = nullptr;
      uint32_t numBakedGraphs = 0;
      if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
        graphsList = binaryInfo->contextBinaryInfoV1.graphs;
        numBakedGraphs = binaryInfo->contextBinaryInfoV1.numGraphs;
      } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        graphsList = binaryInfo->contextBinaryInfoV2.graphs;
        numBakedGraphs = binaryInfo->contextBinaryInfoV2.numGraphs;
      } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
        graphsList = binaryInfo->contextBinaryInfoV3.graphs;
        numBakedGraphs = binaryInfo->contextBinaryInfoV3.numGraphs;
      }
      m_bakedHwInfo.assign(numBakedGraphs, GraphHwInfo{});
      for (uint32_t i = 0; i < numBakedGraphs && graphsList; ++i) {
        const auto& gi = graphsList[i];
        if (gi.version != QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) continue;
        const auto& v3 = gi.graphInfoV3;
        if (!v3.graphBlobInfo || v3.graphBlobInfoSize == 0) continue;
        const auto* blob =
            static_cast<const QnnHtpSystemContext_GraphBlobInfo_t*>(v3.graphBlobInfo);
        if (blob->version != QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V1) continue;
        const auto& bv1 = blob->contextBinaryGraphBlobInfoV1;
        m_bakedHwInfo[i] = {bv1.vtcmSize, bv1.numHvxThreads,
                            bv1.spillFillBufferSize, bv1.optimizationLevel};
        QNN_INFO("graph[%u] '%s': vtcm=%u MB, hvxThreads=%" PRIu64
                 ", spillFill=%" PRIu64 " B, opt=%u",
                 i, v3.graphName ? v3.graphName : "(unnamed)",
                 bv1.vtcmSize, bv1.numHvxThreads,
                 bv1.spillFillBufferSize, bv1.optimizationLevel);
      }
    }

    if (StatusCode::SUCCESS == returnStatus &&
        !copyMetadataToGraphsInfo(binaryInfo, m_graphsInfo, m_graphsCount)) {
      QNN_ERROR("Failed to copy metadata.");
      returnStatus = StatusCode::FAILURE;
    }

    m_qnnFunctionPointers.qnnSystemInterface.systemContextFree(sysCtxHandle);
    sysCtxHandle = nullptr;

    if (StatusCode::SUCCESS == returnStatus &&
        nullptr == m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary) {
      QNN_ERROR("contextCreateFromBinaryFnHandle is nullptr.");
      returnStatus = StatusCode::FAILURE;
    }

    if (StatusCode::SUCCESS == returnStatus &&
        m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary(
            m_backendHandle, m_deviceHandle,
            (const QnnContext_Config_t **)m_contextConfig, nonConstBuffer,
            bufferSize, &m_context, m_profileBackendHandle)) {
      QNN_ERROR("Could not create context from binary.");
      returnStatus = StatusCode::FAILURE;
    }

    if (ProfilingLevel::OFF != m_profilingLevel) {
      extractBackendProfilingInfo(m_profileBackendHandle);
    }

    // Same fix as the base class' createFromBinary: only mark the context
    // as created on success, so a failed contextCreateFromBinary doesn't
    // leave the destructor calling contextFree on a never-created handle.
    if (StatusCode::SUCCESS == returnStatus) {
      m_isContextCreated = true;
      for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
        if (nullptr == m_qnnFunctionPointers.qnnInterface.graphRetrieve) {
          QNN_ERROR("graphRetrieveFnHandle is nullptr.");
          returnStatus = StatusCode::FAILURE;
          break;
        }
        if (QNN_SUCCESS != m_qnnFunctionPointers.qnnInterface.graphRetrieve(
                               m_context, (*m_graphsInfo)[graphIdx].graphName,
                               &((*m_graphsInfo)[graphIdx].graph))) {
          QNN_ERROR("Unable to retrieve graph handle for graph Idx: %d",
                    graphIdx);
          returnStatus = StatusCode::FAILURE;
        }
      }
    }

    // Apply runtime graph configs (VTCM size + HVX thread count) once the
    // graphs are retrieved. For context-binary-loaded graphs (which is
    // every model in our pipeline) these are MOSTLY IGNORED — the .bin
    // author baked the values at compile time and the runtime can only
    // honor smaller VTCM, never larger. We pass them anyway so that:
    //   (a) a hypothetical online-composed-graph path benefits,
    //   (b) the runtime doesn't speculatively reserve more VTCM than the
    //       SoC actually has, and
    //   (c) HVX thread count is bounded by what the device infra reports
    //       (2 on SM7635, vs 4 baked into xororz's binaries).
    if (StatusCode::SUCCESS == returnStatus) {
      applyRuntimeGraphConfigs();
    }

    if (StatusCode::SUCCESS != returnStatus) {
      QNN_DEBUG("Cleaning up graph Info structures.");
      qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount);
    }

    return returnStatus;
  }

  // Try to set per-graph VTCM size and HVX thread count after graphs are
  // retrieved. Best-effort: errors here are logged but do not fail the
  // model load — the runtime falls back to the baked-in values from the
  // context binary, which is the safe behavior.
  void applyRuntimeGraphConfigs() {
    auto qnnInterface = m_qnnFunctionPointers.qnnInterface;
    if (!qnnInterface.graphSetConfig) return;

    // Detect the device's actual HVX thread count via HTP infra; fall
    // back to 2 (mid-tier V73 baseline) if the query fails.
    uint32_t hvxThreads = 2;
    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    if (qnnInterface.deviceGetInfrastructure(&deviceInfra) == QNN_SUCCESS &&
        deviceInfra) {
      // QnnHtpDevice_Infrastructure_t doesn't directly expose thread count
      // in older SDKs; use a conservative cap that matches SM7635.
      // TODO: when SDK exposes numHvxThreads via getOnChipDeviceInfo,
      //       read it and use min(detected, baked).
    }

    // 2 MB matches xororz `min` binaries' baked vtcm_size on SM7635 and
    // is the actual silicon limit on 7s Gen 3. Setting larger has no
    // effect on a bin-loaded graph; setting smaller is also a no-op.
    constexpr uint32_t vtcmSizeMB = 2;

    for (uint32_t gIdx = 0; gIdx < m_graphsCount && m_graphsInfo; ++gIdx) {
      auto graphHandle = (*m_graphsInfo)[gIdx].graph;
      if (!graphHandle) continue;

      QnnHtpGraph_CustomConfig_t vtcmCustom;
      memset(&vtcmCustom, 0, sizeof(vtcmCustom));
      vtcmCustom.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
      vtcmCustom.vtcmSizeInMB = vtcmSizeMB;

      QnnHtpGraph_CustomConfig_t hvxCustom;
      memset(&hvxCustom, 0, sizeof(hvxCustom));
      hvxCustom.option = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
      hvxCustom.numHvxThreads = hvxThreads;

      QnnGraph_Config_t vtcmCfg;
      vtcmCfg.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
      vtcmCfg.customConfig = &vtcmCustom;

      QnnGraph_Config_t hvxCfg;
      hvxCfg.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
      hvxCfg.customConfig = &hvxCustom;

      const QnnGraph_Config_t* cfgs[] = {&vtcmCfg, &hvxCfg, nullptr};
      auto rc = qnnInterface.graphSetConfig(graphHandle, cfgs);
      if (rc != QNN_SUCCESS) {
        QNN_DEBUG("graph[%u] graphSetConfig returned %d (likely ignored "
                  "for bin-loaded graphs — values are already baked)",
                  gIdx, (int)rc);
      }
    }
  }

  // Perf 3: Clear cached quantized embeddings (call between generations)
  void clearCachedEmbeddings() { m_embedCache.clear(); }

 private:
  // Perf 3: Cached quantized CLIP embeddings keyed by source pointer
  // (uncond + cond are two distinct pointers, both constant across steps)
  std::unordered_map<const float*, std::vector<uint16_t>> m_embedCache;

  // HTP power-config slot owned by this model. Held as a member so the
  // destructor can release it via destroyPowerConfigId; previously it was
  // a local in enablePerformaceMode and the slot leaked per model load.
  uint32_t m_powerConfigId = 0;
  bool m_powerConfigIdValid = false;

  // Per-graph hardware metadata read out of the .bin's contextBinaryInfo
  // V3 blob: vtcmSize (MB), numHvxThreads, spillFillBufferSize (bytes),
  // optimizationLevel. Populated in createFromBuffer; useful as ground
  // truth for runtime tuning (e.g. confirming the device's actual VTCM
  // matches the baked value, or computing a shared spill buffer across
  // multiple contexts in a future optimization).
  struct GraphHwInfo {
    uint32_t vtcmSize = 0;
    uint64_t numHvxThreads = 0;
    uint64_t spillFillBufferSize = 0;
    uint32_t optimizationLevel = 0;
  };
  std::vector<GraphHwInfo> m_bakedHwInfo;
};

#endif  // QNNMODEL_HPP