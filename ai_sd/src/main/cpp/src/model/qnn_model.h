#ifndef QNNMODEL_HPP
#define QNNMODEL_HPP

#include <HTP/QnnHtpDevice.h>
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

  StatusCode enablePerformaceMode() {
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

    // input_ids
    {
      uint32_t elementCount = 1 * 77;
      memcpy(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data, input_ids,
             elementCount * sizeof(int32_t));
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

    // latents
    {
      uint16_t *latents_uint16 =
          static_cast<uint16_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data);
      int elementCount = 1 * 4 * sample_width * sample_height;
      qnn::tools::datautil::floatToTfN(
          latents_uint16, latents,
          inputs[0].v1.quantizeParams.scaleOffsetEncoding.offset,
          inputs[0].v1.quantizeParams.scaleOffsetEncoding.scale, elementCount);
    }

    // position/timestep
    {
      int32_t *positionData =
          static_cast<int32_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[1]).data);
      positionData[0] = timestep;
    }

    // text_embedding — Perf 3: cache quantized embeddings per source pointer
    // (embeddings are constant across all denoising steps)
    {
      uint16_t *text_embedding_uint16 =
          static_cast<uint16_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[2]).data);
      int elementCount = 1 * 77 * text_embedding_size;
      auto& cache = m_embedCache[text_embedding];
      if (cache.empty()) {
        cache.resize(elementCount);
        qnn::tools::datautil::floatToTfN(
            cache.data(), text_embedding,
            inputs[2].v1.quantizeParams.scaleOffsetEncoding.offset,
            inputs[2].v1.quantizeParams.scaleOffsetEncoding.scale, elementCount);
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
      int elementCount = 1 * 4 * sample_width * sample_height;
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

    // pixel_values
    {
      uint16_t *pixel_values_uint16 =
          static_cast<uint16_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data);
      int elementCount = 1 * 3 * output_width * output_height;
      qnn::tools::datautil::floatToTfN(
          pixel_values_uint16, pixel_values,
          inputs[0].v1.quantizeParams.scaleOffsetEncoding.offset,
          inputs[0].v1.quantizeParams.scaleOffsetEncoding.scale, elementCount);
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
      int elementCount = 1 * 4 * sample_width * sample_height;
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.convertToFloat(mean, elementCount, &outputs[0])) {
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.convertToFloat(std, elementCount, &outputs[1])) {
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

    // latents
    {
      uint16_t *latents_uint16 =
          static_cast<uint16_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data);
      int elementCount = 1 * 4 * sample_width * sample_height;
      qnn::tools::datautil::floatToTfN(
          latents_uint16, latents,
          inputs[0].v1.quantizeParams.scaleOffsetEncoding.offset,
          inputs[0].v1.quantizeParams.scaleOffsetEncoding.scale, elementCount);
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
      size_t elementCount = 1 * 3 * output_width * output_height;
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

    // input_image (quantized to uint8, 1x3x192x192)
    {
      // uint8_t *input_uint8 =
      //     static_cast<uint8_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data);
      // int elementCount = 1 * 3 * 192 * 192;
      // qnn::tools::datautil::floatToTfN(
      //     input_uint8, input_image,
      //     inputs[0].v1.quantizeParams.scaleOffsetEncoding.offset,
      //     inputs[0].v1.quantizeParams.scaleOffsetEncoding.scale,
      //     elementCount);
      memcpy(static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data),
             input_image, 1 * 3 * 192 * 192 * sizeof(float));
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

    // get output
    // if (StatusCode::SUCCESS == returnStatus) {
    //   float *tmp = nullptr;
    //   int elementCount = 1 * 3 * 768 * 768;
    //   if (qnn::tools::iotensor::StatusCode::SUCCESS !=
    //       m_ioTensor.convertToFloat(&tmp, &outputs[0])) {
    //     returnStatus = StatusCode::FAILURE;
    //     return returnStatus;
    //   }
    //   memcpy(output_image, tmp, elementCount * sizeof(float));
    //   free(tmp);
    // }
    memcpy(output_image,
           static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(outputs[0]).data),
           1 * 3 * 768 * 768 * sizeof(float));
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

    m_isContextCreated = true;

    if (StatusCode::SUCCESS == returnStatus) {
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

    if (StatusCode::SUCCESS != returnStatus) {
      QNN_DEBUG("Cleaning up graph Info structures.");
      qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount);
    }

    return returnStatus;
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
};

#endif  // QNNMODEL_HPP