// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2022 Second State INC

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#ifdef WASMEDGE_WASINN_BUILD_OPENVINO
#include "common/log.h"
#include <c_api/ie_c_api.h>
#endif

namespace WasmEdge {
namespace Host {
namespace WASINN {

enum class ErrNo : uint32_t {
  Success = 0,         // No error occurred.
  InvalidArgument = 1, // Caller module passed an invalid argument.
  MissingMemory = 2,   // Caller module is missing a memory export.
  Busy = 3             // Device or resource busy.
};

enum class Backend : uint8_t {
  OpenVINO = 0,
};

using Graph = uint32_t;
using GraphEncoding = uint8_t;
using ExecutionTarget = uint8_t;
using GraphExecutionContext = uint32_t;

#ifdef WASMEDGE_WASINN_BUILD_OPENVINO
class OpenVINOSession {
public:
  ~OpenVINOSession() {
    if (InferRequest != nullptr) {
      ie_infer_request_free(&(InferRequest));
    }
  }
  ie_network_t *Network = nullptr;
  ie_executable_network_t *ExeNetwork = nullptr;
  ie_infer_request_t *InferRequest = nullptr;
};
#endif

class WasiNNContext {
public:
  WasiNNContext() noexcept {
#ifdef WASMEDGE_WASINN_BUILD_OPENVINO
    if (ie_core_create("", &OpenVINOCore) != IEStatusCode::OK) {
      spdlog::error(
          "[WASI-NN] Error happened when initializing OpenVINO core.");
    }
#endif
    GraphBackends.reserve(16U);
    GraphContextBackends.reserve(16U);
  }
  ~WasiNNContext() {
#ifdef WASMEDGE_WASINN_BUILD_OPENVINO
    if (OpenVINOCore != nullptr) {
      ie_core_free(&OpenVINOCore);
    }
    for (auto &I : OpenVINONetworks) {
      ie_network_free(&I);
    }
    for (auto &I : OpenVINOExecutions) {
      ie_exec_network_free(&I);
    }
    for (auto &I : OpenVINOInfers) {
      if (I != nullptr) {
        delete I;
      }
    }
    for (auto &I : OpenVINOModelWeights) {
      if (I != nullptr) {
        ie_blob_free(&I);
      }
    }
#endif
  }

  // context for implementing WASI-NN
  std::vector<Backend> GraphBackends;
  std::vector<Backend> GraphContextBackends;
#ifdef WASMEDGE_WASINN_BUILD_OPENVINO
  ie_core_t *OpenVINOCore = nullptr;
  std::vector<ie_network_t *> OpenVINONetworks;
  std::vector<ie_executable_network_t *> OpenVINOExecutions;
  std::vector<ie_blob_t *> OpenVINOModelWeights;
  std::vector<OpenVINOSession *> OpenVINOInfers;
#endif
};

} // namespace WASINN
} // namespace Host
} // namespace WasmEdge
