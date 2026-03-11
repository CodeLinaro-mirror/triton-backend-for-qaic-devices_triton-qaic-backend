// Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Changes from Qualcomm Innovation Center are provided under the following license:
// Copyright (c) 2023,2026 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#ifndef TRITON_BACKEND_QAIC_MODEL_STATE_H_
#define TRITON_BACKEND_QAIC_MODEL_STATE_H_

#include <list>
#include <vector>
#include <string>
#include <filesystem>
#include "QAicApi.hpp"
#include "QAicCc.hpp"

namespace qaicrt = ::qaic::rt;

#define LOG_IF_FALSE_AND_RETURN(condition, msg, ret_value)           \
  if (!condition) {                                                 \
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string(msg).c_str())); \
    return ret_value;                                                \
  }

namespace triton { namespace backend { namespace qaic {
extern qaicrt::shContext global_rt_context;

// Configuration for model input and output
struct ModelInputOutput {
  std::string name;
  std::string d_type;
  std::vector<int64_t> dims;
  int32_t size;
  int32_t index;
};

class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);

  // Generate unique ID for inference submission
  static int GetUID();
  // Retrieve index of an output from qpc mappings
  uint32_t GetIndex(std::string name);
  TRITONSERVER_Error* TensorShape(std::vector<int64_t>& shape);
  const std::vector<int64_t>& TensorNonBatchShape() const { return nb_shape_; }
  const std::list<ModelInputOutput>& GetConfigOutputs() { return config_outputs_; }
  const std::list<ModelInputOutput>& GetQpcInputs() { return qpc_inputs_; }
  const std::list<ModelInputOutput>& GetQpcOutputs() { return qpc_outputs_; }
  qaicrt::shQpc GetQpc() const { return qpc_; }
  qaicrt::shContext GetContext() const { return rt_context_; }
  int GetSetSize() const { return set_size_; }
  int GetActivations() const { return no_of_activations_; }
  std::optional<QID> GetDeviceId() const { return device_id_; }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  // Map config json into ModelInputOutput
  TRITONSERVER_Error* ConvertJsonMappings(
      common::TritonJson::Value& json_list,
      std::list<ModelInputOutput>& out_list);
  TRITONSERVER_Error* PopulateQpcMappings();
  TRITONSERVER_Error* PopulateConfigMappings();
  // Load the model (parse config) and populate backend objects
  TRITONSERVER_Error* LoadModel();

  TRITONSERVER_Error* AutoCompleteConfig();
  TRITONSERVER_Error* AutoCompleteIO(const char* key);

  bool shape_initialized_;
  std::vector<int64_t> nb_shape_;
  std::vector<int64_t> shape_;
  qaicrt::shQpc qpc_;
  std::list<ModelInputOutput> config_inputs_;
  std::list<ModelInputOutput> config_outputs_;
  std::list<ModelInputOutput> qpc_inputs_;
  std::list<ModelInputOutput> qpc_outputs_;
  uint32_t num_nsp_;
  uint32_t batch_size_;
  qaicrt::shContext rt_context_;
  int set_size_;
  int no_of_activations_;
  std::optional<QID> device_id_;
};

}}}  // namespace triton::backend::qaic

#endif  // TRITON_BACKEND_QAIC_MODEL_STATE_H_