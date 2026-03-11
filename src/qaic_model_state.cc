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

#include "triton/backend/backend_model.h"
#include "qaic_model_state.h"
#include "QAicApi.hpp"

namespace fs = std::filesystem;
namespace qaicrt = ::qaic::rt;

namespace triton { namespace backend { namespace qaic {

/* Below are the default of parameters which can be passed through config.pbtxt file.*/
namespace config {
// set_size denotes no of execution objects per loop, default is set to 20.
constexpr const int default_set_size = 20;
// activation denotes a runtime instance of model network, default is set to 1.
constexpr const int default_activations = 1;
// default version number
constexpr const int default_version = 1;
}

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  // Auto-complete configuration if required
  bool auto_complete_config = false;
  RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(
      triton_model, &auto_complete_config));
  if (auto_complete_config) {
    RETURN_IF_ERROR((*state)->AutoCompleteConfig());

    triton::common::TritonJson::WriteBuffer json_buffer;
    (*state)->ModelConfig().Write(&json_buffer);

    TRITONSERVER_Message* message;
    RETURN_IF_ERROR(TRITONSERVER_MessageNewFromSerializedJson(
        &message, json_buffer.Base(), json_buffer.Size()));
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetConfig(
        triton_model, config::default_version /* config_version */, message));
  }
  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model), shape_initialized_(false)
{
  THROW_IF_BACKEND_MODEL_ERROR(LoadModel());
  THROW_IF_BACKEND_MODEL_ERROR(PopulateConfigMappings());
  THROW_IF_BACKEND_MODEL_ERROR(PopulateQpcMappings());
}

TRITONSERVER_Error*
ModelState::LoadModel()
{
  fs::path dir_path = RepositoryPath();
  fs::path qpc_file_name = "programqpc.bin";
  std::string qpc_path;
  // Search for the first instance of qpc in model path
  // If file does not exist, return with an error
  for (const auto& dir_entry : fs::recursive_directory_iterator(dir_path))
  {
    if (dir_entry.path().filename() == qpc_file_name){
      qpc_path = dir_entry.path().parent_path();
      break;
    }
  }
  bool exists;
  FileExists(qpc_path, &exists);
  RETURN_ERROR_IF_FALSE(
        exists, TRITONSERVER_ERROR_INTERNAL,
        std::string("Could not find qpc file in recursive search within") + dir_path.string());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string(" Using qpc at location: ") + std::string(qpc_path)).c_str());

  // Parse backend parameters
  int device_id;
  int set_size;
  int no_of_activations;
  bool device_id_specified = false;
  triton::common::TritonJson::Value params;

  if (this->ModelConfig().Find("parameters", &params)) {
    //Check if device_id parameter exists.
    triton::common::TritonJson::Value device_param;
    if (params.Find("device_id", &device_param)) {
      THROW_IF_BACKEND_MODEL_ERROR(
          TryParseModelStringParameter(params, "device_id", &device_id, 0));
      device_id_specified = true;
    }
    THROW_IF_BACKEND_MODEL_ERROR(
        TryParseModelStringParameter(params, "set_size", &set_size, config::default_set_size));
    THROW_IF_BACKEND_MODEL_ERROR(
        TryParseModelStringParameter(params, "no_of_activations", &no_of_activations, config::default_activations));
  }
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("+----------------------------+").c_str()));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("| Configured QAIC parameters |").c_str()));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("+----------------------------+").c_str()));
  if (device_id_specified) {
      LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string(" device_id: ") + std::to_string(device_id)).c_str());}
  else {
      LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string(" device_id: not specified (auto-device picking enabled)").c_str()));}
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string(" set_size: ") + std::to_string(set_size)).c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string(" no_of_activations: ") + std::to_string(no_of_activations)).c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("+----------------------------+").c_str()));

  std::optional <QID> aic_device_id;

  // Initialize device id for auto-device picker in case default is selected/no device configured.
  if (device_id_specified) {
  aic_device_id = static_cast<QID>(device_id);
  LOG_MESSAGE(
    TRITONSERVER_LOG_INFO,
    (std::string("Using device ID: ") + std::to_string(device_id)).c_str());
  } else {
  aic_device_id = std::nullopt; // Setting device_id to null which invokes auto-device picker
  }

  try{
    this->qpc_ = qaicrt::Qpc::Factory(qpc_path);
    RETURN_ERROR_IF_FALSE(
      (static_cast<bool>(this->qpc_)), TRITONSERVER_ERROR_INTERNAL,
      std::string("Invalid qpc object"));

    this->rt_context_ = global_rt_context;

    RETURN_ERROR_IF_FALSE(
        (static_cast<bool>(this->rt_context_)), TRITONSERVER_ERROR_INTERNAL,
        std::string("Invalid runtime context"));

    this->set_size_ = set_size;
    this->no_of_activations_ = no_of_activations;
    this->device_id_ = aic_device_id;
  }
  catch (std::exception &e) {
    return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INTERNAL,
      (std::string("qaic backend exception: ") + e.what()).c_str());
  }

  return nullptr;
}


TRITONSERVER_Error*
ModelState::PopulateQpcMappings()
{
  qaicrt::shQpcInfo info = (this->qpc_)->getInfo();
  this->batch_size_ = info->program[0].batchSize;
  this->num_nsp_ = info->program[0].numCores;

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("+----------------------------+").c_str()));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("| Configured QPC parameters  |").c_str()));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("+----------------------------+").c_str()));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string(" batch size: ") + std::to_string(this->batch_size_)).c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string(" num NSP: ") + std::to_string(this->num_nsp_)).c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("+----------------------------+").c_str()));

  qaicrt::BufferMappings buffer_mappings;
  buffer_mappings = (this->qpc_)->getBufferMappings();

  for (auto const& m : buffer_mappings) {
    ModelInputOutput in_out;
    in_out.name = m.bufferName;
    in_out.d_type = "";
    in_out.size = m.size;
    in_out.index = m.index;

    if (m.ioType == BUFFER_IO_TYPE_OUTPUT)
      this->qpc_outputs_.push_back(in_out);

    if (m.ioType == BUFFER_IO_TYPE_INPUT)
      this->qpc_inputs_.push_back(in_out);
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelState::ConvertJsonMappings(
    common::TritonJson::Value& json_list, std::list<ModelInputOutput>& out_list)
{
  for (size_t i = 0; i < json_list.ArraySize(); i++) {
    common::TritonJson::Value value;
    RETURN_IF_ERROR(json_list.IndexAsObject(i, &value));

    std::string name, dtype;
    std::vector<int64_t> shape;

    RETURN_IF_ERROR(value.MemberAsString("name", &name));
    RETURN_IF_ERROR(value.MemberAsString("data_type", &dtype));
    RETURN_IF_ERROR(backend::ParseShape(value, "dims", &shape));

    ModelInputOutput in_out;
    in_out.name = name;
    in_out.d_type = dtype;
    in_out.dims = shape;
    in_out.size = -1;
    in_out.index = -1;

    out_list.push_back(in_out);
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelState::PopulateConfigMappings()
{
  common::TritonJson::Value cfg_inputs, cfg_outputs;
  RETURN_IF_ERROR(
      ModelConfig().MemberAsArray("input", &cfg_inputs));
  RETURN_IF_ERROR(
      ModelConfig().MemberAsArray("output", &cfg_outputs));

  this->ConvertJsonMappings(cfg_inputs, this->config_inputs_);
  this->ConvertJsonMappings(cfg_outputs, this->config_outputs_);

  return nullptr;
}

TRITONSERVER_Error*
ModelState::TensorShape(std::vector<int64_t>& shape)
{
  if (!shape_initialized_) {
    bool supports_first_dim_batching;
    RETURN_IF_ERROR(SupportsFirstDimBatching(&supports_first_dim_batching));
    if (supports_first_dim_batching) {
      shape_.push_back(-1);
    }

    shape_.insert(shape_.end(), nb_shape_.begin(), nb_shape_.end());
    shape_initialized_ = true;
  }

  shape = shape_;

  return nullptr;  // success
}

int
ModelState::GetUID()
{
  static std::atomic<std::uint32_t> uid{0};
  return ++uid;
}

uint32_t
ModelState::GetIndex(std::string name)
{
  for (auto const& output : this->GetQpcOutputs()) {
    if (name.compare(output.name) == 0)
      return output.index;
  }
  return 0;
}

std::string
QAicBufferDataTypeToModelConfigDataType(QAicBufferDataTypeEnum_t data_type){
  switch (data_type){
    case BUFFER_DATA_TYPE_FLOAT:        // 32-bit float type (float)
      return "TYPE_FP32";
      break;
    case BUFFER_DATA_TYPE_FLOAT16:      // 16-bit float type (half, fp16)
      return "TYPE_FP16";
      break;
    case BUFFER_DATA_TYPE_INT8Q:        // 8-bit quantized type (int8_t)
      return "TYPE_INT8";
      break;
    case BUFFER_DATA_TYPE_UINT8Q:       // unsigned 8-bit quantized type (uint8_t)
      return "TYPE_UINT8";
      break;
    case BUFFER_DATA_TYPE_INT16Q:       // 16-bit quantized type (int16_t)
      return "TYPE_INT16";
      break;
    case BUFFER_DATA_TYPE_INT32Q:       // 32-bit quantized type (int32_t)
      return "TYPE_INT32";
      break;
    case BUFFER_DATA_TYPE_INT32I:       // 32-bit index type (int32_t)
      return "TYPE_INT32";
      break;
    case BUFFER_DATA_TYPE_INT64I:       // 64-bit index type (int64_t)
      return "TYPE_INT64";
      break;
    case BUFFER_DATA_TYPE_INT8:         // 8-bit type (int8_t)
      return "TYPE_INT8";
      break;
    default:
      return "TYPE_INVALID";
      break;
  }
}

TRITONSERVER_Error*
ModelState::AutoCompleteConfig()
{
  // Check if input and output is specified
  size_t input_count = 0;
  size_t output_count = 0;
  {
    triton::common::TritonJson::Value inputs;
    if(ModelConfig().Find("input", &inputs)){
      input_count = inputs.ArraySize();
    }

    triton::common::TritonJson::Value outputs;
    if(ModelConfig().Find("output", &outputs)){
      output_count = outputs.ArraySize();
    }
  }

  if ((input_count > 0) && (output_count > 0)){
    LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Skipping auto-complete configuration for '") + Name() +
      "': input and output specified").c_str());
    return nullptr;
  }

  if (input_count == 0){
    RETURN_IF_ERROR(AutoCompleteIO("input"));
  }
  if (output_count == 0){
    RETURN_IF_ERROR(AutoCompleteIO("output"));
  }

  if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
    triton::common::TritonJson::WriteBuffer buffer;
    RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("post auto-complete:\n") + buffer.Contents()).c_str());
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelState::AutoCompleteIO(const char* keys){
  // Obtaining Buffer Mappings
  qaicrt::v2::BufferMappings buffer_mappings;
  buffer_mappings = this->qpc_->getBufferMappingsV2();
  // Checking for existing ios
  triton::common::TritonJson::Value existing_ios;
  bool found_ios = ModelConfig().Find(keys, &existing_ios);
  // Setting Max Batch Size
  if (MaxBatchSize() == 0){
    const int max_bs = buffer_mappings[0].ioShapes[0].dims[0];
    SetMaxBatchSize(max_bs);
  }
  triton::common::TritonJson::Value ios(
      ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
  QAicBufferIoTypeEnum_t io_type;
  if (strcmp(keys,"output") == 0) {
    io_type = BUFFER_IO_TYPE_OUTPUT;
  }
  else {
    io_type = BUFFER_IO_TYPE_INPUT;
  }

  for (auto const &m : buffer_mappings){
      triton::common::TritonJson::Value io(
        ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
      if ((m.bufferMapping.ioType == io_type)){
        RETURN_IF_ERROR(io.AddString("name", m.bufferMapping.bufferName));
        RETURN_IF_ERROR(io.AddString("data_type", QAicBufferDataTypeToModelConfigDataType(m.bufferMapping.dataType)));
        triton::common::TritonJson::Value dims(
            ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
        for (size_t i = 0; i < m.ioShapes[0].dims.size();i++) {
          RETURN_IF_ERROR(dims.AppendInt(m.ioShapes[0].dims[i]));
        }
        RETURN_IF_ERROR(io.Add("dims", std::move(dims)));
        RETURN_IF_ERROR(ios.Append(std::move(io)));
      }
  }

  if (found_ios) {
    existing_ios.Swap(ios);
  } else {
    ModelConfig().Add(keys, std::move(ios));
  }

  return nullptr;
}

}}}  // namespace triton::backend::qaic