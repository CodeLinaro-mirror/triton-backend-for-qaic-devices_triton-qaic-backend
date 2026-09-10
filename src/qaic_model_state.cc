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
    : BackendModel(triton_model),
      shape_initialized_(false),
      has_specializations_(false)
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
  std::string device_id_str;
  int set_size;
  int no_of_activations;
  bool device_id_specified = false;
  triton::common::TritonJson::Value params;

  if (this->ModelConfig().Find("parameters", &params)) {
    //Check if device_id parameter exists.
    triton::common::TritonJson::Value device_param;
    if (params.Find("device_id", &device_param)) {
      THROW_IF_BACKEND_MODEL_ERROR(device_param.MemberAsString("string_value", &device_id_str));
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
      (std::string(" device_id: ") + device_id_str).c_str());}
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

  std::optional<std::string> aic_device_id;

  // Initialize device id for auto-device picker in case default is selected/no device configured.
  if (device_id_specified) {
  aic_device_id = device_id_str;
  LOG_MESSAGE(
    TRITONSERVER_LOG_INFO,
    (std::string("Using device ID: ") + device_id_str).c_str());
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

  // Extract network specializations
  qaicrt::v2::BufferMappings buffer_mappings_v2;
  buffer_mappings_v2 = (this->qpc_)->getBufferMappingsV2();

  // Check for multiple specializations
  if (!buffer_mappings_v2.empty() &&
      buffer_mappings_v2[0].ioShapes.size() > 1) {
    this->has_specializations_ = true;
    size_t num_specs = buffer_mappings_v2[0].ioShapes.size();

    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        (std::string("+----------------------------+")).c_str());
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        (std::string("| Network Specializations    |")).c_str());
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        (std::string("+----------------------------+")).c_str());
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        (std::string(" Found ") + std::to_string(num_specs) +
         " specializations").c_str());

    // Extract batch sizes and log dimensions for each specialization
    std::string batch_sizes_str;
    for (size_t i = 0; i < num_specs; i++) {
      uint32_t batch_size = buffer_mappings_v2[0].ioShapes[i].dims[0];
      this->available_batch_sizes_.push_back(batch_size);
      this->batch_size_to_spec_index_[batch_size] = i;

      if (i > 0) batch_sizes_str += ", ";
      batch_sizes_str += std::to_string(batch_size);

      LOG_MESSAGE(TRITONSERVER_LOG_INFO,
          (std::string(" Specialization [") + std::to_string(i) +
           "] batch_size=" + std::to_string(batch_size)).c_str());

      for (size_t buf_idx = 0; buf_idx < buffer_mappings_v2.size(); buf_idx++) {
        const auto& mapping = buffer_mappings_v2[buf_idx];
        std::string io_type_str = (mapping.bufferMapping.ioType == BUFFER_IO_TYPE_INPUT) ? "INPUT " : "OUTPUT";
        std::string dims_str = "[";
        for (size_t d = 0; d < mapping.ioShapes[i].dims.size(); d++) {
          if (d > 0) dims_str += ", ";
          dims_str += std::to_string(mapping.ioShapes[i].dims[d]);
        }
        dims_str += "]";
        LOG_MESSAGE(TRITONSERVER_LOG_INFO,
            (std::string("   ") + io_type_str + " '" + mapping.bufferMapping.bufferName +
             "': " + dims_str).c_str());
      }
    }

    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        (std::string(" Available batch sizes: [") + batch_sizes_str + "]").c_str());

    // Store all specializations
    this->all_specializations_.push_back(buffer_mappings_v2);

    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        (std::string("+----------------------------+")).c_str());
  } else {
    this->has_specializations_ = false;
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        "No network specializations found - using default batch size");
  }

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

  // Set max_batch_size before calling AutoCompleteIO
  qaicrt::v2::BufferMappings buffer_mappings = this->qpc_->getBufferMappingsV2();
  int max_bs = GetMaxBatchSizeFromBufferMappings(buffer_mappings);
  if (max_bs > 0 && max_bs != MaxBatchSize()) {
    SetMaxBatchSize(max_bs);
    // update the JSON config
    triton::common::TritonJson::Value max_batch_size_value;
    ModelConfig().Find("max_batch_size", &max_batch_size_value);
    max_batch_size_value.SetInt(max_bs);
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        (std::string("Auto-completed max_batch_size=") +
         std::to_string(max_bs) + " (maximum across all specializations, overriding config value)").c_str());
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
    int max_bs = GetMaxBatchSizeFromBufferMappings(buffer_mappings);
    if (max_bs > 0) {
      SetMaxBatchSize(max_bs);
      LOG_MESSAGE(TRITONSERVER_LOG_INFO,
          (std::string("Auto-completed max_batch_size=") +
           std::to_string(max_bs) + " (maximum across all specializations)").c_str());
    }
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
        // Skip the first dimension (batch dimension) for Triton config
        // Triton expects dims without the batch dimension
        // Standard practice for all Triton backends
        for (size_t i = 1; i < m.ioShapes[0].dims.size(); i++) {
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

// Get max batch size from buffer mappings
int ModelState::GetMaxBatchSizeFromBufferMappings(
    const qaicrt::v2::BufferMappings& buffer_mappings) const {
  int max_bs = 0;
  // Find first input buffer and check all its specializations
  for (const auto& mapping : buffer_mappings) {
    if (mapping.bufferMapping.ioType == BUFFER_IO_TYPE_INPUT) {
      for (const auto& io_shape : mapping.ioShapes) {
        if (!io_shape.dims.empty()) {
          int batch_size = io_shape.dims[0];
          if (batch_size > max_bs) {
            max_bs = batch_size;
          }
        }
      }
      // Only need to check first input buffer to get max batch size
      // batch dimension for batched models, or first data dimension for non-batched models
      break;
    }
  }
  return max_bs;
}

// Function to get element size from data type
static uint32_t GetElementSizeFromDataType(QAicBufferDataTypeEnum_t dataType) {
  switch (dataType) {
    case BUFFER_DATA_TYPE_FLOAT:      // 32-bit float
    case BUFFER_DATA_TYPE_INT32Q:     // 32-bit quantized
    case BUFFER_DATA_TYPE_INT32I:     // 32-bit index
      return 4;
    case BUFFER_DATA_TYPE_FLOAT16:    // 16-bit float
    case BUFFER_DATA_TYPE_INT16Q:     // 16-bit quantized
    case BUFFER_DATA_TYPE_BFLOAT16:   // 16-bit bfloat
      return 2;
    case BUFFER_DATA_TYPE_INT8Q:      // 8-bit quantized
    case BUFFER_DATA_TYPE_UINT8Q:     // unsigned 8-bit quantized
    case BUFFER_DATA_TYPE_INT8:       // 8-bit
    case BUFFER_DATA_TYPE_UINT8:      // unsigned 8-bit
      return 1;
    case BUFFER_DATA_TYPE_INT64I:     // 64-bit index
    case BUFFER_DATA_TYPE_FLOAT64C:   // 64-bit complex float
      return 8;
    default:
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
          (std::string("Unknown data type: ") + std::to_string(dataType)).c_str());
      return 4;  // Default to 4 bytes (FP32) which prevents failure while still alerting the problem via logs
  }
}

// Get specialized buffer size for a given specialization
size_t ModelState::GetSpecializedBufferSize(size_t spec_index, size_t buffer_index) const {
  if (!has_specializations_ || all_specializations_.empty()) {
    if (buffer_index < qpc_inputs_.size()) {
      auto it = qpc_inputs_.begin();
      std::advance(it, buffer_index);
      return it->size;
    } else {
      size_t output_idx = buffer_index - qpc_inputs_.size();
      if (output_idx >= qpc_outputs_.size()) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
            (std::string("Output buffer index ") + std::to_string(output_idx) +
             " out of range (total outputs: " + std::to_string(qpc_outputs_.size()) + ")").c_str());
        return 0;
      }
      auto it = qpc_outputs_.begin();
      std::advance(it, output_idx);
      return it->size;
    }
  }

  const auto& buffer_mappings = all_specializations_[0];

  if (buffer_index >= buffer_mappings.size()) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
        (std::string("Buffer index ") + std::to_string(buffer_index) +
         " out of range").c_str());
    return 0;
  }

  const auto& mapping = buffer_mappings[buffer_index];
  if (spec_index >= mapping.ioShapes.size()) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
        (std::string("Specialization index ") + std::to_string(spec_index) +
         " out of range").c_str());
    return 0;
  }

  const auto& io_shape = mapping.ioShapes[spec_index];
  size_t total_elements = 1;
  for (size_t i = 0; i < io_shape.dims.size(); i++) {
    total_elements *= io_shape.dims[i];
  }

  // Get element size and multiply to get total bytes
  uint32_t element_size = GetElementSizeFromDataType(io_shape.dataType);
  return total_elements * element_size;
}

// Get specialized dimensions for all buffers
std::vector<std::pair<uint32_t, std::vector<uint32_t>>>
ModelState::GetSpecializedDimensions(size_t spec_index) const {
  std::vector<std::pair<uint32_t, std::vector<uint32_t>>> buffer_dims;

  if (!has_specializations_ || all_specializations_.empty()) {
    return buffer_dims;
  }

  const auto& buffer_mappings = all_specializations_[0];

  for (size_t buf_idx = 0; buf_idx < buffer_mappings.size(); buf_idx++) {
    const auto& mapping = buffer_mappings[buf_idx];

    if (spec_index >= mapping.ioShapes.size()) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
          (std::string("Specialization index ") + std::to_string(spec_index) +
           " out of range for buffer " + std::to_string(buf_idx)).c_str());
      continue;
    }

    const auto& io_shape = mapping.ioShapes[spec_index];

    uint32_t element_size = GetElementSizeFromDataType(io_shape.dataType);

    // Get dimensions
    std::vector<uint32_t> dims;
    for (const auto& dim : io_shape.dims) {
      dims.push_back(static_cast<uint32_t>(dim));
    }

    buffer_dims.push_back(std::make_pair(element_size, dims));
  }

  return buffer_dims;
}

// Specialization Selection Method
TRITONSERVER_Error* ModelState::SelectSpecialization(
    uint32_t requested_batch_size,
    size_t& spec_index,
    uint32_t& actual_batch_size) const {

  if (!has_specializations_) {
    // No specializations, use default
    spec_index = 0;
    actual_batch_size = requested_batch_size;
    return nullptr;
  }

  // Check for exact match
  auto it = batch_size_to_spec_index_.find(requested_batch_size);
  if (it != batch_size_to_spec_index_.end()) {
    spec_index = it->second;
    actual_batch_size = requested_batch_size;
    return nullptr;
  }

  // No exact match - return error
  std::string available_sizes;
  for (size_t i = 0; i < available_batch_sizes_.size(); i++) {
    if (i > 0) available_sizes += ", ";
    available_sizes += std::to_string(available_batch_sizes_[i]);
  }

  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INVALID_ARG,
      (std::string("Requested batch size ") +
       std::to_string(requested_batch_size) +
       " not supported. Available: [" + available_sizes + "]").c_str());
}

}}}  // namespace triton::backend::qaic