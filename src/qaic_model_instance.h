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
// Copyright (c) 2023 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

namespace triton { namespace backend { namespace qaic {

class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);

  virtual ~ModelInstanceState() = default;

  QStatus ExecuteInference(
      std::vector<QBuffer>& input_buffers, std::vector<QBuffer>& output_buffers,
      ::qaic::rt::shInferenceHandle& completed_inf_handle);

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }
  // conversion method for string datatype from config file to triton datatypes
  TRITONSERVER_datatype_enum GetTritonDatatype(std::string d_type);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance)
      : BackendModelInstance(model_state, triton_model_instance),
        model_state_(model_state)
  {
  }

  ModelState* model_state_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }
  return nullptr;  // success
}

TRITONSERVER_datatype_enum
ModelInstanceState::GetTritonDatatype(std::string d_type)
{
  if (d_type.compare("TYPE_BOOL") == 0)
    return TRITONSERVER_TYPE_BOOL;
  if (d_type.compare("TYPE_UINT8") == 0)
    return TRITONSERVER_TYPE_UINT8;
  if (d_type.compare("TYPE_UINT16") == 0)
    return TRITONSERVER_TYPE_UINT16;
  if (d_type.compare("TYPE_UINT32") == 0)
    return TRITONSERVER_TYPE_UINT32;
  if (d_type.compare("TYPE_UINT64") == 0)
    return TRITONSERVER_TYPE_UINT64;
  if (d_type.compare("TYPE_INT8") == 0)
    return TRITONSERVER_TYPE_INT8;
  if (d_type.compare("TYPE_INT16") == 0)
    return TRITONSERVER_TYPE_INT16;
  if (d_type.compare("TYPE_INT32") == 0)
    return TRITONSERVER_TYPE_INT32;
  if (d_type.compare("TYPE_INT64") == 0)
    return TRITONSERVER_TYPE_INT64;
  if (d_type.compare("TYPE_FP16") == 0)
    return TRITONSERVER_TYPE_FP16;
  if (d_type.compare("TYPE_FP32") == 0)
    return TRITONSERVER_TYPE_FP32;
  if (d_type.compare("TYPE_FP64") == 0)
    return TRITONSERVER_TYPE_FP64;
  if (d_type.compare("TYPE_STRING") == 0)
    return TRITONSERVER_TYPE_BYTES;
  return TRITONSERVER_TYPE_INVALID;
}

}}}  // namespace triton::backend::qaic