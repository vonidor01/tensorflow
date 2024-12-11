// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/quantize_op_legalization.h"

#include <cmath>
#include <limits>
#include <string>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_element_type.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/common.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/graph_mapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/util.h"

namespace litert::qnn {

static constexpr absl::string_view kQnnConvertOpTypeName = "Convert";
static constexpr absl::string_view kDefaultQnnOpPackageName = "qti.aisw";
static constexpr absl::string_view kConvertOpFmt = "q_convert_%d";

static constexpr absl::string_view kQnnQuantizeOpTypeName = "Quantize";
static constexpr absl::string_view kQuantizeOpFmt = "quantize_%d";

static constexpr absl::string_view kQnnCastOpTypeName = "Cast";
static constexpr absl::string_view kCastOpFmt = "q_cast_%d";

// SFIXED_8 and UFIXED_8 offset diff
static constexpr int kSUFixed8OffsetDiff = 128;
// SFIXED_16 and UFIXED_16 offset diff
static constexpr int kSUFixed16OffsetDiff = 32768;

LiteRtStatus QuantizeOpLegalization::ConvertLeaglization(
    const litert::Op& src, Qnn_OpConfig_t& dest, GraphMapper& graph_mapper) {
  std::string op_name = absl::StrFormat(kConvertOpFmt, op_counter_++);
  LITERT_RETURN_STATUS_IF_NOT_OK(SetOpInfo(op_name.c_str(),
                                           kDefaultQnnOpPackageName.data(),
                                           kQnnConvertOpTypeName.data(), dest));
  LITERT_RETURN_STATUS_IF_NOT_OK(LegalizeSimpleOp(src, dest, graph_mapper));
  LITERT_LOG(LITERT_INFO, "Legalized quantize op to Convert Op");
  return kLiteRtStatusOk;
}

LiteRtStatus QuantizeOpLegalization::CastLeaglization(
    const litert::Op& src, Qnn_OpConfig_t& dest, GraphMapper& graph_mapper) {
  std::string op_name = absl::StrFormat(kCastOpFmt, op_counter_++);
  LITERT_RETURN_STATUS_IF_NOT_OK(SetOpInfo(op_name.c_str(),
                                           kDefaultQnnOpPackageName.data(),
                                           kQnnCastOpTypeName.data(), dest));
  LITERT_RETURN_STATUS_IF_NOT_OK(LegalizeSimpleOp(src, dest, graph_mapper));
  LITERT_LOG(LITERT_INFO, "Legalized quantize op to Cast Op");
  return kLiteRtStatusOk;
}

LiteRtStatus QuantizeOpLegalization::QuantizeLeaglization(
    const litert::Op& src, Qnn_OpConfig_t& dest, GraphMapper& graph_mapper) {
  std::string op_name = absl::StrFormat(kQuantizeOpFmt, op_counter_++);
  LITERT_RETURN_STATUS_IF_NOT_OK(
      SetOpInfo(op_name.c_str(), kDefaultQnnOpPackageName.data(),
                kQnnQuantizeOpTypeName.data(), dest));
  LITERT_RETURN_STATUS_IF_NOT_OK(LegalizeSimpleOp(src, dest, graph_mapper));
  LITERT_LOG(LITERT_INFO, "Legalized quantize op to Quantize Op");
  return kLiteRtStatusOk;
}

inline bool IsTensorUInt8(Tensor& tensor) {
  return tensor.RankedTensorType().ElementType() == ElementType::UInt8;
}
inline bool IsTensorInt8(Tensor& tensor) {
  return tensor.RankedTensorType().ElementType() == ElementType::Int8;
}
inline bool IsTensorUInt16(Tensor& tensor) {
  return tensor.RankedTensorType().ElementType() == ElementType::UInt16;
}
inline bool IsTensorInt16(Tensor& tensor) {
  return tensor.RankedTensorType().ElementType() == ElementType::Int16;
}

inline bool isTensorPerTensorQuantized(Tensor& tensor) {
  return (IsTensorInt8(tensor) || IsTensorUInt8(tensor) ||
          IsTensorInt16(tensor) || IsTensorUInt16(tensor)) &&
         tensor.QTypeId() == kLiteRtQuantizationPerTensor;
}

inline bool WithinCastRange(Tensor& input_tensor, Tensor& output_tensor,
                            const int offst_diff) {
  return (std::fabs(input_tensor.PerTensorQuantization().scale -
                    output_tensor.PerTensorQuantization().scale)) <
             std::numeric_limits<float>::epsilon() &&
         std::abs(input_tensor.PerTensorQuantization().zero_point -
                  output_tensor.PerTensorQuantization().zero_point) ==
             offst_diff;
}

LiteRtStatus QuantizeOpLegalization::LegalizeOp(const litert::Op& src,
                                                Qnn_OpConfig_t& dest,
                                                GraphMapper& graph_mapper) {
  if (src.Code() != kLiteRtOpCodeTflQuantize) {
    return kLiteRtStatusLegalizeNoMatch;
  }

  if (isTensorPerTensorQuantized(src.Inputs().front()) &&
      isTensorPerTensorQuantized(src.Outputs().front())) {
    if ((IsTensorInt8(src.Inputs().front()) &&
         IsTensorUInt8(src.Outputs().front())) ||
        (IsTensorUInt8(src.Inputs().front()) &&
         IsTensorInt8(src.Outputs().front()))) {
      if (WithinCastRange(src.Inputs().front(), src.Outputs().front(),
                          kSUFixed8OffsetDiff)) {
        LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
            CastLeaglization(src, dest, graph_mapper));
        return kLiteRtStatusOk;
      }
    } else if ((IsTensorInt16(src.Inputs().front()) &&
                IsTensorUInt16(src.Outputs().front())) ||
               (IsTensorUInt16(src.Inputs().front()) &&
                IsTensorInt16(src.Outputs().front()))) {
      if (WithinCastRange(src.Inputs().front(), src.Outputs().front(),
                          kSUFixed16OffsetDiff)) {
        LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
            CastLeaglization(src, dest, graph_mapper));
        return kLiteRtStatusOk;
      }
    }
    LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
        ConvertLeaglization(src, dest, graph_mapper));
    return kLiteRtStatusOk;
  }

  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      QuantizeLeaglization(src, dest, graph_mapper));
  LITERT_LOG(LITERT_INFO, "Legalized quantize op");
  return kLiteRtStatusOk;
}

}  // namespace litert::qnn
