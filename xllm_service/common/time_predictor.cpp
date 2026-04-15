/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "time_predictor.h"
#include <glog/logging.h>
#include <sstream>

namespace {
constexpr int32_t kDegree = 2;

std::string EigenVectorToString(const Eigen::VectorXd& vec) {
  std::ostringstream oss;
  oss << "[";
  for (int32_t i = 0; i < vec.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << vec(i);
  }
  oss << "]";
  return oss.str();
}
}  // namespace

namespace xllm_service {

TimePredictor::TimePredictor(
    const std::vector<std::pair<int32_t, double>>& ttft_profiling_data,
    const std::vector<std::tuple<int32_t, int32_t, double>>&
        tpot_profiling_data,
    const std::vector<double>& coefficients) {
  if (!ttft_profiling_data.empty()) {
    // construct Vandermonde matrix
    int32_t m = ttft_profiling_data.size();
    int32_t n = kDegree + 1;
    Eigen::MatrixXd matrix(m, n);
    for (int32_t i = 0; i < m; ++i) {
      for (int32_t j = 0; j < n; ++j) {
        matrix(i, j) = std::pow(ttft_profiling_data[i].first, j);
      }
    }

    // construct target vector
    Eigen::VectorXd target(m);
    for (int32_t i = 0; i < m; ++i) {
      target(i) = ttft_profiling_data[i].second;
    }

    // get coefficients
    ttft_coefficients_ = matrix.colPivHouseholderQr().solve(target);
  } else {
    ttft_coefficients_ = Eigen::VectorXd::Zero(1);
  }

  if (!tpot_profiling_data.empty()) {
    int32_t m = tpot_profiling_data.size();
    int32_t n = kDegree + 1;
    Eigen::MatrixXd matrix(m, n);
    for (int32_t i = 0; i < m; ++i) {
      int32_t avg_length = std::get<0>(tpot_profiling_data[i]);
      int32_t batch_size = std::get<1>(tpot_profiling_data[i]);

      matrix(i, 0) = 1.0;  // the index 0 is always for constant
      matrix(i, 1) = batch_size;
      matrix(i, 2) = batch_size * (avg_length - 1);
    }

    // construct target vector
    Eigen::VectorXd target(m);
    for (int32_t i = 0; i < m; ++i) {
      target(i) = std::get<2>(tpot_profiling_data[i]);
    }

    // get coefficients
    tpot_coefficients_ = matrix.colPivHouseholderQr().solve(target);
  } else {
    tpot_coefficients_ = Eigen::VectorXd::Zero(3);
  }

  if (!coefficients.empty()) {
    general_coefficient_ = Eigen::VectorXd::Zero(coefficients.size());
    for (size_t i = 0; i < coefficients.size(); ++i) {
      general_coefficient_(i) = coefficients[i];
    }
  } else if (ttft_coefficients_.size() > 0) {
    general_coefficient_ = ttft_coefficients_;
  } else {
    general_coefficient_ = Eigen::VectorXd::Zero(0);
  }

  LOG(INFO) << "TimePredictor initialized. input_coefficients_size="
            << coefficients.size()
            << ", general_coefficient=" << EigenVectorToString(general_coefficient_)
            << ", ttft_coefficients=" << EigenVectorToString(ttft_coefficients_)
            << ", tpot_coefficients=" << EigenVectorToString(tpot_coefficients_);
}

double TimePredictor::get_constant_overhead() {
  double result = 0.0;
  if (general_coefficient_.size() > 0) {
    result = general_coefficient_(0);
  } else if (ttft_coefficients_.size() > 0) {
    result = ttft_coefficients_(0);
  }
  if (result < 0) {
    LOG(ERROR) << "Negative constant term: " << result;
    result = 0.0;
  }
  return result;
}

double TimePredictor::predict_ttft(int32_t length,
                                   bool if_need_add_constant_term) {
  double result = 0.0;
  if (if_need_add_constant_term && ttft_coefficients_.size() > 0) {
    result = ttft_coefficients_(0);
  }

  double power = length;
  for (int32_t i = 1; i < ttft_coefficients_.size(); ++i) {
    result += ttft_coefficients_(i) * power;
    power *= length;
  }

  return result;
}

double TimePredictor::predict_tpot(int32_t total_length,
                                   int32_t batch_size,
                                   bool if_need_add_constant_term) {
  double result = 0.0;
  if (if_need_add_constant_term && tpot_coefficients_.size() > 0) {
    result = tpot_coefficients_(0);
  }

  if (tpot_coefficients_.size() > 1) {
    result += tpot_coefficients_(1) * batch_size;
  }

  if (tpot_coefficients_.size() > 2) {
    result += tpot_coefficients_(2) * total_length;
  }

  return result;
}

double TimePredictor::predict_step_time(int32_t length,
                                        int32_t prefix_length,
                                        bool if_need_add_constant_term) {
  if (general_coefficient_.size() == 0) {
    if (prefix_length > 0) {
      return predict_tpot(length, prefix_length, if_need_add_constant_term);
    }
    return predict_ttft(length, if_need_add_constant_term);
  }

  // Compat path: keep old decode-style invocation
  // (length=0, prefix_length=batch_size) working after API switch.
  int32_t effective_length = length;
  int32_t effective_prefix_length = prefix_length;
  if (length <= 0 && prefix_length > 0) {
    effective_length = prefix_length;
    effective_prefix_length = 0;
  }

  double result = 0.0;
  if (if_need_add_constant_term && general_coefficient_.size() > 0) {
    result = general_coefficient_(0);
  }

  if (general_coefficient_.size() >= 5) {
    int32_t diff = effective_length - effective_prefix_length;
    result += (general_coefficient_(1) * diff * diff +
               general_coefficient_(2) * diff +
               general_coefficient_(3) * diff * effective_prefix_length +
               general_coefficient_(4) * effective_prefix_length);
  } else {
    int32_t effective_token_length = effective_length - effective_prefix_length;
    double power = effective_token_length;
    for (int32_t i = 1; i < general_coefficient_.size(); ++i) {
      result += general_coefficient_(i) * power;
      power *= effective_token_length;
    }
  }

  if (result < 0) {
    LOG(ERROR) << "Negative step time prediction: " << result
               << ". Input param: length:" << length
               << " prefix_length:" << prefix_length;
    result = 0.0;
  }
  return result;
}

double TimePredictor::predict_decode_term(int32_t decode_request_num) {
  if (general_coefficient_.size() <= 2) {
    LOG(ERROR) << "general_coefficient size is less than 3 for decode term "
                  "prediction. size="
               << general_coefficient_.size();
    return 0.0;
  }

  double result = general_coefficient_(2) * decode_request_num;
  if (result < 0) {
    LOG(ERROR) << "Negative decode term prediction: " << result
               << ". Input param: decode_request_num:" << decode_request_num;
    result = 0.0;
  }
  return result;
}

}  // namespace xllm_service
