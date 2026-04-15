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

#pragma once

#include <Eigen/Dense>
#include <tuple>
#include <vector>

namespace xllm_service {

// Predictor for predicting TTFT and TPOT
class TimePredictor final {
 public:
  TimePredictor(
      const std::vector<std::pair<int32_t, double>>& ttft_profiling_data,
      const std::vector<std::tuple<int32_t, int32_t, double>>&
          tpot_profiling_data,
      const std::vector<double>& coefficients = {});
  ~TimePredictor() = default;

  double predict_ttft(int32_t length,
                      bool if_need_add_constant_term = true);

  double predict_tpot(int32_t total_length,
                      int32_t batch_size,
                      bool if_need_add_constant_term = true);

  double predict_step_time(int32_t length,
                           int32_t prefix_length = 0,
                           bool if_need_add_constant_term = true);
  double predict_decode_term(int32_t decode_request_num);

  double get_constant_overhead();

 private:
  Eigen::VectorXd ttft_coefficients_;
  Eigen::VectorXd tpot_coefficients_;
  Eigen::VectorXd general_coefficient_;
};

}  // namespace xllm_service
