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

#include "common/global_gflags.h"
#include "common/macros.h"
#include "loadbalance_policy.h"
#include <unordered_map>
#include <vector>
#include <string>
#include <atomic>
#include <thread>
#include <shared_mutex>

namespace xllm_service {

class PriorityRouting final : public LoadBalancePolicy {
  using RunningRequestMap = std::unordered_map<std::string, std::vector<std::shared_ptr<Request>>>;
 public:
  PriorityRouting(std::shared_ptr<InstanceMgr> instance_mgr, const Options& options)
      : LoadBalancePolicy(instance_mgr, options) {
        if_pd_disagg_ = FLAGS_if_pd_disagg;
      }

  virtual ~PriorityRouting() = default;

  double get_latency_budget_and_request_order(TtftPredictor& ttft_predictor, std::vector<std::shared_ptr<Request>>& running_queue);

  double get_raw_total_exec_time(std::vector<std::shared_ptr<Request>>& running_queue);

  std::string get_max_gain_instance(std::unordered_map<std::string, int32_t>& decode_request_num_map, std::unordered_map<std::string, absl::Time>& update_time_map, std::unordered_map<std::string, TtftPredictor>& ttft_predictors, RunningRequestMap& prefill_running_requests_map, std::unordered_map<std::string,std::string>& strategies, std::unordered_map<std::string,double>& budgets, std::shared_ptr<Request> request);

  bool select_instances_pair(std::shared_ptr<Request> request) override;

  int32_t get_gain_for_running_queue(bool is_pre, std::vector<std::shared_ptr<Request>>& running_queue, double latency_budget, double constant_overhead, double executed_time, double total_exec_time);

  double get_estimate_exec_time(bool is_pre, double executed_time, double exec_time, double constant_overhead, int32_t num_sequences, double latency_budget);

 private:
  DISALLOW_COPY_AND_ASSIGN(PriorityRouting);

  std::unordered_map<std::string, TtftPredictor> ttft_predictors_;

  std::atomic<size_t> num_warmup_request_num_{0};

  bool if_pd_disagg_ = false;

  std::mutex request_metrics_mutex_;
  // std::shared_mutex request_metrics_mutex_;



  
};

}  // namespace xllm_service
