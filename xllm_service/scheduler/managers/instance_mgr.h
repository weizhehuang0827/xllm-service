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

#include <brpc/channel.h>

#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/macros.h"
#include "common/options.h"
#include "common/threadpool.h"
#include "common/time_predictor.h"
#include "common/types.h"
#include "request/request.h"
#include "scheduler/etcd_client/etcd_client.h"
#include "xllm_rpc_service.pb.h"

namespace xllm_service {
class Scheduler;

class InstanceMgr final {
 public:
  explicit InstanceMgr(const Options& options,
                       const std::shared_ptr<EtcdClient>& etcd_client,
                       const bool is_master_service,
                       Scheduler* scheduler);

  ~InstanceMgr();

  InstanceMetaInfo get_instance_info(const std::string& instance_name);

  bool get_next_instance_pair(Routing* routing);

  std::vector<std::string> get_static_decode_list(
      const std::string& instance_name);

  std::vector<std::string> get_static_prefill_list(
      const std::string& instance_name);

  void get_load_metrics(LoadBalanceInfos* infos);

  std::shared_ptr<brpc::Channel> get_channel(const std::string& instance_name);

  void record_load_metrics_update(const std::string& instance_name,
                                  const proto::LoadMetrics& load_metrics);
  bool upload_load_metrics();

  // update the recent token latency metrics for the corresponding instance
  void update_latency_metrics(const std::string& instance_name,
                              const proto::LatencyMetrics& latency_metrics);

  // update request metrics under different actions
  void update_request_metrics(std::shared_ptr<Request> request,
                              RequestAction action);

  // select instances based on the SLO
  bool select_instance_pair_on_slo(std::shared_ptr<Request> request);

  void set_as_master();

  bool get_min_load_decode_instance(Routing* routing);

  bool get_min_load_prefill_instance(Routing* routing);

  std::unordered_map<std::string, TtftPredictor> get_time_predictors();

  std::unordered_map<std::string, absl::Time> get_prefill_instance_update_time_map();

  std::unordered_map<std::string, std::vector<std::shared_ptr<Request>>> get_prefill_running_requests_map();

  std::unordered_map<std::string, int32_t> get_decode_request_num_map();

  double predict_step_time(std::shared_ptr<Request> request);

  double get_constant_overhead(std::string instance_name);

 private:
  DISALLOW_COPY_AND_ASSIGN(InstanceMgr);

  void init();

  bool create_channel(const std::string& target_uri);
  // use etcd as ServiceDiscovery
  void update_instance_metainfo(const etcd::Response& response,
                                const uint64_t& prefix_len);

  void update_load_metrics(const etcd::Response& response,
                           const uint64_t& prefix_len);

  TimePredictor& get_time_predictor(const std::string& instance_name);

  void flip_prefill_to_decode(std::string& instance_name);
  void flip_decode_to_prefill(std::string& instance_name);

  // Register a new instance with all necessary resources and connections
  bool register_instance(const std::string& name, InstanceMetaInfo& info);
  // Remove an instance and clean up its resources and connections
  void deregister_instance(const std::string& name);
  // Initialize internal resources for an instance (predictors, metrics)
  void add_instance_resources(const std::string& name,
                              const InstanceMetaInfo& info);
  // Release internal resources for an instance
  void remove_instance_resources(const std::string& name);
  // Internal helper to establish bidirectional links with peer instances
  bool link_instance_internal(const std::string& name, InstanceMetaInfo& info);
  // Internal helper to break bidirectional links with peer instances
  void unlink_instance_internal(const std::string& name,
                                const InstanceMetaInfo& info);
  // Add instance to prefill or decode index according to its type
  void add_instance_to_index(const std::string& name, InstanceMetaInfo& info);
  // Remove instance from prefill or decode index
  void remove_instance_from_index(const std::string& name,
                                  const InstanceMetaInfo& info);
  bool call_link_instance(const std::string& target_rpc_addr,
                          const InstanceMetaInfo& peer_info);
  bool call_unlink_instance(const std::string& target_rpc_addr,
                            const InstanceMetaInfo& peer_info);

 private:
  Options options_;

  bool exited_ = false;
  bool use_etcd_ = false;
  std::atomic_bool is_master_service_ = false;

  std::shared_ptr<EtcdClient> etcd_client_;

  std::shared_mutex inst_mutex_;
  std::unordered_map<std::string, InstanceMetaInfo> instances_;
  std::vector<std::string> prefill_index_;
  std::vector<std::string> decode_index_;
  uint64_t next_prefill_index_ = 0;
  uint64_t next_decode_index_ = 0;

  std::shared_mutex load_metric_mutex_;
  std::unordered_map<std::string, LoadMetrics> load_metrics_;
  std::unordered_map<std::string, std::shared_ptr<brpc::Channel>>
      cached_channels_;

  std::mutex update_mutex_;
  std::unordered_map<std::string, LoadMetrics> updated_metrics_;
  std::unordered_set<std::string> removed_instance_;

  // "instance name" -> "TimePredictor" map
  std::mutex time_predictor_mutex_;
  std::unordered_map<std::string, TimePredictor> time_predictors_;

  // Record the latest token latency metrics for each instance, including TTFT
  // and TBT.
  std::mutex latency_metrics_mutex_;
  std::unordered_map<std::string, LatencyMetrics> latency_metrics_;

  // Record the request metrics for each instance, including prefill token
  // count, prefill request count, estimated prefill execution time, decode
  // token count, and decode request count.
  std::mutex request_metrics_mutex_;
  std::unordered_map<std::string, RequestMetrics> request_metrics_;

  // not own
  // NOTE: need to refactor with scheduler in future
  Scheduler* scheduler_;
  std::atomic<size_t> satisfied_slo_request_num_{0};
  std::atomic<size_t> total_request_num_{0};
  std::atomic<size_t> prev_satisfied_slo_request_num_{0};
  std::atomic<size_t> prev_actual_satisfied_slo_request_num_{0};
  std::atomic<size_t> actual_satisfied_slo_request_num_{0};

  // for defualt instance
  std::unordered_map<std::string, int32_t> decode_request_num_map_;
  std::shared_mutex update_map_mutex_;
  std::unordered_map<std::string, absl::Time> update_time_map_;
  using RunningRequestMap =
      std::unordered_map<std::string, std::unordered_map<std::string, std::shared_ptr<Request>>>;
  // currently only support prefill instance
  RunningRequestMap running_requests_map_;

  ThreadPool threadpool_;
};

}  // namespace xllm_service
