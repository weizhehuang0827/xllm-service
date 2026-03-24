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

#include "priority_routing.h"
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include "request/priority_comparator.h"

namespace xllm_service {

double PriorityRouting::get_latency_budget_and_request_order(TtftPredictor& ttft_predictor, std::vector<std::shared_ptr<Request>>& running_queue){
  
  double latency_budget = 0;
  for (auto& request : running_queue){
    // request->set_estimated_latency(ttft_predictor.predict_step_time(request->token_ids.size(), false));
    request->set_urgency(Urgency::NORMAL);
    request->set_elapsed_time_ms();
    // request->set_deadline_ms();
  }


    //这些策略不需要感知slo和latency
  if (options_.priority_strategy() == "fcfs" || options_.priority_strategy() == "fcfs1" || options_.priority_strategy() == "priority"){
    std::sort(running_queue.begin(), running_queue.end(), create_sort_comparator(options_.priority_strategy()));
    if (if_pd_disagg_){ //fcfs
      latency_budget = std::numeric_limits<int32_t>::max();
    }
    else { //chunkedprefill+fcfs/priority
      latency_budget = running_queue.size() > 0 ? running_queue[0]->get_tpot_slo_ms() : 100;
    }
  }

  auto constant_overhead = ttft_predictor.get_constant_overhead();
  
  // 获得decode请求的remaining_time最小值
  // 决定latency budget
  double exec_latency_sum = 0;

  int32_t min_remaining_time = std::numeric_limits<int32_t>::max();

  if (!if_pd_disagg_ && running_queue.size() > 0){
    min_remaining_time = running_queue[0]->get_tpot_slo_ms();
  }

  auto min_remaining_time_iter = running_queue.end();
  for (auto it = running_queue.begin(); it != running_queue.end(); it++){
    auto request =  *it;

    auto remaining_time = request->get_remaining_time();
    //remaining time需要大于估计的执行时间，否则已经完不成了，目前是跳过
    // TODO:后续可以加个early reject机制 like mooncake
    exec_latency_sum += request->get_estimated_latency();

    if (remaining_time < request->get_estimated_latency() + constant_overhead){
      request->set_urgency(Urgency::URGENT); //已经超时的，放在最后面
      continue;
    }
    if (remaining_time < min_remaining_time){ 
      min_remaining_time = static_cast<int32_t>(remaining_time); 
      min_remaining_time_iter = it;
    }
  }
  double total_exec_time = exec_latency_sum;
  
  int32_t yuzhi = static_cast<int32_t>(4 * constant_overhead); 
  latency_budget = std::max(min_remaining_time, yuzhi);

  // 排序策略
  for (auto& request : running_queue){
    // 下个batch就要超时的请求优先
    if (request->get_remaining_time()< total_exec_time*latency_budget / (latency_budget - constant_overhead)){
      request->set_urgency(Urgency::URGENT); //下一个回合就超时的，所以尽量这回合就加进去
    }
    else {
      request->set_urgency(Urgency::NORMAL); //正常的
    }
  }

  // 排序
  std::sort(running_queue.begin(), running_queue.end(), create_sort_comparator(options_.priority_strategy()));
  return latency_budget;
}

double PriorityRouting::get_raw_total_exec_time(std::vector<std::shared_ptr<Request>>& running_queue){
  double exec_queue_time = 0;
  for (int32_t i = 0 ; i<running_queue.size(); i++){
    exec_queue_time += running_queue[i]->get_estimated_latency();
    // running_queue[i]->set_exec_queue_time(exec_queue_time);
  }
  return exec_queue_time;
}

double PriorityRouting::get_estimate_exec_time(bool is_pre, double executed_time, double exec_time, double constant_overhead, int32_t num_sequences, double latency_budget){
  // 假设撑满整个ttft slo
  // double result = exec_time + ceil(exec_time / (ttft_slo - constant_overhead)) * constant_overhead;
  // double result = exec_time + ceil(exec_time / (latency_budget - constant_overhead)) * constant_overhead;
  // 假设是一条一条做
  double result = 0.0;
  if (if_pd_disagg_){
    if (options_.priority_strategy() == "fcfs1" || options_.priority_strategy() == "fcfs"){
      // constrain seq = 1
      if (options_.priority_strategy() == "fcfs1"){
        result = exec_time + constant_overhead*num_sequences;
      }
      // no constrain seq
      else {
        // result = exec_time + constant_overhead;
        result = is_pre? (exec_time + constant_overhead) : (exec_time + constant_overhead*num_sequences);
      }

    }
    else {
      // result = exec_time + ceil(exec_time / (latency_budget - constant_overhead)) * constant_overhead;
      result = exec_time + constant_overhead*num_sequences;
    }
  }
  else{
    if (options_.priority_strategy() == "priority" || options_.priority_strategy() == "fcfs"){
      //chunkedprefill + fcfs/priority
      // result = ceil(exec_time / (latency_budget - constant_overhead)) * latency_budget;
      result = exec_time + constant_overhead*num_sequences;
    }
    else {
      //urgency_density
      result = exec_time + constant_overhead*num_sequences;
    }
    
  }
  result -= executed_time;
  // LOG(INFO) << "exec_time: " << exec_time << ", ttft_slo: " << ttft_slo << ", constant_overhead: " << constant_overhead << ", result: " << result;
  return result;
}


int32_t PriorityRouting::get_gain_for_running_queue(bool is_pre, std::vector<std::shared_ptr<Request>>& running_queue, double latency_budget, double constant_overhead, double executed_time, double total_exec_time){
  int32_t gain = 0;
  double alpha = is_pre? 1:0.8;
  if (if_pd_disagg_){
    if (options_.priority_strategy() == "fcfs1" || options_.priority_strategy() == "fcfs"){
      // constrain seq = 1
      if (options_.priority_strategy() == "fcfs1"){
        if (!is_pre){
          auto request = running_queue.back();
          if (alpha*request->get_remaining_time() > total_exec_time){
            gain+=request->get_ttft_priority_weight();
          }
        }
      }

      // no constrain seq
      else {
        for (int32_t i = 0 ; i<running_queue.size(); i++){
          auto request = running_queue[i];
          if (alpha*request->get_remaining_time() > total_exec_time){
            gain+=request->get_ttft_priority_weight();
          }
        }
      }
    }
    // other strategies: urgency_density
    else {
      for (int32_t i = 0 ; i<running_queue.size(); i++){
        auto request = running_queue[i];
        if (alpha*request->get_remaining_time() > total_exec_time){
          gain+=request->get_ttft_priority_weight();
        }
      }
    }

  }
  else {
    // pd mixed
    // use total time
    for (int32_t i = 0 ; i<running_queue.size(); i++){
      auto request = running_queue[i];
      if (alpha*request->get_remaining_time() > total_exec_time){
        gain+=request->get_ttft_priority_weight();
      }
    }

    // std::vector<double> running_queue_exec;
    // for (int32_t i = 0 ; i<running_queue.size(); i++){
    //   exec_queue_time += running_queue[i]->get_estimated_latency();
    //   double actual_time;
    //   double zhengchu = ceil(exec_queue_time / (latency_budget - constant_overhead));
    //   double duochu =  zhengchu*(latency_budget - constant_overhead) - exec_queue_time;
    //   int32_t j = i+1;
    //   while (j <  running_queue.size() && duochu > 0){
    //     duochu -=  running_queue[j]->get_estimated_latency();
    //     j += 1;
    //   }
    //   if (duochu > 0){
    //     actual_time = zhengchu*constant_overhead+ exec_queue_time;
    //   }
    //   else{
    //     actual_time = zhengchu * latency_budget;
    //   }
    //   actual_time -= executed_time;

    //   // double actual_time = ceil(exec_queue_time / (latency_budget - constant_overhead)) * latency_budget;
    //   if (0.8*running_queue[i]->get_remaining_time() > actual_time){
    //     gain+=running_queue[i]->get_ttft_priority_weight();
    //   }
    // }
  }
  return gain;
}


std::string PriorityRouting::get_max_gain_instance(std::unordered_map<std::string, int32_t>& decode_request_num_map, std::unordered_map<std::string, absl::Time>& update_time_map, std::unordered_map<std::string, TtftPredictor>& ttft_predictors, RunningRequestMap& prefill_running_requests_map,std::unordered_map<std::string,std::string>& strategies,std::unordered_map<std::string,double>& budgets, std::shared_ptr<Request> request){
  double up_ratio = 0.9;
  double low_ratio = 0.1;
  int32_t ttft_slo = request->get_ttft_slo_ms();
  double max_delta_gain = std::numeric_limits<double>::lowest();
  std::string max_gain_instance = "";
  std::string max_gain_instance_strategy = "";
  // 未来可以并行计算不同实例上的delta_gain，然后选gain最大的
  std::vector<double> running_queue_exec_time; 
  std::vector<double> running_queue_exec_time_insert; 
  std::vector<int32_t> running_queue_delta_gain; 
  std::vector<std::string> instances_name; 
  
  LOG(INFO) << "request info: " <<  "prompt length: " << request->token_ids.size();
  
  for (auto& pair : prefill_running_requests_map){

    auto budget_it = budgets.find(pair.first);
    if (budget_it == budgets.end()) {
      LOG(ERROR) << "Failed to find instance budgets, instance name : "
                 << pair.first;
      continue;
    }
    auto latency_budget = budget_it->second;

    auto it = ttft_predictors.find(pair.first);
    if (it == ttft_predictors.end()) {
      LOG(ERROR) << "Failed to find instance ttft predictor, instance name : "
                 << pair.first;
      continue;
    }
    auto time_it = update_time_map.find(pair.first);
    if (time_it == update_time_map.end()) {
      LOG(ERROR) << "Failed to find instance update time, instance name : "
                 << pair.first;
      continue;
    }
    auto decode_num_it = decode_request_num_map.find(pair.first);
    if (decode_num_it == decode_request_num_map.end()) {
      LOG(ERROR) << "Failed to find instance decode num, instance name : "
                 << pair.first;
      continue;
    }

    auto& ttft_predictor = it->second;
    auto& running_queue = pair.second;
    double executed_time =0.0;
    if (running_queue.size() >0) {
      executed_time = absl::ToDoubleSeconds(absl::Now() - time_it->second) * 1000; // ms
    }
    int32_t decode_request_num = decode_num_it->second;
    double constant_overhead = ttft_predictor.get_constant_overhead();
    if (!if_pd_disagg_){
      constant_overhead += ttft_predictor.predict_step_time(decode_request_num, false); // only for disagg_
    }

    instances_name.push_back(pair.first);

    // 计算当前这个request的指标
    request->set_estimated_latency(ttft_predictor.predict_step_time(request->token_ids.size(), false));
    request->set_elapsed_time_ms();
    // if (request->get_remaining_time()< total_exec_time*latency_budget / (latency_budget - constant_overhead)){
    //   request->set_urgency(Urgency::URGENT); //下一个回合就超时的，所以尽量这回合就加进去
    // }
    // else {
    //   request->set_urgency(Urgency::NORMAL); //正常的
    // }

    // 获取排序以后的下标
    auto cur_request_pos = std::upper_bound(running_queue.begin(), running_queue.end(), request, create_sort_comparator(options_.priority_strategy()));



    // 估算前后的收益
    // 估算插入前后的总体执行时间
    int32_t pre_gain=0,post_gain=0;
    auto pre_raw_total_exec_time = get_raw_total_exec_time(running_queue);
    auto pre_total_exec_time = get_estimate_exec_time(true,executed_time, pre_raw_total_exec_time,constant_overhead,running_queue.size(),latency_budget);
    running_queue_exec_time.push_back(pre_total_exec_time);
    pre_gain = get_gain_for_running_queue(true,running_queue, latency_budget, constant_overhead, executed_time,pre_total_exec_time);
    LOG(INFO) << "In instance " << pair.first << ", running queue size: " << running_queue.size() << ", total exec_time: " << pre_total_exec_time << ", current request position: " << (cur_request_pos - running_queue.begin());

    running_queue.insert(cur_request_pos,request); // use sort to insert position
    auto post_raw_total_exec_time = get_raw_total_exec_time(running_queue);
    auto post_total_exec_time = get_estimate_exec_time(false,executed_time, post_raw_total_exec_time,constant_overhead,running_queue.size(),latency_budget);
    running_queue_exec_time_insert.push_back(post_total_exec_time);
    post_gain = get_gain_for_running_queue(false,running_queue, latency_budget, constant_overhead, executed_time,post_total_exec_time);
    

    int32_t delta_gain = post_gain - pre_gain;
    running_queue_delta_gain.push_back(delta_gain);
    if (delta_gain >= max_delta_gain){
      max_delta_gain = delta_gain;
      max_gain_instance = pair.first;
    }
    LOG(INFO) << "In instance " << pair.first << ", pre_gain: " << pre_gain << ", post_gain: " << post_gain << ", delta_gain: " << delta_gain;
    
  }

  // 先选取收益差不多的实例作为候选集candidates
  double alpha=0.9;
  std::vector<size_t> candidate_index;
  for (int32_t i=0; i<running_queue_delta_gain.size();i++ ){
    if (max_delta_gain * alpha < running_queue_delta_gain[i]){
      candidate_index.push_back(i);
    }
  }
  //获取最小的remaining time
  // std::vector<double> min_remain_time_vec;
  // for  (auto index : candidate_index){
  //   auto instance_name = instances_name[index];
  //   auto running_queue = prefill_running_requests_map[instance_name];
  //     double min_remain_time = std::numeric_limits<double>::max();
  //     for (auto& req : running_queue){
  //       if (req->get_remaining_time() < min_remain_time){
  //         min_remain_time = req->get_remaining_time();
  //       }
  //     }
  //     min_remain_time_vec.push_back(min_remain_time);
  // }

  if (max_delta_gain>0 ){
    request->can_satisfy_slo = true;
    // 收益最大的实例是低负载模式的实例，且收益大于0（表示该请求能够完成）

    // 先选取符合上述条件的实例中，总执行时间非常低的实例，防止有实例过于空闲。如果有，则选取最低的
    bool has_low_ratio_instance = false;
    double min_exec_time = std::numeric_limits<int32_t>::max();
    for (auto index : candidate_index){
      if (running_queue_exec_time[index] <  low_ratio*ttft_slo && running_queue_exec_time[index] < min_exec_time){
        has_low_ratio_instance = true;
        min_exec_time = running_queue_exec_time[index];
        max_gain_instance = instances_name[index];
      }
    }
    if (has_low_ratio_instance){
      LOG(INFO) << "Branch: 1-too low";
      return max_gain_instance;
    }

    //再选取收益差不多且总执行时间*up_ratio不超过ttft_slo的实例
    std::vector<size_t> nh_candidate_index;
    for (auto index : candidate_index){
      if (running_queue_exec_time_insert[index] <= up_ratio*ttft_slo){
        nh_candidate_index.push_back(index);
      }
    }
    if(nh_candidate_index.empty()){
      // 如果没有符合条件的实例，则退化成选取收益差不多的实例里负载最低的
      double min_exec_time = std::numeric_limits<int32_t>::max();
      for (auto index : candidate_index){
        if (running_queue_exec_time[index] < min_exec_time){
          min_exec_time = running_queue_exec_time[index];
          max_gain_instance = instances_name[index];
        }
      }
      LOG(INFO) << "Branch: 2-all high";
      return max_gain_instance;
    }
    else{
      // 否则选取收益差不多实例中，执行时间最长的实例，预留空间给未来可能的长请求
      double max_exec_time = std::numeric_limits<int32_t>::min();
      for (auto index : nh_candidate_index){
        if (running_queue_exec_time[index] >  max_exec_time){
          max_exec_time = running_queue_exec_time[index];
          max_gain_instance = instances_name[index];
        }
      }
      LOG(INFO) << "Branch: 3-max medium";
      return max_gain_instance;
    }
  }
  else {
    // 该请求无法完成，则直接调度到负载最低（也就是执行时间最小）的那个实例
    double min_exec_time = std::numeric_limits<int32_t>::max();
    for (int32_t index=0;index<running_queue_exec_time.size();index++){
      if (running_queue_exec_time[index] < min_exec_time){
        min_exec_time = running_queue_exec_time[index];
        max_gain_instance = instances_name[index];
      }
    }
    LOG(INFO) << "Branch: 4-fall back";
  }

  return max_gain_instance;

}

bool PriorityRouting::select_instances_pair(std::shared_ptr<Request> request) {
  // for warm up
  size_t prev_count = num_warmup_request_num_.fetch_add(1, std::memory_order_relaxed);
  if  (prev_count < 500) {
    return instance_mgr_->get_next_instance_pair(&request->routing);
  }

  std::lock_guard<std::mutex> metric_lock(request_metrics_mutex_);
  
  // find prefill
  // 拷贝成map->vector
  RunningRequestMap prefill_running_requests_map;
  std::unordered_map<std::string, absl::Time> update_time_map;
  std::unordered_map<std::string, int32_t> decode_request_num_map;
  std::unordered_map<std::string, TtftPredictor> ttft_predictors;
  prefill_running_requests_map = instance_mgr_->get_prefill_running_requests_map();
  update_time_map = instance_mgr_->get_prefill_instance_update_time_map();
  decode_request_num_map = instance_mgr_->get_decode_request_num_map();
  ttft_predictors = instance_mgr_->get_ttft_predictors();
  if (prefill_running_requests_map.empty()){
    LOG(ERROR) << "No prefill instance found!";
    return false;
  }

  //request order
  std::unordered_map<std::string,std::string> strategies;
  std::unordered_map<std::string,double> budgets;
  for (auto& pair : prefill_running_requests_map) {
    auto it = ttft_predictors.find(pair.first);
    if (it == ttft_predictors.end()) {
      LOG(ERROR) << "Failed to find instance ttft predictor, instance name : "
                 << pair.first;
      continue;
    }

    budgets.emplace(pair.first, get_latency_budget_and_request_order(it->second, pair.second));
  }

  auto max_prefill_gain_instance = get_max_gain_instance(decode_request_num_map, update_time_map, ttft_predictors, prefill_running_requests_map, strategies, budgets, request);
  request->routing.prefill_name = max_prefill_gain_instance;
  return instance_mgr_->get_min_load_decode_instance(&request->routing);

}

}  // namespace xllm_service
