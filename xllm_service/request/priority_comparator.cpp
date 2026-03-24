/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "priority_comparator.h"

#include "glog/logging.h"

namespace xllm_service {

// implement operator()
bool FCFSComparator::operator()(const std::shared_ptr<Request>& a,
                                const std::shared_ptr<Request>& b) const {
  return a->get_created_time() > b->get_created_time();
}

bool StrictPriorityComparator::operator()(
    const std::shared_ptr<Request>& a,
    const std::shared_ptr<Request>& b) const {
  auto priority_a = a->get_priority();
  auto priority_b = b->get_priority();
  if (priority_a != priority_b) {
    return priority_a > priority_b;  // HIGH(1) < NORMAL(2) < LOW(3)
  }
  return a->get_created_time() > b->get_created_time();
}

bool DeadlineComparator::operator()(const std::shared_ptr<Request>& a,
                                    const std::shared_ptr<Request>& b) const {

  auto remain_time_a = a->get_remaining_time();
  auto remain_time_b = b->get_remaining_time();
  
  return remain_time_a > remain_time_b;
}

bool Deadline2Comparator::operator()(const std::shared_ptr<Request>& a,
                                    const std::shared_ptr<Request>& b) const {
  // auto& sequence_a = a->sequences()[0];
  // auto& sequence_b = b->sequences()[0];

  int32_t remain_time_a = a->get_remaining_time() - static_cast<int32_t>(a->get_estimated_latency());
  int32_t remain_time_b = b->get_remaining_time() - static_cast<int32_t>(b->get_estimated_latency());
  
  return remain_time_a > remain_time_b;
}

// bool DecodeDeadlineComparator::operator()(const std::shared_ptr<Request>& a,
//                                     const std::shared_ptr<Request>& b) const {
//   auto& sequence_a = a->sequences()[0];
//   auto& sequence_b = b->sequences()[0];

//   if (sequence_a->stage() == sequence_b->stage()) {
//     return DeadlineComparator()(a,b);
//   }
  
//   return sequence_a->stage() < sequence_b->stage();
// }

// bool UrgencyDecodeDensityComparator::operator()(const std::shared_ptr<Request>& a,
//                                     const std::shared_ptr<Request>& b) const {

//   if (a->urgency() == b->urgency()) {
//     return DecodeDensityComparator()(a, b);
//   }
//   return a->urgency() < b->urgency();
// }

bool UrgencyDensityComparator::operator()(const std::shared_ptr<Request>& a,
                                    const std::shared_ptr<Request>& b) const {
  if (!a || !b) {
    // 处理空指针情况
    LOG(INFO) << "One of the requests is nullptr";
  }

  if (a->get_urgency() == b->get_urgency()) {
    if (a->get_urgency() == Urgency::URGENT){ // 紧急的：下一个回合就要超时的
      return DensityComparator()(a, b);
      // return DeadlineComparator()(a, b);
      // return FCFSComparator()(a, b);
    }
    if (a->get_urgency() == Urgency::NORMAL){ // 不紧急的
      return DeadlineComparator()(a, b);
      // return Deadline2Comparator()(a, b);
      // return DensityComparator()(a, b);
      // return FCFSComparator()(a, b);
    }
    if (a->get_urgency() == Urgency::TIMEOUT){ // 超时的
      // return FCFSComparator()(a, b);
      return DensityComparator()(a, b);
      // return DensityComparator()(a, b);
    }
    return FCFSComparator()(a, b);
  }
  return a->get_urgency() < b->get_urgency();
}

bool DensityComparator::operator()(const std::shared_ptr<Request>& a,
                                    const std::shared_ptr<Request>& b) const {


  const double epsilon = 1e-9;  // 设置一个合适的容差
  double density_a, density_b;
  
  density_a = a->get_ttft_priority_weight() * 1.0 / a->get_estimated_latency();
  density_b = b->get_ttft_priority_weight() * 1.0 / b->get_estimated_latency();
  
  // 使用容差比较
  if (std::abs(density_a - density_b) < epsilon) {
    // 如果密度非常接近，使用稳定的比较标准（如指针地址）
    return a->get_created_time() > b->get_created_time();
  }
  
  // 对于排序来说，<表示小的在前面；对于优先队列来说，<表示大的在前面
  
  return density_a < density_b;
}

bool SJFComparator::operator()(const std::shared_ptr<Request>& a,
                                    const std::shared_ptr<Request>& b) const {


  const double epsilon = 1e-9;  // 设置一个合适的容差
  double density_a, density_b;
  
  density_a =  1.0 / a->get_estimated_latency();
  density_b =  1.0 / b->get_estimated_latency();
  
  // 使用容差比较
  if (std::abs(density_a - density_b) < epsilon) {
    // 如果密度非常接近，使用稳定的比较标准（如指针地址）
    return a->get_created_time() > b->get_created_time();
  }
  
  // 对于排序来说，<表示小的在前面；对于优先队列来说，<表示大的在前面
  
  return density_a < density_b;
}

std::function<bool(const std::shared_ptr<Request>&,
                   const std::shared_ptr<Request>&)>
create_comparator(const std::string& priority_strategy) {
  if (priority_strategy == "fcfs" || priority_strategy == "fcfs1") {
    return [](const std::shared_ptr<Request>& a,
              const std::shared_ptr<Request>& b) {
      return FCFSComparator()(a, b);
    };
  } else if (priority_strategy == "priority") {
    return [](const std::shared_ptr<Request>& a,
              const std::shared_ptr<Request>& b) {
      return StrictPriorityComparator()(a, b);
    };
  } else if (priority_strategy == "deadline") {
    return [](const std::shared_ptr<Request>& a,
              const std::shared_ptr<Request>& b) {
      return DeadlineComparator()(a, b);
    };
  } else if (priority_strategy == "deadline2") {
    return [](const std::shared_ptr<Request>& a,
              const std::shared_ptr<Request>& b) {
      return Deadline2Comparator()(a, b);
    };
  } else if (priority_strategy == "sjf") {
    return [](const std::shared_ptr<Request>& a,
              const std::shared_ptr<Request>& b) {
      return SJFComparator()(a, b);
    };
  } else if (priority_strategy == "density") {
    return [](const std::shared_ptr<Request>& a,
              const std::shared_ptr<Request>& b) {
      return DensityComparator()(a, b);
    };
  } else if (priority_strategy == "urgency_density") {
    return [](const std::shared_ptr<Request>& a,
              const std::shared_ptr<Request>& b) {
      return UrgencyDensityComparator()(a, b);
    };
  } else {
    LOG(FATAL) << "Unknown strategy: " << priority_strategy;
    return nullptr;
  }
}

std::function<bool(const std::shared_ptr<Request>&,
                   const std::shared_ptr<Request>&)>
create_sort_comparator(const std::string& priority_strategy) {
  if (priority_strategy == "fcfs" || priority_strategy == "fcfs1") {
    return [](const std::shared_ptr<Request>& a,
              const std::shared_ptr<Request>& b) {
      return !FCFSComparator()(a, b);
    };
  }  else if (priority_strategy == "priority") {
    return [](const std::shared_ptr<Request>& a,
              const std::shared_ptr<Request>& b) {
      return StrictPriorityComparator()(a, b);
    };
  } else if (priority_strategy == "deadline") {
    return [](const std::shared_ptr<Request>& a,
              const std::shared_ptr<Request>& b) {
      return !DeadlineComparator()(a, b);
    };
  } else if (priority_strategy == "deadline2") {
    return [](const std::shared_ptr<Request>& a,
              const std::shared_ptr<Request>& b) {
      return !Deadline2Comparator()(a, b);
    };
  } else if (priority_strategy == "sjf") {
    return [](const std::shared_ptr<Request>& a,
              const std::shared_ptr<Request>& b) {
      return !SJFComparator()(a, b);
    };
  }else if (priority_strategy == "density") {
    return [](const std::shared_ptr<Request>& a,
              const std::shared_ptr<Request>& b) {
      return !DensityComparator()(a, b);
    };
  } else if (priority_strategy == "urgency_density") {
    return [](const std::shared_ptr<Request>& a,
              const std::shared_ptr<Request>& b) {
      return !UrgencyDensityComparator()(a, b);
    };
  } else {
    LOG(FATAL) << "Unknown strategy: " << priority_strategy;
    return nullptr;
  }
}

}  // namespace xllm_service