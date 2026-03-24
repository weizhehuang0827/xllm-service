include(cc_library)

cc_library(
  NAME
    request
  HDRS
    request.h
    priority_comparator.h
  SRCS
    priority_comparator.cpp
  DEPS
    :common
    :chat_template
)