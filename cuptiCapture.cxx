#include "cuptiCapture.h"

#include <sys/syscall.h>
#include <unistd.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "cupti.h"

constexpr size_t BUFFER_SIZE = 5 << 20;
CUpti_SubscriberHandle m_subscriber;
void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size,
                              size_t* maxNumRecords);

// callback called when buffer is full with activity records
void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t* buffer,
                              size_t size, size_t validSize);
void CUPTIAPI trace_callback(void* userdata, CUpti_CallbackDomain domain,
                             CUpti_CallbackId cbid, const void* cbdata);

void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size,
                              size_t* maxNumRecords) {
  *buffer = (uint8_t*)malloc(sizeof(uint8_t) * BUFFER_SIZE);
  *size = BUFFER_SIZE;
  *maxNumRecords = 0;
}

class LoggedCount {
 public:
  LoggedCount(const std::string& msg) : count(0), m_msg(msg){};
  ~LoggedCount() { std::cerr << m_msg << " counter is " << count << std::endl; }
  uint64_t count;
  std::string m_msg;
};

#define CUPTI_CALL(call)                                                \
  do {                                                                  \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
      const char* errstr;                                               \
      cuptiGetResultString(_status, &errstr);                           \
      std::cerr << "Cupti call " << #call << " failed with " << errstr; \
    }                                                                   \
  } while (0)

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t* buffer,
                              size_t size, size_t validSize) {
  CUpti_Activity* record = nullptr;
  CUptiResult status;
  static LoggedCount bc("Completed buffers");
  static LoggedCount rc("Encountered records");
  bc.count++;
  if (validSize) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        rc.count++;
      } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        break;
      else {
        CUPTI_CALL(status);
      }
    } while (true);
  }
  free(buffer);
}

// Callback called on every CUDA API call entry
__always_inline void OnDriverApiEnter(CUpti_CallbackDomain domain,
                                      CUpti_driver_api_trace_cbid cbid,
                                      const CUpti_CallbackData* cbdata) {
  // record current timestamp (used in OnDriverAPIExit)
  cuptiGetTimestamp(cbdata->correlationData);
}

// Callback called on every CUDA API call exit

__always_inline void OnDriverApiExit(CUpti_CallbackDomain domain,
                                     CUpti_CallbackId cbid,
                                     const CUpti_CallbackData* cbdata) {
  // get current timestamp
  uint64_t end_tsc;
  cuptiGetTimestamp(&end_tsc);
  static LoggedCount api_time("Time spent on driver API ");
  api_time.count += (end_tsc - *(cbdata->correlationData));
  if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
    if ((cbid == CUPTI_DRIVER_TRACE_CBID_cuCtxGetDevice) ||
        (cbid ==
         CUPTI_DRIVER_TRACE_CBID_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) ||
        (cbid == CUPTI_DRIVER_TRACE_CBID_cuFuncGetAttribute)) {
      return;
    }
  }
}

void CUPTIAPI trace_callback(void* userdata, CUpti_CallbackDomain domain,
                             CUpti_CallbackId cbid, const void* cbdata) {
  // record callstacks in every CUDA API call
  // CUDA API call entry
  const CUpti_CallbackData* cbInfo = (CUpti_CallbackData*)cbdata;
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    cuptiGetTimestamp(cbInfo->correlationData);
  } else {
    OnDriverApiExit(domain, cbid, cbInfo);
  }
}

void CuptiCapture::Start() {
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION));

  // register callback
  CUPTI_CALL(
      cuptiSubscribe(&m_subscriber, (CUpti_CallbackFunc)trace_callback, this));
  CUPTI_CALL(cuptiEnableDomain(1, m_subscriber, CUPTI_CB_DOMAIN_DRIVER_API));

  //  register callbacks for buffer requests and for buffers completed by
  //  CUPTI.
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
}

void CuptiCapture::Stop() {
  CUPTI_CALL(cuptiEnableDomain(0, m_subscriber, CUPTI_CB_DOMAIN_DRIVER_API));
  CUPTI_CALL(cuptiUnsubscribe(m_subscriber));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONTEXT));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_NAME));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION));
  CUPTI_CALL(cuptiActivityFlushAll(1));
}
