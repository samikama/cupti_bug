#pragma once
#include <cstddef>
class CuptiCapture {
 public:
  void Start();
  void Stop();
  static CuptiCapture& instance() {
    static CuptiCapture inst;
    return inst;
  }

 private:
  CuptiCapture() = default;
  ~CuptiCapture() = default;
};
