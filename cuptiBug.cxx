#include <map>
#include <memory>
#include <sstream>
#include <string>

#include "cuptiCapture.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"
namespace py = pybind11;
PYBIND11_MODULE(cupti_bug, m) {
  py::class_<CuptiCapture, std::unique_ptr<CuptiCapture, py::nodelete>>(
      m, "CuptiCapture")
      .def_static("instance", &CuptiCapture::instance,
                  py::return_value_policy::reference)
      .def("start_profiling", &CuptiCapture::Start)
      .def("stop_profiling", &CuptiCapture::Stop);
}
