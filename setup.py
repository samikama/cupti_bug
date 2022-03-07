import os
import sys
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
  """
    This class defines a Extension built by CMake (instead of distutils).
    """

  def __init__(self,
               name,
               cmakelists_dir=".",
               is_library=False,
               sources=None,
               *args,
               **kw):
    """
        Constructor.
        :param name: Name of the CMake build target.
        :param cmakelists_dir: Location of the CMakeLists.txt that defined the target.
        :param is_library: False for Python C Extension, True for normal shared library.
        :param sources: None because no source file will be built by Distutils.
        :param args: Additional arguments
        :param kw: Additional keyword-arguments.
        """
    Extension.__init__(self, name, sources=sources or [], *args, **kw)
    self.sourcedir = os.path.abspath(cmakelists_dir)
    self.is_library = is_library


class CMakeBuild(build_ext):
  """
    This class defines a build_ext that builds Python C Extensions by CMake.
    """

  # copied parts of this from pybind cmake build examples
  def build_extension(self, ext):
    ext_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
    if not ext_dir.endswith(os.path.sep):
      ext_dir += os.path.sep
    #TODO(SAMI): change this to release
    cfg = "Debug" if self.debug else "RelWithDebInfo"
    # CMake lets you override the generator - we need to check this.
    # Can be set with Conda-Build, for example.
    cmake_generator = os.environ.get("CMAKE_GENERATOR", "")
    # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
    # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
    # from Python.
    cmake_args = [
        "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(ext_dir),
        "-DPYTHON_EXECUTABLE={}".format(sys.executable),
        "-DCMAKE_BUILD_TYPE={}".format(cfg),
        "-DCMAKE_VERBOSE_MAKEFILE=ON",
    ]
    cmake_cuda_compiler = os.environ.get("CMAKE_CUDA_COMPILER", "")
    cmake_cudatoolkit_root = os.environ.get("CUDAToolkit_ROOT", "")
    if os.path.exists(cmake_cudatoolkit_root):
      if not os.path.exists(cmake_cuda_compiler):
        cmake_cuda_compiler = os.path.join(cmake_cudatoolkit_root, "bin/nvcc")
      cmake_args.append("-DCUDAToolkit_ROOT={}".format(cmake_cudatoolkit_root))
    if os.path.exists(cmake_cuda_compiler):
      cmake_args.append(f"-DCMAKE_CUDA_COMPILER={cmake_cuda_compiler}")
    build_args = []
    # Comment ninja generator for the time being since it fails more often without hints while developing with pip
    # if not cmake_generator:
    #     try:
    #         import ninja  # noqa: F401
    #         cmake_args += ["-GNinja"]
    #     except ImportError:
    #         pass

    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)

    subprocess.check_call(["cmake", ext.sourcedir] + cmake_args,
                          cwd=self.build_temp)
    subprocess.check_call(["cmake", "--build", "."] + build_args,
                          cwd=self.build_temp)



with open("README.md", "r") as fh:
  long_description = fh.read()

test_require = ["pytest", "pytest-pspec"]

setup(
    name="cupti_bug",
    version="0.2.0",
    description="repro for cupti issues",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Operating System :: POSIX :: Linux",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    python_requires=">=3.6",
    package_dir={"cupti_bug": ""},
    packages=["cupti_bug"],
    ext_modules=[
        CMakeExtension("cupti_bug"),
    ],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    extras_require={"test": test_require},
)
