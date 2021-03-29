CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

SET (CMAKE_CXX_STANDARD 11)
SET (CMAKE_CXX_STANDARD_REQUIRED ON)

PROJECT (zhang-cnn CXX)

# some user settings
SET (TF_SOURCE $ENV{HOME}/build/tensorflow CACHE PATH "Path to Tensorflow source directory")

# get Tensorflow C++ librarys
FIND_LIBRARY (TF_CC libtensorflow_cc.so PATHS ${TF_SOURCE}/bazel-bin PATH_SUFFIXES tensorflow )
FIND_LIBRARY (TF_F libtensorflow_framework.so PATHS ${TF_SOURCE}/bazel-bin PATH_SUFFIXES tensorflow )
# This lib provides only the C functions... use TF_CC for C++ linking.
# However, this is needed when compiling on Gentoo.  As it doesn't hurt too
# much we can leav it on by default.
FIND_LIBRARY (TF libtensorflow.so PATHS ${TF_SOURCE}/bazel-bin PATH_SUFFIXES tensorflow)

IF (NOT TF_CC OR NOT TF_F)
    MESSAGE (FATAL_ERROR "missing tensorflow shared libraries, did you compile them?")
ENDIF ()

# Get CUDA
SET (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# XXX It is possible to run tensorflow without CUDA and CUDANN, so let's
#     avoid REQUIRED here, and do something differently when they are
#     found on the system.
SET (CUDA_CUDART_LIBRARY "")  # FIXME If I don't set these two, cmake complains...
SET (CUDA_TOOLKIT_INCLUDE "")
FIND_PACKAGE (CUDA)
IF (CUDA_FOUND)
    FIND_PACKAGE (CUDNN)
ENDIF ()
FIND_PACKAGE (MNIST REQUIRED)

# In case the default protbuf library doesn't work, we have to specify
# the path where to find the working one.  Otherwise run CMake with
# -DSYSTEM_PROTOBUF=ON
IF (NOT SYSTEM_PROTOBUF)
    # On should manually specify `-DProtobuf_LIBRARY` in order to get the source built version (and
    # not the system one)!
    IF (NOT Protobuf_LIBRARY)
         MESSAGE (WARNING "You should set `Protobuf_LIBRARY` to the "
                          "location of the Protobuf lib compiled by Tensorflow Bazel!")
    ENDIF ()
    SET (Protobuf_USE_STATIC_LIBS ON)
ENDIF ()
FIND_PACKAGE (Protobuf REQUIRED)

IF (MKL)
    MESSAGE (STATUS "Using MKL")
    SET (MKL_LIBRARY_DIR ${TF_SOURCE}/bazel-tensorflow-mkl/external/mkl_linux/lib)
    SET (MKL_INCLUDE_DIR ${TF_SOURCE}/bazel-tensorflow-mkl/external/mkl_linux/include)
    SET (MKL_LIBRARIES -liomp5 -lmklml_gnu -lmklml_intel) 
ENDIF ()

# add include dirs
# FIXME should be generic
INCLUDE_DIRECTORIES (SYSTEM ${TF_SOURCE}/bazel-genfiles/tensorflow/include)
INCLUDE_DIRECTORIES (SYSTEM ${TF_SOURCE}/bazel-genfiles/tensorflow/include/external/protobuf_archive/src) # local protobuf build
INCLUDE_DIRECTORIES (SYSTEM ${TF_SOURCE}/bazel-genfiles/tensorflow/include/external/com_google_absl) # local absl build
INCLUDE_DIRECTORIES (SYSTEM ${MKL_INCLUDE_DIR})
# XXX I need these pathes on gentoo.  Having more pathes doesn't hurt, so I am
#     leaving them on by default.
INCLUDE_DIRECTORIES (SYSTEM /usr/include/tensorflow)
INCLUDE_DIRECTORIES (SYSTEM /usr/include/tensorflow/external/com_google_absl)
INCLUDE_DIRECTORIES (SYSTEM ${CUDNN_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${Protobuf_INCLUDE_DIRS} ${MNIST_INCLUDE_DIR})
INCLUDE_DIRECTORIES (${PROJECT_SOURCE_DIR})
IF (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.0")
    ADD_DEFINITIONS (-D_GLIBCXX_USE_CXX11_ABI=0) # with GCC 6.3.0 this needs to be zero.
ENDIF ()
LINK_DIRECTORIES (${MKL_LIBRARY_DIR})
ADD_COMPILE_OPTIONS (-march=native -mtune=native -O3)

ADD_EXECUTABLE (zhang-cnn zhang.cc)
TARGET_LINK_LIBRARIES(zhang-cnn ${TF_CC} ${TF_F} ${Protobuf_LIBRARIES} ${CUDNN_LIBRARIES} ${CUDA_LIBRARIES} ${MKL_LIBRARIES})
TARGET_COMPILE_DEFINITIONS (zhang-cnn PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
