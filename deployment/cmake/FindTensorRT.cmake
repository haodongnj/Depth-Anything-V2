#[=======================================================================[.rst:
FindTensorRT
------------

Find the NVIDIA TensorRT inference SDK.

Imported targets
^^^^^^^^^^^^^^^^
``TensorRT::nvinfer``

Result variables
^^^^^^^^^^^^^^^^
``TensorRT_FOUND``
``TensorRT_INCLUDE_DIRS``
``TensorRT_LIBRARIES``

Search order
^^^^^^^^^^^^
1. $ENV{TRT_ROOT}
2. /usr (apt install)
3. /usr/local/tensorrt (tar install)
4. /opt/tensorrt (JetPack)
#]=======================================================================]

include(FindPackageHandleStandardArgs)

# Build the search list
set(_TRT_SEARCH_PATHS
    /usr
    /usr/local/tensorrt
    /opt/tensorrt
)

if(DEFINED ENV{TRT_ROOT})
    list(INSERT _TRT_SEARCH_PATHS 0 "$ENV{TRT_ROOT}")
endif()

# Find include dir
find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS ${_TRT_SEARCH_PATHS}
    PATH_SUFFIXES include include/x86_64-linux-gnu
    DOC "TensorRT include directory"
)

# Find library
find_library(TensorRT_LIBRARY
    NAMES nvinfer
    PATHS ${_TRT_SEARCH_PATHS}
    PATH_SUFFIXES lib lib/x86_64-linux-gnu
    DOC "TensorRT inference library"
)

find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_LIBRARY TensorRT_INCLUDE_DIR
)

if(TensorRT_FOUND AND NOT TARGET TensorRT::nvinfer)
    add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvinfer PROPERTIES
        IMPORTED_LOCATION "${TensorRT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_LIBRARY)
