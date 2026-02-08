cmake_minimum_required(VERSION 3.18)
project(ObjectDetectionLibtorchForms CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LibTorch
if (NOT Torch_DIR)
    set(LIBTORCH_LOCAL_PATH "${PROJECT_SOURCE_DIR}/libtorch")
    if (EXISTS "${LIBTORCH_LOCAL_PATH}/share/cmake/Torch/TorchConfig.cmake")
        list(PREPEND CMAKE_PREFIX_PATH "${LIBTORCH_LOCAL_PATH}")
        set(Torch_DIR "${LIBTORCH_LOCAL_PATH}/share/cmake/Torch")
    endif ()
endif ()
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# YoloTrainer Library (Native C++)
add_library(YoloTrainer STATIC
    src/YoloTrainer/YoloModel.cpp
    src/YoloTrainer/YoloTrainer.cpp
    src/YoloTrainer/DataLoader.cpp
)

target_link_libraries(YoloTrainer
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

# Set properties for LibTorch
set_property(TARGET YoloTrainer PROPERTY CXX_STANDARD 17)

# Copy DLLs for Windows
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET YoloTrainer
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:YoloTrainer>)
endif (MSVC)

# Installation
install(TARGETS YoloTrainer
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# Print configuration
message(STATUS "PyTorch version: ${TORCH_VERSION}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
