cmake_minimum_required(VERSION 3.20)

project(neural_network_assignment LANGUAGES CXX)

option(NN_BUILD_BRAIN_MRI "Build brain_mri executable (requires OpenCV)" ON)
option(NN_BUILD_BRAIN_MRI_PGM "Build brain_mri_pgm executable (no OpenCV; expects preprocessed PGM/manifests)" ON)
option(NN_ENABLE_TESTS "Build CTest-based tests" ON)
option(NN_ENABLE_CUDA "Enable CUDA backend" ON)
option(NN_FORCE_CPU "Force CPU backend even when CUDA is available" OFF)
option(NN_BUILD_BENCHMARKS "Build benchmark executables" ON)
option(NN_STRICT_CUDA "Fail configure if NN_ENABLE_CUDA=ON and CUDA compiler is not found" ON)
set(NN_CUDA_ARCHS "75;86;89" CACHE STRING "CUDA architectures to compile for")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(NN_PUBLIC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(NN_CORE_SOURCES
    src/layers/dense.cpp
    src/layers/convolutional.cpp
    src/layers/reshape.cpp
)

add_library(nn_core STATIC ${NN_CORE_SOURCES})
target_include_directories(nn_core
    PUBLIC
        ${NN_PUBLIC_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
)

if(NN_FORCE_CPU)
    set(NN_ENABLE_CUDA OFF)
    message(STATUS "NN_FORCE_CPU=ON: CUDA backend disabled.")
endif()

if(NN_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
        # RTX 4060 = Ada Lovelace (SM 89). Keep a few common archs for portability.
        if(NN_CUDA_ARCHS)
            set(CMAKE_CUDA_ARCHITECTURES "${NN_CUDA_ARCHS}" CACHE STRING "CUDA architectures" FORCE)
        endif()
        message(STATUS "CUDA enabled. Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        if(NN_STRICT_CUDA)
            message(FATAL_ERROR "NN_ENABLE_CUDA=ON but CUDA compiler not found. On Visual Studio generators, configure with: -A x64 -T \"cuda=%CUDA_PATH%\" or use the cuda-ready preset.")
        else()
            message(WARNING "NN_ENABLE_CUDA=ON but CUDA compiler not found. Continuing with CPU build.")
        endif()
    endif()
endif()

set(NN_CUDA_BACKEND_AVAILABLE OFF)
if(NN_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    add_library(nn_cuda_backend STATIC
        src/backend/cuda_dot.cu
        src/backend/cuda_conv2d.cu
    )
    set_target_properties(nn_cuda_backend PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON)
    target_include_directories(nn_cuda_backend
        PUBLIC
            ${NN_PUBLIC_INCLUDE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set(NN_CUDA_BACKEND_AVAILABLE ON)
endif()

function(nn_apply_cuda_backend target_name)
    if(NN_CUDA_BACKEND_AVAILABLE)
        target_link_libraries(${target_name} PRIVATE nn_cuda_backend)
        target_compile_definitions(${target_name} PRIVATE NN_USE_CUDA_DOT=1 NN_USE_CUDA_CONV2D=1)
    endif()
endfunction()

nn_apply_cuda_backend(nn_core)

function(nn_add_app_target target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE nn_core)
    nn_apply_cuda_backend(${target_name})
endfunction()

nn_add_app_target(xor apps/xor.cxx)
nn_add_app_target(mnist apps/mnist.cxx)
nn_add_app_target(mnist_conv apps/mnist_conv.cxx)

if(NN_BUILD_BENCHMARKS)
    nn_add_app_target(benchmark_dot benchmarks/benchmark_dot.cxx)
    nn_add_app_target(benchmark_conv2d benchmarks/benchmark_conv2d.cxx)
endif()

if(NN_BUILD_BRAIN_MRI)
    find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
    if(OpenCV_FOUND)
        add_executable(brain_mri apps/brain_mri.cxx)
        target_link_libraries(brain_mri PRIVATE nn_core)
        nn_apply_cuda_backend(brain_mri)
        target_link_libraries(brain_mri PRIVATE ${OpenCV_LIBS})
        target_include_directories(brain_mri PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_compile_definitions(brain_mri PRIVATE NN_WITH_OPENCV=1)
    else()
        message(WARNING "OpenCV not found. brain_mri target will be skipped.")
    endif()
endif()

if(NN_BUILD_BRAIN_MRI_PGM)
    add_executable(brain_mri_pgm apps/brain_mri_pgm.cxx)
    target_link_libraries(brain_mri_pgm PRIVATE nn_core)
    nn_apply_cuda_backend(brain_mri_pgm)
endif()

include(GNUInstallDirs)
install(TARGETS nn_core xor mnist mnist_conv
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
if(TARGET brain_mri)
    install(TARGETS brain_mri RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
if(TARGET brain_mri_pgm)
    install(TARGETS brain_mri_pgm RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
if(TARGET benchmark_dot)
    install(TARGETS benchmark_dot RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
if(TARGET benchmark_conv2d)
    install(TARGETS benchmark_conv2d RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(FILES README.md docs/windows_machine_run_test.md docs/architecture.md
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(NN_ENABLE_TESTS)
    enable_testing()
    add_executable(test_core tests/test_core.cxx)
    target_link_libraries(test_core PRIVATE nn_core)
    nn_apply_cuda_backend(test_core)
    add_test(NAME test_core COMMAND test_core)

    add_executable(test_dense_gpu tests/test_dense_gpu.cxx)
    target_link_libraries(test_dense_gpu PRIVATE nn_core)
    nn_apply_cuda_backend(test_dense_gpu)
    add_test(NAME test_dense_gpu COMMAND test_dense_gpu)
    set_tests_properties(test_dense_gpu PROPERTIES TIMEOUT 60)

    add_test(NAME xor_smoke COMMAND xor 2000 0)
    set_tests_properties(xor_smoke PROPERTIES TIMEOUT 180)

    if(TARGET benchmark_dot)
        add_test(NAME benchmark_dot_smoke COMMAND benchmark_dot 128 128 128 2)
        set_tests_properties(benchmark_dot_smoke PROPERTIES TIMEOUT 60)
    endif()

    add_executable(test_conv_gpu tests/test_conv_gpu.cxx)
    target_link_libraries(test_conv_gpu PRIVATE nn_core)
    nn_apply_cuda_backend(test_conv_gpu)
    add_test(NAME test_conv_gpu COMMAND test_conv_gpu)
    set_tests_properties(test_conv_gpu PROPERTIES TIMEOUT 120)

    if(TARGET benchmark_conv2d)
        add_test(NAME benchmark_conv2d_smoke COMMAND benchmark_conv2d 64 64 3 2)
        set_tests_properties(benchmark_conv2d_smoke PROPERTIES TIMEOUT 60)
    endif()
endif()
