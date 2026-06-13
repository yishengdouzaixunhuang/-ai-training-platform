cmake_minimum_required(VERSION 3.14)
project(cpp_infer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)

add_executable(cpp_infer infer.cpp)
target_link_libraries(cpp_infer opencv_world onnxruntime)
target_include_directories(cpp_infer PRIVATE )
