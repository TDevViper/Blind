cmake_minimum_required(VERSION 3.15)
project(BlindAssistantVC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package (ensure OpenCV_DIR is set to your OpenCV build/x64/vc16 directory)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(BlindAssistantVC BlindAssistant_VC.cpp)

# Link OpenCV and Windows Native Speech API / COM libraries
target_link_libraries(BlindAssistantVC PRIVATE ${OpenCV_LIBS} ole32 sapi)

# MSVC compiler flags for optimization and standard exception handling
if(MSVC)
    target_compile_options(BlindAssistantVC PRIVATE /W4 /O2 /EHsc)
else()
    target_compile_options(BlindAssistantVC PRIVATE -Wall -Wextra -O3)
endif()
