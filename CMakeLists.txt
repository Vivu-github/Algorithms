cmake_minimum_required(VERSION 3.16)
project(PeriodicStripeDetector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

add_executable(periodic_stripe_tool
    src/main.cpp
    src/stripe_detector.cpp
    src/stripe_detector.hpp
)

target_include_directories(periodic_stripe_tool PRIVATE ${OpenCV_INCLUDE_DIRS} src)
target_link_libraries(periodic_stripe_tool PRIVATE ${OpenCV_LIBS})

if(MSVC)
  target_compile_options(periodic_stripe_tool PRIVATE /W4)
else()
  target_compile_options(periodic_stripe_tool PRIVATE -Wall -Wextra -Wpedantic)
endif()
