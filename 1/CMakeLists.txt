cmake_minimum_required(VERSION 3.10)
project(1)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找OpenCV库
find_package(OpenCV REQUIRED)

# 包含OpenCV头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件
add_executable(1 main.cpp)

# 链接OpenCV库
target_link_libraries(1 ${OpenCV_LIBS})
