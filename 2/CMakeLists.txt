cmake_minimum_required(VERSION 3.10)
project(2)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找OpenCV库
find_package(OpenCV REQUIRED)

# 包含OpenCV头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件
add_executable(2 main.cpp)

# 链接OpenCV库
target_link_libraries(2 ${OpenCV_LIBS})
