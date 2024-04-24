cmake_minimum_required(VERSION 3.10)
project(BA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3" )


# 查找OpenCV包
find_package(OpenCV REQUIRED)


# 查找Eigen包
find_package(Eigen3 REQUIRED)


#查找g2o
LIST( APPEND CMAKE_MODULE_PATH /home/taka/g2o/g2o/cmake_modules )
set(G2O_ROOT /usr/local/include/g2o)
find_package( G2O REQUIRED )
find_package( CSparse REQUIRED )


find_package( Ceres REQUIRED )


find_package( Sophus REQUIRED )
include_directories(${PROJECT_SOURCE_DIR}/include
                    ${OpenCV_INCLUDE_DIRS}
                    ${Sophus_INCLUDE_DIRS}
                    ${G2O_INCLUDE_DIRS}
                    ${CSPARSE_INCLUDE_DIR}
                    )
include_directories("/usr/include/eigen3")
include_directories( ${CERES_INCLUDE_DIRS} )


# 添加可执行文件
link_directories(${PROJECT_SOURCE_DIR}/lib)
add_executable(BA main.cpp)  # 替换为您的源代码文件名
target_link_libraries(BA ${CERES_LIBRARIES} ${OpenCV_LIBS} ${Sophus_LIBRARIES} ${CSPARSE_LIBRARY} g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension fmt)