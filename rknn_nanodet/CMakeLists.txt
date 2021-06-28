
project(NanoDet)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


file(GLOB SRCS "./src/*.c*")

find_package(OpenCV REQUIRED)

# message("find opencv version : "${OpenCV_VERSION})

# add_subdirectory(../rknn_api_sdk)

include_directories(
    include/
    ${Rknn_Net_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_library(
    ${PROJECT_NAME}
    ${SRCS}
)

target_link_libraries(
    ${PROJECT_NAME}
    ${Rknn_Net_LIBS}
    ${OpenCV_LIBS}
)


set(NanoDet_INCLUDE_DIRS  
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    # ${Rknn_Net_HEADERS}
    PARENT_SCOPE
)

set(NanoDet_LIBS
    ${PROJECT_NAME}
    # ${CMAKE_CURRENT_SOURCE_DIR}/rknn_api/${ARCH_DIR}/lib64/librknn_api.so
    PARENT_SCOPE
)

