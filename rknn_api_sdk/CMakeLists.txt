cmake_minimum_required(VERSION 3.10)

project(Rknn_Net)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ARCH_DIR x86)
else()
    set(ARCH_DIR arm)
endif()

set(RKNN_API_HEADERS 
${CMAKE_CURRENT_SOURCE_DIR}/rknn_api/${ARCH_DIR}/include/
)
set(RKNN_API_LIBS    
    ${CMAKE_CURRENT_SOURCE_DIR}/rknn_api/${ARCH_DIR}/lib64/librknn_api.so
    pthread
)
include_directories(
    ${RKNN_API_HEADERS}
    ./include/
)
file(GLOB Rknn_Net_API_SRCS  "./src/*.c*")

add_library(
    ${PROJECT_NAME}
    ${Rknn_Net_API_SRCS}
)
target_link_libraries(
    ${PROJECT_NAME}
    ${RKNN_API_LIBS}
)

set(Rknn_Net_INCLUDE_DIRS  
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${RKNN_API_HEADERS}
    PARENT_SCOPE
)
set(Rknn_Net_LIBS
    ${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/rknn_api/${ARCH_DIR}/lib64/librknn_api.so
    PARENT_SCOPE
)


