project(nanodet_rknn_demo)



add_subdirectory(rknn_api_sdk)
include_directories(${Rknn_Net_INCLUDE_DIRS})
link_libraries(${Rknn_Net_LIBS})

add_subdirectory(rknn_nanodet)
include_directories(${NanoDet_INCLUDE_DIRS})
link_libraries(${NanoDet_LIBS})

find_package(OpenCV 4 REQUIRED)

add_executable(
    nanodet_rknn_demo
    ./demo.cpp
)


