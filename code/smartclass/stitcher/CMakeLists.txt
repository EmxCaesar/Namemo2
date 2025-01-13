
add_library(stitch STATIC stitcher.cpp stitching.cpp stitcher_observer.cpp)

target_link_libraries(stitch opencv_world)

