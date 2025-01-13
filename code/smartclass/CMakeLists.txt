add_subdirectory(camera)
add_subdirectory(stitcher)
add_subdirectory(face)

add_library(smartclass STATIC smartclass.cpp)

target_link_libraries(smartclass opencv_world)
target_link_libraries(smartclass stitch)
target_link_libraries(smartclass face)
target_link_libraries(smartclass camera)
target_link_libraries(smartclass ssl crypto)

