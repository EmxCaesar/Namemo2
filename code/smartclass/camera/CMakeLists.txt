add_subdirectory(ipcamera)

add_library(camera STATIC camera_tour.cpp)
target_link_libraries(camera ssl crypto)
target_link_libraries(camera ipcamera)
target_link_libraries(camera opencv_world)
