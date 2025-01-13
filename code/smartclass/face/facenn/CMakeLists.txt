cuda_add_library(retinadecode STATIC  retinadecode.cu)
target_link_libraries(retinadecode nvinfer cudart)

add_library(facenn STATIC retinaface.cpp arcface.cpp)
target_link_libraries(facenn nvinfer)
target_link_libraries(facenn cudart)
target_link_libraries(facenn opencv_world)
target_link_libraries(facenn retinadecode )
