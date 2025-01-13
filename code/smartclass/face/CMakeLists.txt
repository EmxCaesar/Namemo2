add_subdirectory(facenn)

add_library(face STATIC face_detector.cpp face_align.cpp
    face_recognizer.cpp face_database.cpp face_observer.cpp
    face_namelist.cpp)

target_link_libraries(face opencv_world)
target_link_libraries(face facenn)
