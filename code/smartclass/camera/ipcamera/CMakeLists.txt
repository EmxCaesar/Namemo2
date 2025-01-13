include_directories(${CMAKE_CURRENT_SOURCE_DIR}/onvif)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/onvif/custom)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/onvif/plugin)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/onvif/soapProxy)

AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR} ipcamera_src)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/onvif src)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/onvif/custom custom_src)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/onvif/plugin plugin_src)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/onvif/soapProxy soapProxy_src)

set(sourcecode ${ipcamera_src} ${src} ${custom_src} ${plugin_src} ${soapProxy_src})

add_library(ipcamera STATIC ${sourcecode})
target_link_libraries(ipcamera ssl crypto)
target_link_libraries(ipcamera opencv_world)
