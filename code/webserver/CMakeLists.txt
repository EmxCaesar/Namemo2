AUX_SOURCE_DIRECTORY(buffer buffer_src)
AUX_SOURCE_DIRECTORY(http http_src)
AUX_SOURCE_DIRECTORY(lock lock_src)
AUX_SOURCE_DIRECTORY(log log_src)
AUX_SOURCE_DIRECTORY(epoller epoller_src)
AUX_SOURCE_DIRECTORY(sqlconnpool sql_src)
AUX_SOURCE_DIRECTORY(threadpool thread_src)
AUX_SOURCE_DIRECTORY(timer timer_src)

add_library(webserver ${buffer_src} ${http_src} ${lock_src} ${log_src} ${epoller_src} ${sql_src} ${thread_src} ${timer_src} webserver.cpp)
target_link_libraries(webserver mysqlclient)
target_link_libraries(webserver opencv_world)


