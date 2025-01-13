#include <unistd.h>
#include "webserver/webserver.h"
#include "Resource.h"
#include "smartclass/smartclass.h"

/*
响应模式
    0：连接和监听都是LT
    1：连接ET，监听LT
    2：连接LT，监听ET
    3：连接和监听都是ET
日志等级
    0：DEBUG
    1：INFO
    2：WARN
    3：ERROR
*/

void start_server()
{
    
    Resource::instance()->init();
    WebServer server(
        9006, 3, 60000, false,
        3306, "your_sqlUser", "your_sqlPwd", "your_dbName",
        12, 6, 10000, true, 1, 1024);
    server.start();
}

int main() {
    std::thread(start_server).detach();

    SmartClass* smartclass = SmartClass::instance();
    smartclass->init();
    smartclass->run();
}
