#include "httprequest.h"
#include "../../Resource.h"
#include "../../Status.h"

using namespace std;

const char CRLF[] = "\r\n";

const unordered_set<string> HttpRequest::DEFAULT_HTML
{
    "/index"
};

const unordered_map<string, int> HttpRequest::DEFAULT_HTML_TAG
{
    {"/register",0},
    {"/login", 1},
    {"/classcode", 2},
    {"/resetcode", 3},
    {"/pano",4},
    {"/cmd",5}
};

void HttpRequest::init()
{
    method = path = version = body = "";
    linger = false;
    contentLen = 0;
    state = REQUEST_LINE;
    header.clear();
    post.clear();
}

HTTP_CODE HttpRequest::parse(Buffer& buffer)
{
    // string content(buffer.peek(), buffer.beginWriteConst());
    // cout << content;
    while (buffer.readableBytes())
    {
        const char* lineEnd;
        string line;
        // 除了消息体外，逐行解析
        if (state != BODY)
        {
            //search，找到返回第一个字符串下标，找不到返回最后一下标
            lineEnd = search(buffer.peek(), buffer.beginWriteConst(), CRLF, CRLF + 2);
            //如果没找到CRLF，也不是BODY，那么一定不完整
            if (lineEnd == buffer.beginWrite()) return NO_REQUEST;
            line = string(buffer.peek(), lineEnd);
            buffer.retrieveUntil(lineEnd + 2); // 除消息体外，都有换行符
        }
        else
        {
            // 消息体读取全部内容，同时清空缓存
            body += buffer.retrieveAllToStr();
            if (body.size() < contentLen)
            {
                return NO_REQUEST;
            }
        }

        switch(state)
        {
        case REQUEST_LINE:
        {
            HTTP_CODE ret = parseRequestLine(line);
            if (ret == BAD_REQUEST)
            {
                return BAD_REQUEST;
            }
            parsePath();
            break;
        }
        case HEADERS:
        {
            HTTP_CODE ret = parseHeader(line);            
            //内部根据content-length字段判断请求完整，提前结束
            if (ret == GET_REQUEST)
            {
                return GET_REQUEST;
            }
            break;
        }
        case BODY:
        {
            HTTP_CODE ret = parseBody();
            if (ret == GET_REQUEST)
            {
                return GET_REQUEST;
            }
            break;
        }
        default:
            break;
        }
    }
    LOG_DEBUG("state: %d", (int)state);
    LOG_DEBUG("content length: %d", contentLen);
    LOG_DEBUG("[%s], [%s], [%s]", method.c_str(), path.c_str(), version.c_str());
    //缓存读空了，但请求还不完整，继续读
    return NO_REQUEST;
}

/* 解析地址 */
void HttpRequest::parsePath()
{
    if (path == "/") path = "/index.html";
    else if (DEFAULT_HTML.count(path))
        path += ".html";
}

/* 解析请求行 */
HTTP_CODE HttpRequest::parseRequestLine(const string& line)
{
    regex pattern("^([^ ]*) ([^ ]*) HTTP/([^ ]*)$");
    smatch subMatch;
    if (regex_match(line, subMatch, pattern))
    {
        method = subMatch[1];
        path = subMatch[2];
        std::cout << path << std::endl;
        version = subMatch[3];
        state = HEADERS;
        return NO_REQUEST; //request isn't completed
    }
    LOG_ERROR("RequestLine Error");
    return BAD_REQUEST;
}

/* 解析请求头 */
HTTP_CODE HttpRequest::parseHeader(const string& line)
{
    regex pattern("^([^:]*): ?(.*)$");
    smatch subMatch;
    if (regex_match(line, subMatch, pattern))
    {
        header[subMatch[1]] = subMatch[2];
        if (subMatch[1] == "Connection")
            linger = (subMatch[2] == "keep-alive");
        if (subMatch[1] == "Content-Length")
        {
            contentLen = stoi(subMatch[2]);
        }
        return NO_REQUEST;
    }
    else if (contentLen)
    {
        state = BODY;
        return NO_REQUEST;
    }
    else
    {
        return GET_REQUEST;
    }
}

/* 解析请求消息体，根据消息类型解析内容 */
HTTP_CODE HttpRequest::parseBody()
{
    //key-value
    if (method == "POST" && header["Content-Type"] == "application/x-www-form-urlencoded")
    {
        parseFromUrlEncoded(); //解析post请求数据
        if (DEFAULT_HTML_TAG.count(path))
        {
            //tag=1:login,tag=0:register
            int tag = DEFAULT_HTML_TAG.find(path)->second;
            LOG_DEBUG("Tag:%d", tag);
            if(tag == 0 || tag == 1){
                if (userVerify(post["username"], post["password"], tag))
                {
                    LOG_INFO("success!");
                    path = "/Response/True.txt";
                }
                else
                {
                    LOG_INFO("failed!");
                    path = "/Response/False.txt";
                }
            }

            if(tag == 2)//set classcode
            {
                std::cout << "Set classcode : "<<post["classcode"] << std::endl;
                StatusMachine::instance()->config(post["classcode"]);
                path = "/Response/True.txt";

            }

            if(tag == 3)//reset classcode
            {
                std::cout <<"Reset classcode !" << std::endl;
                StatusMachine::instance()->dummy();
                path = "/Response/True.txt";
            }

            if(tag == 4) //pano
            {
                int idx = atoi(post["idx"].c_str());// client current pano idx
                //std::cout << idx << std::endl;
                int idx_resrc = Resource::instance()->idx;
                if(0 <= idx && idx < idx_resrc){//legal
                    std::cout << "send pano" << std::endl;
                    path = "/pano_resrc";
                }else{
                    //std::cout << "send empty" << std::endl;
                    path = "/pano_empty";
                }
            }

            if(tag == 5)//cmd
            {
                std::cout << post["cmd"] << std::endl;
                string cmd = post["cmd"];
                if(cmd == std::string("start")){
                    StatusMachine::instance()->start();
                }
                if(cmd == std::string("stop")){
                    StatusMachine::instance()->pause();
                }

                path = "/Response/True.txt";
            }
        }
    }
    LOG_DEBUG("Body:%s len:%d", body.c_str(), body.size());
    return GET_REQUEST;
}

/* 十六进制转十进制 */
int HttpRequest::convertHex(char ch)
{
    if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
    if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
    return ch;
}

/* 解析urlEncoded类型数据 */
void HttpRequest::parseFromUrlEncoded()
{
    if (body.size() == 0) return;

    string key, value;
    int num = 0;
    int n = body.size();
    //key=value&key=value
    //%:hex,+:blank
    //i:right,j:left
    int i = 0, j = 0;
    for (; i < n; i ++)
    {
        char ch = body[i];
        switch (ch)
        {
        case '=':
            key = body.substr(j, i - j);
            j = i + 1;
            break;
        case '+':
            body[i] = ' ';
            break;
        case '%':
            num = convertHex(body[i + 1] * 16 + convertHex(body[i + 2]));
            body[i + 2] = num % 10 + '0';
            body[i + 1] = num / 10 + '0';
            i += 2;
            break;
        case '&':
            value = body.substr(j, i - j);
            j = i + 1;
            post[key] = value;
            //std::cout << key << ":" << value << std::endl;
        default:
            break;
        }
    }
    assert(j <= i);
    //last value
    if (!post.count(key) && j < i)
    {
        value = body.substr(j, i - j);
        post[key] = value;
        //std::cout << key << ":" << value << std::endl;
    }
}

bool HttpRequest::userVerify(const string &name, const string &pwd, bool isLogin) {
    if(name == "" || pwd == "") { return false; }
    LOG_DEBUG("Verify name:%s pwd:%s", name.c_str(), pwd.c_str());
    MYSQL* sql;
    SqlConnect(&sql, SqlConnPool::instance());
    assert(sql);
    bool flag = false;

    char order[256] = { 0 };
    MYSQL_RES *res = nullptr;
    
    if(!isLogin) { flag = true; }
    /* 查找用户是否存在 */
    snprintf(order, 256, "SELECT username, passwd FROM user WHERE username='%s' LIMIT 1", name.c_str());
    LOG_DEBUG("%s", order);

    //失败返回非0,成功返回0
    //失败释放结果集
    if(mysql_query(sql, order)) { 
        mysql_free_result(res);
        return false; 
    }
    //保存结果集
    res = mysql_store_result(sql);

    //遍历结果集，如果为空，那么判断为假
    //这里可以写if
    while(MYSQL_ROW row = mysql_fetch_row(res)) {
        LOG_DEBUG("MYSQL ROW: %s %s", row[0], row[1]);
        string password(row[1]);
        if(isLogin) {
            if(pwd == password) { flag = true; }
            else {
                flag = false;
                LOG_DEBUG("pwd error!");
            }
        } 
        else { 
            flag = false; 
            LOG_DEBUG("user used!");
        }
    }
    //释放结果集
    mysql_free_result(res);

    /* 用户名未被使用，继续注册 */
    if(!isLogin && flag == true) {
        LOG_DEBUG("regirster!");
        bzero(order, 256);
        snprintf(order, 256,"INSERT INTO user(username, passwd) VALUES('%s','%s')", name.c_str(), pwd.c_str());
        LOG_DEBUG( "%s", order);
        if(mysql_query(sql, order)) { 
            LOG_DEBUG( "Insert error!");
            flag = false; 
        }
        else flag = true;
    }
    //RAII机制，不需要手动释放连接
    //SqlConnPool::instance()->releaseConn(sql);
    LOG_DEBUG( "UserVerify success!!");
    return flag;
}

bool HttpRequest:: isKeepAlive() const
{
    if (header.count("Connection"))
        return linger;
    return false;
}
