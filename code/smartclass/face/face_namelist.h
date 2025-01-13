#ifndef _FACE_NAMELIST_H_
#define _FACE_NAMELIST_H_

#include <mutex>

// store fixed size data, for transmission
class NameInfo
{
public:
    static const int m_max_name_len = 32;
    static const int m_max_id_len = 32;
    static const int m_unkonw_id = 999;

    int m_index;
    char m_id[m_max_id_len];
    char m_name[m_max_name_len];
    int m_will;

    NameInfo();
    NameInfo(int index, const char* id, const char* name,int will);
};

class NameList
{
public:
    NameList();
    ~NameList();

    NameList& operator=(const NameList& namelist);

    std::string classcode;
    NameInfo* data;
    int length(){ return size; }

    int load(const std::string& namelist_path, std::string code);
    void clear();
    bool empty(){
        return data == nullptr;
    }

    void lock(){ mtx.lock();}
    void unlock(){mtx.unlock();}

private:
    std::mutex mtx;
    int size;

};

#endif
