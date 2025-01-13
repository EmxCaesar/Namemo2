#include  "face_namelist.h"
#include <iostream>
#include <cstring>
#include <fstream>

#define NAME_LOG 1

const int NameInfo::m_max_name_len;
const int NameInfo::m_max_id_len;
const int NameInfo::m_unkonw_id;

NameInfo::NameInfo()
{
    m_index = 0;
    memset(m_id, 0 ,sizeof(m_id));
    memset(m_name, 0 ,sizeof(m_name));
    m_will = 0;
}

NameInfo::NameInfo(int index, const char* id, const char* name, int will):
    m_index(index), m_will(will)
{
    memset(m_id, 0 ,sizeof(m_id));
    memset(m_name, 0 ,sizeof(m_name));

    int id_len = strlen(id);
    if(id_len < m_max_id_len){
        strcpy(m_id, id);
    }else{
        memcpy(m_id, id, sizeof(m_id));
        m_id[m_max_id_len-1] = 0;
    }

    int name_len = strlen(name);
    if(name_len < m_max_name_len){
        strcpy(m_name, name);
    }else{
        memcpy(m_name, name, sizeof(m_name));
        m_name[m_max_name_len-1] = 0;
    }
}


NameList::NameList()
{
    classcode = "null";
    size = 0;
    data = nullptr;
}

NameList::~NameList()
{
    if(data != nullptr)
        delete [] data;
}

NameList& NameList::operator=(const NameList& namelist)
{
    classcode = namelist.classcode;
    size = namelist.size;
    data = new NameInfo[size];
    std::copy(namelist.data, namelist.data+size, data);

    return *this;
}

int NameList::load(const std::string& namelist_path, std::string code)
{
    lock();
    classcode = code;

    std::ifstream namelist_file(namelist_path);
    int namelist_len;
    namelist_file >> namelist_len;
    size = namelist_len;

#if NAME_LOG
    std::cout << std::endl;
    std::cout << "NameList : length " << namelist_len << std::endl;
#endif

    if(data != nullptr)
        delete [] data;
   data = new NameInfo[namelist_len];

    for(int i = 0; i< namelist_len; ++i)
    {
        int index;
        int will;
        std::string id_code;
        std::string name_str;
        std::string alias_str;

        namelist_file >> index;
        namelist_file >> id_code;
        namelist_file >> name_str;
        namelist_file >> alias_str;
        namelist_file >> will;

#if NAME_LOG
        std::cout << index<<" "<<id_code <<" "<<name_str <<" "<<alias_str << " "<<will <<std::endl;
#endif

        NameInfo tempNamePair(index,id_code.c_str(), name_str.c_str(), will);
        data[i] = tempNamePair;
    }

    unlock();
    printf("NameList : load namelist success\n");
    return namelist_len;
}

void NameList::clear()
{
    lock();

    if(data != nullptr){
        delete [] data;
        data = nullptr;
        size = 0;
        classcode = "null";
    }

    unlock();
}


