#ifndef _FACE_DATABASE_H_
#define _FACE_DATABASE_H_

#include <string>

void face_database_make(const char* pics_folder_dir,
                        int person_num, int  pose_num, const char* pic_type = "jpg");
void face_database_check(float threshold = 0.4f);
//void face_database_get(const char* path, float* buffer,
                       //int person_num, int pose_num);

class FaceDB
{
public:
    FaceDB():data(nullptr), classcode("null"), person(0), pose(0){}
    ~FaceDB();
    FaceDB& operator=(const FaceDB& db);

    float* data;
    std::string classcode;
    int person;
    int pose;

    void load(const std::string& path, std::string code, int person_num, int pose_num);
    void clear();
    bool empty(){
        return data == nullptr;
    }
};

#endif
