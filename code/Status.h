#ifndef _STATUS_H_
#define _STATUS_H_

#include "smartclass/smartclass.h"

class StatusMachine
{
    enum Status{RUNNING, PAUSE, DUMMY, CONFIGDONE };
    Status status;

    StatusMachine():status(DUMMY){}
    ~StatusMachine(){}

public:
     static StatusMachine* instance(){
        static StatusMachine m;
        return &m;
    }

    void start()
    {
        if(status == CONFIGDONE || status == PAUSE){
            SmartClass::instance()->start();
            status = RUNNING;
            std::cout << "status : running" << std::endl;
        }
    }

    void config(std::string& code)
    {
        if(status == DUMMY){
            SmartClass::instance()->config(code);
            status = CONFIGDONE;
            std::cout << "status : configdone" << std::endl;
        }
    }

    void dummy()
    {
        if(status == PAUSE ||status == RUNNING || status == CONFIGDONE){
            SmartClass::instance()->dummy();
            status = DUMMY;
            std::cout << "status : dummy" << std::endl;
        }
    }

    void pause()
    {
        if(status == RUNNING){
            SmartClass::instance()->pause();
            status = PAUSE;
            std::cout << "status : pause" << std::endl;
        }
    }
};

#endif
