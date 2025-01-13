#ifndef _CHRONO_TIMER_H_
#define _CHRONO_TIMER_H_

#include <chrono>
#include <iostream>

class ChronoTimer
{
	std::chrono::time_point<std::chrono::system_clock> m_start;
	std::chrono::time_point<std::chrono::system_clock> m_end;

public:
	// description: start time_point 
	void start(){
            m_start = std::chrono::system_clock::now();
	}

	// description: end time_point 
	void end(){
            m_end = std::chrono::system_clock::now();
	}
	
        void end_print(std::string func_name){
            m_end = std::chrono::system_clock::now();
            std::cout << func_name << " :";
            print_elapse();
        }

	// description: print the elapsed time
	// param time_type: 0 seconds ,1 milliseconds,2 microseconds
	//
	void print_elapse(int time_type = 1){
		switch(time_type){
			case 0: std::cout << std::chrono::duration_cast<std::chrono::seconds>(m_end - m_start).count() << "s" << std::endl; break;
			case 1: std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(m_end - m_start).count() << "ms" << std::endl;break;
			case 2: std::cout << std::chrono::duration_cast<std::chrono::microseconds>(m_end - m_start).count() << "us" << std::endl;break;
			default: //std::cout << "wrong time type! print in milliseconds"<< std::endl;
					std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(m_end - m_start).count() << "ms" << std::endl;break;
		}//switch end		
	}
};

#endif
