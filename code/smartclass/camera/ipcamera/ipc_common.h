#include <iostream>
#include "onvif/stdsoap2.h"
//
// Created by allforgot on 2020/3/28.
//

#ifndef ONVIFCLIENT_GSOAP_COMMON_H
#define ONVIFCLIENT_GSOAP_COMMON_H

#define USERNAME "admin"
#define PASSWORD "cvlab123456"
#define HOSTNAME "http://192.168.1.66/onvif/device_service"
#define PROFILETOKEN "Profile_1"


inline void CLEANUP_SOAP(struct soap* soap) {
    if (soap != nullptr) {
        soap_destroy(soap);
        soap_end(soap);
        soap_free(soap);
    }
}

inline void report_error(struct soap *soap) {
    std::cout << "Oops, something went wrong:" << std::endl;
    soap_stream_fault(soap, std::cerr);
    //xit(EXIT_FAILURE);
}


#endif //ONVIFCLIENT_GSOAP_COMMON_H
