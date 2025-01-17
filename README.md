# Namemo

NaMemo provides teachers with a panoramic view of the classroom, enabling them to click on face boxes to access students’ information. 
This system aims to support teacher-student interaction by facilitating teachers in recognizing students by name.

This repository contains the source code and data used for the paper:

**"Enhancing Teacher-Student Interaction: A Vision-based Name-indicating System Using Advanced Image Processing Techniques"**

## Prerequisites
Ubuntu 20.04

OpenCV 4.4.0

CUDA-10.2 + cuDNN-8.0

Python 3.6.5

Mysql

Onvif https://github.com/Allforgot/OnvifClient_gSOAP/blob/master/GENERATE_ONVIF_CODE.md

CMake

## Getting Started

### Preparation
1. Add the corresponding database in MySQL, where the database stores the login username and password.
2. Collect face images and student information. The face images include one frontal photo and one 45° side photo. Student information includes name, student ID, and willingness to interact (WtoI).
3. Extract face descriptors from the face images and store the descriptors along with the student information in the database.
4. Set up PTCamera network connection, configure LAN and WLAN IPs, and ensure it is connected to the edge computing platform in advance.

### Installation
1.Clone the repo
```bash
git clone https://github.com/EmxCaesar/Namemo.git
```
2.mkdir and compile
```bash
mkdir build
cd build
cmake ..
make
```
3.run the program
```bash
cd ..
./build/webserver
```

### Explanation


The code consists of two main components: **WebServer** and **SmartClass**. The **SmartClass** component includes functionalities such as camera control, image stitching, and face recognition. The **WebServer** component waits for client requests, and upon receiving a request, it coordinates with the **SmartClass** component to complete the tasks, generate a panoramic image (pano), and send it back to the client.
