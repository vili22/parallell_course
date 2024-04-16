#!/bin/bash
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
unzip opencv.zip
unzip opencv_contrib.zip

mkdir build

cmake -DOPENCV_EXTRA_MODULES_PATH=opencv_contrib-4.x/modules -DCMAKE_INSTALL_PREFIX=/home/vvirkkal/Libraries/opencv -S opencv-4.x -B build
cmake --build build
cmake --install build

rm -r build
rm -r opencv-4.x
rm -r opencv_contrib-4.x
rm opencv.zip
rm opencv_contrib.zip
